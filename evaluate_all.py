"""
Re-evaluate ALL existing prediction files to produce accurate, consistent results.

Handles:
- Loading each prediction JSON
- Filtering truncated predictions (incomplete code)
- Computing VSR and IoU_best through the standard metrics
- Writing proper results files
- Rebuilding experiment_summary.json

Usage:
    python evaluate_all.py
    python evaluate_all.py --skip-iou   # Fast mode: VSR only
"""

import os
import sys
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)

from tqdm import tqdm
from src.data_loader import get_eval_subset
from metrics.valid_syntax_rate import evaluate_syntax_rate
from metrics.best_iou import get_iou_best

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Map of experiment name -> prediction file
EXPERIMENTS = {
    "baseline_zero_shot": "baseline_zero_shot_predictions.json",
    "enhanced_rag": "enhanced_rag_predictions.json",
    "enhanced_visual": "enhanced_visual_predictions.json",
    "mcts": "mcts_predictions.json",
}


def is_truncated(code: str) -> bool:
    """Detect if a prediction was truncated by the VLM (incomplete code)."""
    if not code or not code.strip():
        return True

    stripped = code.rstrip()

    # Ends mid-assignment (e.g., "full_height =")
    if re.search(r'=\s*$', stripped):
        return True

    # Ends with an incomplete string literal
    if stripped.endswith(('\\', '\\"', "\\'")):
        return True

    # Ends mid-line with a comma suggesting more was coming
    if stripped.endswith(',') and stripped.count('(') > stripped.count(')'):
        return True

    # Unbalanced parentheses strongly suggest truncation
    open_parens = stripped.count('(') - stripped.count(')')
    if open_parens > 2:  # Small imbalance might be intentional, large is truncation
        return True

    return False


def compute_iou_single(args):
    """Compute IoU for a single sample in a subprocess for crash isolation."""
    import subprocess
    item_id, gt_code, pred_code, project_root = args

    if not pred_code.strip():
        return item_id, None, "Empty prediction"

    script = f"""
import sys
import os
sys.path.insert(0, {repr(project_root)})
from metrics.best_iou import get_iou_best

gt = {repr(gt_code)}
pred = {repr(pred_code)}

try:
    iou = get_iou_best(gt, pred)
    print(f"IOU_RESULT={{iou}}")
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    try:
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60
        )
        for line in res.stdout.splitlines():
            if line.startswith("IOU_RESULT="):
                return item_id, float(line.split("=")[1]), None
        return item_id, None, res.stderr.strip() or "No IOU result produced"
    except subprocess.TimeoutExpired:
        return item_id, None, "Timeout (60s)"


def evaluate_experiment(
    experiment_name: str,
    predictions: dict,
    gt_codes: dict,
    skip_iou: bool = False,
) -> dict:
    """Evaluate a single experiment's predictions."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_name}")
    print(f"{'='*60}")

    total = len(predictions)
    non_empty = sum(1 for v in predictions.values() if v.strip())
    truncated_count = sum(1 for v in predictions.values() if is_truncated(v))

    print(f"  Total predictions: {total}")
    print(f"  Non-empty: {non_empty}")
    print(f"  Truncated: {truncated_count}")

    # --- VSR ---
    print("\n  Computing VSR...")
    start = time.time()
    vsr_result = evaluate_syntax_rate(predictions, verbose=False)
    vsr_time = time.time() - start
    print(f"  VSR: {vsr_result['vsr']:.3f} ({vsr_result['successful']}/{vsr_result['total']}) [{vsr_time:.1f}s]")

    # --- IoU ---
    iou_result = {"mean_iou": 0.0, "ious": {}, "valid_count": 0, "total_count": 0, "errors": {}}

    if not skip_iou:
        print("\n  Computing IoU_best (subprocess-isolated, parallel)...")
        ids = sorted(set(gt_codes.keys()) & set(predictions.keys()))
        iou_result["total_count"] = len(ids)

        project_root = PROJECT_ROOT.replace("\\", "/")
        args_list = [
            (item_id, gt_codes[item_id], predictions[item_id], project_root)
            for item_id in ids
            if predictions[item_id].strip() and not is_truncated(predictions[item_id])
        ]

        skipped = len(ids) - len(args_list)
        if skipped > 0:
            print(f"  Skipping {skipped} empty/truncated predictions for IoU")

        ious = {}
        errors = {}

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(compute_iou_single, arg): arg[0] for arg in args_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="  IoU"):
                item_id, iou_val, error = future.result()
                if iou_val is not None:
                    ious[item_id] = iou_val
                elif error:
                    errors[item_id] = error

        mean_iou = sum(ious.values()) / len(ious) if ious else 0.0
        iou_result = {
            "mean_iou": mean_iou,
            "ious": ious,
            "valid_count": len(ious),
            "total_count": len(ids),
            "errors": errors,
        }
        print(f"  Mean IoU_best: {mean_iou:.4f} ({len(ious)}/{len(ids)} valid)")

    # --- Compile results ---
    results = {
        "experiment_name": experiment_name,
        "num_samples": total,
        "num_non_empty": non_empty,
        "num_truncated": truncated_count,
        "vsr": vsr_result["vsr"],
        "vsr_successful": vsr_result["successful"],
        "vsr_total": vsr_result["total"],
        "vsr_failed_ids": vsr_result["failed_ids"],
        "mean_iou_best": iou_result["mean_iou"],
        "iou_valid_count": iou_result["valid_count"],
        "iou_total_count": iou_result["total_count"],
        "iou_per_sample": iou_result.get("ious", {}),
        "iou_errors": iou_result.get("errors", {}),
    }

    # Save results
    results_path = os.path.join(RESULTS_DIR, f"{experiment_name}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {results_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Re-evaluate all prediction files")
    parser.add_argument("--skip-iou", action="store_true", help="Skip IoU computation (VSR only)")
    parser.add_argument("--experiment", type=str, default=None, help="Evaluate only this experiment")
    args = parser.parse_args()

    # Load ground truth
    print("Loading ground truth (hundred_subset)...")
    eval_data_hundred = get_eval_subset(split="test", use_hundred=True)
    gt_codes_hundred = {k: v["cadquery"] for k, v in eval_data_hundred.items()}
    print(f"  Loaded {len(gt_codes_hundred)} ground truth samples (hundred_subset)")

    # Also load full test set for experiments that may not use hundred_subset
    print("Loading ground truth (full test set)...")
    eval_data_full = get_eval_subset(split="test", use_hundred=False)
    gt_codes_full = {k: v["cadquery"] for k, v in eval_data_full.items()}
    print(f"  Loaded {len(gt_codes_full)} ground truth samples (full)")

    experiments_to_run = EXPERIMENTS
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        experiments_to_run = {args.experiment: EXPERIMENTS[args.experiment]}

    all_results = []

    for exp_name, pred_file in experiments_to_run.items():
        pred_path = os.path.join(RESULTS_DIR, pred_file)
        if not os.path.exists(pred_path):
            print(f"\n  ⚠ Skipping {exp_name}: {pred_file} not found")
            continue

        # Load predictions
        with open(pred_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            predictions = data.get("predictions", data)

        # Choose the right GT set
        # If predictions keys overlap more with hundred_subset, use that
        hundred_overlap = len(set(predictions.keys()) & set(gt_codes_hundred.keys()))
        full_only_overlap = len(set(predictions.keys()) - set(gt_codes_hundred.keys()))

        if full_only_overlap > 0:
            gt_codes = gt_codes_full
            print(f"  Using full test set GT ({full_only_overlap} predictions outside hundred_subset)")
        else:
            gt_codes = gt_codes_hundred

        results = evaluate_experiment(
            experiment_name=exp_name,
            predictions=predictions,
            gt_codes=gt_codes,
            skip_iou=args.skip_iou,
        )
        all_results.append(results)

    # Print comparison table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"{'Experiment':<25} {'VSR':>8} {'IoU_best':>10} {'Samples':>10} {'Truncated':>10}")
        print(f"{'-'*70}")
        for r in all_results:
            print(
                f"{r['experiment_name']:<25} "
                f"{r['vsr']:>8.3f} "
                f"{r['mean_iou_best']:>10.4f} "
                f"{r['num_samples']:>10} "
                f"{r['num_truncated']:>10}"
            )
        print(f"{'='*70}")

    # Rebuild experiment_summary.json
    summary = []
    for r in all_results:
        summary.append({
            "experiment": r["experiment_name"],
            "vsr": r["vsr"],
            "mean_iou_best": r["mean_iou_best"],
            "num_samples": r["num_samples"],
            "num_non_empty": r["num_non_empty"],
            "num_truncated": r["num_truncated"],
            "iou_valid_count": r["iou_valid_count"],
        })

    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nExperiment summary saved to {summary_path}")


if __name__ == "__main__":
    main()
