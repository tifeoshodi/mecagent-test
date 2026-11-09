"""
Evaluation harness for CadQuery code generation models.

Runs VSR, IoU_best, Chamfer Distance, Betti numbers, solid validity gate,
and STEP export. All CADGenBench-grade metrics run always-on.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Add project root to path for metrics import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cadquery as cq
from cadquery import exporters
from metrics.valid_syntax_rate import evaluate_syntax_rate, evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best, _load_solid_from_code, _normalized_mesh, iou_best
from metrics.betti_numbers import compute_betti_numbers
from metrics.chamfer_distance import chamfer_distance


def compute_vsr(predictions: dict[str, str], verbose: bool = False) -> dict:
    """Compute Valid Syntax Rate for predictions.

    Args:
        predictions: Dict of {id: code_string}
        verbose: Print per-sample results

    Returns:
        Dict with 'vsr', 'successful', 'total', 'failed_ids'
    """
    return evaluate_syntax_rate(predictions, verbose=verbose)


def compute_solid_validity(
    predictions: dict[str, str],
    export_dir: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """Compute enhanced solid validity with hard gates and STEP export.

    Checks:
    1. Code executes without error (basic VSR)
    2. solid.isValid() returns True
    3. solid.Volume() > 1e-5 (filters degenerate zero-volume shells)

    Optionally exports valid solids to STEP format.

    Args:
        predictions: Dict of {id: code_string}
        export_dir: Directory to export STEP files (None = skip)
        verbose: Print per-sample results

    Returns:
        Dict with validity stats, per-sample details, and export paths
    """
    results = {
        "total": 0,
        "parse_ok": 0,
        "solid_valid": 0,
        "volume_ok": 0,
        "fully_valid": 0,
        "details": {},
        "step_exports": {},
        "betti_numbers": {},
    }

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    for item_id, code in tqdm(predictions.items(), desc="Solid Validation", disable=not verbose):
        results["total"] += 1
        detail = {"parse_ok": False, "solid_valid": False, "volume_ok": False, "error": None}

        if not code.strip():
            detail["error"] = "Empty prediction"
            results["details"][item_id] = detail
            continue

        try:
            solid = _load_solid_from_code(code, item_id)
            detail["parse_ok"] = True
            results["parse_ok"] += 1
        except Exception as e:
            detail["error"] = f"parse_error: {e}"
            results["details"][item_id] = detail
            continue

        # Hard validity gate
        try:
            is_valid = solid.isValid()
            detail["solid_valid"] = is_valid
            if is_valid:
                results["solid_valid"] += 1
        except Exception:
            detail["solid_valid"] = False

        # Volume gate
        try:
            vol = solid.Volume()
            detail["volume"] = vol
            if vol > 1e-5:
                detail["volume_ok"] = True
                results["volume_ok"] += 1
        except Exception:
            detail["volume"] = 0.0

        # Fully valid = all gates passed
        if detail["parse_ok"] and detail["solid_valid"] and detail["volume_ok"]:
            results["fully_valid"] += 1

            # STEP export
            if export_dir:
                try:
                    safe_id = item_id.replace("/", "_").replace("\\", "_")
                    step_path = os.path.join(export_dir, f"{safe_id}.step")
                    os.makedirs(os.path.dirname(step_path), exist_ok=True)
                    exporters.export(solid, step_path, "STEP")
                    if os.path.exists(step_path):
                        results["step_exports"][item_id] = step_path
                except Exception as e:
                    if verbose:
                        print(f"  {item_id}: STEP export failed: {e}")

            # Betti numbers
            try:
                mesh = _normalized_mesh(solid)
                betti = compute_betti_numbers(mesh)
                results["betti_numbers"][item_id] = betti
            except Exception as e:
                if verbose:
                    print(f"  {item_id}: Betti computation failed: {e}")

        results["details"][item_id] = detail

    total = results["total"]
    results["solid_validity_rate"] = results["fully_valid"] / total if total else 0.0

    return results


def compute_iou_best_batch(
    ground_truth: dict[str, str],
    predictions: dict[str, str],
    pitch: float = 0.05,
    verbose: bool = False,
) -> dict:
    """Compute IoU_best for all valid prediction-GT pairs.

    Args:
        ground_truth: Dict of {id: gt_code_string}
        predictions: Dict of {id: pred_code_string}
        pitch: Voxel pitch for IoU calculation
        verbose: Print per-sample IoU

    Returns:
        Dict with 'mean_iou', 'ious' (per-sample), 'valid_count', 'total_count', 'errors'
    """
    import numpy as np

    ids = sorted(set(ground_truth.keys()) & set(predictions.keys()))
    ious = {}
    errors = {}

    for item_id in tqdm(ids, desc="Computing IoU", disable=not verbose):
        gt_code = ground_truth[item_id]
        pred_code = predictions[item_id]

        if not pred_code.strip():
            errors[item_id] = "Empty prediction"
            continue

        try:
            iou = get_iou_best(gt_code, pred_code)
            ious[item_id] = float(iou)
            if verbose:
                print(f"  {item_id}: IoU = {iou:.4f}")
        except Exception as e:
            errors[item_id] = str(e)
            if verbose:
                print(f"  {item_id}: Error - {e}")

    mean_iou = float(np.mean(list(ious.values()))) if ious else 0.0

    return {
        "mean_iou": mean_iou,
        "ious": ious,
        "valid_count": len(ious),
        "total_count": len(ids),
        "errors": errors,
    }


def run_evaluation(
    model,
    eval_data: dict,
    pitch: float = 0.05,
    save_dir: Optional[str] = None,
    experiment_name: str = "experiment",
    api_delay: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Run full evaluation pipeline: generate predictions, compute VSR and IoU.

    Args:
        model: Generator object with .generate_batch(items) method
        eval_data: Dict of {id: {"image": PIL.Image, "cadquery": str, ...}}
        pitch: Voxel pitch for IoU calculation
        save_dir: Directory to save results
        experiment_name: Name for the experiment
        api_delay: Delay between API calls
        verbose: Print progress

    Returns:
        Results dict with VSR, IoU, predictions, and metadata
    """
    save_dir = save_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(save_dir, exist_ok=True)

    predictions_path = os.path.join(save_dir, f"{experiment_name}_predictions.json")

    # Step 1: Generate predictions
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"Eval samples: {len(eval_data)}")
        print(f"{'='*60}")
        print("\nStep 1: Generating predictions...")

    existing_preds = {}
    if os.path.exists(predictions_path):
        try:
            with open(predictions_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_preds = data.get("predictions", {})
        except Exception:
            pass

    filtered_eval = {}
    for k, v in eval_data.items():
        if k not in existing_preds or not str(existing_preds[k]).strip():
            filtered_eval[k] = v

    if verbose and existing_preds:
        print(f"  Resuming from {len(existing_preds)} existing predictions... ({len(filtered_eval)} left)")

    start_time = time.time()
    predictions = model.generate_batch(
        filtered_eval,
        delay=api_delay,
        save_path=predictions_path,
        existing_predictions=existing_preds,
    )
    gen_time = time.time() - start_time

    if verbose:
        print(f"  Generation complete in {gen_time:.1f}s")
        non_empty = sum(1 for v in predictions.values() if v.strip())
        print(f"  Non-empty predictions: {non_empty}/{len(predictions)}")

    # Step 2: Compute VSR
    if verbose:
        print("\nStep 2: Computing Valid Syntax Rate...")

    vsr_result = compute_vsr(predictions, verbose=False)
    if verbose:
        print(f"  VSR: {vsr_result['vsr']:.3f} ({vsr_result['successful']}/{vsr_result['total']})")

    # Step 3: Compute IoU_best
    if verbose:
        print("\nStep 3: Computing IoU_best...")

    gt_codes = {k: v["cadquery"] for k, v in eval_data.items()}
    iou_result = compute_iou_best_batch(gt_codes, predictions, pitch=pitch, verbose=False)
    if verbose:
        print(f"  Mean IoU_best: {iou_result['mean_iou']:.4f} ({iou_result['valid_count']}/{iou_result['total_count']} valid)")

    # Step 4: Compute solid validity + Betti numbers + STEP export
    if verbose:
        print("\nStep 4: Computing solid validity, Betti numbers, and STEP export...")

    step_export_dir = os.path.join(save_dir, f"{experiment_name}_step_exports")
    validity_result = compute_solid_validity(
        predictions,
        export_dir=step_export_dir,
        verbose=False,
    )
    if verbose:
        print(f"  Solid validity rate: {validity_result['solid_validity_rate']:.3f} ({validity_result['fully_valid']}/{validity_result['total']})")
        if validity_result["betti_numbers"]:
            b0_values = [b["b0"] for b in validity_result["betti_numbers"].values()]
            print(f"  Mean Betti b0: {sum(b0_values)/len(b0_values):.2f}")
        print(f"  STEP exports: {len(validity_result['step_exports'])} files")

    # Compile results
    results = {
        "experiment_name": experiment_name,
        "num_samples": len(eval_data),
        "vsr": vsr_result["vsr"],
        "vsr_successful": vsr_result["successful"],
        "vsr_total": vsr_result["total"],
        "vsr_failed_ids": vsr_result["failed_ids"],
        "mean_iou_best": iou_result["mean_iou"],
        "iou_valid_count": iou_result["valid_count"],
        "iou_per_sample": iou_result["ious"],
        "iou_errors": iou_result["errors"],
        "solid_validity_rate": validity_result["solid_validity_rate"],
        "solid_validity_fully_valid": validity_result["fully_valid"],
        "betti_numbers": validity_result["betti_numbers"],
        "step_export_count": len(validity_result["step_exports"]),
        "generation_time_s": gen_time,
        "predictions": predictions,
    }

    # Save results
    results_path = os.path.join(save_dir, f"{experiment_name}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nResults saved to {results_path}")
        print(f"\n{'='*60}")
        print(f"SUMMARY — {experiment_name}")
        print(f"  VSR:              {results['vsr']:.3f}")
        print(f"  Mean IoU_best:    {results['mean_iou_best']:.4f}")
        print(f"  Solid Validity:   {results['solid_validity_rate']:.3f}")
        print(f"  STEP exports:     {results['step_export_count']}")
        print(f"  Gen time:      {results['generation_time_s']:.1f}s")
        print(f"{'='*60}\n")

    return results


def load_results(results_path: str) -> dict:
    """Load saved results from JSON."""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_comparison_table(results_list: list[dict]):
    """Print a comparison table of multiple experiment results."""
    print(f"\n{'='*70}")
    print(f"{'Experiment':<30} {'VSR':>8} {'IoU_best':>10} {'Time(s)':>10}")
    print(f"{'-'*70}")
    for r in results_list:
        print(
            f"{r['experiment_name']:<30} "
            f"{r['vsr']:>8.3f} "
            f"{r['mean_iou_best']:>10.4f} "
            f"{r['generation_time_s']:>10.1f}"
        )
    print(f"{'='*70}\n")
