"""
Single sample offline CAD evaluation runner.

Evaluates a previously processed CAD sample against Ground Truth using
the CADGenBench-grade evaluation pipeline (Solid Validity Gate, STEP export,
Betti numbers, Voxel IoU_best, and Chamfer Distance) with ZERO API calls.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_eval_subset
from src.evaluate import compute_solid_validity
from metrics.best_iou import _load_solid_from_code, _normalized_mesh, iou_best
from metrics.betti_numbers import compute_betti_numbers, betti_match_score
from metrics.chamfer_distance import chamfer_distance


def evaluate_single_sample(
    sample_id: str = "0046/00461826",
    predictions_file: str = "results/enhanced_rag_predictions.json",
    export_step: bool = True,
    pitch: float = 0.05,
    eval_data: Optional[dict] = None,
    preds: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """Evaluate a single CAD prediction against ground truth offline.

    Args:
        sample_id: DeepCAD ID (e.g. '0046/00461826')
        predictions_file: Path to predictions JSON
        export_step: Whether to export STEP CAD model
        pitch: Voxel resolution for IoU
        eval_data: Optional pre-loaded evaluation dict
        preds: Optional pre-loaded predictions dict
        verbose: Whether to print detailed per-sample breakdown

    Returns:
        Dictionary of computed metrics
    """
    if verbose:
        print(f"\n{'=' * 65}")
        print(f"OFFLINE CAD EVALUATION: Sample {sample_id}")
        print(f"Predictions source: {predictions_file}")
        print(f"{'=' * 65}")

    # Load Ground Truth
    if eval_data is None:
        eval_data = get_eval_subset(use_hundred=True)
    if sample_id not in eval_data:
        raise KeyError(f"Sample '{sample_id}' not found in test evaluation subset.")
    gt_code = eval_data[sample_id]["cadquery"]

    # Load Prediction
    if preds is None:
        full_pred_path = os.path.join(PROJECT_ROOT, predictions_file) if not os.path.isabs(predictions_file) else predictions_file
        with open(full_pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f).get("predictions", {})

    if sample_id not in preds:
        raise KeyError(f"Sample '{sample_id}' not found in {predictions_file}.")
    pred_code = preds[sample_id]

    # Step 1: Solid Validity & STEP Export
    print("\n[Step 1] Solid Validity Gate & STEP Export:")
    export_dir = os.path.join(PROJECT_ROOT, "results", "offline_step_exports") if export_step else None
    validity_res = compute_solid_validity({sample_id: pred_code}, export_dir=export_dir, verbose=False)
    detail = validity_res["details"].get(sample_id, {})

    parse_ok = detail.get("parse_ok", False)
    solid_valid = detail.get("solid_valid", False)
    volume = detail.get("volume")
    volume_ok = detail.get("volume_ok", False)
    step_path = validity_res["step_exports"].get(sample_id)

    print(f"  * Code Syntax / Execution: {'PASS' if parse_ok else 'FAIL'}")
    print(f"  * Topological Validity:     {'PASS (solid.isValid() == True)' if solid_valid else 'FAIL'}")
    vol_str = f"{volume:.4f}" if volume is not None else "N/A"
    print(f"  * Solid Volume:             {vol_str} {'(PASS)' if volume_ok else '(FAIL)'}")
    if step_path:
        print(f"  * STEP Export:              {step_path} ({os.path.getsize(step_path)} bytes)")

    if not (parse_ok and solid_valid and volume_ok):
        print("\nPrediction failed solid validity gate. Alignment metrics skipped.")
        return {
            "sample_id": sample_id,
            "fully_valid": False,
            "error": detail.get("error"),
        }

    # Step 2: Betti Topological Invariants
    print("\n[Step 2] Topological Invariant Analysis (Betti Numbers):")
    solid_gt = _load_solid_from_code(gt_code, "gt")
    solid_pred = _load_solid_from_code(pred_code, "pred")

    mesh_gt = _normalized_mesh(solid_gt)
    mesh_pred = _normalized_mesh(solid_pred)

    betti_gt = compute_betti_numbers(mesh_gt)
    betti_pred = compute_betti_numbers(mesh_pred)
    b_score = betti_match_score(betti_gt, betti_pred)

    print(f"  * Ground Truth:  b0={betti_gt['b0']} (components), b1={betti_gt['b1']} (tunnels), b2={betti_gt['b2']} (voids), chi={betti_gt['euler_char']}")
    print(f"  * Predicted:     b0={betti_pred['b0']} (components), b1={betti_pred['b1']} (tunnels), b2={betti_pred['b2']} (voids), chi={betti_pred['euler_char']}")
    print(f"  * Invariant Match Score: {b_score * 100:.1f}%")

    # Step 3: 3D Alignment (IoU and Chamfer Distance)
    print("\n[Step 3] 3D Metric Evaluation:")
    iou = iou_best(mesh_gt, mesh_pred, pitch=pitch)
    cd = chamfer_distance(mesh_gt, mesh_pred, n_samples=5000)

    print(f"  * Principal-Axis Voxel IoU_best: {iou:.4f}")
    print(f"  * Surface Chamfer Distance:      {cd:.4f}")

    print(f"\n{'=' * 65}")
    print("Test run finished successfully with ZERO API credits used.")
    print(f"{'=' * 65}\n")

    return {
        "sample_id": sample_id,
        "fully_valid": True,
        "betti_gt": betti_gt,
        "betti_pred": betti_pred,
        "betti_match_score": b_score,
        "iou_best": iou,
        "chamfer_distance": cd,
        "step_path": step_path,
    }


def evaluate_batch_offline(
    max_samples: int = 10,
    predictions_file: str = "results/enhanced_rag_predictions.json",
    export_step: bool = True,
    pitch: float = 0.05,
) -> list[dict]:
    """Evaluate a batch of samples offline against ground truth."""
    full_pred_path = os.path.join(PROJECT_ROOT, predictions_file) if not os.path.isabs(predictions_file) else predictions_file
    with open(full_pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f).get("predictions", {})

    eval_data = get_eval_subset(use_hundred=True)
    common_ids = [k for k in sorted(eval_data.keys()) if k in preds and str(preds[k]).strip()]
    selected_ids = common_ids[:max_samples]

    print(f"\n{'=' * 75}")
    print(f"OFFLINE CAD BENCHMARK: {len(selected_ids)} SAMPLES")
    print(f"Predictions source: {predictions_file}")
    print(f"{'=' * 75}\n")

    results = []
    for i, sid in enumerate(selected_ids, 1):
        try:
            res = evaluate_single_sample(
                sample_id=sid,
                predictions_file=predictions_file,
                export_step=export_step,
                pitch=pitch,
                eval_data=eval_data,
                preds=preds,
                verbose=False,
            )
            results.append(res)
            if res.get("fully_valid"):
                print(f"[{i:02d}/{len(selected_ids):02d}] {sid} -> VALID   | IoU: {res['iou_best']:.4f} | CD: {res['chamfer_distance']:.4f} | Betti: {res['betti_match_score']*100:.0f}% | STEP: Yes")
            else:
                err = res.get("error", "Invalid solid")
                print(f"[{i:02d}/{len(selected_ids):02d}] {sid} -> INVALID | Reason: {err[:50]}")
        except Exception as e:
            print(f"[{i:02d}/{len(selected_ids):02d}] {sid} -> FAILED  | Error: {e}")
            results.append({"sample_id": sid, "fully_valid": False, "error": str(e)})

    # Summary Statistics
    total = len(results)
    valid_count = sum(1 for r in results if r.get("fully_valid", False))
    valid_ious = [r["iou_best"] for r in results if r.get("fully_valid", False) and "iou_best" in r]
    valid_cds = [r["chamfer_distance"] for r in results if r.get("fully_valid", False) and "chamfer_distance" in r]
    valid_bettis = [r["betti_match_score"] for r in results if r.get("fully_valid", False) and "betti_match_score" in r]

    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    mean_cd = sum(valid_cds) / len(valid_cds) if valid_cds else 0.0
    mean_betti = sum(valid_bettis) / len(valid_bettis) if valid_bettis else 0.0
    exported_count = sum(1 for r in results if r.get("step_path"))

    print(f"\n{'=' * 75}")
    print(f"CONSOLIDATED {total}-SAMPLE OFFLINE BENCHMARK RESULTS")
    print(f"{'=' * 75}\n")
    print("| # | Sample ID | Solid Valid | IoU_best | Chamfer Dist | Betti Match | STEP Export |")
    print("|:--|:----------|:------------|:---------|:-------------|:------------|:------------|")
    for idx, r in enumerate(results, 1):
        if r.get("fully_valid"):
            print(f"| {idx:02d} | `{r['sample_id']}` | PASS | {r['iou_best']:.4f} | {r['chamfer_distance']:.4f} | {r['betti_match_score']*100:.0f}% | Yes |")
        else:
            print(f"| {idx:02d} | `{r['sample_id']}` | FAIL | N/A | N/A | N/A | No |")
    
    print("\n--- Summary Metrics ---")
    print(f"Solid Validity Rate:          {valid_count}/{total} ({valid_count / total * 100:.1f}%)")
    print(f"Mean Principal-Axis IoU_best: {mean_iou:.4f} (over valid solids)")
    print(f"Mean Surface Chamfer Dist:    {mean_cd:.4f}")
    print(f"Mean Topological Match:       {mean_betti * 100:.1f}%")
    print(f"Valid STEP Models Exported:   {exported_count}")
    print(f"{'=' * 75}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Offline CAD evaluation runner")
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="DeepCAD sample ID (e.g. 0046/00461826)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate in batch (default: 10 if sample-id not set)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="results/enhanced_rag_predictions.json",
        help="Predictions JSON file (default: results/enhanced_rag_predictions.json)",
    )
    parser.add_argument(
        "--no-step",
        action="store_true",
        help="Disable STEP CAD export",
    )
    args = parser.parse_args()

    if args.sample_id:
        evaluate_single_sample(
            sample_id=args.sample_id,
            predictions_file=args.predictions,
            export_step=not args.no_step,
        )
    else:
        n = args.max_samples if args.max_samples is not None else 10
        evaluate_batch_offline(
            max_samples=n,
            predictions_file=args.predictions,
            export_step=not args.no_step,
        )


if __name__ == "__main__":
    main()

