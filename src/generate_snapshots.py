"""
Generate side-by-side visual snapshot comparisons between Ground Truth
and Model-Predicted 3D CAD geometries.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import json
import argparse
import atexit
from PIL import Image, ImageDraw

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_eval_subset
from src.visual_feedback import render_stl
from metrics.best_iou import _load_solid_from_code, _normalized_mesh
from metrics.betti_numbers import compute_betti_numbers, betti_match_score
from metrics.chamfer_distance import chamfer_distance
import cadquery as cq


def create_side_by_side_snapshot(
    sample_id: str,
    gt_image: Image.Image,
    pred_image: Image.Image,
    iou: float,
    cd: float,
    betti_match: float,
    out_path: str,
):
    """Combine GT and Predicted images into a single annotated side-by-side comparison."""
    target_size = (480, 480)
    img_gt = gt_image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    img_pred = pred_image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)

    # Canvas dimensions
    banner_height = 60
    label_height = 30
    canvas_w = 480 * 2 + 30
    canvas_h = banner_height + label_height + 480 + 20

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    # Top banner background
    draw.rectangle([0, 0, canvas_w, banner_height], fill=(30, 41, 59))
    
    # Text in banner
    title = f"Sample: {sample_id}  |  IoU: {iou:.4f}  |  Chamfer: {cd:.4f}  |  Betti Match: {betti_match*100:.0f}%"
    draw.text((20, 20), title, fill=(255, 255, 255))

    # Panel labels
    y_labels = banner_height + 8
    draw.text((20, y_labels), "[GROUND TRUTH TARGET]", fill=(15, 23, 42))
    draw.text((510, y_labels), "[PREDICTED 3D CAD MODEL]", fill=(15, 23, 42))

    # Paste images
    y_img = banner_height + label_height
    canvas.paste(img_gt, (15, y_img))
    canvas.paste(img_pred, (505, y_img))

    # Borders around images
    draw.rectangle([14, y_img - 1, 15 + 480, y_img + 480], outline=(203, 213, 225), width=1)
    draw.rectangle([504, y_img - 1, 505 + 480, y_img + 480], outline=(203, 213, 225), width=1)

    canvas.save(out_path, format="PNG")


if os.name == "nt":
    atexit.register(lambda: (sys.stdout.flush(), sys.stderr.flush(), os._exit(0)))

TIER1_IDS = [
    "0046/00461826", "0067/00675497", "0078/00789379", "0082/00824840",
    "0094/00948991", "0041/00417409", "0097/00976443", "0036/00369046",
    "0041/00410256", "0023/00234447", "0007/00070015"
]

TIER2_IDS = [
    "0095/00955962", "0027/00278384", "0014/00141107", "0060/00600430", "0021/00211465"
]

TIER3_IDS = [
    "0002/00024819", "0003/00039769", "0007/00070021", "0011/00118368", "0014/00144344",
    "0016/00167991", "0018/00189332", "0021/00217029", "0023/00231272", "0024/00248776",
    "0030/00302501", "0031/00315650", "0034/00342885", "0034/00344795", "0034/00347202",
    "0042/00427410", "0045/00450000", "0046/00467787", "0050/00507123", "0051/00512113",
    "0054/00540970", "0057/00574088", "0057/00578255", "0061/00618731", "0062/00629922",
    "0064/00642208", "0066/00669299", "0070/00707758", "0074/00747180", "0075/00753045",
    "0076/00763349", "0076/00765253", "0077/00773053", "0081/00819437", "0082/00824990",
    "0082/00827834", "0082/00828865", "0085/00852119", "0090/00904404", "0094/00946705",
    "0095/00950919", "0095/00959048"
]

ALL_CLEAN_IDS = TIER1_IDS + TIER2_IDS + TIER3_IDS


def generate_snapshots(
    sample_ids: list[str],
    predictions_file: str = "results/enhanced_hybrid_predictions.json",
    out_dir: str = "results/snapshots",
    skip_existing: bool = True,
):
    """Generate snapshot comparisons for a list of sample IDs."""
    os.makedirs(out_dir, exist_ok=True)
    temp_stl = os.path.join(PROJECT_ROOT, "temp_snapshot_render.stl")

    eval_data = get_eval_subset(use_hundred=True)
    with open(os.path.join(PROJECT_ROOT, predictions_file), "r", encoding="utf-8") as f:
        preds = json.load(f).get("predictions", {})

    ious_data = {}
    results_file = predictions_file.replace("_predictions.json", "_results.json")
    if os.path.exists(os.path.join(PROJECT_ROOT, results_file)):
        with open(os.path.join(PROJECT_ROOT, results_file), "r", encoding="utf-8") as f:
            ious_data = json.load(f).get("iou_per_sample", {})

    print(f"\n{'=' * 70}")
    print(f"GENERATING SNAPSHOTS: {len(sample_ids)} SAMPLES")
    print(f"Predictions: {predictions_file}")
    print(f"Output directory: {out_dir}")
    print(f"{'=' * 70}\n")

    saved_paths = []
    for i, sid in enumerate(sample_ids, 1):
        safe_sid = sid.replace("/", "_")
        out_file = os.path.join(out_dir, f"{safe_sid}_comparison.png")

        if skip_existing and os.path.exists(out_file):
            print(f"[{i:02d}/{len(sample_ids):02d}] {sid} -> ALREADY EXISTS: {out_file}")
            saved_paths.append(out_file)
            continue

        if sid not in eval_data or sid not in preds:
            print(f"[{i:02d}/{len(sample_ids):02d}] {sid} -> SKIPPED (not in dataset or predictions)")
            continue

        gt_img = eval_data[sid]["image"]
        pred_code = preds[sid]

        try:
            solid = _load_solid_from_code(pred_code, sid)
            if not solid.isValid() or solid.Volume() <= 1e-5:
                print(f"[{i:02d}/{len(sample_ids):02d}] {sid} -> SKIPPED (invalid solid)")
                continue

            cq.exporters.export(solid, temp_stl)
            pred_img = render_stl(temp_stl)

            iou = ious_data.get(sid, 0.0)
            
            # Compute true metrics
            try:
                gt_code = eval_data[sid]["cadquery"]
                solid_gt = _load_solid_from_code(gt_code, f"{sid}_gt")
                mesh_gt = _normalized_mesh(solid_gt)
                mesh_pred = _normalized_mesh(solid)
                
                betti_gt = compute_betti_numbers(mesh_gt)
                betti_pred = compute_betti_numbers(mesh_pred)
                betti = betti_match_score(betti_gt, betti_pred)
                
                cd = chamfer_distance(mesh_gt, mesh_pred)
            except Exception as metric_err:
                print(f"[{i:02d}/{len(sample_ids):02d}] {sid} -> Metric computation failed: {metric_err}")
                cd = 0.0
                betti = 0.0

            create_side_by_side_snapshot(
                sample_id=sid,
                gt_image=gt_img,
                pred_image=pred_img,
                iou=iou,
                cd=cd,
                betti_match=betti,
                out_path=out_file,
            )
            print(f"[{i:02d}/{len(sample_ids):02d}] {sid} -> SAVED: {out_file} (IoU: {iou:.4f})")
            saved_paths.append(out_file)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{i:02d}/{len(sample_ids):02d}] {sid} -> FAILED: {e}")

    # Clean up temp STL
    if os.path.exists(temp_stl):
        try:
            os.remove(temp_stl)
        except Exception:
            pass

    print(f"\nSuccessfully verified/generated {len(saved_paths)} comparison snapshots in '{out_dir}'.\n")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Generate GT vs Predicted CAD comparison snapshots")
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Specific sample IDs to render",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render all clean samples across all tiers",
    )
    parser.add_argument(
        "--predictions",
        default="results/enhanced_hybrid_predictions.json",
        help="Predictions JSON path",
    )
    parser.add_argument(
        "--out-dir",
        default="results/snapshots",
        help="Output directory for snapshot images",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 0],
        default=0,
        help="Render specific tier (1: Tier 1 only, 2: Tier 2 only, 3: Tier 3 only, 0: All 58 clean samples)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-render even if snapshot already exists",
    )
    args = parser.parse_args()

    results_file = args.predictions.replace("_predictions.json", "_results.json")
    if args.samples:
        sids = args.samples
    else:
        # Load all samples with valid IoU from the results file
        sids = []
        if os.path.exists(os.path.join(PROJECT_ROOT, results_file)):
            with open(os.path.join(PROJECT_ROOT, results_file), "r", encoding="utf-8") as f:
                ious_data = json.load(f).get("iou_per_sample", {})
                sids = sorted(list(ious_data.keys()))
        else:
            sids = ALL_CLEAN_IDS

    generate_snapshots(
        sample_ids=sids,
        predictions_file=args.predictions,
        out_dir=args.out_dir,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()

