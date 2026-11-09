"""
Generate side-by-side diagnostic snapshots for failed CAD generation benchmark parts.
Left Panel: Ground Truth Target Image
Right Panel: Diagnostic Card (Error Category, Exception Message, Predicted Code Preview)
Outputs to: results/failed_snapshots/
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_eval_subset
from metrics.best_iou import _load_solid_from_code
from src.generate_snapshots import ALL_CLEAN_IDS


def get_fonts():
    """Load system fonts with graceful fallback."""
    try:
        font_title = ImageFont.truetype("arial.ttf", 16)
        font_bold = ImageFont.truetype("arialbd.ttf", 14)
        font_regular = ImageFont.truetype("arial.ttf", 12)
        font_code = ImageFont.truetype("consola.ttf", 11)
        return font_title, font_bold, font_regular, font_code
    except Exception:
        default = ImageFont.load_default()
        return default, default, default, default


def categorize_error(code: str, error_msg: str) -> tuple[str, str]:
    """Classify the failure into a concise category and readable message."""
    clean_code = code.strip()
    if not clean_code:
        return "EMPTY_PREDICTION", "Model returned 0 tokens (context/rate-limit drop)."

    if "No CadQuery objects" in error_msg:
        return "NO_CAD_OBJECT", "Script contains comments or variables without creating a CadQuery solid."

    if "has no attribute" in error_msg:
        attr = error_msg.split("has no attribute")[-1].strip()
        return "API_MISMATCH", f"Invalid CadQuery method {attr} called on Workplane."

    if "invalid syntax" in error_msg or "never closed" in error_msg:
        return "SYNTAX_ERROR", error_msg.replace("Error executing script ", "").split(":")[-1].strip()

    if "is not defined" in error_msg:
        var = error_msg.split("name")[-1].split("is not defined")[0].strip()
        return "UNDEFINED_NAME", f"Reference to undefined/truncated identifier {var}."

    if "Cannot find a solid" in error_msg or "Standard_Failure" in error_msg:
        return "TOPOLOGY_KERNEL_ERROR", "OpenCASCADE geometric kernel failed to create manifold solid."

    return "EXECUTION_ERROR", error_msg.replace("Error executing script ", "")[:100]


def render_diagnostic_card(
    sample_id: str,
    error_category: str,
    error_detail: str,
    pred_code: str,
    card_size: tuple[int, int] = (480, 480),
) -> Image.Image:
    """Render a failure diagnostic card with error message and code preview."""
    card = Image.new("RGB", card_size, color=(15, 23, 42))  # slate-900
    draw = ImageDraw.Draw(card)
    _, font_bold, font_regular, font_code = get_fonts()

    # Red warning badge
    draw.rounded_rectangle([16, 16, 280, 42], radius=4, fill=(185, 28, 28))  # red-700
    draw.text((26, 21), "[GENERATION FAILED]", fill=(255, 255, 255), font=font_bold)

    # Error Category
    draw.text((16, 52), f"Error Category: {error_category}", fill=(248, 113, 113), font=font_bold)

    # Error Detail (wrapped to 2 lines)
    words = error_detail.split()
    line1, line2 = "", ""
    for w in words:
        if len(line1) + len(w) < 55 and not line2:
            line1 += (w + " ")
        else:
            line2 += (w + " ")
    draw.text((16, 75), line1.strip(), fill=(226, 232, 240), font=font_regular)
    if line2:
        draw.text((16, 92), line2.strip()[:65], fill=(226, 232, 240), font=font_regular)

    # Code Preview Section Header
    y_code_start = 120
    draw.text((16, y_code_start), "[PREDICTED CODE PREVIEW]", fill=(56, 189, 248), font=font_bold)

    # Code background box
    draw.rounded_rectangle(
        [16, y_code_start + 24, card_size[0] - 16, card_size[1] - 16],
        radius=6,
        fill=(30, 41, 59),  # slate-800
        outline=(51, 65, 85),  # slate-700
        width=1,
    )

    # Code text lines
    code_lines = pred_code.strip().split("\n")
    if not pred_code.strip():
        code_lines = ["<Empty model response / 0 tokens>"]

    y_line = y_code_start + 32
    max_lines = 18
    for idx, line in enumerate(code_lines[:max_lines], 1):
        line_num = f"{idx:2d} | "
        draw.text((24, y_line), line_num, fill=(100, 116, 139), font=font_code)
        draw.text((58, y_line), line[:52], fill=(241, 245, 249), font=font_code)
        y_line += 17

    if len(code_lines) > max_lines:
        draw.text((58, y_line), f"... ({len(code_lines) - max_lines} more lines)", fill=(148, 163, 184), font=font_code)

    return card


def create_failed_snapshot(
    sample_id: str,
    gt_image: Image.Image,
    error_category: str,
    error_detail: str,
    pred_code: str,
    out_path: str,
):
    """Combine GT target image and Failure Diagnostic Card into side-by-side comparison."""
    target_size = (480, 480)
    img_gt = gt_image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    img_card = render_diagnostic_card(sample_id, error_category, error_detail, pred_code, target_size)

    # Canvas dimensions
    banner_height = 60
    label_height = 30
    canvas_w = 480 * 2 + 30
    canvas_h = banner_height + label_height + 480 + 20

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    font_title, _, font_regular, _ = get_fonts()

    # Top banner background (dark red/maroon)
    draw.rectangle([0, 0, canvas_w, banner_height], fill=(69, 10, 10))  # red-950

    # Text in banner
    title = f"FAILED SAMPLE: {sample_id}  |  STATUS: CODE EXECUTION FAILED  |  {error_category}"
    draw.text((20, 20), title, fill=(254, 202, 202), font=font_title)

    # Panel labels
    y_labels = banner_height + 8
    draw.text((20, y_labels), "[GROUND TRUTH TARGET]", fill=(15, 23, 42), font=font_regular)
    draw.text((510, y_labels), "[PREDICTED MODEL DIAGNOSTIC]", fill=(15, 23, 42), font=font_regular)

    # Paste images
    y_img = banner_height + label_height
    canvas.paste(img_gt, (15, y_img))
    canvas.paste(img_card, (505, y_img))

    # Borders around panels
    draw.rectangle([14, y_img - 1, 15 + 480, y_img + 480], outline=(203, 213, 225), width=1)
    draw.rectangle([504, y_img - 1, 505 + 480, y_img + 480], outline=(239, 68, 68), width=1)

    canvas.save(out_path, format="PNG")


# Curated list of representative failed samples covering all failure modes
KEY_FAILED_IDS = [
    "0014/00144800",  # API mismatch: 'Workplane' object has no attribute 'cutEx'
    "0081/00812431",  # API mismatch: Workplane attribute error
    "0096/00966577",  # Topology error: Cannot find a solid
    "0035/00354124",  # Syntax error: unclosed parenthesis
    "0010/00106662",  # Syntax error: invalid syntax
    "0015/00152464",  # Syntax error: invalid syntax on line 29
    "0012/00129614",  # Undefined name: SHORT_
    "0014/00144677",  # Undefined name: lower
    "0044/00442545",  # Undefined name: box_
    "0047/00472674",  # Undefined name: fil
    "0000/00002718",  # No CAD object: comments only
    "0001/00014960",  # No CAD object: comments only
    "0005/00059989",  # No CAD object: comments only
    "0020/00201504",  # Empty prediction: 0 tokens
]


def generate_failed_snapshots(
    sample_ids: list[str] = None,
    predictions_file: str = "results/enhanced_hybrid_predictions.json",
    out_dir: str = "results/failed_snapshots",
    render_all: bool = False,
):
    """Generate diagnostic snapshot comparisons for failed samples."""
    os.makedirs(out_dir, exist_ok=True)

    eval_data = get_eval_subset(use_hundred=True)
    with open(os.path.join(PROJECT_ROOT, predictions_file), "r", encoding="utf-8") as f:
        preds = json.load(f).get("predictions", {})

    results_file = predictions_file.replace("_predictions.json", "_results.json")
    clean_ids = []
    if os.path.exists(os.path.join(PROJECT_ROOT, results_file)):
        with open(os.path.join(PROJECT_ROOT, results_file), "r", encoding="utf-8") as f:
            clean_ids = list(json.load(f).get("iou_per_sample", {}).keys())

    all_failed_ids = [k for k in preds if k not in clean_ids]

    if sample_ids:
        target_ids = sample_ids
    elif render_all:
        target_ids = all_failed_ids
    else:
        target_ids = all_failed_ids

    print(f"\n{'=' * 70}")
    print(f"GENERATING FAILED SAMPLE SNAPSHOTS: {len(target_ids)} SAMPLES")
    print(f"Predictions: {predictions_file}")
    print(f"Output directory: {out_dir}")
    print(f"{'=' * 70}\n")

    saved_paths = []
    for i, sid in enumerate(target_ids, 1):
        if sid not in eval_data or sid not in preds:
            print(f"[{i:02d}/{len(target_ids):02d}] {sid} -> SKIPPED (not in dataset or predictions)")
            continue

        gt_img = eval_data[sid]["image"]
        pred_code = preds[sid]

        try:
            _load_solid_from_code(pred_code, sid)
            err_msg = "No error detected during load"
        except Exception as e:
            err_msg = str(e)

        cat, detail = categorize_error(pred_code, err_msg)

        safe_sid = sid.replace("/", "_")
        out_file = os.path.join(out_dir, f"{safe_sid}_failed_comparison.png")

        create_failed_snapshot(
            sample_id=sid,
            gt_image=gt_img,
            error_category=cat,
            error_detail=detail,
            pred_code=pred_code,
            out_path=out_file,
        )
        print(f"[{i:02d}/{len(target_ids):02d}] {sid} -> SAVED: {out_file} [{cat}]")
        saved_paths.append(out_file)

    print(f"\nSuccessfully generated {len(saved_paths)} failed snapshot comparisons in '{out_dir}'.\n")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic snapshots for failed CAD samples")
    parser.add_argument("--samples", nargs="+", default=None, help="Specific failed sample IDs to render")
    parser.add_argument("--predictions", default="results/enhanced_hybrid_predictions.json", help="Predictions JSON path")
    parser.add_argument("--out-dir", default="results/failed_snapshots", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Render all failed samples")
    args = parser.parse_args()

    generate_failed_snapshots(
        sample_ids=args.samples,
        predictions_file=args.predictions,
        out_dir=args.out_dir,
        render_all=args.all,
    )


if __name__ == "__main__":
    main()
