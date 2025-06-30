from datasets import load_dataset
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best
from tqdm import tqdm


def evaluate_baseline(num_samples: int = 100) -> None:
    """Load test examples and run baseline metrics."""
    # Load a subset of the test split
    try:
        dataset = load_dataset(
            "CADCODER/GenCAD-Code",
            split=f"test[:{num_samples}]",
        )
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        return

    # Simple constant prediction used for every example
    baseline_prediction = 'result = cq.Workplane("XY").box(1, 1, 1)'

    gt_codes = {}
    pred_codes = {}
    for i, ex in enumerate(dataset):
        cad_code = ex.get("cadquery")
        if cad_code is None:
            print(f"Skipping sample {i}: missing 'cadquery' field")
            continue
        key = str(i)
        gt_codes[key] = cad_code
        pred_codes[key] = baseline_prediction

    # Valid Syntax Rate on predictions
    try:
        vsr = evaluate_syntax_rate_simple(pred_codes)
    except Exception as exc:
        print(f"Failed to compute VSR: {exc}")
        vsr = 0.0

    # Mean IOU against ground-truth
    ious = []
    for k in tqdm(gt_codes.keys(), desc="IOU"):
        try:
            iou_val = get_iou_best(gt_codes[k], pred_codes[k])
            ious.append(iou_val)
        except Exception as exc:
            print(f"Sample {k}: IOU computation failed -> {exc}")
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    # Print formatted result
    print(f"Baseline — VSR: {vsr:.3f}, IOU: {mean_iou:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate baseline model")
    parser.add_argument(
        "num_samples",
        nargs="?",
        type=int,
        default=100,
        help="Number of test samples to evaluate",
    )

    args = parser.parse_args()
    evaluate_baseline(args.num_samples)
