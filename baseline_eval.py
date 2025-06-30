from datasets import load_dataset
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best


def evaluate_baseline(num_samples: int = 100) -> None:
    """Load test examples and run baseline metrics."""
    # Load a subset of the test split
    dataset = load_dataset("CADCODER/GenCAD-Code", split=f"test[:{num_samples}]")

    # Simple constant prediction used for every example
    baseline_prediction = 'result = cq.Workplane("XY").box(1, 1, 1)'

    gt_codes = {str(i): ex["cadquery"] for i, ex in enumerate(dataset)}
    pred_codes = {str(i): baseline_prediction for i in range(len(dataset))}

    # Valid Syntax Rate on predictions
    vsr = evaluate_syntax_rate_simple(pred_codes)

    # Mean IOU against ground-truth
    ious = [get_iou_best(gt_codes[k], pred_codes[k]) for k in gt_codes]
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    # Print formatted result
    print(f"Baseline — VSR: {vsr:.3f}, IOU: {mean_iou:.3f}")


if __name__ == "__main__":
    evaluate_baseline()
