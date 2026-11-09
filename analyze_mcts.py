import os
import sys
import json
from src.data_loader import get_eval_subset
from src.evaluate import compute_vsr, compute_iou_best_batch

def analyze():
    predictions_path = "results/mcts_predictions.json"
    if not os.path.exists(predictions_path):
        print(f"Predictions file not found at {predictions_path}")
        return

    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        predictions = data.get("predictions", {})

    print(f"Loaded {len(predictions)} predictions.")

    # Load evaluation ground truth
    eval_data = get_eval_subset(split="test", use_hundred=True)
    gt_codes = {k: v["cadquery"] for k, v in eval_data.items() if k in predictions}

    print("\nStep 1: Computing Valid Syntax Rate (VSR)...")
    vsr_result = compute_vsr(predictions, verbose=False)
    print(f"  VSR: {vsr_result['vsr']:.3f} ({vsr_result['successful']}/{vsr_result['total']})")

    print("\nStep 2: Computing IoU_best...")
    iou_result = compute_iou_best_batch(gt_codes, predictions, pitch=0.05, verbose=False)
    print(f"  Mean IoU_best: {iou_result['mean_iou']:.4f} ({iou_result['valid_count']}/{iou_result['total_count']} valid)")

    # Save aggregated results
    results = {
        "num_samples": len(predictions),
        "vsr": vsr_result["vsr"],
        "vsr_successful": vsr_result["successful"],
        "vsr_total": vsr_result["total"],
        "mean_iou_best": iou_result["mean_iou"],
        "iou_valid_count": iou_result["valid_count"],
    }
    
    with open("results/mcts_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    analyze()
