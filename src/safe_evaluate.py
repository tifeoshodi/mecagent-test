import os
import json
import sys
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_item(args):
    item_id, gt_code, pred_code, project_root = args
    if not pred_code.strip():
        return item_id, None

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
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""

    try:
        res = subprocess.run(
            [sys.executable, "-c", script], 
            capture_output=True, 
            text=True, 
            timeout=45
        )
        for line in res.stdout.splitlines():
            if line.startswith("IOU_RESULT="):
                return item_id, float(line.split("=")[1])
        return item_id, None
    except subprocess.TimeoutExpired:
        print(f"\\nTimeout for {item_id}", flush=True)
        return item_id, None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", default="results/enhanced_visual_predictions.json", help="Path to predictions JSON")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__)).replace(chr(92), '/')
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_loader import get_eval_subset
    from metrics.valid_syntax_rate import evaluate_syntax_rate

    eval_data = get_eval_subset(split="test", use_hundred=False)
    
    preds_path = args.preds
    with open(preds_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        predictions = data.get("predictions", data)
    
    gt_codes = {k: v["cadquery"] for k, v in eval_data.items()}
    
    vsr_res = evaluate_syntax_rate(predictions, verbose=False)
    print(f"VSR: {vsr_res['vsr']:.3f} ({vsr_res['successful']}/{vsr_res['total']})")
    
    ids = sorted(set(gt_codes.keys()) & set(predictions.keys()))
    ious = {}
    
    args_list = [(item_id, gt_codes[item_id], predictions[item_id], project_root) for item_id in ids]
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_item, arg): arg[0] for arg in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing IoU with timeout (Parallel)"):
            item_id, result = future.result()
            if result is not None:
                ious[item_id] = result
            
    if ious:
        mean_iou = sum(ious.values()) / len(ious)
    else:
        mean_iou = 0.0
        
    print(f"Mean IoU_best: {mean_iou:.4f} ({len(ious)} valid)")
    
    results = {
        "vsr": vsr_res["vsr"],
        "mean_iou_best": mean_iou
    }
    with open("results/enhanced_visual_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
