import argparse
from datasets import load_dataset
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best
from tqdm import tqdm
import transformers


def evaluate_enhanced(model_dir: str, num_samples: int = 10):
    model = transformers.VisionEncoderDecoderModel.from_pretrained(model_dir)
    feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(model_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)

    dataset = load_dataset("CADCODER/GenCAD-Code", split=f"test[:{num_samples}]")

    gt_codes = {}
    pred_codes = {}
    for i, ex in enumerate(dataset):
        code = ex.get("cadquery")
        image = ex.get("image")
        if code is None or image is None:
            continue
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        outputs = model.generate(pixel_values)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        key = str(i)
        gt_codes[key] = code
        pred_codes[key] = pred

    vsr = evaluate_syntax_rate_simple(pred_codes)
    ious = []
    for k in tqdm(gt_codes.keys(), desc="IOU"):
        try:
            iou_val = get_iou_best(gt_codes[k], pred_codes[k])
            ious.append(iou_val)
        except Exception as exc:
            print(f"Sample {k}: IOU computation failed -> {exc}")
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    print(f"Enhanced — VSR: {vsr:.3f}, IOU: {mean_iou:.3f}")
    return vsr, mean_iou


def main():
    parser = argparse.ArgumentParser(description="Evaluate enhanced model")
    parser.add_argument("model_dir", help="Path to trained model")
    parser.add_argument("--samples", type=int, default=10, help="Number of test samples")
    args = parser.parse_args()
    evaluate_enhanced(args.model_dir, args.samples)


if __name__ == "__main__":
    main()
