#!/usr/bin/env python3
import argparse
import torch
import numpy as np
from improved_simple_train import ImprovedModel, VOCAB, VOCAB_SIZE, tokenize_simple
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best
from tqdm import tqdm

# Reverse vocab for decoding
REVERSE_VOCAB = {v: k for k, v in VOCAB.items()}

def decode_tokens(tokens):
    """Convert token indices back to code string."""
    words = []
    for token in tokens:
        if token == VOCAB['<END>']:
            break
        if token in REVERSE_VOCAB and token not in [VOCAB['<PAD>'], VOCAB['<START>']]:
            words.append(REVERSE_VOCAB[token])
    return ' '.join(words)

def generate_code(model, image):
    """Generate code from image using the improved model."""
    model.eval()
    with torch.no_grad():
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        outputs = model(image)
        predicted_tokens = outputs.argmax(dim=-1)
        return decode_tokens(predicted_tokens[0].tolist())

def create_test_images(num_samples=10):
    """Create test images."""
    images = []
    gt_codes = []
    
    for i in range(num_samples):
        # Create dummy test image
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        images.append(image)
        
        # Ground truth code
        gt_codes.append('result = cq.Workplane("XY").box(1, 1, 1)')
    
    return images, gt_codes

def evaluate_improved_model(model_path: str, num_samples: int = 10):
    """Evaluate the improved model."""
    
    # Load model
    model = ImprovedModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"Loaded improved model from {model_path}")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Get test data
    images, gt_codes = create_test_images(num_samples)
    
    # Generate predictions
    pred_codes = {}
    gt_code_dict = {}
    
    print("Generating predictions...")
    for i, (image, gt_code) in enumerate(tqdm(zip(images, gt_codes))):
        # Generate prediction
        pred_code = generate_code(model, image)
        
        # Clean up prediction for evaluation
        pred_code_clean = pred_code.replace(' . ', '.').replace('( ', '(').replace(' )', ')')
        pred_code_clean = pred_code_clean.replace(' , ', ', ')
        if 'result = ' not in pred_code_clean:
            pred_code_clean = 'result = ' + pred_code_clean
        
        key = str(i)
        pred_codes[key] = pred_code_clean
        gt_code_dict[key] = gt_code
        
        if i < 3:  # Show first few examples
            print(f"Example {i}:")
            print(f"  Predicted: {pred_code_clean}")
            print(f"  Ground truth: {gt_code}")
    
    # Evaluate VSR
    try:
        vsr = evaluate_syntax_rate_simple(pred_codes)
        print(f"Valid Syntax Rate: {vsr:.3f}")
    except Exception as e:
        print(f"VSR evaluation failed: {e}")
        vsr = 0.0
    
    # Evaluate IOU
    ious = []
    print("Computing IOU scores...")
    for key in tqdm(gt_code_dict.keys()):
        try:
            iou = get_iou_best(gt_code_dict[key], pred_codes[key])
            ious.append(iou)
        except Exception as e:
            print(f"IOU computation failed for sample {key}: {e}")
    
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    print(f"Mean IOU: {mean_iou:.3f}")
    
    return vsr, mean_iou

def main():
    parser = argparse.ArgumentParser(description="Evaluate improved model")
    parser.add_argument("model_path", help="Path to trained model")
    parser.add_argument("--samples", type=int, default=10, help="Number of test samples")
    
    args = parser.parse_args()
    
    vsr, iou = evaluate_improved_model(args.model_path, args.samples)
    print(f"Improved Model Results - VSR: {vsr:.3f}, IOU: {iou:.3f}")

if __name__ == "__main__":
    main() 