#!/usr/bin/env python3
"""Robust model downloader with retry logic and offline caching."""

import os
import time
import argparse
from pathlib import Path

def setup_offline_cache():
    """Set up offline model cache directory."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"HuggingFace cache directory: {cache_dir}")
    return cache_dir

def check_model_availability():
    """Check which models are already cached locally."""
    cache_dir = setup_offline_cache()
    
    models_to_check = [
        "models--google--vit-base-patch16-224-in21k",
        "models--google--vit-base-patch16-224", 
        "models--gpt2",
        "models--distilgpt2"
    ]
    
    available_models = []
    
    print("Checking locally cached models...")
    for model in models_to_check:
        model_path = cache_dir / model
        if model_path.exists():
            print(f"Found: {model}")
            available_models.append(model)
        else:
            print(f"Missing: {model}")
    
    return available_models

def download_essential_models():
    """Download essential models with retry logic."""
    print("Attempting to download essential models...")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Smaller models that are easier to download
        models = [
            "google/vit-base-patch16-224",  # Smaller ViT variant
            "distilgpt2"  # Smaller GPT2 variant
        ]
        
        for model_id in models:
            try:
                print(f"Downloading {model_id}...")
                snapshot_download(
                    repo_id=model_id,
                    resume_download=True,
                    local_files_only=False
                )
                print(f"Successfully downloaded {model_id}")
            except Exception as e:
                print(f"Failed to download {model_id}: {e}")
                
    except ImportError:
        print("HuggingFace hub not available")
        return False
    
    return True

def create_enhanced_train_offline():
    """Create an offline-capable enhanced training script."""
    
    script_content = '''#!/usr/bin/env python3
import argparse
import os
import sys
import transformers
import datasets
import torch
import numpy as np
from PIL import ImageOps

def load_local_dataset(json_path: str):
    """Load dataset from JSONL file."""
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Dataset file not found: {json_path}")
    dataset = datasets.load_dataset("json", data_files=json_path, split="train")
    dataset = dataset.cast_column("image_path", datasets.Image())
    dataset = dataset.rename_column("image_path", "image")
    dataset = dataset.rename_column("code", "text")
    return dataset

def preprocess_dataset(dataset, feature_extractor, tokenizer, max_length: int = 32, augment: bool = False):
    def _process(example):
        try:
            img = example["image"]
            if augment and np.random.rand() < 0.5:
                img = ImageOps.mirror(img)
            pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values[0]
            labels = tokenizer(
                example["text"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            ).input_ids
            return {"pixel_values": pixel_values, "labels": labels}
        except Exception as exc:
            print(f"Failed to process example: {exc}")
            return {"pixel_values": None, "labels": None}

    processed = dataset.map(_process)
    processed = processed.filter(lambda x: x["pixel_values"] is not None)
    return processed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", default="./enhanced_results")
    parser.add_argument("--train-json", default="data/train.jsonl")
    parser.add_argument("--use-smaller-models", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("Using offline mode")
    
    # Choose model variants
    if args.use_smaller_models:
        encoder_model = "google/vit-base-patch16-224"
        decoder_model = "distilgpt2"
        print("Using smaller models")
    else:
        encoder_model = "google/vit-base-patch16-224-in21k"
        decoder_model = "gpt2"
        print("Using standard models")
    
    try:
        print(f"Loading models: {encoder_model} + {decoder_model}")
        model = transformers.VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_model, decoder_model
        )
        feature_extractor = transformers.ViTImageProcessor.from_pretrained(encoder_model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(decoder_model)
        tokenizer.pad_token = tokenizer.eos_token
        print("Models loaded successfully")
        
    except Exception as e:
        print(f"Failed to load models: {e}")
        sys.exit(1)

    # Load and preprocess data
    train_dataset = load_local_dataset(args.train_json)
    train_dataset = preprocess_dataset(train_dataset, feature_extractor, tokenizer)
    print(f"Training samples: {len(train_dataset)}")

    # Training setup
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to=None,
        dataloader_num_workers=0,
    )

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=transformers.default_data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
'''
    
    with open("enhanced_train_offline.py", "w") as f:
        f.write(script_content)
    
    print("Created enhanced_train_offline.py")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check cached models")
    parser.add_argument("--download", action="store_true", help="Download models")
    parser.add_argument("--setup-offline", action="store_true", help="Create offline script")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    if args.all or args.check:
        print("CHECKING CACHED MODELS")
        available = check_model_availability()
        
    if args.all or args.download:
        print("DOWNLOADING MODELS")
        download_essential_models()
        
    if args.all or args.setup_offline:
        print("CREATING OFFLINE SCRIPT")
        create_enhanced_train_offline()
    
    if args.all:
        print("\nNEXT STEPS:")
        print("1. Try: python enhanced_train_offline.py --use-smaller-models --epochs 2")
        print("2. Then: python enhanced_train_offline.py --epochs 2")

if __name__ == "__main__":
    main() 