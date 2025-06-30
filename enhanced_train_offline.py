#!/usr/bin/env python3
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
    parser = argparse.ArgumentParser(description="Enhanced training with offline support")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--output-dir", default="./enhanced_results", help="Output directory")
    parser.add_argument("--train-json", default="data/train.jsonl", help="Training data path")
    parser.add_argument("--use-smaller-models", action="store_true", help="Use smaller model variants")
    parser.add_argument("--offline", action="store_true", help="Use offline mode")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
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
        
        # Using ViTImageProcessor instead of deprecated ViTFeatureExtractor
        try:
            feature_extractor = transformers.ViTImageProcessor.from_pretrained(encoder_model)
        except:
            feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(encoder_model)
            
        tokenizer = transformers.AutoTokenizer.from_pretrained(decoder_model)
        tokenizer.pad_token = tokenizer.eos_token
        print("Models loaded successfully")
        
    except Exception as e:
        print(f"Failed to load models: {e}")
        print("Try using --use-smaller-models flag")
        sys.exit(1)

    # Load and preprocess data
    try:
        print(f"Loading training dataset from {args.train_json}")
        train_dataset = load_local_dataset(args.train_json)
        train_dataset = preprocess_dataset(train_dataset, feature_extractor, tokenizer, augment=args.augment)
        print(f"Training samples: {len(train_dataset)}")
    except Exception as e:
        print(f"Failed to load training data: {e}")
        sys.exit(1)

    # Training setup
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=max(1, len(train_dataset) // args.batch_size // 4),  # Log 4 times per epoch
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to=None,
        dataloader_num_workers=0,  # Avoid multiprocessing issues 
        fp16=False,  # Disable mixed precision to avoid potential issues
    )

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=transformers.default_data_collator,
    )
    
    print("Starting enhanced training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
