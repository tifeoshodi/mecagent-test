import argparse
import csv
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoTokenizer,
    ViTFeatureExtractor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    TrainerCallback,
)


class CSVLogger(TrainerCallback):
    """Log trainer metrics to a CSV file."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._initialized = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_local_process_zero:
            return
        if not self._initialized:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch"] + sorted(logs.keys()))
                writer.writeheader()
            self._initialized = True
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch"] + sorted(logs.keys()))
            writer.writerow({"epoch": state.epoch, **{k: float(v) for k, v in logs.items()}})


def create_dummy_dataset(num_samples: int = 10, image_size: int = 64) -> Dataset:
    images = []
    captions = []
    for _ in range(num_samples):
        array = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        images.append(Image.fromarray(array))
        captions.append("dummy caption")
    return Dataset.from_dict({"image": images, "text": captions})


def preprocess_dataset(dataset: Dataset, feature_extractor, tokenizer, max_length: int = 32) -> Dataset:
    def _process(example):
        pixel_values = feature_extractor(images=example["image"], return_tensors="pt").pixel_values[0]
        labels = tokenizer(example["text"], max_length=max_length, padding="max_length", truncation=True).input_ids
        return {"pixel_values": pixel_values, "labels": labels}

    return dataset.map(_process)


def main():
    parser = argparse.ArgumentParser(
        description="Train a VisionEncoderDecoderModel on dummy data"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size per device"
    )
    parser.add_argument(
        "--output-dir", default="./results", help="Directory for trainer outputs"
    )
    parser.add_argument(
        "--csv-path", default="metrics.csv", help="Path to CSV metrics file"
    )
    args = parser.parse_args()

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224-in21k", "gpt2"
    )
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = create_dummy_dataset()
    dataset = preprocess_dataset(dataset, feature_extractor, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_steps=10,
        remove_unused_columns=False,
        report_to=None,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.add_callback(CSVLogger(args.csv_path))
    trainer.train()


if __name__ == "__main__":
    main()
