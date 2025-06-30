import argparse
import csv
import os
from typing import Any, Dict

from PIL import ImageOps


class _LazyModule:
    """Utility to lazily import modules when they are first needed."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module = None

    def __getattr__(self, attr: str):
        if self._module is None:
            import importlib

            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, attr)


np = _LazyModule("numpy")
datasets = _LazyModule("datasets")
Image = _LazyModule("PIL.Image")
transformers = _LazyModule("transformers")


class CSVLogger:
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


def create_dummy_dataset(num_samples: int = 10, image_size: int = 64):
    images = []
    captions = []
    for _ in range(num_samples):
        array = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        images.append(Image.fromarray(array))
        captions.append("dummy caption")
    return datasets.Dataset.from_dict({"image": images, "text": captions})


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
    parser.add_argument(
        "--augment", action="store_true", help="Apply random data augmentation"
    )
    args = parser.parse_args()

    model = transformers.VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224-in21k", "gpt2"
    )
    feature_extractor = transformers.ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = create_dummy_dataset()
    dataset = preprocess_dataset(dataset, feature_extractor, tokenizer, augment=args.augment)

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_steps=10,
        remove_unused_columns=False,
        report_to=None,
    )

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=transformers.default_data_collator,
    )
    trainer.add_callback(CSVLogger(args.csv_path))
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
