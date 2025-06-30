import argparse
import csv
import os
from typing import Any, Dict


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


def load_local_dataset(json_path: str):
    """Load dataset metadata from a JSONL file with image paths and code."""
    json_path = os.path.expanduser(json_path)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Dataset file not found: {json_path}")
    dataset = datasets.load_dataset("json", data_files=json_path, split="train")
    dataset = dataset.cast_column("image_path", datasets.Image())
    dataset = dataset.rename_column("image_path", "image")
    dataset = dataset.rename_column("code", "text")
    return dataset


def preprocess_dataset(dataset, feature_extractor, tokenizer, max_length: int = 32):
    def _process(example):
        try:
            pixel_values = feature_extractor(images=example["image"], return_tensors="pt").pixel_values[0]
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
        "--train-json",
        default="data/train.jsonl",
        help="Path to training metadata JSONL",
    )
    parser.add_argument(
        "--eval-json",
        default="data/test.jsonl",
        help="Path to evaluation metadata JSONL",
    )
    args = parser.parse_args()

    model = transformers.VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224-in21k", "gpt2"
    )
    feature_extractor = transformers.ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_local_dataset(args.train_json)
    train_dataset = preprocess_dataset(train_dataset, feature_extractor, tokenizer)

    eval_dataset = None
    if args.eval_json:
        try:
            eval_dataset = load_local_dataset(args.eval_json)
            eval_dataset = preprocess_dataset(eval_dataset, feature_extractor, tokenizer)
        except FileNotFoundError as exc:
            print(f"Evaluation dataset not found: {exc}")

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to=None,
    )

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset is not None else train_dataset,
        data_collator=transformers.default_data_collator,
    )
    trainer.add_callback(CSVLogger(args.csv_path))
    trainer.train()


if __name__ == "__main__":
    main()
