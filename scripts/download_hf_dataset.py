#!/usr/bin/env python

import json
from pathlib import Path
from datasets import load_dataset

from PIL import Image


def save_split(dataset_iter, out_dir: Path, limit: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for idx, sample in enumerate(dataset_iter):
        if idx >= limit:
            break
        image: Image.Image = sample["image"]
        code: str = sample["code"]
        img_path = out_dir / f"{idx:05d}.png"
        image.save(img_path)
        meta.append({"image_path": str(img_path), "code": code})
    return meta


def main():
    data_dir = Path("data")
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    data_dir.mkdir(exist_ok=True)

    ds = load_dataset("lyleokoth/image-to-code-v1", split="train", streaming=True)

    train_meta = save_split(ds.take(2000), train_dir, 2000)
    test_meta = save_split(ds.skip(2000).take(500), test_dir, 500)

    with open(data_dir / "train.jsonl", "w") as f:
        for row in train_meta:
            json.dump(row, f)
            f.write("\n")

    with open(data_dir / "test.jsonl", "w") as f:
        for row in test_meta:
            json.dump(row, f)
            f.write("\n")


if __name__ == "__main__":
    main()
