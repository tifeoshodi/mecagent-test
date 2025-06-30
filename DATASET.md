# Dataset

This project relies on an image-to-code dataset hosted on [HuggingFace](https://huggingface.co/datasets/lyleokoth/image-to-code-v1). Each entry provides a rendered image and the corresponding CadQuery code.

## Download

Use the helper script to fetch a small subset for local experiments:

```bash
python scripts/download_hf_dataset.py
```

The script streams the dataset and creates a `data` directory with the following layout:

```
data/
├── train/       # training images
├── test/        # testing images
├── train.jsonl  # metadata mapping image paths to CadQuery code
└── test.jsonl
```

`train.jsonl` and `test.jsonl` contain one JSON object per line with `image_path` and `code` fields.

Other training or evaluation scripts should load images and code from these files and directories.
