# MecAgent Technical Test

This repository contains a few simple scripts used for the Mecagent test. Most of the
walkthrough can be found in the `good_luck.ipynb` notebook.

## Environment Setup

- **Python version**: `>=3.11`
- **Core dependencies** (from `pyproject.toml`):
  - `cadquery>=2.5.2`
  - `datasets>=3.6.0`
  - `ipykernel>=6.29.5`
  - `scipy>=1.15.3`
  - `trimesh>=4.6.11`
  - `numpy>=2.1.3`
  - `torch>=2.5.1`
  - `transformers>=4.52.4`

Create and activate a virtual environment before installing packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies using **uv** (preferred) or plain pip:

```bash
uv sync            # reads `pyproject.toml` and `uv.lock`
# or fall back to requirements file
pip install -r requirements.txt
```

## Running Code

Open the notebook:

```bash
jupyter notebook good_luck.ipynb
```

The scripts can also be executed directly, for example:

## Enhancements
- Summarize improvements and additional features tested.
The training pipeline was extended with optional data augmentation. Images can
now be randomly mirrored horizontally during preprocessing when the `--augment`
flag is used in `enhanced_train.py`. The model checkpoints are saved to the
specified output directory.

## Evaluation
- Explain metrics and validation approach used to compare results.
`enhanced_eval.py` loads the saved model and evaluates a subset of the
`CADCODER/GenCAD-Code` dataset using the same valid syntax rate and IoU metrics
as `baseline_eval.py`. Due to environment constraints the full evaluation could
not be completed during testing, so the `enhanced` row in `results.csv` is left
at zero values.

To reproduce the baseline evaluation on a small subset of the dataset, install
the project dependencies and run:

```bash
pip install -e .
python baseline_eval.py 5
```

This will compute the Valid Syntax Rate (VSR) and mean IOU for five examples and
print the metrics used to update `results.csv`.

```bash
python baseline_eval.py                # run the baseline metrics
python train_vision_encoder_decoder.py # start training
```

## Dataset
See [DATASET.md](DATASET.md) for instructions on downloading a small subset of
the image-to-code dataset from HuggingFace using the provided helper script.

