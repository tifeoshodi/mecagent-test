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

```bash
python baseline_eval.py                # run the baseline metrics
python train_vision_encoder_decoder.py # start training
```

