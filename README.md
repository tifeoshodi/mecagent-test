
### Everything is explained in the good_luck.ipynb file.


# Project Overview


## Environment Setup
- Describe the required Python version and dependencies.
- Instructions for creating and activating a virtual environment.

## Baseline Model
- Outline the initial model architecture and training process.

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

## Report Structure
- Provide an overview of how to present findings and conclusions.

## Dataset
See [DATASET.md](DATASET.md) for instructions on downloading a small subset of
the image-to-code dataset from HuggingFace using the provided helper script.

