# Dataset

This project uses the [image-to-code dataset](https://huggingface.co/datasets/lyleokoth/image-to-code-v1) hosted on HuggingFace. The dataset contains pairs of 3D rendered images and their corresponding CadQuery code implementations.

## Dataset Overview

- **Source**: HuggingFace dataset `lyleokoth/image-to-code-v1`
- **Task**: Image-to-code generation for 3D CAD models
- **Format**: Images (PNG) paired with CadQuery Python code
- **Size**: Large dataset, but we use a subset for this project

## Local Dataset Setup

### Quick Download

Use the provided script to fetch a manageable subset:

```bash
python scripts/download_hf_dataset.py
```

This downloads:
- **2000 training samples** from the beginning of the dataset
- **500 test samples** from the next portion of the dataset

### Dataset Structure

After running the download script, you'll have:

```
data/
├── train/           # Training images (00000.png, 00001.png, ...)
├── test/            # Test images (00000.png, 00001.png, ...)
├── train.jsonl      # Training metadata (image_path -> code mapping)
└── test.jsonl       # Test metadata (image_path -> code mapping)
```

### Metadata Format

The `.jsonl` files contain one JSON object per line with the structure:

```json
{"image_path": "data/train/00000.png", "code": "result = cq.Workplane(\"XY\").box(1, 1, 1)"}
{"image_path": "data/train/00001.png", "code": "result = cq.Workplane(\"XY\").cylinder(5, 2)"}
```

Each entry maps:
- `image_path`: Path to the corresponding 3D rendered image
- `code`: CadQuery Python code that generates the 3D model

## Usage in Training Scripts

### Loading the Dataset

The training scripts automatically load the dataset using these paths:
- `data/train.jsonl` - Training metadata
- `data/test.jsonl` - Evaluation metadata

### Command Line Options

You can specify custom dataset paths:

```bash
python train_vision_encoder_decoder.py \
    --train-json /path/to/your/train.jsonl \
    --eval-json /path/to/your/test.jsonl
```

### Data Processing

Images are automatically:
1. Loaded using PIL
2. Preprocessed with ViT feature extractor (224x224 resolution)
3. Paired with tokenized CadQuery code (max 32 tokens by default)

## Dataset Characteristics

### Images
- **Format**: PNG
- **Content**: 3D rendered CAD models from various viewpoints
- **Resolution**: Variable (automatically resized to 224x224 for ViT)

### Code
- **Language**: Python (CadQuery library)
- **Style**: Functional CAD modeling commands
- **Complexity**: Ranges from simple primitives to complex assemblies

### Examples

**Simple box:**
```python
result = cq.Workplane("XY").box(10, 10, 10)
```

**Box with hole:**
```python
result = (
    cq.Workplane("XY")
    .box(20, 20, 10)
    .faces(">Z")
    .workplane()
    .hole(5)
)
```

## Evaluation Datasets

### Baseline Evaluation
The `baseline_eval.py` script uses the online `CADCODER/GenCAD-Code` dataset directly from HuggingFace for comparison purposes.

### Enhanced Evaluation  
The `enhanced_eval.py` script can use either:
- Local test set from `data/test.jsonl` 
- Online `CADCODER/GenCAD-Code` dataset



```python
# Change these lines to adjust dataset size
train_meta = save_split(ds.take(5000), train_dir, 5000)  # More training data
test_meta = save_split(ds.skip(5000).take(1000), test_dir, 1000)  # More test data
```
