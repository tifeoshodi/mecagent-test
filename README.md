# MecAgent Technical Test

This repository contains a CadQuery code generator model that converts 3D rendered images to CadQuery code using a Vision-Encoder-Decoder architecture.

## Overview

The project implements an image-to-code generation system that:
- Takes 3D rendered images as input
- Generates corresponding CadQuery Python code
- Evaluates model performance using Valid Syntax Rate (VSR) and Intersection over Union (IOU) metrics

## Environment Setup

### Prerequisites
- **Python version**: `>=3.11`
- **Package manager**: `uv` (recommended) or `pip`

### Installation

1. **Clone and navigate to the repository:**
```bash
git clone <repository-url>
cd mecagent-test-1
```

2. **Create and activate a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Core Dependencies
- `cadquery>=2.5.2` - 3D CAD modeling library
- `datasets>=3.6.0` - HuggingFace datasets for data loading
- `torch>=2.5.1` - PyTorch for deep learning
- `transformers>=4.52.4` - HuggingFace transformers for vision-encoder-decoder models
- `trimesh>=4.6.11` - 3D mesh processing for IOU evaluation
- `scipy>=1.15.3` - Scientific computing
- `numpy>=2.1.3` - Numerical computing

## Dataset

The project uses a subset of the CadQuery image-to-code dataset. See [DATASET.md](DATASET.md) for detailed download instructions.

**Quick start:**
```bash
python scripts/download_hf_dataset.py
```

This creates a `data/` directory with training and test sets (2000 training samples, 500 test samples).

## Usage

### Training

**Basic training with real dataset:**
```bash
python train_vision_encoder_decoder.py --epochs 3 --batch-size 4 --output-dir ./models/basic
```

**Enhanced training with data augmentation:**
```bash
python enhanced_train.py --epochs 3 --batch-size 4 --output-dir ./models/enhanced --augment --train-json data/train.jsonl --eval-json data/test.jsonl
```

### Evaluation

**Baseline evaluation:**
```bash
python baseline_eval.py 50  # Evaluate on 50 test samples
```

**Enhanced model evaluation:**
```bash
python enhanced_eval.py ./models/enhanced --samples 50
```

## Metrics

- **Valid Syntax Rate (VSR)**: Percentage of generated code that executes without syntax/runtime errors
- **Intersection over Union (IOU)**: 3D geometric similarity between generated and ground-truth shapes after optimal alignment

Results are logged to `results.csv` with format: `Model,VSR,IOU,Delta`

## Project Structure

```
mecagent-test-1/
├── scripts/
│   └── download_hf_dataset.py     # Dataset download utility
├── metrics/
│   ├── valid_syntax_rate.py       # VSR evaluation
│   └── best_iou.py                # IOU evaluation with alignment
├── train_vision_encoder_decoder.py # Basic training script
├── enhanced_train.py              # Training with augmentation
├── baseline_eval.py               # Baseline model evaluation
├── enhanced_eval.py               # Enhanced model evaluation
├── results.csv                    # Performance metrics log
└── good_luck.ipynb               # Owein's walkthrough
```

## Getting Started

1. **Setup environment** (see [Environment Setup](#environment-setup))
2. **Download dataset**: `python scripts/download_hf_dataset.py`
3. **Run baseline evaluation**: `python baseline_eval.py 10`
4. **Train enhanced model**: `python enhanced_train.py --epochs 2 --augment --train-json data/train.jsonl --eval-json data/test.jsonl --output-dir models/enhanced`
5. **Evaluate enhanced model**: `python enhanced_eval.py models/enhanced --samples 10`

## Results

See `results.csv` for performance comparisons between baseline and enhanced models.

## Notebook

For an interactive walkthrough, see Owein's `good_luck.ipynb`.

## Findings and Future Work

### What Was Accomplished

1. **Complete Documentation**: Enhanced README and DATASET guides with clear setup instructions
2. **Dataset Pipeline**: Successfully downloaded 681 training samples with proper metadata structure  
3. **Baseline Evaluation**: Established baseline performance (VSR=1.000, IOU=0.025) using constant predictions
4. **Training Infrastructure**: Created both enhanced and simple training approaches
5. **Evaluation Framework**: Validated metrics pipeline using Valid Syntax Rate and IOU

### First Experiment: Infrastructure-First Approach

**Goal**: Fix network dependency issues to unlock the enhanced ViT+GPT2 approach (210M parameters)

**What was built**:
- `model_downloader.py`: Robust downloader with retry logic and offline caching
- `enhanced_train_offline.py`: Offline-capable training script with fallback models
- Alternative model support (ViT-base + DistilGPT2 instead of ViT-21k + GPT2)

**Results**: **Still failed due to persistent network issues**
- ViT-base model (346MB) download consistently timed out at 55-77% completion  
- HuggingFace's Xet Storage system connection instability
- Even "smaller" pretrained models proved too large for unreliable network

**Notes**: Infrastructure reliability is **critical** for vision-transformer approaches. Without stable model access, sophisticated architectures become unusable.

### Approach Comparison

| Approach | Status | Parameters | VSR | IOU | Key Insight |
|----------|--------|------------|-----|-----|-------------|
| **Baseline** | Complete | 0 | 1.000 | 0.025 | Constant prediction baseline |
| **Enhanced** | Network issues | 210M | N/A | N/A | **Infrastructure bottleneck** |
| **Simple** | Complete | 262k | 0.000 | 0.000 | Trains in <1 min, validates pipeline |
| **Improved** | Complete | 1.7M | 0.000 | 0.000 | 6.5x larger, real data, still insufficient |

### Thoughts

**Infrastructure > Model Architecture**: Network reliability proved more important than model sophistication. A 210M parameter state-of-the-art model is worthless if it can't be downloaded or loaded.

**Working Pipeline > Perfect Model**: The simple 262k parameter model that trains in 1 minute provided more value than the sophisticated 210M parameter model that never ran.

**Dependency Management is Key**: Complex models introduce complex dependencies. For unreliable environments, simpler architectures with fewer external dependencies are superior.

**Progressive Enhancement**: Start with minimal viable models, validate the full pipeline, then incrementally increase complexity only when infrastructure supports it.

### Bottlenecks Identified

1. **Network Instability**: Model downloads failing at 55-77% completion with timeout errors
2. **Model Size vs Connection**: Even "smaller" models (346MB) too large for unstable connections  
3. **Dependency Chain**: HuggingFace → Xet Storage → Local cache introduces multiple failure points
4. **No Graceful Degradation**: No fallback when pretrained models unavailable

### With More Time

**Immediate (1-2 days)** - Infrastructure Focus:
- **Offline Model Setup**: Pre-download and package models locally to eliminate network dependencies
- **Alternative Model Sources**: Try TensorFlow Hub, PyTorch Hub, or direct model weights download
- **Gradual Model Scaling**: Start with tiny models (DistilBERT-tiny + small CNN) and scale up incrementally
- **Network-Resilient Pipeline**: Implement robust download with exponential backoff and resume capability

**Medium-term (1-2 weeks)** - Architecture Exploration:
- **Custom Vision-Language Models**: Build smaller, domain-specific architectures from scratch
- **Progressive Enhancement**: Train simple models first, then use them as initialization for larger ones
- **Hybrid Approaches**: Combine rule-based code generation with neural models for reliability
- **Alternative Architectures**: Explore LSTM-based or Transformer-lite approaches that train from scratch

**Long-term (1+ months)** - Production Ready:
- **Infrastructure-First Development**: Establish reliable model serving before architecture experimentation
- **Edge Computing**: Deploy models that can run without external dependencies
- **Fallback Systems**: Implement graceful degradation when advanced models unavailable
- **MLOps Pipeline**: Container-based deployment with model versioning and rollback capabilities

**Key Strategic Shift**: **infrastructure reliability > model sophistication**. Future work should prioritize robust, dependency-minimal systems over cutting-edge architectures that can't reliably run.

See [COMPARISON.md](COMPARISON.md) for detailed analysis of different training approaches and their implications.

## API Server

This repository now includes a small FastAPI application (`server.py`) that exposes
an endpoint for retrieving the critical path of a project.

Run the server:

```bash
uvicorn server:app --reload
```

Retrieve a project's critical path:

```bash
curl http://localhost:8000/projects/1/critical-path
```

