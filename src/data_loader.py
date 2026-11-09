"""
Data loading utilities for the GenCAD-Code dataset.
"""

import os
from typing import Optional
from datasets import load_dataset
from PIL import Image


# Default cache directory (local to project)
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "huggingface")


def load_gencad_dataset(
    split: str = "test",
    cache_dir: Optional[str] = None,
    num_proc: int = 4,
):
    """Load the CADCODER/GenCAD-Code dataset for a given split.

    Args:
        split: One of "train", "test", "validation"
        cache_dir: HuggingFace cache directory. Defaults to ./cache/huggingface
        num_proc: Number of processes for dataset loading

    Returns:
        HuggingFace Dataset object
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    ds = load_dataset(
        "CADCODER/GenCAD-Code",
        split=split,
        num_proc=num_proc,
        cache_dir=cache_dir,
    )
    return ds


def get_eval_subset(
    split: str = "test",
    use_hundred: bool = True,
    cache_dir: Optional[str] = None,
) -> dict:
    """Get evaluation data as a dictionary keyed by deepcad_id.

    Args:
        split: Dataset split to load
        use_hundred: If True, filter to the hundred_subset only
        cache_dir: HuggingFace cache directory

    Returns:
        Dict of {deepcad_id: {"image": PIL.Image, "cadquery": str, "prompt": str}}
    """
    ds = load_gencad_dataset(split=split, cache_dir=cache_dir)

    if use_hundred:
        ds = ds.filter(lambda x: x["hundred_subset"])

    eval_data = {}
    for item in ds:
        eval_data[item["deepcad_id"]] = {
            "image": item["image"],
            "cadquery": item["cadquery"],
            "prompt": item.get("prompt", ""),
            "token_count": item.get("token_count", 0),
        }

    return eval_data


def sample_few_shot_examples(
    n: int = 5,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    max_token_count: int = 500,
) -> list[dict]:
    """Sample diverse training examples for few-shot prompting.

    Selects examples with moderate token counts (not too simple, not too complex)
    for effective few-shot demonstrations.

    Args:
        n: Number of examples to sample
        seed: Random seed for reproducibility
        cache_dir: HuggingFace cache directory
        max_token_count: Maximum token count to consider (filter very long codes)

    Returns:
        List of {image: PIL.Image, cadquery: str, deepcad_id: str} dicts
    """
    ds = load_gencad_dataset(split="train", cache_dir=cache_dir)

    # Filter to moderate-length examples
    ds_filtered = ds.filter(
        lambda x: 50 <= x.get("token_count", 0) <= max_token_count
    )

    # Shuffle and take n
    ds_sampled = ds_filtered.shuffle(seed=seed).select(range(min(n, len(ds_filtered))))

    examples = []
    for item in ds_sampled:
        examples.append({
            "image": item["image"],
            "cadquery": item["cadquery"],
            "deepcad_id": item["deepcad_id"],
            "prompt": item.get("prompt", ""),
        })

    return examples


def get_training_data_streaming(
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    """Load training data with optional sample limit for memory efficiency.

    Args:
        max_samples: Maximum number of samples to load. None = all.
        cache_dir: HuggingFace cache directory

    Returns:
        HuggingFace Dataset object (possibly truncated)
    """
    ds = load_gencad_dataset(split="train", cache_dir=cache_dir)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds
