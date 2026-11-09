"""
Data exploration script for the GenCAD-Code dataset.

Prints dataset statistics, schema, sample entries, and token count distribution.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    from src.data_loader import load_gencad_dataset
    import numpy as np

    print("=" * 60)
    print("GenCAD-Code Dataset Exploration")
    print("=" * 60)

    # Load test split (smaller, faster)
    print("\n1. Loading test split...")
    test_ds = load_gencad_dataset(split="test")
    print(f"   Test split size: {len(test_ds)}")
    print(f"   Columns: {test_ds.column_names}")

    # Sample entry
    print("\n2. Sample entry:")
    sample = test_ds[0]
    print(f"   deepcad_id:      {sample['deepcad_id']}")
    print(f"   cadquery length: {len(sample['cadquery'])} chars")
    print(f"   token_count:     {sample['token_count']}")
    print(f"   prompt:          {sample.get('prompt', 'N/A')[:100]}...")
    print(f"   hundred_subset:  {sample['hundred_subset']}")
    print(f"   image size:      {sample['image'].size}")

    # Hundred subset count
    hundred_ds = test_ds.filter(lambda x: x["hundred_subset"])
    print(f"\n3. Hundred subset count: {len(hundred_ds)}")

    # Token count statistics
    print("\n4. Token count distribution (test split):")
    token_counts = np.array(test_ds["token_count"])
    print(f"   Min:    {token_counts.min()}")
    print(f"   Max:    {token_counts.max()}")
    print(f"   Mean:   {token_counts.mean():.1f}")
    print(f"   Median: {np.median(token_counts):.1f}")
    print(f"   P95:    {np.percentile(token_counts, 95):.1f}")
    print(f"   P99:    {np.percentile(token_counts, 99):.1f}")

    # Code length distribution
    print("\n5. CadQuery code length distribution:")
    code_lengths = np.array([len(item["cadquery"]) for item in test_ds])
    print(f"   Min:    {code_lengths.min()}")
    print(f"   Max:    {code_lengths.max()}")
    print(f"   Mean:   {code_lengths.mean():.1f}")
    print(f"   Median: {np.median(code_lengths):.1f}")

    # Show a sample CadQuery code
    print("\n6. Sample CadQuery code (first 15 lines):")
    print("-" * 40)
    for line in sample["cadquery"].split("\n")[:15]:
        print(f"   {line}")
    print("   ...")

    # Train split size (just info, don't load fully)
    print("\n7. Loading train split info...")
    train_ds = load_gencad_dataset(split="train")
    print(f"   Train split size: {len(train_ds)}")

    print("\n" + "=" * 60)
    print("Exploration complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
