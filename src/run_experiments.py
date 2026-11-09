"""
Master experiment runner for CadQuery code generation.

Runs baseline and enhanced models, evaluates them, and produces comparison results.

Usage:
    python src/run_experiments.py --experiment baseline_zero_shot
    python src/run_experiments.py --experiment baseline_few_shot
    python src/run_experiments.py --experiment enhanced_rag
    python src/run_experiments.py --experiment all
"""

import os
import sys
import argparse
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_gencad_dataset, get_eval_subset, sample_few_shot_examples
from src.evaluate import run_evaluation, print_comparison_table, load_results


RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


def run_baseline_zero_shot(eval_data: dict, api_delay: float = 1.0) -> dict:
    """Run baseline zero-shot experiment."""
    from src.baseline_model import BaselineGeminiGenerator

    model = BaselineGeminiGenerator(
        model="gemini-2.5-flash",
        n_shots=0,
        use_cot=False,
        include_reference=False,
        temperature=0.1,
    )

    return run_evaluation(
        model=model,
        eval_data=eval_data,
        save_dir=RESULTS_DIR,
        experiment_name="baseline_zero_shot",
        api_delay=api_delay,
    )


def run_baseline_few_shot(eval_data: dict, api_delay: float = 1.0) -> dict:
    """Run baseline few-shot experiment with fixed examples."""
    from src.baseline_model import BaselineGeminiGenerator
    from src.data_loader import sample_few_shot_examples

    print("Loading few-shot examples from training data...")
    few_shot_examples = sample_few_shot_examples(n=3, seed=42)
    print(f"  Loaded {len(few_shot_examples)} examples")

    model = BaselineGeminiGenerator(
        model="gemini-2.5-flash",
        n_shots=3,
        few_shot_examples=few_shot_examples,
        use_cot=False,
        include_reference=True,
        temperature=0.1,
    )

    return run_evaluation(
        model=model,
        eval_data=eval_data,
        save_dir=RESULTS_DIR,
        experiment_name="baseline_few_shot",
        api_delay=api_delay,
    )


def run_enhanced_rag(eval_data: dict, api_delay: float = 1.5) -> dict:
    """Run enhanced RAG experiment with dynamic retrieval + CoT + repair."""
    from src.enhanced_model import EnhancedRAGGenerator
    from src.image_embedder import ImageEmbedder
    from src.retriever import SimilarityRetriever, build_train_lookup
    import numpy as np

    embeddings_cache = os.path.join(CACHE_DIR, "clip_embeddings.npz")

    # Step 1: Load training data for retrieval index
    print("Loading training data for RAG index...")
    # Use a subset for CPU feasibility
    max_train_samples = 5000
    train_ds = load_gencad_dataset(split="train")
    train_ds_subset = train_ds.select(range(min(max_train_samples, len(train_ds))))

    # Build lookup dict
    train_lookup = build_train_lookup(train_ds_subset)
    print(f"  Loaded {len(train_lookup)} training samples for RAG")

    # Step 2: Build CLIP embedding index
    print("Building CLIP embedding index (CPU, may take a few minutes)...")
    embedder = ImageEmbedder()
    index_embeddings, index_ids = embedder.build_index(
        train_ds_subset,
        cache_path=embeddings_cache,
        max_samples=max_train_samples,
    )
    print(f"  Index shape: {index_embeddings.shape}")

    # Step 3: Create retriever
    retriever = SimilarityRetriever(
        embedder=embedder,
        index_embeddings=index_embeddings,
        index_ids=index_ids,
        train_data_lookup=train_lookup,
        k=3,
    )

    # Step 4: Create enhanced model
    model = EnhancedRAGGenerator(
        retriever=retriever,
        model="gemini-2.5-flash",
        n_shots=3,
        use_cot=True,
        include_reference=True,
        enable_repair=True,
        repair_max_attempts=2,
        temperature=0.1,
    )

    return run_evaluation(
        model=model,
        eval_data=eval_data,
        save_dir=RESULTS_DIR,
        experiment_name="enhanced_rag",
        api_delay=api_delay,
    )


def run_enhanced_hybrid(
    eval_data: dict,
    api_delay: float = 1.5,
    generator_model: str = "gemini-3.1-pro-preview",
    repair_model: str = "gemini-2.5-flash",
    use_dinov2: bool = True,
) -> dict:
    """Run enhanced hybrid experiment with Pro+Flash dual-model and DINOv2."""
    from src.enhanced_model import EnhancedRAGGenerator
    from src.retriever import SimilarityRetriever, build_train_lookup
    import numpy as np

    embeddings_cache = os.path.join(CACHE_DIR, "dinov2_embeddings.npz" if use_dinov2 else "clip_embeddings.npz")
    max_train_samples = 5000

    # Step 1: Load training data for retrieval index
    print("Loading training data for RAG index...")
    train_ds = load_gencad_dataset(split="train")
    train_ds_subset = train_ds.select(range(min(max_train_samples, len(train_ds))))
    train_lookup = build_train_lookup(train_ds_subset)
    print(f"  Loaded {len(train_lookup)} training samples for RAG")

    # Step 2: Build embedding index (DINOv2 or CLIP)
    if use_dinov2:
        from src.image_embedder import DINOv2Embedder
        print("Building DINOv2 embedding index (CPU, may take a few minutes)...")
        embedder = DINOv2Embedder()
    else:
        from src.image_embedder import ImageEmbedder
        print("Building CLIP embedding index (CPU, may take a few minutes)...")
        embedder = ImageEmbedder()

    index_embeddings, index_ids = embedder.build_index(
        train_ds_subset,
        cache_path=embeddings_cache,
        max_samples=max_train_samples,
    )
    print(f"  Index shape: {index_embeddings.shape}")

    # Step 3: Create retriever
    retriever = SimilarityRetriever(
        embedder=embedder,
        index_embeddings=index_embeddings,
        index_ids=index_ids,
        train_data_lookup=train_lookup,
        k=3,
    )

    # Step 4: Create enhanced model with dual-model architecture
    model = EnhancedRAGGenerator(
        retriever=retriever,
        model=generator_model,
        repair_model=repair_model,
        n_shots=3,
        use_cot=True,
        include_reference=True,
        enable_repair=True,
        repair_max_attempts=2,
        temperature=0.1,
    )

    return run_evaluation(
        model=model,
        eval_data=eval_data,
        save_dir=RESULTS_DIR,
        experiment_name="enhanced_hybrid",
        api_delay=api_delay,
    )


def main():
    parser = argparse.ArgumentParser(description="Run CadQuery code generation experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["baseline_zero_shot", "baseline_few_shot", "enhanced_rag", "enhanced_hybrid", "all"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Primary generation model ID (default: gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--repair-model",
        type=str,
        default="gemini-2.5-flash",
        help="Constrained repair model ID (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--use-dinov2",
        action="store_true",
        default=False,
        help="Use DINOv2 embeddings instead of CLIP for retrieval",
    )
    parser.add_argument(
        "--export-step",
        action="store_true",
        default=False,
        help="Export STEP files alongside STL for valid solids",
    )
    parser.add_argument(
        "--use-hundred",
        action="store_true",
        default=True,
        help="Use the hundred_subset for evaluation (default: True)",
    )
    parser.add_argument(
        "--api-delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of evaluation samples to avoid API quotas",
    )
    args = parser.parse_args()

    # Load evaluation data
    print("Loading evaluation data...")
    eval_data = get_eval_subset(split="test", use_hundred=args.use_hundred)
    
    if args.max_samples:
        # Just take the first N keys
        keys = sorted(eval_data.keys())[:args.max_samples]
        eval_data = {k: eval_data[k] for k in keys}
        
    print(f"  Evaluation samples: {len(eval_data)}")

    all_results = []

    if args.experiment in ("baseline_zero_shot", "all"):
        print("\n" + "=" * 70)
        print("RUNNING: Baseline Zero-Shot")
        print("=" * 70)
        r = run_baseline_zero_shot(eval_data, api_delay=args.api_delay)
        all_results.append(r)

    if args.experiment in ("baseline_few_shot", "all"):
        print("\n" + "=" * 70)
        print("RUNNING: Baseline Few-Shot")
        print("=" * 70)
        r = run_baseline_few_shot(eval_data, api_delay=args.api_delay)
        all_results.append(r)

    if args.experiment in ("enhanced_rag", "all"):
        print("\n" + "=" * 70)
        print("RUNNING: Enhanced RAG + CoT + Repair")
        print("=" * 70)
        r = run_enhanced_rag(eval_data, api_delay=args.api_delay)
        all_results.append(r)

    if args.experiment in ("enhanced_hybrid", "all"):
        print("\n" + "=" * 70)
        print("RUNNING: Enhanced Hybrid (Pro+Flash + DINOv2)")
        print("=" * 70)
        r = run_enhanced_hybrid(
            eval_data,
            api_delay=args.api_delay,
            generator_model=args.generator_model,
            repair_model=args.repair_model,
            use_dinov2=args.use_dinov2,
        )
        all_results.append(r)

    # Print comparison
    if len(all_results) > 1:
        print_comparison_table(all_results)

    # Save combined summary
    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    summary = []
    for r in all_results:
        summary.append({
            "experiment": r["experiment_name"],
            "vsr": r["vsr"],
            "mean_iou_best": r["mean_iou_best"],
            "num_samples": r["num_samples"],
            "generation_time_s": r["generation_time_s"],
        })
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
