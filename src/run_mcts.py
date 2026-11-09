"""Run MCTS-based CAD generation benchmark."""

import os
import sys
import json
import argparse
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_eval_subset, load_gencad_dataset
from src.mcts_generator import MCTSGenerator
from src.image_embedder import ImageEmbedder
from src.baseline_model import _save_json


def main():
    parser = argparse.ArgumentParser(description="Run MCTS CAD generation benchmark")
    parser.add_argument("--n", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--budget", type=int, default=20, help="MCTS iterations per sample")
    parser.add_argument("--candidates", type=int, default=3, help="Candidate steps per expansion")
    parser.add_argument("--max-depth", type=int, default=12, help="Max CSG tree depth")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    parser.add_argument("--api-provider", type=str, default="gemini", choices=["gemini", "ollama"], help="API provider to use")
    args = parser.parse_args()

    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Smart defaults for Ollama
    if args.api_provider == "ollama":
        if args.budget == 20: args.budget = 5
        if args.candidates == 3: args.candidates = 2
        model_name = "llava"
    else:
        model_name = "gemini-2.5-flash"

    # Build CLIP embedder for scoring
    print("Building CLIP embedder for MCTS scoring...")
    embedder = ImageEmbedder()

    print(f"Initializing MCTS Generator ({args.api_provider}, model={model_name}, budget={args.budget}, candidates={args.candidates})...")
    model = MCTSGenerator(
        model=model_name,
        budget=args.budget,
        n_candidates=args.candidates,
        max_depth=args.max_depth,
        temperature=0.4,
        embedder=embedder,
        api_provider=args.api_provider,
    )

    # Load evaluation data
    use_hundred = args.n >= 100
    eval_data = get_eval_subset(split="test", use_hundred=use_hundred)

    if args.n < len(eval_data):
        keys = sorted(eval_data.keys())[:args.n]
        eval_data = {k: eval_data[k] for k in keys}

    out_path = os.path.join(RESULTS_DIR, f"mcts_{args.api_provider}_predictions.json")

    existing_predictions = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_predictions = data.get("predictions", {})
                print(f"Loaded {len(existing_predictions)} existing predictions from {out_path}. Resuming...")
        except Exception as e:
            print(f"Could not load existing predictions: {e}")

    # Filter out items we've already predicted
    keys_to_predict = [k for k in eval_data.keys() if k not in existing_predictions]
    eval_data = {k: eval_data[k] for k in keys_to_predict}

    if not eval_data:
        print("All samples already predicted. Exiting.")
        sys.exit(0)

    print(f"Generating predictions for {len(eval_data)} samples...")
    results = model.generate_batch(
        eval_data,
        delay=1.5,
        save_path=out_path,
        verbose=args.verbose,
        existing_predictions=existing_predictions
    )

    _save_json(out_path, {"predictions": results, "errors": {}})
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
