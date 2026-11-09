import os
import sys
import json
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_eval_subset, load_gencad_dataset
from src.enhanced_model import EnhancedRAGGenerator
from src.image_embedder import ImageEmbedder
from src.retriever import SimilarityRetriever, build_train_lookup
from src.baseline_model import _save_json

def main():
    CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    embeddings_cache = os.path.join(CACHE_DIR, "clip_embeddings.npz")

    print("Loading training data for RAG index...")
    max_train_samples = 5000
    train_ds = load_gencad_dataset(split="train")
    train_ds_subset = train_ds.select(range(min(max_train_samples, len(train_ds))))
    train_lookup = build_train_lookup(train_ds_subset)

    print("Building CLIP embedding index...")
    embedder = ImageEmbedder()
    index_embeddings, index_ids = embedder.build_index(
        train_ds_subset,
        cache_path=embeddings_cache,
        max_samples=max_train_samples,
    )

    retriever = SimilarityRetriever(
        embedder=embedder,
        index_embeddings=index_embeddings,
        index_ids=index_ids,
        train_data_lookup=train_lookup,
        k=3,
    )

    print("Initializing EnhancedRAGGenerator with Visual Feedback...")
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

    eval_data = get_eval_subset(split="test", use_hundred=True)
    
    # Just take 10 samples to test visual feedback quickly to save API tokens and time, 
    # but the user requested a benchmark. A benchmark of 100 takes 20-30 mins.
    # We will run the full 100.
    out_path = os.path.join(RESULTS_DIR, "enhanced_visual_predictions.json")
    
    existing_preds = {}
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            data = json.load(f)
            existing_preds = data.get("predictions", {})
            
    filtered_eval_data = {}
    for k, v in eval_data.items():
        if k not in existing_preds or not existing_preds[k].strip():
            filtered_eval_data[k] = v
            
    print(f"Generating predictions for {len(filtered_eval_data)} samples (skipped {len(eval_data) - len(filtered_eval_data)})...")
    
    # Generate batch handles tracking and saving
    results = model.generate_batch(
        filtered_eval_data, 
        delay=1.5, 
        save_path=out_path,
        existing_predictions=existing_preds
    )
    
    print(f"Predictions saved to {out_path}")

if __name__ == "__main__":
    main()
