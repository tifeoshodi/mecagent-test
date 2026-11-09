"""
Similarity-based retriever for finding training examples most similar
to a query image using pre-computed embeddings (CLIP or DINOv2).
"""

import numpy as np
from PIL import Image
from typing import Optional, Union

from src.image_embedder import ImageEmbedder, DINOv2Embedder


class SimilarityRetriever:
    """Retrieve top-k similar training examples for a query image."""

    def __init__(
        self,
        embedder: Union[ImageEmbedder, DINOv2Embedder],
        index_embeddings: np.ndarray,
        index_ids: list[str],
        train_data_lookup: dict,
        k: int = 3,
    ):
        """
        Args:
            embedder: ImageEmbedder instance for query embedding
            index_embeddings: Pre-computed embeddings (N, embed_dim)
            index_ids: IDs corresponding to each embedding row
            train_data_lookup: Dict of {id: {"image": PIL.Image, "cadquery": str, ...}}
            k: Number of examples to retrieve
        """
        self.embedder = embedder
        self.index_embeddings = index_embeddings
        self.index_ids = index_ids
        self.train_data_lookup = train_data_lookup
        self.k = k

    def retrieve(self, query_image: Image.Image) -> list[dict]:
        """Retrieve the top-k most similar training examples.

        Args:
            query_image: Query PIL Image

        Returns:
            List of dicts [{image, cadquery, deepcad_id, similarity}]
        """
        # Embed query
        query_embedding = self.embedder.embed_single(query_image)

        # Cosine similarity (embeddings are already L2-normalised)
        similarities = self.index_embeddings @ query_embedding

        # Top-k indices
        top_k_indices = np.argsort(similarities)[::-1][: self.k]

        results = []
        for idx in top_k_indices:
            item_id = self.index_ids[idx]
            if item_id in self.train_data_lookup:
                data = self.train_data_lookup[item_id]
                results.append({
                    "image": data["image"],
                    "cadquery": data["cadquery"],
                    "deepcad_id": item_id,
                    "similarity": float(similarities[idx]),
                })

        return results


def build_train_lookup(train_dataset, max_samples: Optional[int] = None) -> dict:
    """Build a lookup dictionary from training dataset.

    Args:
        train_dataset: HuggingFace dataset
        max_samples: Limit number of samples

    Returns:
        Dict of {deepcad_id: {"image": PIL.Image, "cadquery": str}}
    """
    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))

    lookup = {}
    for item in train_dataset:
        lookup[item["deepcad_id"]] = {
            "image": item["image"],
            "cadquery": item["cadquery"],
            "prompt": item.get("prompt", ""),
        }
    return lookup
