"""
Image embedding using CLIP for similarity-based retrieval.

Computes CLIP embeddings for CAD object images to enable
retrieval-augmented few-shot prompting.
"""

import os
import numpy as np
from PIL import Image
from typing import Optional
from tqdm import tqdm


def _load_clip_model(model_name: str = "openai/clip-vit-base-patch32"):
    """Load CLIP model and processor. Lazy import to avoid torch at module level."""
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return model, processor


class ImageEmbedder:
    """Compute CLIP embeddings for images for similarity search."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Args:
            model_name: HuggingFace CLIP model identifier
        """
        self.model_name = model_name
        self.model, self.processor = _load_clip_model(model_name)

    def embed_single(self, image: Image.Image) -> np.ndarray:
        """Compute embedding for a single image.

        Args:
            image: PIL Image

        Returns:
            1D numpy array of shape (embed_dim,)
        """
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        
        # Extract tensor depending on output type
        if hasattr(outputs, "detach"):
            embedding = outputs.detach().cpu().numpy()[0]
        elif hasattr(outputs, "image_embeds"):
            embedding = outputs.image_embeds.detach().cpu().numpy()[0]
        elif hasattr(outputs, "pooler_output"):
            embedding = outputs.pooler_output.detach().cpu().numpy()[0]
        else:
            embedding = outputs[1].detach().cpu().numpy()[0]

        # L2 normalise
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def embed_batch(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute embeddings for a batch of images.

        Args:
            images: List of PIL Images
            batch_size: Processing batch size
            show_progress: Show tqdm progress bar

        Returns:
            numpy array of shape (N, embed_dim)
        """
        import torch

        all_embeddings = []
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding images", total=len(images) // batch_size + 1)

        for start in iterator:
            batch_images = images[start : start + batch_size]
            inputs = self.processor(images=batch_images, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            
            # Extract tensor depending on output type
            if hasattr(outputs, "detach"):
                embeddings = outputs.detach().cpu().numpy()
            elif hasattr(outputs, "image_embeds"):
                embeddings = outputs.image_embeds.detach().cpu().numpy()
            elif hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output.detach().cpu().numpy()
            else:
                embeddings = outputs[1].detach().cpu().numpy()
            # L2 normalise each row
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def build_index(
        self,
        train_data,
        cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Build an embedding index from training data.

        Args:
            train_data: HuggingFace dataset or dict of {id: {"image": ...}}
            cache_path: Path to save/load cached embeddings
            max_samples: Limit number of samples to embed

        Returns:
            Tuple of (embeddings array, list of IDs)
        """
        # Check cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return data["embeddings"], data["ids"].tolist()

        # Extract images and IDs
        if isinstance(train_data, dict):
            ids = sorted(train_data.keys())
            if max_samples:
                ids = ids[:max_samples]
            images = [train_data[i]["image"] for i in ids]
        else:
            # HuggingFace Dataset
            if max_samples:
                train_data = train_data.select(range(min(max_samples, len(train_data))))
            ids = [item["deepcad_id"] for item in train_data]
            images = [item["image"] for item in train_data]

        print(f"Building embedding index for {len(images)} images...")
        embeddings = self.embed_batch(images)

        # Cache to disk
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, embeddings=embeddings, ids=np.array(ids))
            print(f"Saved embeddings to {cache_path}")

        return embeddings, ids


class DINOv2Embedder:
    """DINOv2-based image embeddings for geometric structure retrieval.
    
    Uses self-supervised vision features (no text alignment) that encode
    actual spatial structure rather than semantic similarity. Better for
    mechanical geometry and CAD sketch retrieval than CLIP.
    
    Output dimension: 768 (dinov2-base) or 384 (dinov2-small).
    """

    def __init__(self, model_name: str = "facebook/dinov2-base"):
        """
        Args:
            model_name: HuggingFace DINOv2 model identifier
        """
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_single(self, image: Image.Image) -> np.ndarray:
        """Compute DINOv2 embedding for a single image.

        Args:
            image: PIL Image

        Returns:
            1D numpy array of shape (embed_dim,) — L2 normalised
        """
        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use CLS token embedding (first token of last hidden state)
        embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def embed_batch(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute DINOv2 embeddings for a batch of images.

        Args:
            images: List of PIL Images
            batch_size: Processing batch size
            show_progress: Show tqdm progress bar

        Returns:
            numpy array of shape (N, embed_dim)
        """
        import torch

        all_embeddings = []
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding images (DINOv2)", total=len(images) // batch_size + 1)

        for start in iterator:
            batch_images = images[start : start + batch_size]
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # CLS token for each image in batch
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            # L2 normalise each row
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def build_index(
        self,
        train_data,
        cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Build an embedding index from training data.

        Uses versioned cache paths (_dinov2.npz) to prevent loading
        stale CLIP embeddings.

        Args:
            train_data: HuggingFace dataset or dict of {id: {"image": ...}}
            cache_path: Path to save/load cached embeddings
            max_samples: Limit number of samples to embed

        Returns:
            Tuple of (embeddings array, list of IDs)
        """
        # Version the cache path to avoid CLIP/DINOv2 mismatch
        if cache_path:
            if not cache_path.endswith("_dinov2.npz"):
                cache_path = cache_path.replace(".npz", "_dinov2.npz")

        # Check cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached DINOv2 embeddings from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return data["embeddings"], data["ids"].tolist()

        # Extract images and IDs
        if isinstance(train_data, dict):
            ids = sorted(train_data.keys())
            if max_samples:
                ids = ids[:max_samples]
            images = [train_data[i]["image"] for i in ids]
        else:
            # HuggingFace Dataset
            if max_samples:
                train_data = train_data.select(range(min(max_samples, len(train_data))))
            ids = [item["deepcad_id"] for item in train_data]
            images = [item["image"] for item in train_data]

        print(f"Building DINOv2 embedding index for {len(images)} images...")
        embeddings = self.embed_batch(images)

        # Cache to disk
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, embeddings=embeddings, ids=np.array(ids))
            print(f"Saved DINOv2 embeddings to {cache_path}")

        return embeddings, ids
