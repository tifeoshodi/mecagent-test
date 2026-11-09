"""
Chamfer Distance metric for 3D mesh comparison.

Computes symmetric Chamfer Distance by sampling point clouds from
mesh surfaces. Auto-detects CUDA for GPU acceleration; falls back
to CPU numpy on systems without NVIDIA GPUs.
"""

import numpy as np
import trimesh


def chamfer_distance(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    n_samples: int = 10000,
) -> float:
    """Compute symmetric Chamfer Distance between two meshes.
    
    Samples n_samples points from each mesh surface and computes
    the average nearest-neighbor distance in both directions.
    
    Auto-detects CUDA GPU; uses numpy on CPU if unavailable.
    
    Args:
        mesh1: First mesh (e.g., ground truth)
        mesh2: Second mesh (e.g., prediction)
        n_samples: Number of surface points to sample per mesh
        
    Returns:
        Symmetric Chamfer Distance (float, lower is better)
    """
    # Sample point clouds from mesh surfaces
    pts1_np = mesh1.sample(n_samples)
    pts2_np = mesh2.sample(n_samples)

    # Try GPU-accelerated path
    _use_cuda = False
    try:
        import torch
        if torch.cuda.is_available():
            _use_cuda = True
    except ImportError:
        pass

    if _use_cuda:
        import torch
        device = torch.device("cuda")
        pts1 = torch.tensor(pts1_np, dtype=torch.float32, device=device)
        pts2 = torch.tensor(pts2_np, dtype=torch.float32, device=device)

        # Pairwise distances via torch.cdist
        dist1 = torch.cdist(pts1.unsqueeze(0), pts2.unsqueeze(0)).squeeze(0).min(dim=1).values
        dist2 = torch.cdist(pts2.unsqueeze(0), pts1.unsqueeze(0)).squeeze(0).min(dim=1).values

        return (dist1.mean() + dist2.mean()).item() / 2.0
    else:
        # CPU numpy path (default on systems without NVIDIA GPU)
        from scipy.spatial import cKDTree

        tree1 = cKDTree(pts1_np)
        tree2 = cKDTree(pts2_np)

        dist1, _ = tree2.query(pts1_np)  # For each point in pts1, nearest in pts2
        dist2, _ = tree1.query(pts2_np)  # For each point in pts2, nearest in pts1

        return (np.mean(dist1) + np.mean(dist2)) / 2.0
