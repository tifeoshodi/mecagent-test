"""
Topological invariant (Betti number) computation for 3D meshes.

Computes:
- b0: Number of connected components
- b1: Number of tunnels/holes (genus-related)
- b2: Number of enclosed void cavities
- Euler characteristic: χ = V - E + F
"""

import numpy as np
import trimesh


def compute_betti_numbers(mesh: trimesh.Trimesh) -> dict:
    """Compute topological Betti numbers for a triangulated mesh.
    
    Uses the Euler characteristic relation for closed orientable surfaces:
        χ = V - E + F = 2(b0) - 2(genus)  [for closed surfaces]
        b1 = 2*b0 - χ - b2                [derived]
    
    Args:
        mesh: A trimesh.Trimesh object
        
    Returns:
        Dict with keys: b0, b1, b2, euler_char, is_watertight
    """
    # b0: connected components
    try:
        components = trimesh.graph.connected_components(
            mesh.face_adjacency, min_len=1
        )
        b0 = len(components)
    except Exception:
        # Fallback: if adjacency fails, treat as single component
        b0 = 1

    # Euler characteristic: χ = V - E + F
    V = len(mesh.vertices)
    E = len(mesh.edges_unique)
    F = len(mesh.faces)
    euler_char = V - E + F

    # b2: number of enclosed voids (for watertight meshes, one per component)
    is_watertight = mesh.is_watertight
    b2 = b0 if is_watertight else 0

    # b1: tunnels/holes
    # From χ = b0 - b1 + b2  =>  b1 = b0 - χ + b2
    b1 = b0 - euler_char + b2
    b1 = max(b1, 0)  # Clamp to non-negative

    return {
        "b0": b0,
        "b1": b1,
        "b2": b2,
        "euler_char": euler_char,
        "is_watertight": is_watertight,
    }


def betti_match_score(betti_gt: dict, betti_pred: dict) -> float:
    """Compute a simple match score between two Betti number dicts.
    
    Returns 1.0 if all Betti numbers match, 0.0 if completely different.
    Weighted: b0 (40%), b1 (40%), b2 (20%) since b0 and b1 are most
    discriminative for CAD parts.
    
    Args:
        betti_gt: Ground truth Betti numbers
        betti_pred: Predicted Betti numbers
    
    Returns:
        Float score in [0.0, 1.0]
    """
    score = 0.0
    if betti_gt["b0"] == betti_pred["b0"]:
        score += 0.4
    if betti_gt["b1"] == betti_pred["b1"]:
        score += 0.4
    if betti_gt["b2"] == betti_pred["b2"]:
        score += 0.2
    return score
