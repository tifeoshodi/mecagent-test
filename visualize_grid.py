import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
from typing import Iterable


def _plot_trimesh(ax, mesh: trimesh.Trimesh) -> None:
    """Render a trimesh object on a 3D axis without axis ticks."""
    if not mesh.is_watertight:
        mesh = mesh.convex_hull
    faces = mesh.faces
    verts = mesh.vertices
    poly3d = [[verts[idx] for idx in face] for face in faces]
    collection = Poly3DCollection(poly3d, facecolor="lightblue", edgecolor="k", linewidths=0.1, alpha=0.8)
    ax.add_collection3d(collection)
    scale = verts.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()


def plot_comparison_grid(images: Iterable[np.ndarray], baseline_meshes: Iterable[trimesh.Trimesh], enhanced_meshes: Iterable[trimesh.Trimesh]):
    """Display a grid with columns: image, baseline mesh, enhanced mesh."""
    images = list(images)
    baseline_meshes = list(baseline_meshes)
    enhanced_meshes = list(enhanced_meshes)
    assert len(images) == len(baseline_meshes) == len(enhanced_meshes), "All inputs must have the same length"

    n_rows = len(images)
    fig = plt.figure(figsize=(9, 3 * n_rows))

    for row in range(n_rows):
        # Image
        ax_img = fig.add_subplot(n_rows, 3, row * 3 + 1)
        ax_img.imshow(images[row])
        ax_img.set_title("Image")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Baseline mesh
        ax_base = fig.add_subplot(n_rows, 3, row * 3 + 2, projection="3d")
        _plot_trimesh(ax_base, baseline_meshes[row])
        ax_base.set_title("Baseline")

        # Enhanced mesh
        ax_enh = fig.add_subplot(n_rows, 3, row * 3 + 3, projection="3d")
        _plot_trimesh(ax_enh, enhanced_meshes[row])
        ax_enh.set_title("Enhanced")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage with dummy data
    imgs = [np.random.rand(64, 64, 3) for _ in range(6)]
    cube = trimesh.creation.box()
    sphere = trimesh.creation.icosphere()
    bases = [cube for _ in range(6)]
    enh = [sphere for _ in range(6)]
    plot_comparison_grid(imgs, bases, enh)
    plt.show()
