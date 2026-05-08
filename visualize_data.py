"""Render ModelNet10 point clouds + rotation visualizations for the README."""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import (
    CLASSES,
    list_modelnet10_paths,
    load_mesh_as_pointcloud,
    random_rotation_matrix_so3,
    rotate_pointcloud,
    rotation_matrix_yaw,
)


def plot_pc(ax, pts, title=None, color_axis: int = 2):
    pts = pts.detach().cpu().numpy() if torch.is_tensor(pts) else pts
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, color_axis], cmap="viridis")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if title is not None:
        ax.set_title(title, fontsize=9)


def main(out_dir: str = "assets"):
    os.makedirs(out_dir, exist_ok=True)
    n_points = 1024

    # 1) One sample per class
    fig = plt.figure(figsize=(20, 8))
    for i, c in enumerate(CLASSES):
        paths = list_modelnet10_paths(split="train", classes=[c])
        pts = load_mesh_as_pointcloud(paths[0], n_points=n_points)
        ax = fig.add_subplot(2, 5, i + 1, projection="3d")
        plot_pc(ax, pts, title=c)
    fig.suptitle("ModelNet10 — 1024 surface-sampled points per object", fontsize=12)
    fig.tight_layout()
    out = os.path.join(out_dir, "modelnet10_samples.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # 2) Yaw rotation sweep (around z)
    chair = load_mesh_as_pointcloud(
        list_modelnet10_paths(split="train", classes=["chair"])[0], n_points
    )
    n = 8
    thetas = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    R = rotation_matrix_yaw(thetas)
    pts_rot = rotate_pointcloud(chair.unsqueeze(0).expand(n, -1, -1), R)
    fig = plt.figure(figsize=(2.4 * n, 2.6))
    for i in range(n):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        plot_pc(ax, pts_rot[i], title=f"theta={thetas[i].item():.2f}")
    fig.suptitle("Yaw rotation sweep (around z-axis), single chair", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, "yaw_rotation_sweep.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # 3) Random SO(3) rotations
    sofa = load_mesh_as_pointcloud(
        list_modelnet10_paths(split="train", classes=["sofa"])[0], n_points
    )
    torch.manual_seed(0)
    n_rots = 8
    R_so3 = random_rotation_matrix_so3(n_rots)
    pts_rot = rotate_pointcloud(sofa.unsqueeze(0).expand(n_rots, -1, -1), R_so3)
    fig = plt.figure(figsize=(2.4 * (n_rots + 1), 2.6))
    ax = fig.add_subplot(1, n_rots + 1, 1, projection="3d")
    plot_pc(ax, sofa, title="original")
    for i in range(n_rots):
        ax = fig.add_subplot(1, n_rots + 1, i + 2, projection="3d")
        plot_pc(ax, pts_rot[i], title=f"rand R{i}")
    fig.suptitle("Random SO(3) rotations, single sofa", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, "so3_random_rotations.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # 4) Training pairs: (x, R(x)) across classes
    pair_classes = ["chair", "table", "sofa", "toilet", "monitor", "bed"]
    torch.manual_seed(1)
    R_pairs = random_rotation_matrix_so3(len(pair_classes))
    fig = plt.figure(figsize=(2.4 * len(pair_classes), 4.8))
    for i, c in enumerate(pair_classes):
        pts = load_mesh_as_pointcloud(
            list_modelnet10_paths(split="train", classes=[c])[0], n_points
        )
        pts_r = rotate_pointcloud(pts.unsqueeze(0), R_pairs[i : i + 1]).squeeze(0)
        ax_top = fig.add_subplot(2, len(pair_classes), i + 1, projection="3d")
        plot_pc(ax_top, pts, title=c)
        ax_bot = fig.add_subplot(
            2, len(pair_classes), len(pair_classes) + i + 1, projection="3d"
        )
        plot_pc(ax_bot, pts_r, title="rotated")
    fig.suptitle("Training pairs: x (top), rotate(x, R) with random R in SO(3) (bottom)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, "training_pairs_3d.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
