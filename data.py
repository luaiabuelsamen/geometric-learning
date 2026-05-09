"""ModelNet10 point cloud dataset + 3D rotation utilities.

Loads .off meshes, samples uniformly on the surface, centers + scales to
unit sphere. Provides yaw (SO(2)) and full SO(3) rotation matrices.
"""
import glob
import math
import os
from typing import List, Optional

import numpy as np
import torch
import trimesh

DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "data_files", "ModelNet10")
CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]


def load_mesh_as_pointcloud(off_path: str, n_points: int = 1024) -> torch.Tensor:
    """Load .off mesh, sample uniformly on surface, center, scale to unit sphere."""
    mesh = trimesh.load(off_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    pts = torch.from_numpy(np.asarray(points)).float()
    pts = pts - pts.mean(dim=0, keepdim=True)
    scale = pts.norm(dim=-1).max().clamp_min(1e-8)
    return pts / scale


def list_modelnet10_paths(
    root: str = DEFAULT_ROOT,
    split: str = "train",
    classes: Optional[List[str]] = None,
) -> List[str]:
    if classes is None:
        classes = CLASSES
    paths: List[str] = []
    for c in classes:
        paths.extend(sorted(glob.glob(os.path.join(root, c, split, "*.off"))))
    return paths


class PointCloudDataset(torch.utils.data.Dataset):
    """In-memory cached point cloud dataset."""

    def __init__(self, paths: List[str], n_points: int = 1024, cache: bool = True):
        self.paths = paths
        self.n_points = n_points
        self.cache = cache
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.cache and idx in self._cache:
            return self._cache[idx]
        pts = load_mesh_as_pointcloud(self.paths[idx], self.n_points)
        if self.cache:
            self._cache[idx] = pts
        return pts


def rotation_matrix_yaw(theta: torch.Tensor) -> torch.Tensor:
    """Yaw rotation matrices around z-axis. theta: (B,) -> (B, 3, 3)."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    return torch.stack(
        [
            torch.stack([c, -s, z], dim=-1),
            torch.stack([s, c, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ],
        dim=1,
    )


def random_rotation_matrix_so3(n: int = 1, device: str = "cpu") -> torch.Tensor:
    """Uniform random rotation matrices on SO(3) via QR of Gaussian.

    Returns (n, 3, 3) with det = +1.
    """
    A = torch.randn(n, 3, 3, device=device)
    Q, R = torch.linalg.qr(A)
    # Make Q have the same sign convention as R's diagonal so that det(Q) = +1
    d = torch.diagonal(R, dim1=-2, dim2=-1).sign()
    Q = Q * d.unsqueeze(-2)
    det = torch.det(Q)
    flip = det < 0
    if flip.any():
        Q[flip, :, 0] = -Q[flip, :, 0]
    return Q


def rotate_pointcloud(pts: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply rotation R to point cloud pts.

    pts: (..., n, 3)
    R:   (..., 3, 3)
    Each row vector p in pts becomes p @ R^T (= R p as a column vector).
    """
    return pts @ R.transpose(-1, -2)


def sample_pair_batch(
    dataset: PointCloudDataset,
    batch_size: int,
    device: str = "cpu",
    rotation: str = "so3",
):
    """Return (x, R, x_rot) where x_rot = rotate(x, R).

    rotation: 'so3' or 'yaw'.
    """
    idx = torch.randint(0, len(dataset), (batch_size,))
    x = torch.stack([dataset[int(i)] for i in idx]).to(device)
    if rotation == "so3":
        R = random_rotation_matrix_so3(batch_size, device=device)
    elif rotation == "yaw":
        theta = torch.rand(batch_size, device=device) * 2 * math.pi
        R = rotation_matrix_yaw(theta)
    else:
        raise ValueError(rotation)
    x_rot = rotate_pointcloud(x, R)
    return x, R, x_rot


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula. axis_angle: (..., 3) -> (..., 3, 3).

    Direction = rotation axis, magnitude = rotation angle (radians).
    """
    angle = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = axis_angle / angle
    # angle for trig — keep one extra dim for broadcasting against (..., 3, 3)
    angle_b = angle.unsqueeze(-1)
    ax, ay, az = axis.unbind(dim=-1)
    zeros = torch.zeros_like(ax)
    K = torch.stack(
        [
            torch.stack([zeros, -az, ay], dim=-1),
            torch.stack([az, zeros, -ax], dim=-1),
            torch.stack([-ay, ax, zeros], dim=-1),
        ],
        dim=-2,
    )
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand_as(K)
    R = I + torch.sin(angle_b) * K + (1.0 - torch.cos(angle_b)) * (K @ K)
    return R


def sample_trajectory_batch(
    dataset: PointCloudDataset,
    batch_size: int,
    T: int,
    action_mag_range,
    device: str = "cpu",
):
    """Generate trajectories of point clouds rotated by per-step axis-angle actions.

    Returns:
        xs: (B, T+1, N, 3) cumulative rotated point clouds
        actions: (B, T, 3) axis-angle actions, ‖a_t‖ ∈ action_mag_range
    """
    idx = torch.randint(0, len(dataset), (batch_size,))
    x0 = torch.stack([dataset[int(i)] for i in idx]).to(device)

    axes = torch.randn(batch_size, T, 3, device=device)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    lo, hi = action_mag_range
    mags = torch.rand(batch_size, T, 1, device=device) * (hi - lo) + lo
    actions = axes * mags  # (B, T, 3)

    xs = [x0]
    x_curr = x0
    for t in range(T):
        R = axis_angle_to_matrix(actions[:, t])  # (B, 3, 3)
        x_next = rotate_pointcloud(x_curr, R)
        xs.append(x_next)
        x_curr = x_next
    xs = torch.stack(xs, dim=1)  # (B, T+1, N, 3)
    return xs, actions
