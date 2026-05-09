"""Vector-Neurons encoder + FoldingNet decoder.

Encoder is hard SO(3)-equivariant by construction — every layer commutes
with rotation. Decoder explicitly factors out the rotation: it folds a
fixed 2D grid into a canonical-frame point cloud conditioned on z_content,
then applies z_pose as a final 3×3 matmul. So `dec(z_c, R @ z_p) =
R @ dec(z_c, z_p)` is an architectural identity, not a soft loss.
"""
import torch
import torch.nn as nn

from vn_layers import (
    VNLinear,
    VNLinearLeakyReLU,
    VNStdFeature,
    vn_mean_pool,
)


class VNEncoder(nn.Module):
    def __init__(self, content_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.content_dim = content_dim
        # Lift the 3D point (1 vector feature) to `hidden` vector features
        self.lift = VNLinear(1, hidden)
        self.conv1 = VNLinearLeakyReLU(hidden, hidden * 2)
        self.conv2 = VNLinearLeakyReLU(hidden * 2, hidden * 2)
        self.conv3 = VNLinearLeakyReLU(hidden * 2, hidden * 4)
        # Heads
        # Pose head: 3 vector features stacked as columns of a 3x3 matrix
        self.pose_head = VNLinear(hidden * 4, 3)
        # Content head: invariant scalars via learned frame projection
        self.std_feature = VNStdFeature(hidden * 4, dim=3, normalize_frame=True)
        self.content_head = nn.Sequential(
            nn.Linear(hidden * 4 * 3, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, content_dim),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, N, 3) -> z_content: (B, content_dim), z_pose: (B, 3, 3)
        x_v = x.unsqueeze(-2)  # (B, N, 1, 3) — each point is one vector feature
        h = self.lift(x_v)  # (B, N, hidden, 3)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        # Mean-pool over points (equivariant)
        h_global = vn_mean_pool(h, dim=1)  # (B, hidden*4, 3)
        # Pose: 3 vector features stacked as a matrix
        z_pose = self.pose_head(h_global)  # (B, 3, 3) where each row is a vector feature
        # Re-arrange so that z_pose acts as left-multiply: rows of z_pose ARE the
        # 3 equivariant vectors that should rotate with the input. Use as columns
        # of the rotation operator — i.e. transpose to put vectors in columns.
        z_pose = z_pose.transpose(-1, -2)
        # Content: invariant scalars
        x_std, _ = self.std_feature(h_global)  # (B, C, 3) invariant
        z_content = self.content_head(x_std.flatten(-2))
        return z_content, z_pose


class FoldingDecoder(nn.Module):
    """FoldingNet-style decoder with explicit rotation factorization.

    1. Fold a fixed 2D grid into a canonical-frame point cloud conditioned on z_content.
    2. Apply z_pose @ x_canonical^T at the end.

    This makes `dec(z_c, R @ z_p) = R @ dec(z_c, z_p)` an architectural identity.
    """

    def __init__(self, content_dim: int = 32, n_points_per_side: int = 23, hidden: int = 256):
        super().__init__()
        n = n_points_per_side
        self.n_grid = n * n
        coords = torch.linspace(-1.0, 1.0, n)
        gx, gy = torch.meshgrid(coords, coords, indexing="ij")
        grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (n_grid, 2)
        self.register_buffer("grid", grid)

        self.fold1 = nn.Sequential(
            nn.Linear(content_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.fold2 = nn.Sequential(
            nn.Linear(content_dim + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, z_content: torch.Tensor, z_pose: torch.Tensor) -> torch.Tensor:
        B = z_content.size(0)
        n = self.n_grid
        z_c_rep = z_content.unsqueeze(1).expand(B, n, -1)  # (B, n, content_dim)
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)  # (B, n, 2)
        x1 = self.fold1(torch.cat([z_c_rep, grid], dim=-1))  # (B, n, 3) canonical
        x2 = self.fold2(torch.cat([z_c_rep, x1], dim=-1))  # (B, n, 3) refined canonical
        # Apply rotation: each point as a row, apply z_pose on the right
        # so that p_rot = p @ z_pose.T
        x_rotated = x2 @ z_pose.transpose(-1, -2)
        return x_rotated


def rotate_pose_matrix(z_pose: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return R @ z_pose


def chamfer_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    diff = p.unsqueeze(2) - q.unsqueeze(1)
    d2 = (diff**2).sum(-1)
    p_to_q = d2.min(dim=2).values.mean(dim=1)
    q_to_p = d2.min(dim=1).values.mean(dim=1)
    return (p_to_q + q_to_p).mean()
