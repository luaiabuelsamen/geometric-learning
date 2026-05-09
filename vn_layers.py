"""Vector Neurons primitive layers (Deng et al. 2021).

Lifted from FlyingGiraffe/vnn. All operations preserve SO(3) equivariance:
if x is a vector feature of shape (..., C, 3) and R ∈ SO(3), then
applying any layer here to (R · x) is the same as R · (layer(x)).

Convention: vector features are (..., C, 3) where C is the channel dim.
"""
import torch
import torch.nn as nn

EPS = 1e-7


class VNLinear(nn.Module):
    """Channel-wise linear layer that preserves the 3D vector structure."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C_in, 3) -> (..., C_out, 3)
        return self.map_to_feat(x.transpose(-1, -2)).transpose(-1, -2)


class VNLeakyReLU(nn.Module):
    """Equivariant LeakyReLU: project onto a learned direction, half-space gating."""

    def __init__(
        self,
        in_channels: int,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.negative_slope = negative_slope
        out_dirs = 1 if share_nonlinearity else in_channels
        self.map_to_dir = nn.Linear(in_channels, out_dirs, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C, 3)
        d = self.map_to_dir(x.transpose(-1, -2)).transpose(-1, -2)  # (..., C or 1, 3)
        dotprod = (x * d).sum(dim=-1, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(dim=-1, keepdim=True)
        x_out = (
            self.negative_slope * x
            + (1.0 - self.negative_slope)
            * (mask * x + (1.0 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))
        )
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.linear = VNLinear(in_channels, out_channels)
        self.batchnorm = VNBatchNorm(out_channels, dim=4)
        self.leaky = VNLeakyReLU(
            out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope
        )

    def forward(self, x):
        return self.leaky(self.batchnorm(self.linear(x)))


class VNBatchNorm(nn.Module):
    """Batch-norm on vector features, normalising by L2-norm.

    Accepts (B, C, 3) or (B, N, C, 3); detects shape from the input rank.
    """

    def __init__(self, num_features: int, dim: int = 4):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1) + EPS  # (B, ..., C)
        if norm.dim() == 2:  # (B, C)
            norm_bn = self.bn(norm)
        elif norm.dim() == 3:  # (B, N, C)
            B, N, C = norm.shape
            norm_bn = self.bn(norm.reshape(B * N, C)).reshape(B, N, C)
        else:
            raise ValueError(f"unexpected norm shape {norm.shape}")
        return x * (norm_bn / norm).unsqueeze(-1)


class VNMaxPool(nn.Module):
    """Equivariant max-pool: pick the vector with largest projection onto a learned direction."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C, 3) -> (B, C, 3)
        d = self.map_to_dir(x.transpose(-1, -2)).transpose(-1, -2)  # (B, N, C, 3)
        dotprod = (x * d).sum(dim=-1)  # (B, N, C)
        idx = dotprod.max(dim=1).indices  # (B, C)
        # Gather vectors at those indices
        idx_e = idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, 3)  # (B, 1, C, 3)
        out = torch.gather(x, dim=1, index=idx_e).squeeze(1)
        return out


def vn_mean_pool(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return x.mean(dim=dim)


class VNStdFeature(nn.Module):
    """Project vector features into rotation-invariant scalars via a learned frame.

    Returns: (x_std, frame) where x_std is invariant of shape (..., C, 3) and
    frame is the predicted rotation matrix (..., 3, 3).
    """

    def __init__(
        self,
        in_channels: int,
        dim: int = 4,
        normalize_frame: bool = False,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        self.vn1 = VNLinearLeakyReLU(
            in_channels, in_channels // 2,
            share_nonlinearity=share_nonlinearity, negative_slope=negative_slope,
        )
        self.vn2 = VNLinearLeakyReLU(
            in_channels // 2, in_channels // 4,
            share_nonlinearity=share_nonlinearity, negative_slope=negative_slope,
        )
        self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (B, N, C, 3) or (B, C, 3) — predict a (B, 3, 3) or (B, N, 3, 3) frame
        z = self.vn1(x)
        z = self.vn2(z)
        z = self.vn_lin(z.transpose(-1, -2)).transpose(-1, -2)  # (..., 3, 3)

        if self.normalize_frame:
            v1 = z[..., 0, :]
            v1 = v1 / (v1.norm(dim=-1, keepdim=True) + EPS)
            v2 = z[..., 1, :] - (v1 * z[..., 1, :]).sum(dim=-1, keepdim=True) * v1
            v2 = v2 / (v2.norm(dim=-1, keepdim=True) + EPS)
            v3 = torch.cross(v1, v2, dim=-1)
            z = torch.stack([v1, v2, v3], dim=-2)

        # Project x onto the frame: (..., C, 3) @ (..., 3, 3)^T
        x_std = torch.einsum("...cj,...kj->...ck", x, z)
        return x_std, z
