"""Perceiver-style point cloud autoencoder with SO(3)-structured latent.

Encoder: 1024(or 512) input points -> embed -> cross-attend into K=32 learned
latents -> N self-attention layers -> mean pool -> (z_content, z_pose=3x3 mat).

Decoder: (z_content, z_pose) -> project to K latent tokens -> self-attention ->
cross-attended by N learned point queries -> per-token linear -> 3D coordinates.

Group action: R @ z_pose for the latent rotation; equivariance loss + cycle loss
make the pose latent track the rotation of the input.
"""
import torch
import torch.nn as nn


class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, q, kv):
        a, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        q = q + a
        return q + self.mlp(self.norm2(q))


class SelfAttnBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x_n = self.norm1(x)
        a, _ = self.attn(x_n, x_n, x_n, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class PointCloudEncoder(nn.Module):
    def __init__(
        self,
        content_dim: int = 32,
        dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        n_latents: int = 32,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.point_embed = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.latents = nn.Parameter(torch.zeros(1, n_latents, dim))
        self.cross_attn = CrossAttnBlock(dim, n_heads)
        self.self_attn = nn.ModuleList(
            [SelfAttnBlock(dim, n_heads) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, content_dim + 9)
        nn.init.trunc_normal_(self.latents, std=0.02)

    def forward(self, x: torch.Tensor):
        # x: (B, N, 3)
        B = x.size(0)
        kv = self.point_embed(x)
        q = self.latents.expand(B, -1, -1)
        q = self.cross_attn(q, kv)
        for blk in self.self_attn:
            q = blk(q)
        z = self.norm(q.mean(dim=1))
        h = self.head(z)
        z_content = h[:, : self.content_dim]
        z_pose = h[:, self.content_dim :].view(B, 3, 3)
        return z_content, z_pose


class PointCloudDecoder(nn.Module):
    def __init__(
        self,
        content_dim: int = 32,
        dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        n_latents: int = 32,
        n_points: int = 512,
    ):
        super().__init__()
        self.dim = dim
        self.n_latents = n_latents
        self.n_points = n_points
        self.input_proj = nn.Linear(content_dim + 9, n_latents * dim)
        self.latent_pos = nn.Parameter(torch.zeros(1, n_latents, dim))
        self.self_attn = nn.ModuleList(
            [SelfAttnBlock(dim, n_heads) for _ in range(n_layers)]
        )
        self.queries = nn.Parameter(torch.zeros(1, n_points, dim))
        self.cross_attn = CrossAttnBlock(dim, n_heads)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 3)
        nn.init.trunc_normal_(self.latent_pos, std=0.02)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, z_content: torch.Tensor, z_pose: torch.Tensor) -> torch.Tensor:
        B = z_content.size(0)
        z = torch.cat([z_content, z_pose.reshape(B, 9)], dim=-1)
        latents = self.input_proj(z).view(B, self.n_latents, self.dim) + self.latent_pos
        for blk in self.self_attn:
            latents = blk(latents)
        out = self.cross_attn(self.queries.expand(B, -1, -1), latents)
        return self.head(self.norm(out))  # (B, n_points, 3)


def rotate_pose_matrix(z_pose: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Group action on the pose latent: R @ M.

    z_pose: (B, 3, 3), R: (B, 3, 3) -> (B, 3, 3).
    Each column of M transforms as a 3D vector under R.
    """
    return R @ z_pose


def chamfer_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance for unordered point sets.

    p: (B, N, 3), q: (B, M, 3) -> scalar.
    """
    diff = p.unsqueeze(2) - q.unsqueeze(1)
    d2 = (diff**2).sum(-1)
    p_to_q = d2.min(dim=2).values.mean(dim=1)
    q_to_p = d2.min(dim=1).values.mean(dim=1)
    return (p_to_q + q_to_p).mean()
