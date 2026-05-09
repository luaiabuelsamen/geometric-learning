"""Latent dynamics models for the rollout experiment.

Two variants share the same `(z_content, z_pose, action) -> (z_content', z_pose')`
interface but differ in inductive bias:

- HybridDynamics: z_pose evolves by exact group action `R(a) @ z_pose`.
  Only the content latent has a small learned residual MLP. The rigid part
  is closed-form, not learned, and so extrapolates exactly to action
  magnitudes never seen at training time.

- BaselineDynamics: a single MLP predicts deltas to both z_content and the
  flattened z_pose. Must learn rotation dynamics from data.
"""
import torch
import torch.nn as nn

from data import axis_angle_to_matrix


class HybridDynamics(nn.Module):
    def __init__(self, content_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.content_dim = content_dim
        self.content_mlp = nn.Sequential(
            nn.Linear(content_dim + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, content_dim),
        )
        nn.init.zeros_(self.content_mlp[-1].weight)
        nn.init.zeros_(self.content_mlp[-1].bias)

    def forward(self, z_content, z_pose, action):
        R = axis_angle_to_matrix(action)
        z_pose_next = R @ z_pose
        delta_c = self.content_mlp(torch.cat([z_content, action], dim=-1))
        z_content_next = z_content + delta_c
        return z_content_next, z_pose_next


class PureHybridDynamics(nn.Module):
    """Zero learnable params: z_pose evolves by exact group action, z_content frozen.

    Cleanest possible test of the architectural claim — any divergence here
    is encoder/decoder noise, not dynamics-model error.
    """

    def forward(self, z_content, z_pose, action):
        R = axis_angle_to_matrix(action)
        return z_content, R @ z_pose


class BaselineDynamics(nn.Module):
    def __init__(self, content_dim: int = 32, hidden: int = 256, depth: int = 3):
        super().__init__()
        self.content_dim = content_dim
        in_dim = content_dim + 9 + 3  # z_c, z_p flat, action
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, content_dim + 9))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z_content, z_pose, action):
        z_p_flat = z_pose.reshape(z_pose.size(0), 9)
        delta = self.mlp(torch.cat([z_content, z_p_flat, action], dim=-1))
        z_content_next = z_content + delta[:, : self.content_dim]
        z_pose_next = z_pose + delta[:, self.content_dim :].view(-1, 3, 3)
        return z_content_next, z_pose_next
