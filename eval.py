"""Evaluate the trained SO(3) point cloud autoencoder.

Plots:
  - z_pose entries vs yaw angle (should track R_z(theta) @ z_pose_0 entries)
  - content invariance (std of z_content over rotation, per dim)
  - cycle reconstruction grid (input vs encode->decode vs cycle through R(theta))
  - numeric report: equivariance error, |z_pose|, content std
"""
import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import (
    list_modelnet10_paths,
    load_mesh_as_pointcloud,
    rotate_pointcloud,
    rotation_matrix_yaw,
)
from model import PointCloudDecoder, PointCloudEncoder, rotate_pose_matrix


def plot_pc(ax, pts, title=None):
    pts = pts.detach().cpu().numpy() if torch.is_tensor(pts) else pts
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 2], cmap="viridis")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if title is not None:
        ax.set_title(title, fontsize=8)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
        os.path.join(args.ckpt_dir, "model.pt"),
        map_location=device,
        weights_only=False,
    )
    saved = ckpt["args"]

    enc = PointCloudEncoder(
        content_dim=saved["content_dim"],
        dim=saved["dim"],
        n_heads=saved["n_heads"],
        n_layers=saved["n_layers"],
        n_latents=saved["n_latents"],
    ).to(device)
    dec = PointCloudDecoder(
        content_dim=saved["content_dim"],
        dim=saved["dim"],
        n_heads=saved["n_heads"],
        n_layers=saved["n_layers"],
        n_latents=saved["n_latents"],
        n_points=saved["n_points"],
    ).to(device)
    enc.load_state_dict(ckpt["enc"])
    dec.load_state_dict(ckpt["dec"])
    enc.eval()
    dec.eval()
    n_points = saved["n_points"]

    chair_path = list_modelnet10_paths(split="test", classes=["chair"])[0]
    chair = load_mesh_as_pointcloud(chair_path, n_points=n_points).to(device)

    n_steps = 32
    thetas = torch.linspace(0, 2 * math.pi, n_steps + 1, device=device)[:-1]
    R_yaw = rotation_matrix_yaw(thetas)
    chair_rep = chair.unsqueeze(0).expand(n_steps, -1, -1).contiguous()
    x_rot = rotate_pointcloud(chair_rep, R_yaw)

    with torch.no_grad():
        z_c, z_p = enc(x_rot)
        z_c0, z_p0 = enc(chair.unsqueeze(0))
        z_c0_rep = z_c0.expand(n_steps, -1)
        z_p0_rep = z_p0.expand(n_steps, -1, -1)
        z_p_cyc = rotate_pose_matrix(z_p0_rep, R_yaw)
        x_cyc = dec(z_c0_rep, z_p_cyc)
        x_rot_recon = dec(z_c, z_p)

    # 1) Pose entries vs theta — overlay encoded vs predicted-by-group-action
    z_p_pred = rotate_pose_matrix(z_p0_rep, R_yaw)  # what the group action says
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
    thetas_np = thetas.cpu().numpy()
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            ax.plot(thetas_np, z_p[:, i, j].cpu().numpy(), "-o", markersize=3,
                    label="enc(R x)", color="C0")
            ax.plot(thetas_np, z_p_pred[:, i, j].cpu().numpy(), "--",
                    label="R . enc(x)", color="C3", alpha=0.8)
            ax.set_title(f"z_pose[{i},{j}]", fontsize=10)
            ax.axhline(0, color="gray", lw=0.5)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Pose latent under input yaw — solid: encoder, dashed: predicted by R . enc(x_0)", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(args.ckpt_dir, "pose_orbit_yaw.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # 2) Content invariance
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    std_per_dim = z_c.std(dim=0).cpu().numpy()
    ax.bar(range(len(std_per_dim)), std_per_dim)
    ax.set_xlabel("content dim")
    ax.set_ylabel("std over yaw in [0, 2pi]")
    ax.set_title("Content invariance under input rotation (lower = more invariant)")
    fig.tight_layout()
    out_path = os.path.join(args.ckpt_dir, "content_invariance.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # 3) Cycle reconstruction grid
    n_show = 6
    show_idx = torch.linspace(0, n_steps - 1, n_show).long()
    fig = plt.figure(figsize=(2.4 * n_show, 7.2))
    for j, idx in enumerate(show_idx):
        ax_in = fig.add_subplot(3, n_show, j + 1, projection="3d")
        plot_pc(ax_in, x_rot[idx], title=f"input theta={thetas[idx].item():.2f}")
        ax_recon = fig.add_subplot(3, n_show, n_show + j + 1, projection="3d")
        plot_pc(ax_recon, x_rot_recon[idx], title="enc -> dec")
        ax_cyc = fig.add_subplot(3, n_show, 2 * n_show + j + 1, projection="3d")
        plot_pc(ax_cyc, x_cyc[idx], title="dec(z_c, R(theta) z_p)")
    fig.suptitle(
        "Top: rotated input. Mid: full encode->decode. Bottom: cycle through latent group action.",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.join(args.ckpt_dir, "cycle_grid.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # Numeric report
    equiv_err = ((z_p - z_p_pred) ** 2).mean().item()
    print(f"avg equivariance error on yaw sweep (lower=better): {equiv_err:.6f}")
    print(f"avg |z_pose|_F: {z_p.flatten(1).norm(dim=-1).mean().item():.4f}")
    print(f"avg z_content std across rotation: {z_c.std(dim=0).mean().item():.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, default="ckpts")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
