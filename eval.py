"""Autoencoder evaluation.

Hard equivariance check, content invariance, cycle reconstruction grid,
per-class reconstruction grid, and a smooth rotation sweep showing the
latent group action decoded back to the input space.
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
from model import FoldingDecoder, VNEncoder, chamfer_distance, rotate_pose_matrix


def plot_pc(ax, pts, title=None, color_axis: int = 2):
    pts = pts.detach().cpu().numpy() if torch.is_tensor(pts) else pts
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, color_axis], cmap="viridis")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    if title is not None:
        ax.set_title(title, fontsize=8)


def load_model(ckpt_dir, device):
    ck = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device, weights_only=False)
    sa = ck["args"]
    n_per_side = int(math.sqrt(sa["n_points"]))
    enc = VNEncoder(content_dim=sa["content_dim"], hidden=sa["vn_hidden"]).to(device).eval()
    dec = FoldingDecoder(content_dim=sa["content_dim"], n_points_per_side=n_per_side,
                         hidden=sa["dec_hidden"]).to(device).eval()
    enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
    return enc, dec, sa


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, dec, sa = load_model(args.ckpt_dir, device)
    n_points = sa["n_points"]
    out_dir = args.ckpt_dir
    classes = ["bathtub", "bed", "chair", "desk", "dresser",
               "monitor", "night_stand", "sofa", "table", "toilet"]

    # ---- Hard equivariance check (machine epsilon expected) ----
    chair = load_mesh_as_pointcloud(
        list_modelnet10_paths(split="test", classes=["chair"])[0], n_points=n_points
    ).to(device)
    n_steps = 32
    thetas = torch.linspace(0, 2 * math.pi, n_steps + 1, device=device)[:-1]
    R_yaw = rotation_matrix_yaw(thetas)
    chair_rep = chair.unsqueeze(0).expand(n_steps, -1, -1).contiguous()
    x_rot = rotate_pointcloud(chair_rep, R_yaw)

    with torch.no_grad():
        z_c, z_p = enc(x_rot)
        z_c0, z_p0 = enc(chair.unsqueeze(0))
        z_p_pred = rotate_pose_matrix(z_p0.expand(n_steps, -1, -1), R_yaw)
        x_cyc = dec(z_c0.expand(n_steps, -1), z_p_pred)
        x_recon = dec(z_c, z_p)

    pose_eq_err = (z_p - z_p_pred).abs().mean().item()
    pose_eq_max = (z_p - z_p_pred).abs().max().item()
    content_inv = z_c.std(dim=0).mean().item()
    print(f"pose equivariance err  (mean abs): {pose_eq_err:.3e}")
    print(f"pose equivariance err  (max  abs): {pose_eq_max:.3e}")
    print(f"content invariance std (mean over dim): {content_inv:.3e}")

    # ---- Plot 1: pose orbit overlay ----
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
    fig.suptitle(
        f"Pose latent under input yaw — solid: encoder, dashed: predicted by R . enc(x_0)\n"
        f"hard equivariance: mean err {pose_eq_err:.2e}, max err {pose_eq_max:.2e}",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "pose_orbit.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # ---- Plot 2: cycle reconstruction grid ----
    n_show = 8
    show_idx = torch.linspace(0, n_steps - 1, n_show).long()
    fig = plt.figure(figsize=(2.4 * n_show, 7.2))
    for j, idx in enumerate(show_idx):
        ax_in = fig.add_subplot(3, n_show, j + 1, projection="3d")
        plot_pc(ax_in, x_rot[idx], title=f"input theta={thetas[idx].item():.2f}")
        ax_recon = fig.add_subplot(3, n_show, n_show + j + 1, projection="3d")
        plot_pc(ax_recon, x_recon[idx], title="enc -> dec")
        ax_cyc = fig.add_subplot(3, n_show, 2 * n_show + j + 1, projection="3d")
        plot_pc(ax_cyc, x_cyc[idx], title="dec(z_c, R(theta) z_p)")
    fig.suptitle(
        "Cycle: input rotated by theta (top), enc->dec (mid), dec(z_c, R(theta)·z_p) (bottom)",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "cycle_grid.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # ---- Plot 3: per-class reconstruction grid ----
    fig = plt.figure(figsize=(2.4 * len(classes), 4.8))
    with torch.no_grad():
        for i, c in enumerate(classes):
            paths = list_modelnet10_paths(split="test", classes=[c])
            pts = load_mesh_as_pointcloud(paths[0], n_points=n_points).to(device)
            z_c, z_p = enc(pts.unsqueeze(0))
            recon = dec(z_c, z_p)[0]
            cd = chamfer_distance(recon.unsqueeze(0), pts.unsqueeze(0)).item()
            ax_top = fig.add_subplot(2, len(classes), i + 1, projection="3d")
            plot_pc(ax_top, pts, title=c)
            ax_bot = fig.add_subplot(2, len(classes), len(classes) + i + 1, projection="3d")
            plot_pc(ax_bot, recon, title=f"recon (cd={cd:.4f})")
    fig.suptitle("Per-class reconstruction (top: ground truth, bottom: enc -> dec)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, "per_class_recon.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # ---- Plot 4: rotation interpolation strip ----
    n_show = 12
    thetas_strip = torch.linspace(0, 2 * math.pi, n_show + 1, device=device)[:-1]
    R_strip = rotation_matrix_yaw(thetas_strip)
    chair_strip = chair.unsqueeze(0).expand(n_show, -1, -1).contiguous()
    x_rot_strip = rotate_pointcloud(chair_strip, R_strip)
    with torch.no_grad():
        z_c0_strip, z_p0_strip = enc(chair.unsqueeze(0))
        z_p_strip = rotate_pose_matrix(z_p0_strip.expand(n_show, -1, -1), R_strip)
        x_cyc_strip = dec(z_c0_strip.expand(n_show, -1), z_p_strip)
    fig = plt.figure(figsize=(1.8 * n_show, 3.6))
    for j in range(n_show):
        ax_top = fig.add_subplot(2, n_show, j + 1, projection="3d")
        plot_pc(ax_top, x_rot_strip[j], title=f"{thetas_strip[j].item():.2f}")
        ax_bot = fig.add_subplot(2, n_show, n_show + j + 1, projection="3d")
        plot_pc(ax_bot, x_cyc_strip[j])
    fig.suptitle(
        "Smooth rotation in latent space: top = real rotated chair, bottom = dec(z_c, R(theta) z_p)",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "rotation_sweep.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, default="ckpts")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
