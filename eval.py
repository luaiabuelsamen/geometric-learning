"""Autoencoder evaluation.

Hard equivariance check, content invariance, cycle reconstruction grid,
per-class reconstruction grid, and a smooth rotation sweep showing the
latent group action decoded back to the input space.
"""
import argparse
import io
import math
import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    train_classes = sa.get("classes", "")
    # If trained on a single class, show variety within that class.
    # If trained on all classes, show one of each.
    if train_classes and "," not in train_classes:
        recon_classes = [train_classes] * 10  # 10 instances of the same class
        recon_titles = [f"{train_classes} #{i}" for i in range(10)]
    else:
        recon_classes = ["bathtub", "bed", "chair", "desk", "dresser",
                         "monitor", "night_stand", "sofa", "table", "toilet"]
        recon_titles = recon_classes

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

    # ---- Plot 3: reconstruction grid (10 examples) ----
    n_show = len(recon_classes)
    fig = plt.figure(figsize=(2.4 * n_show, 4.8))
    with torch.no_grad():
        for i, c in enumerate(recon_classes):
            test_paths = list_modelnet10_paths(split="test", classes=[c])
            # If repeating same class, use different test instances
            idx = i if (train_classes and "," not in train_classes) else 0
            idx = min(idx, len(test_paths) - 1)
            pts = load_mesh_as_pointcloud(test_paths[idx], n_points=n_points).to(device)
            z_c, z_p = enc(pts.unsqueeze(0))
            recon = dec(z_c, z_p)[0]
            cd = chamfer_distance(recon.unsqueeze(0), pts.unsqueeze(0)).item()
            ax_top = fig.add_subplot(2, n_show, i + 1, projection="3d")
            plot_pc(ax_top, pts, title=recon_titles[i])
            ax_bot = fig.add_subplot(2, n_show, n_show + i + 1, projection="3d")
            plot_pc(ax_bot, recon, title=f"recon (cd={cd:.4f})")
    fig.suptitle("Reconstruction grid (top: ground truth, bottom: enc -> dec)", fontsize=11)
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

    # ---- Plot 5: canonicalization demo ----
    # Rotate chair multiple ways, encode each, apply z_pose^T to bring back
    # to canonical orientation. All canonicalized chairs should look identical.
    n_can = 8
    thetas_can, x_rot_can, x_canon = make_canonicalization_demo(enc, chair, n_steps=n_can)
    fig = plt.figure(figsize=(2.4 * n_can, 4.8))
    for i in range(n_can):
        ax_top = fig.add_subplot(2, n_can, i + 1, projection="3d")
        plot_pc(ax_top, x_rot_can[i], title=f"input theta={thetas_can[i].item():.2f}")
        ax_bot = fig.add_subplot(2, n_can, n_can + i + 1, projection="3d")
        plot_pc(ax_bot, x_canon[i], title="canonicalized: x @ z_pose^T")
    fig.suptitle(
        "Canonicalization (no decoder): inputs at different rotations encode to z_pose,"
        " applying z_pose^T undoes the rotation — all bottom panels should be identical",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "canonicalization.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # ---- Plot 6: animated GIF of equivariance ----
    out = os.path.join(out_dir, "rotation_cycle.gif")
    make_rotation_gif(enc, chair, out, n_frames=60, fps=15)
    print(f"saved {out}")


def plot_pc_with_frame(ax, pts, frame, title=None, axes_scale: float = 0.8):
    """Plot a point cloud with the inferred z_pose drawn as 3 RGB axes.

    Each column of z_pose is normalised to unit length before drawing,
    so the arrow direction (which is what matters — equivariance lives
    in the direction, not the magnitude) is clearly visible.
    """
    pts = pts.detach().cpu().numpy() if torch.is_tensor(pts) else pts
    frame = frame.detach().cpu().numpy() if torch.is_tensor(frame) else frame
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c="lightgray", alpha=0.5)
    origin = np.zeros(3)
    colors = ["red", "green", "blue"]
    for i, c in enumerate(colors):
        v = frame[:, i]
        v_norm = np.linalg.norm(v) + 1e-8
        v = v / v_norm * axes_scale
        ax.quiver(origin[0], origin[1], origin[2], v[0], v[1], v[2],
                  color=c, arrow_length_ratio=0.2, linewidth=3.0)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    if title is not None:
        ax.set_title(title, fontsize=9)


def make_canonicalization_demo(enc, chair, n_steps=8):
    """Rotate chair k ways, encode each to get z_pose, apply z_pose^T to canonicalize.

    If z_pose correctly captures the rotation, all k canonicalized point clouds
    should look identical regardless of input rotation. This bypasses the decoder.
    """
    device = chair.device
    thetas = torch.linspace(0, 2 * math.pi, n_steps + 1, device=device)[:-1]
    R_yaw = rotation_matrix_yaw(thetas)
    chair_rep = chair.unsqueeze(0).expand(n_steps, -1, -1).contiguous()
    x_rot = rotate_pointcloud(chair_rep, R_yaw)
    with torch.no_grad():
        z_c, z_p = enc(x_rot)
        # Each col of z_pose is an equivariant 3-vector. Treat z_pose as the
        # inferred orientation of a canonical frame; apply its transpose to
        # bring the input back to canonical orientation.
        # Normalize z_p to a proper rotation via SVD so we get a clean R^T.
        U, _, Vt = torch.linalg.svd(z_p)
        R_inferred = U @ Vt
        det = torch.det(R_inferred)
        # Fix sign so det = +1 (proper rotation, not reflection)
        Vt[det < 0, -1, :] = -Vt[det < 0, -1, :]
        R_inferred = U @ Vt
        # Canonicalize: x_canon = R_inferred^T @ x_rot
        x_canon = rotate_pointcloud(x_rot, R_inferred.transpose(-1, -2))
    return thetas, x_rot, x_canon


def make_rotation_gif(enc, chair, out_path, n_frames=60, fps=15):
    """Animate: chair rotating, with the inferred z_pose drawn as RGB axes.

    If equivariance holds, the axes rotate exactly with the chair —
    `z_pose(R · x) = R · z_pose(x)` is visible as the axes tracking the chair.
    """
    device = chair.device
    thetas = torch.linspace(0, 2 * math.pi, n_frames + 1, device=device)[:-1]
    R = rotation_matrix_yaw(thetas)
    chair_rep = chair.unsqueeze(0).expand(n_frames, -1, -1).contiguous()
    x_rot = rotate_pointcloud(chair_rep, R)

    with torch.no_grad():
        _, z_p_all = enc(x_rot)

    frames = []
    for i in range(n_frames):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        plot_pc_with_frame(
            ax, x_rot[i], z_p_all[i],
            title=f"theta = {thetas[i].item():.2f}   (RGB axes = inferred z_pose columns)",
        )
        fig.suptitle("Architectural equivariance: z_pose rotates exactly with the input",
                     fontsize=10)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
    imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, default="ckpts")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
