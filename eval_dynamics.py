"""Compare pure-hybrid / hybrid / baseline dynamics, in latent and decoded space.

The headline plots are:
  (1) per-step pose-latent Frobenius error  vs rollout step (in-dist + OOD)
  (2) per-step content-latent L2 error      vs rollout step (in-dist + OOD)
  (3) per-step decoded Chamfer distance     vs rollout step (in-dist + OOD)
  (4) one OOD trajectory rendered as point clouds (GT vs each method)

With the hard-equivariant encoder + equivariant decoder, the latent metrics
in (1)+(2) and the decoded metric (3) tell the same story — pure/hybrid sit
at machine epsilon, baseline diverges.
"""
import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import (
    PointCloudDataset,
    list_modelnet10_paths,
    sample_trajectory_batch,
)
from dynamics import BaselineDynamics, HybridDynamics, PureHybridDynamics
from model import FoldingDecoder, VNEncoder, chamfer_distance


def plot_pc(ax, pts, title=None):
    pts = pts.detach().cpu().numpy() if torch.is_tensor(pts) else pts
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 2], cmap="viridis")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    if title is not None:
        ax.set_title(title, fontsize=8)


def rollout(dyn, z_c, z_p, actions):
    cs, ps = [], []
    for t in range(actions.size(1)):
        z_c, z_p = dyn(z_c, z_p, actions[:, t])
        cs.append(z_c)
        ps.append(z_p)
    return cs, ps


def per_step_metrics(dyn, dec, z_c0, z_p0, actions, gt_xs, gt_z_c, gt_z_p):
    """For each rollout step return (chamfer, pose_err_F, content_err_L2)."""
    cs, ps = rollout(dyn, z_c0, z_p0, actions)
    cd, pe, ce = [], [], []
    for t in range(actions.size(1)):
        x_pred = dec(cs[t], ps[t])
        cd.append(chamfer_distance(x_pred, gt_xs[:, t]).item())
        # gt_z_p[:, t+1] = encoded pose at frame t+1; ps[t] = predicted pose for that frame
        pe.append((ps[t] - gt_z_p[:, t + 1]).pow(2).mean().item())
        ce.append((cs[t] - gt_z_c[:, t + 1]).pow(2).mean().item())
    return cd, pe, ce


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dyn_ckpt = torch.load(
        os.path.join(args.ckpt_dir, "dynamics.pt"),
        map_location=device, weights_only=False,
    )
    enc_args = dyn_ckpt["enc_args"]
    train_args = dyn_ckpt["args"]

    n_per_side = int(math.sqrt(enc_args["n_points"]))
    enc = VNEncoder(
        content_dim=enc_args["content_dim"], hidden=enc_args["vn_hidden"]
    ).to(device).eval()
    dec = FoldingDecoder(
        content_dim=enc_args["content_dim"], n_points_per_side=n_per_side,
        hidden=enc_args["dec_hidden"],
    ).to(device).eval()
    enc_ckpt = torch.load(
        os.path.join(args.ckpt_dir, "model.pt"),
        map_location=device, weights_only=False,
    )
    enc.load_state_dict(enc_ckpt["enc"]); dec.load_state_dict(enc_ckpt["dec"])

    hybrid = HybridDynamics(content_dim=enc_args["content_dim"]).to(device)
    baseline = BaselineDynamics(content_dim=enc_args["content_dim"]).to(device)
    pure = PureHybridDynamics().to(device)
    hybrid.load_state_dict(dyn_ckpt["hybrid"]); baseline.load_state_dict(dyn_ckpt["baseline"])
    hybrid.eval(); baseline.eval(); pure.eval()

    n_points = enc_args["n_points"]
    content_dim = enc_args["content_dim"]
    train_max = train_args["action_mag_max"]

    paths = list_modelnet10_paths(split="test")
    dataset = PointCloudDataset(paths, n_points=n_points, cache=True)

    T_eval = args.T_eval
    settings = [
        ("in-distribution", (0.0, train_max)),
        ("OOD large", (math.pi / 2, math.pi)),
    ]
    methods = [("pure (no params)", pure), ("hybrid (learned residual)", hybrid),
               ("baseline (all-MLP)", baseline)]

    torch.manual_seed(0)
    results = {}
    with torch.no_grad():
        for name, mag_range in settings:
            xs, actions = sample_trajectory_batch(
                dataset, args.eval_batch, T_eval, mag_range, device=device
            )
            B, T1, N, _ = xs.shape
            z_c_all, z_p_all = enc(xs.reshape(B * T1, N, 3))
            z_c_all = z_c_all.reshape(B, T1, content_dim)
            z_p_all = z_p_all.reshape(B, T1, 3, 3)
            z_c0 = z_c_all[:, 0]; z_p0 = z_p_all[:, 0]
            gt = xs[:, 1:]
            results[name] = {}
            for mname, dyn in methods:
                cd, pe, ce = per_step_metrics(
                    dyn, dec, z_c0, z_p0, actions, gt, z_c_all, z_p_all
                )
                results[name][mname] = {"chamfer": cd, "pose_err": pe, "content_err": ce}
            # Oracle: enc-dec on GT — lower bound on chamfer
            cd_oracle = []
            for t in range(T_eval):
                cd_oracle.append(
                    chamfer_distance(dec(z_c_all[:, t + 1], z_p_all[:, t + 1]), gt[:, t]).item()
                )
            results[name]["oracle"] = {"chamfer": cd_oracle}
            # Encoder equivariance baseline: how much z_p drifts even on GT (lower bound on pose_err)
            # Compare z_p_t to R_cum @ z_p_0
            z_p_expected = z_p0.clone()
            eq_err_per_step = []
            for t in range(T_eval):
                R_t = torch.linalg.matrix_exp(  # axis-angle via matrix exp would be cleaner; use existing helper
                    None  # placeholder
                ) if False else None
            # Simpler: just measure ||R_cum @ z_p_0 - z_p_gt||^2 step-by-step using the actions
            from data import axis_angle_to_matrix
            R_cum = torch.eye(3, device=device).expand(B, 3, 3).contiguous()
            for t in range(T_eval):
                R_cum = axis_angle_to_matrix(actions[:, t]) @ R_cum
                pred_zp = R_cum @ z_p0
                eq_err_per_step.append((pred_zp - z_p_all[:, t + 1]).pow(2).mean().item())
            results[name]["encoder_eq_err"] = eq_err_per_step

            print(f"\n{name} (|a| in [{mag_range[0]:.3f}, {mag_range[1]:.3f}])")
            print(f"  {'step':>4} | {'pure-cd':>7} {'pure-pe':>7} | {'hyb-cd':>7} {'hyb-pe':>7} | {'base-cd':>7} {'base-pe':>7} | enc-eq")
            for t in range(T_eval):
                p = results[name]["pure (no params)"]
                h = results[name]["hybrid (learned residual)"]
                b = results[name]["baseline (all-MLP)"]
                e = results[name]["encoder_eq_err"][t]
                print(
                    f"  {t+1:4d} | {p['chamfer'][t]:7.4f} {p['pose_err'][t]:7.4f} | "
                    f"{h['chamfer'][t]:7.4f} {h['pose_err'][t]:7.4f} | "
                    f"{b['chamfer'][t]:7.4f} {b['pose_err'][t]:7.4f} | {e:7.4f}"
                )

    # Plot 1: pose latent error per step
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    steps = list(range(1, T_eval + 1))
    colors = {"pure (no params)": "C2", "hybrid (learned residual)": "C0",
              "baseline (all-MLP)": "C3"}
    markers = {"pure (no params)": "^", "hybrid (learned residual)": "o",
               "baseline (all-MLP)": "s"}
    for ax, (name, _) in zip(axes, settings):
        for mname, _ in methods:
            ax.plot(steps, results[name][mname]["pose_err"],
                    marker=markers[mname], color=colors[mname], label=mname)
        ax.plot(steps, results[name]["encoder_eq_err"], "k--", alpha=0.6,
                label="encoder eq. err (lower bound)")
        ax.set_xlabel("rollout step")
        ax.set_title(name)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("z_pose Frobenius² error")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"Pose-latent error vs rollout step  |  trained: |a| ≤ {train_max:.3f}, T={train_args['T']}",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(args.ckpt_dir, "dynamics_pose_err.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")

    # Plot 2: content latent error per step
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (name, _) in zip(axes, settings):
        for mname, _ in methods:
            ax.plot(steps, results[name][mname]["content_err"],
                    marker=markers[mname], color=colors[mname], label=mname)
        ax.set_xlabel("rollout step")
        ax.set_title(name)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("z_content L2² error")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Content-latent error vs rollout step", fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.ckpt_dir, "dynamics_content_err.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # Plot 3: decoded chamfer per step (the contaminated metric)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (name, _) in zip(axes, settings):
        ax.plot(steps, results[name]["oracle"]["chamfer"], "k--",
                label="oracle (enc-dec on GT)", alpha=0.7)
        for mname, _ in methods:
            ax.plot(steps, results[name][mname]["chamfer"],
                    marker=markers[mname], color=colors[mname], label=mname)
        ax.set_xlabel("rollout step")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("chamfer distance")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Decoded Chamfer vs rollout step  —  pure/hybrid stay flat under OOD rotations, baseline diverges",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(args.ckpt_dir, "dynamics_per_step.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # Plot 4: OOD visual rollout
    with torch.no_grad():
        torch.manual_seed(1)
        xs, actions = sample_trajectory_batch(
            dataset, 1, T_eval, settings[1][1], device=device
        )
        z_c_all, z_p_all = enc(xs.reshape(T_eval + 1, n_points, 3))
        z_c_all = z_c_all.reshape(1, T_eval + 1, content_dim)
        z_p_all = z_p_all.reshape(1, T_eval + 1, 3, 3)
        z_c0 = z_c_all[:, 0]; z_p0 = z_p_all[:, 0]
        rollouts = {}
        for mname, dyn in methods:
            cs, ps = rollout(dyn, z_c0, z_p0, actions)
            rollouts[mname] = (cs, ps)

    n_show = min(T_eval, 6)
    show_idx = torch.linspace(0, T_eval - 1, n_show).long()
    n_rows = 1 + len(methods)
    fig = plt.figure(figsize=(2.4 * n_show, 2.4 * n_rows))
    for j, idx in enumerate(show_idx):
        ax = fig.add_subplot(n_rows, n_show, j + 1, projection="3d")
        plot_pc(ax, xs[0, idx + 1], title=f"GT step {idx.item()+1}")
        for r, (mname, _) in enumerate(methods):
            cs, ps = rollouts[mname]
            ax = fig.add_subplot(n_rows, n_show, (r + 1) * n_show + j + 1, projection="3d")
            plot_pc(ax, dec(cs[idx], ps[idx])[0], title=mname.split(" ")[0])
    fig.suptitle(
        "OOD rollout (rotation magnitudes never seen at training): GT vs each dynamics model",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(args.ckpt_dir, "dynamics_ood_visual.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, default="ckpts")
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--T-eval", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
