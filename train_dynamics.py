"""Train HybridDynamics and BaselineDynamics on top of frozen encoder/decoder.

Per step: sample a batch of T-step trajectories, encode each frame, roll out
both dynamics models from z_0 using the action sequence, decode each predicted
latent, and accumulate Chamfer distance to the ground-truth trajectory frames.
"""
import argparse
import math
import os
import time

import torch
from tqdm import tqdm

from data import PointCloudDataset, list_modelnet10_paths, sample_trajectory_batch
from dynamics import BaselineDynamics, HybridDynamics
from model import PointCloudDecoder, PointCloudEncoder, chamfer_distance


def prewarm(dataset: PointCloudDataset):
    for i in tqdm(range(len(dataset)), desc="prewarm cache"):
        _ = dataset[i]


def rollout(dyn, z_c0, z_p0, actions):
    """Roll out dyn for T steps. actions: (B, T, 3). Returns lists of (z_c, z_p)."""
    z_c, z_p = z_c0, z_p0
    cs, ps = [], []
    for t in range(actions.size(1)):
        z_c, z_p = dyn(z_c, z_p, actions[:, t])
        cs.append(z_c)
        ps.append(z_p)
    return cs, ps


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Frozen encoder/decoder
    enc_ckpt = torch.load(
        os.path.join(args.enc_ckpt_dir, "model.pt"),
        map_location=device,
        weights_only=False,
    )
    saved = enc_ckpt["args"]
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
    enc.load_state_dict(enc_ckpt["enc"])
    dec.load_state_dict(enc_ckpt["dec"])
    enc.eval()
    dec.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in dec.parameters():
        p.requires_grad_(False)
    n_points = saved["n_points"]
    content_dim = saved["content_dim"]

    paths = list_modelnet10_paths(split="train")
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    dataset = PointCloudDataset(paths, n_points=n_points, cache=True)
    print(f"dataset size: {len(dataset)}")
    prewarm(dataset)

    hybrid = HybridDynamics(content_dim=content_dim, hidden=128).to(device)
    baseline = BaselineDynamics(content_dim=content_dim, hidden=256, depth=3).to(device)
    n_h = sum(p.numel() for p in hybrid.parameters())
    n_b = sum(p.numel() for p in baseline.parameters())
    print(f"hybrid params: {n_h/1e3:.1f}K   baseline params: {n_b/1e3:.1f}K")

    opt_h = torch.optim.Adam(hybrid.parameters(), lr=args.lr)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=args.lr)

    action_range = (args.action_mag_min, args.action_mag_max)
    print(f"training action range: |a| in [{action_range[0]:.3f}, {action_range[1]:.3f}]")

    pbar = tqdm(range(args.steps))
    t0 = time.time()
    for step in pbar:
        xs, actions = sample_trajectory_batch(
            dataset, args.batch_size, args.T, action_range, device=device
        )
        B, T1, N, _ = xs.shape
        T = T1 - 1

        # Encode all frames once with frozen encoder
        with torch.no_grad():
            xs_flat = xs.reshape(B * T1, N, 3)
            z_c_all, z_p_all = enc(xs_flat)
            z_c_all = z_c_all.reshape(B, T1, content_dim)
            z_p_all = z_p_all.reshape(B, T1, 3, 3)
        z_c0 = z_c_all[:, 0]
        z_p0 = z_p_all[:, 0]
        gt = xs[:, 1:]  # (B, T, N, 3)

        # Hybrid rollout + loss
        cs_h, ps_h = rollout(hybrid, z_c0, z_p0, actions)
        loss_h = 0.0
        for t in range(T):
            x_pred = dec(cs_h[t], ps_h[t])
            loss_h = loss_h + chamfer_distance(x_pred, gt[:, t])
        loss_h = loss_h / T

        opt_h.zero_grad()
        loss_h.backward()
        opt_h.step()

        # Baseline rollout + loss
        cs_b, ps_b = rollout(baseline, z_c0, z_p0, actions)
        loss_b = 0.0
        for t in range(T):
            x_pred = dec(cs_b[t], ps_b[t])
            loss_b = loss_b + chamfer_distance(x_pred, gt[:, t])
        loss_b = loss_b / T

        opt_b.zero_grad()
        loss_b.backward()
        opt_b.step()

        if step % args.log_every == 0:
            pbar.set_postfix(
                {"hyb": f"{loss_h.item():.4f}", "base": f"{loss_b.item():.4f}"}
            )

    elapsed = time.time() - t0
    print(f"trained {args.steps} steps in {elapsed:.1f}s ({elapsed/args.steps*1000:.1f}ms/step)")

    out = os.path.join(args.ckpt_dir, "dynamics.pt")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save(
        {
            "hybrid": hybrid.state_dict(),
            "baseline": baseline.state_dict(),
            "args": vars(args),
            "enc_args": saved,
        },
        out,
    )
    print(f"saved {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--T", type=int, default=4, help="trajectory length")
    p.add_argument("--action-mag-min", type=float, default=0.0)
    p.add_argument("--action-mag-max", type=float, default=math.pi / 4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-samples", type=int, default=400)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--enc-ckpt-dir", type=str, default="ckpts")
    p.add_argument("--ckpt-dir", type=str, default="ckpts")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
