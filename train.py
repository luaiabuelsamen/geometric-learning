"""Train the SO(3)-equivariant point cloud autoencoder on ModelNet10."""
import argparse
import math
import os
import time

import torch
from tqdm import tqdm

from data import (
    PointCloudDataset,
    list_modelnet10_paths,
    sample_pair_batch,
)
from model import (
    PointCloudEncoder,
    PointCloudDecoder,
    chamfer_distance,
    rotate_pose_matrix,
)


def prewarm(dataset: PointCloudDataset, max_samples: int):
    """Touch each sample to populate the in-memory cache."""
    n = min(max_samples, len(dataset))
    for i in tqdm(range(n), desc="prewarm cache"):
        _ = dataset[i]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    paths = list_modelnet10_paths(split="train")
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    dataset = PointCloudDataset(paths, n_points=args.n_points, cache=True)
    print(f"dataset size: {len(dataset)}")
    prewarm(dataset, len(dataset))

    enc = PointCloudEncoder(
        content_dim=args.content_dim,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_latents=args.n_latents,
    ).to(device)
    dec = PointCloudDecoder(
        content_dim=args.content_dim,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_latents=args.n_latents,
        n_points=args.n_points,
    ).to(device)
    n_params = sum(p.numel() for p in enc.parameters()) + sum(
        p.numel() for p in dec.parameters()
    )
    print(f"params: {n_params/1e6:.2f}M")

    opt = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()), lr=args.lr
    )

    target_pose_norm = math.sqrt(3.0)  # ||I||_F

    pbar = tqdm(range(args.steps))
    t0 = time.time()
    for step in pbar:
        x, R, x_rot = sample_pair_batch(
            dataset, args.batch_size, device=device, rotation=args.rotation
        )

        z_c, z_p = enc(x)
        z_c_rot, z_p_rot = enc(x_rot)

        x_recon = dec(z_c, z_p)
        x_rot_recon = dec(z_c_rot, z_p_rot)
        x_rot_pred = dec(z_c, rotate_pose_matrix(z_p, R))

        loss_recon = chamfer_distance(x_recon, x) + chamfer_distance(x_rot_recon, x_rot)
        loss_equiv = ((z_p_rot - rotate_pose_matrix(z_p, R)) ** 2).mean()
        loss_inv = ((z_c_rot - z_c) ** 2).mean()
        loss_cycle = chamfer_distance(x_rot_pred, x_rot)
        loss_pose_norm = (
            (z_p.flatten(1).norm(dim=-1) - target_pose_norm).pow(2).mean()
        )

        loss = (
            loss_recon
            + args.w_equiv * loss_equiv
            + args.w_inv * loss_inv
            + args.w_cycle * loss_cycle
            + args.w_norm * loss_pose_norm
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            pbar.set_postfix(
                {
                    "rec": f"{loss_recon.item():.3f}",
                    "eq": f"{loss_equiv.item():.3f}",
                    "inv": f"{loss_inv.item():.3f}",
                    "cyc": f"{loss_cycle.item():.3f}",
                    "|zp|": f"{z_p.flatten(1).norm(dim=-1).mean().item():.2f}",
                }
            )

    elapsed = time.time() - t0
    print(f"trained {args.steps} steps in {elapsed:.1f}s ({elapsed/args.steps*1000:.1f}ms/step)")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    out = os.path.join(args.ckpt_dir, "model.pt")
    torch.save(
        {"enc": enc.state_dict(), "dec": dec.state_dict(), "args": vars(args)}, out
    )
    print(f"saved {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-points", type=int, default=512)
    p.add_argument("--content-dim", type=int, default=32)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-latents", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w-equiv", type=float, default=1.0)
    p.add_argument("--w-inv", type=float, default=1.0)
    p.add_argument("--w-cycle", type=float, default=1.0)
    p.add_argument("--w-norm", type=float, default=0.05)
    p.add_argument("--rotation", type=str, default="so3", choices=["so3", "yaw"])
    p.add_argument("--max-samples", type=int, default=800,
                   help="cap number of training meshes (0 = use all)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-dir", type=str, default="ckpts")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
