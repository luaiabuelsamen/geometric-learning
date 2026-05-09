"""Train the VN-encoder + FoldingNet-decoder autoencoder.

Equivariance and the cycle property are architectural identities, so the
loss is essentially just reconstruction. The cycle term and the soft
equivariance/invariance penalties are kept at small weights as sanity
checks — they should stay near 0 throughout training.
"""
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
from model import FoldingDecoder, VNEncoder, chamfer_distance, rotate_pose_matrix


def prewarm(dataset, n=None):
    if n is None:
        n = len(dataset)
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
    prewarm(dataset)

    enc = VNEncoder(content_dim=args.content_dim, hidden=args.vn_hidden).to(device)
    n_grid_per_side = int(math.sqrt(args.n_points))
    dec = FoldingDecoder(
        content_dim=args.content_dim,
        n_points_per_side=n_grid_per_side,
        hidden=args.dec_hidden,
    ).to(device)

    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(
        f"params: enc={sum(p.numel() for p in enc.parameters())/1e6:.2f}M  "
        f"dec={sum(p.numel() for p in dec.parameters())/1e6:.2f}M  "
        f"total={n_params/1e6:.2f}M"
    )

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)

    pbar = tqdm(range(args.steps))
    t0 = time.time()
    for step in pbar:
        x, R, x_rot = sample_pair_batch(
            dataset, args.batch_size, device=device, rotation="so3"
        )

        z_c, z_p = enc(x)
        z_c_rot, z_p_rot = enc(x_rot)

        x_recon = dec(z_c, z_p)
        x_rot_recon = dec(z_c_rot, z_p_rot)
        # Cycle: rotating the latent should match encoding the rotated input
        x_rot_pred = dec(z_c, rotate_pose_matrix(z_p, R))

        loss_recon = chamfer_distance(x_recon, x) + chamfer_distance(x_rot_recon, x_rot)
        loss_cycle = chamfer_distance(x_rot_pred, x_rot)
        # Soft equivariance check (should be near 0 by architecture)
        loss_equiv = ((z_p_rot - rotate_pose_matrix(z_p, R)) ** 2).mean()
        loss_inv = ((z_c_rot - z_c) ** 2).mean()

        loss = loss_recon + args.w_cycle * loss_cycle + args.w_equiv * loss_equiv + args.w_inv * loss_inv

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            pbar.set_postfix({
                "rec": f"{loss_recon.item():.4f}",
                "cyc": f"{loss_cycle.item():.4f}",
                "eq":  f"{loss_equiv.item():.2e}",
                "inv": f"{loss_inv.item():.2e}",
            })

    elapsed = time.time() - t0
    print(f"trained {args.steps} steps in {elapsed:.1f}s ({elapsed/args.steps*1000:.1f}ms/step)")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    out = os.path.join(args.ckpt_dir, "model.pt")
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "args": vars(args)}, out)
    print(f"saved {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-points", type=int, default=529)  # 23*23 grid
    p.add_argument("--content-dim", type=int, default=64)
    p.add_argument("--vn-hidden", type=int, default=64)
    p.add_argument("--dec-hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w-cycle", type=float, default=1.0)
    p.add_argument("--w-equiv", type=float, default=0.1)  # tiny — should already be ~0 by arch
    p.add_argument("--w-inv", type=float, default=0.1)
    p.add_argument("--max-samples", type=int, default=800)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-dir", type=str, default="ckpts")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
