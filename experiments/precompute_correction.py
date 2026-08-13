"""Precompute residual correction means from training data and save to .npz.

Collects GT residuals (future - hermite_prior) in chord-frame from N training
scenarios, buckets by (heading change, chord length), and saves mean residual
curves to /workspace/data/residual_correction.npz.

This file is loaded at module-import time by data/normalization.py to apply
correction to the Hermite prior (see _hermite_numpy and _hermite_torch).
"""

import sys
import os
import math
import numpy as np
import yaml

sys.path.insert(0, "/workspace")
os.chdir("/workspace")

from data.av2_dataset import Argoverse2Dataset
from data.normalization import (
    _hermite_numpy, residual_to_chord_frame,
    denormalize, denormalize_residual, denormalize_residual_chord, unpack_chord,
)


# Bucketing — must match the constants in data/normalization.py
BUCKET_EDGES_DEG = [0, 15, 45, 90, 120, 180]   # heading change buckets
CHORD_BUCKETS_M = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 1000)]  # chord length buckets
N_HEAD_BUCKETS = len(BUCKET_EDGES_DEG) - 1   # 5
N_CHORD_BUCKETS = len(CHORD_BUCKETS_M)       # 5
N_FUTURE = 60
MIN_COUNT = 20
OUTPUT_PATH = "/workspace/data/residual_correction.npz"


def get_head_bucket(abs_delta_deg):
    for i in range(N_HEAD_BUCKETS):
        if BUCKET_EDGES_DEG[i] <= abs_delta_deg < BUCKET_EDGES_DEG[i + 1]:
            return i
    return N_HEAD_BUCKETS - 1


def get_chord_bucket(chord_len):
    for i, (lo, hi) in enumerate(CHORD_BUCKETS_M):
        if lo <= chord_len < hi:
            return i
    return N_CHORD_BUCKETS - 1


def to_np(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_scenario_data(sample):
    goal = to_np(denormalize(sample["goal"].reshape(1, 2))).flatten()
    history_end = to_np(denormalize(sample["history"][-1:].reshape(1, 2))).flatten()
    h0 = float(sample["start_heading"].item())
    h1 = float(sample["end_heading"].item())
    prior_norm = to_np(sample["prior"])
    prior_m = to_np(denormalize(prior_norm))
    traj_norm = to_np(sample["trajectory"])
    use_chord = float(sample["use_chord_frame"].item()) > 0.5
    if use_chord:
        r_lon_n, r_lat_n = unpack_chord(traj_norm)
        r_lon, r_lat = denormalize_residual_chord(r_lon_n, r_lat_n)
        chord_dir = to_np(sample["chord_dir"])
        perp_dir = np.array([-chord_dir[1], chord_dir[0]], dtype=np.float32)
        residual_m = r_lon[..., None] * chord_dir + r_lat[..., None] * perp_dir
    else:
        residual_m = to_np(denormalize_residual(traj_norm))
    future = prior_m + residual_m
    return history_end, goal, h0, h1, future, prior_m


def collect_residuals(dataset, n_samples, seed=0):
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    # grouped[(hb, cb)] = list of (r_lon (T,), r_lat (T,), chord_len)
    grouped = {(h, c): [] for h in range(N_HEAD_BUCKETS) for c in range(N_CHORD_BUCKETS)}
    skipped = 0
    for idx in indices:
        try:
            sample = dataset[int(idx)]
            history_end, goal, h0, h1, future, prior_m = load_scenario_data(sample)
            chord = goal - history_end
            chord_len = float(np.linalg.norm(chord))
            if chord_len < 1e-3:
                skipped += 1
                continue
            delta = h1 - h0
            delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
            abs_delta_deg = math.degrees(abs(delta))
            hb = get_head_bucket(abs_delta_deg)
            cb = get_chord_bucket(chord_len)
            residual = future - prior_m
            r_lon, r_lat = residual_to_chord_frame(residual, history_end, goal)
            if r_lon is None:
                skipped += 1
                continue
            grouped[(hb, cb)].append((r_lon, r_lat, chord_len))
        except Exception:
            skipped += 1
            continue
    return grouped, skipped


def compute_bucket_means(grouped):
    """Compute mean r_lon, r_lat, count per (hb, cb) bucket.

    Returns arrays of shape (N_HEAD_BUCKETS, N_CHORD_BUCKETS, N_FUTURE) for
    r_lon_mean and r_lat_mean, plus a counts array (N_HEAD_BUCKETS, N_CHORD_BUCKETS).
    Buckets with count < MIN_COUNT are zeroed out (and counts set to 0).
    """
    r_lon_means = np.zeros((N_HEAD_BUCKETS, N_CHORD_BUCKETS, N_FUTURE), dtype=np.float32)
    r_lat_means = np.zeros((N_HEAD_BUCKETS, N_CHORD_BUCKETS, N_FUTURE), dtype=np.float32)
    counts = np.zeros((N_HEAD_BUCKETS, N_CHORD_BUCKETS), dtype=np.int32)
    for (hb, cb), items in grouped.items():
        if len(items) < MIN_COUNT:
            continue
        r_lons = np.stack([it[0] for it in items])
        r_lats = np.stack([it[1] for it in items])
        r_lon_means[hb, cb] = r_lons.mean(axis=0)
        r_lat_means[hb, cb] = r_lats.mean(axis=0)
        counts[hb, cb] = len(items)
    return r_lon_means, r_lat_means, counts


def main(n_samples=5000, seed=0):
    with open("/workspace/config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    dataset = Argoverse2Dataset(
        data_dir=cfg["data"]["train_dir"],
        map_dir=cfg["data"].get("map_dir"),
        n_future=cfg["data"]["n_future"],
        n_history=cfg["data"]["n_history"],
        n_lanes=cfg["data"]["n_lanes"],
        lane_feat_dim=cfg["data"]["lane_feat_dim"],
        n_neighbors=cfg["data"]["n_neighbors"],
        split="eval",
        return_scene_data=True,
        prior_type=cfg["data"].get("prior_type", "hermite"),
        residual_frame=cfg["data"].get("residual_frame", "chord"),
        filter_parking=cfg["data"].get("filter_parking", False),
    )
    dataset.preload_cache()

    print(f"Collecting residuals from {n_samples} scenarios (seed={seed})...")
    grouped, skipped = collect_residuals(dataset, n_samples=n_samples, seed=seed)
    print(f"Skipped: {skipped}")

    print("\nBucket counts:")
    header = "heading\\chord"
    print(f"{header:<14}", end='')
    for lo, hi in CHORD_BUCKETS_M:
        label = f"{lo}-{hi}m" if hi < 1000 else f"{lo}m+"
        print(f"{label:>10}", end='')
    print()
    for hi in range(N_HEAD_BUCKETS):
        hname = f"{BUCKET_EDGES_DEG[hi]}-{BUCKET_EDGES_DEG[hi+1]}°"
        print(f"{hname:<14}", end='')
        for ci in range(N_CHORD_BUCKETS):
            print(f"{len(grouped[(hi, ci)]):>10}", end='')
        print()

    r_lon_means, r_lat_means, counts = compute_bucket_means(grouped)

    print(f"\nBuckets with count >= {MIN_COUNT} (will be applied as correction):")
    for hi in range(N_HEAD_BUCKETS):
        for ci in range(N_CHORD_BUCKETS):
            n = counts[hi, ci]
            if n > 0:
                r_lon_peak = np.abs(r_lon_means[hi, ci]).max()
                r_lat_peak = np.abs(r_lat_means[hi, ci]).max()
                hname = f"{BUCKET_EDGES_DEG[hi]}-{BUCKET_EDGES_DEG[hi+1]}°"
                clo, chi = CHORD_BUCKETS_M[ci]
                clabel = f"{clo}-{chi}m" if chi < 1000 else f"{clo}m+"
                print(f"  {hname}/{clabel} (n={n}): r_lon peak={r_lon_peak:.3f}m, r_lat peak={r_lat_peak:.3f}m")

    # Save to npz
    np.savez(
        OUTPUT_PATH,
        r_lon_means=r_lon_means,
        r_lat_means=r_lat_means,
        counts=counts,
        bucket_edges_deg=np.array(BUCKET_EDGES_DEG, dtype=np.float32),
        chord_buckets_m=np.array(CHORD_BUCKETS_M, dtype=np.float32),
        n_future=np.array(N_FUTURE, dtype=np.int32),
        min_count=np.array(MIN_COUNT, dtype=np.int32),
        n_samples=np.array(n_samples, dtype=np.int32),
        seed=np.array(seed, dtype=np.int32),
    )
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"File contents:")
    print(f"  r_lon_means: shape={r_lon_means.shape}, dtype={r_lon_means.dtype}")
    print(f"  r_lat_means: shape={r_lat_means.shape}, dtype={r_lat_means.dtype}")
    print(f"  counts: shape={counts.shape}, dtype={counts.dtype}")
    print(f"  Total non-empty buckets: {(counts > 0).sum()}/{N_HEAD_BUCKETS * N_CHORD_BUCKETS}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(n_samples=args.n_samples, seed=args.seed)
