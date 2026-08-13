"""Visualize residual prior correction curves and before/after prior vs GT.

Generates:
  1. fig_correction_curves.png — mean r_lon(t), r_lat(t) per (heading, chord) sub-bucket
  2. fig_prior_vs_gt.png — old Hermite vs new (corrected) prior vs GT on representative scenarios
"""

import sys
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, "/workspace")
os.chdir("/workspace")

from data.av2_dataset import Argoverse2Dataset
from data.normalization import (
    compute_hermite_prior, _hermite_numpy,
    residual_to_chord_frame, chord_frame_to_residual, compute_chord_dir,
    denormalize, denormalize_residual, denormalize_residual_chord, unpack_chord,
)


BUCKET_EDGES = [0, 15, 45, 90, 120, 180]
BUCKET_NAMES = ["0-15", "15-45", "45-90", "90-120", "120-180"]
CHORD_BUCKETS = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 1000)]
CHORD_LABELS = ["0-10m", "10-20m", "20-40m", "40-80m", "80m+"]

MIN_COUNT = 20


def get_bucket(abs_delta_deg):
    for i, edge in enumerate(BUCKET_EDGES[:-1]):
        if BUCKET_EDGES[i] <= abs_delta_deg < BUCKET_EDGES[i + 1]:
            return i
    return len(BUCKET_NAMES) - 1


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


def collect_subbucket_stats(dataset, n_samples=2000, seed=0):
    """Collect (r_lon, r_lat, chord_len) per scenario, grouped by (heading, chord) bucket."""
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    grouped = {(h, c): [] for h in range(len(BUCKET_NAMES)) for c in range(len(CHORD_BUCKETS))}
    for idx in indices:
        try:
            sample = dataset[int(idx)]
            history_end, goal, h0, h1, future, prior_m = load_scenario_data(sample)
            chord = goal - history_end
            chord_len = float(np.linalg.norm(chord))
            if chord_len < 1e-3: continue
            delta = h1 - h0
            delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
            abs_delta_deg = math.degrees(abs(delta))
            hb = get_bucket(abs_delta_deg)
            cb = None
            for i, (clo, chi) in enumerate(CHORD_BUCKETS):
                if clo <= chord_len < chi:
                    cb = i
                    break
            if cb is None: continue
            residual = future - prior_m
            r_lon, r_lat = residual_to_chord_frame(residual, history_end, goal)
            if r_lon is None: continue
            grouped[(hb, cb)].append((r_lon, r_lat, chord_len))
        except Exception:
            continue
    return grouped


def compute_subbucket_means(grouped):
    """Compute per-(hb,cb) mean r_lon, r_lat, count, std."""
    means = {}
    for key, items in grouped.items():
        if len(items) < MIN_COUNT:
            means[key] = None
            continue
        r_lons = np.stack([it[0] for it in items])
        r_lats = np.stack([it[1] for it in items])
        means[key] = {
            'r_lon_mean': r_lons.mean(axis=0),
            'r_lon_std': r_lons.std(axis=0),
            'r_lat_mean': r_lats.mean(axis=0),
            'r_lat_std': r_lats.std(axis=0),
            'n': len(items),
        }
    return means


def plot_correction_curves(means, save_path):
    """Plot r_lon_mean(t) and r_lat_mean(t) per (heading bucket, chord bucket)."""
    fig, axes = plt.subplots(len(BUCKET_NAMES), len(CHORD_BUCKETS), figsize=(20, 14), sharex=True, sharey=False)
    t = np.arange(60)
    for hi, hname in enumerate(BUCKET_NAMES):
        for ci, clabel in enumerate(CHORD_LABELS):
            ax = axes[hi, ci]
            entry = means.get((hi, ci))
            if entry is None:
                ax.text(0.5, 0.5, f"n < {MIN_COUNT}", ha='center', va='center', transform=ax.transAxes, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.plot(t, entry['r_lon_mean'], 'b-', linewidth=2, label='r_lon (longitudinal)')
                ax.fill_between(t,
                                entry['r_lon_mean'] - entry['r_lon_std'],
                                entry['r_lon_mean'] + entry['r_lon_std'],
                                color='blue', alpha=0.15)
                ax.plot(t, entry['r_lat_mean'], 'r-', linewidth=2, label='r_lat (lateral)')
                ax.fill_between(t,
                                entry['r_lat_mean'] - entry['r_lat_std'],
                                entry['r_lat_mean'] + entry['r_lat_std'],
                                color='red', alpha=0.15)
                ax.axhline(0, color='k', linewidth=0.5)
                ax.set_title(f"{hname}°, {clabel}  (n={entry['n']})", fontsize=10)
                ax.grid(True, alpha=0.3)
                if hi == 0 and ci == 0:
                    ax.legend(fontsize=8, loc='best')
            if hi == len(BUCKET_NAMES) - 1:
                ax.set_xlabel('t (frame)', fontsize=9)
            if ci == 0:
                ax.set_ylabel('residual (m)', fontsize=9)
    plt.suptitle('Mean GT residual in chord frame, by (heading bucket, chord bucket)\n'
                 'Solid line = mean, shaded = ±1 std. blend=1.0 applies full mean as correction.',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def pick_representative_scenarios(dataset, scenario_indices, grouped_means):
    """Pick representative scenarios: 1-2 per heading bucket (when available)."""
    by_bucket = {i: [] for i in range(len(BUCKET_NAMES))}
    for idx in scenario_indices:
        try:
            sample = dataset[int(idx)]
            history_end, goal, h0, h1, future, prior_m = load_scenario_data(sample)
            chord = goal - history_end
            chord_len = float(np.linalg.norm(chord))
            delta = h1 - h0
            delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
            abs_delta_deg = math.degrees(abs(delta))
            hb = get_bucket(abs_delta_deg)
            by_bucket[hb].append({
                'idx': int(idx), 'history_end': history_end, 'goal': goal,
                'h0': h0, 'h1': h1, 'future': future, 'chord_len': chord_len,
                'abs_delta_deg': abs_delta_deg,
            })
        except Exception:
            continue
    picked = []
    for hb in range(len(BUCKET_NAMES)):
        if by_bucket[hb]:
            # Pick the first one
            picked.append((hb, by_bucket[hb][0]))
            if len(by_bucket[hb]) > 4:
                # Pick a second one with different chord length if available
                first = by_bucket[hb][0]
                for cand in by_bucket[hb][1:]:
                    if abs(cand['chord_len'] - first['chord_len']) > 10:
                        picked.append((hb, cand))
                        break
    return picked


def plot_prior_vs_gt(dataset, picked, means, save_path):
    """Plot old Hermite vs new corrected prior vs GT for picked scenarios."""
    n = len(picked)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)

    for i, (hb, sc) in enumerate(picked):
        ax = axes[i // n_cols][i % n_cols]
        history_end = sc['history_end']
        goal = sc['goal']
        h0 = sc['h0']
        h1 = sc['h1']
        future = sc['future']
        chord_len = sc['chord_len']
        n_future = future.shape[0]

        # Old prior
        prior_old = _hermite_numpy(history_end, goal, h0, h1, n_future)

        # New prior (with correction)
        cb = None
        for j, (clo, chi) in enumerate(CHORD_BUCKETS):
            if clo <= chord_len < chi:
                cb = j
                break
        entry = means.get((hb, cb)) if cb is not None else None
        if entry is not None:
            chord = goal - history_end
            chord_dir = chord / max(chord_len, 1e-8)
            r_lon_abs = entry['r_lon_mean']  # blend=1.0
            r_lat_abs = entry['r_lat_mean']
            residual_m = chord_frame_to_residual(r_lon_abs, r_lat_abs, chord_dir)
            prior_new = prior_old + residual_m.astype(np.float32)
        else:
            prior_new = prior_old  # no correction available

        # Plot
        ax.plot(prior_old[:, 0], prior_old[:, 1], 'b--', linewidth=1.5, label='Hermite (old)')
        ax.plot(prior_new[:, 0], prior_new[:, 1], 'g-', linewidth=2.0, label='Corrected (new)')
        ax.plot(future[:, 0], future[:, 1], 'r-', linewidth=2.0, alpha=0.7, label='GT future')
        ax.scatter([history_end[0]], [history_end[1]], c='k', s=50, zorder=5, label='history_end')
        ax.scatter([goal[0]], [goal[1]], c='orange', s=80, marker='*', zorder=5, label='goal')

        # Compute peak residuals
        res_old = np.linalg.norm(future - prior_old, axis=-1).max()
        res_new = np.linalg.norm(future - prior_new, axis=-1).max()
        ax.set_title(f"idx={sc['idx']} |Δh|={sc['abs_delta_deg']:.1f}° chord={chord_len:.1f}m\n"
                     f"bucket {BUCKET_NAMES[hb]}°/{CHORD_LABELS[cb] if cb is not None else 'N/A'}\n"
                     f"peak res: old={res_old:.2f}m new={res_new:.2f}m ({(res_new-res_old)/res_old*100:+.1f}%)",
                     fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    # Hide unused
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis('off')

    plt.suptitle('Hermite (old) vs Corrected (new) prior vs GT\n'
                 'Green = corrected prior, Red = GT, Blue dashed = old Hermite',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
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

    print("Collecting residuals from 2000 scenarios...")
    grouped = collect_subbucket_stats(dataset, n_samples=2000, seed=0)
    means = compute_subbucket_means(grouped)

    print("\nSub-bucket counts:")
    header = 'heading\\chord'
    print(f"{header:<14}", end='')
    for clabel in CHORD_LABELS:
        print(f"{clabel:>10}", end='')
    print()
    for hi, hname in enumerate(BUCKET_NAMES):
        print(f"{hname+'°':<14}", end='')
        for ci, _ in enumerate(CHORD_BUCKETS):
            entry = means.get((hi, ci))
            n = entry['n'] if entry else 0
            print(f"{n:>10}", end='')
        print()

    # Save correction curves
    plot_correction_curves(means, "/workspace/experiments/figures/fig_correction_curves.png")

    # Pick representative scenarios (from a larger pool)
    np.random.seed(11)
    pool = list(np.random.choice(len(dataset), size=400, replace=False)) + [2532, 4845, 7578, 1234, 5678]
    picked = pick_representative_scenarios(dataset, pool, means)
    print(f"\nPicked {len(picked)} representative scenarios")
    for hb, sc in picked:
        print(f"  idx={sc['idx']} bucket={BUCKET_NAMES[hb]}° chord={sc['chord_len']:.1f}m |Δh|={sc['abs_delta_deg']:.1f}°")

    plot_prior_vs_gt(dataset, picked, means, "/workspace/experiments/figures/fig_prior_vs_gt.png")


if __name__ == "__main__":
    main()
