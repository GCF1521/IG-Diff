"""Residual prior experiment: learn mean GT residual from training data,
then apply it as a correction to the Hermite prior.

Strategy:
- Collect GT residuals (future - hermite_prior) in chord-frame from training scenarios.
- Bucket by absolute heading change (|Δh|).
- Compute mean residual curve per bucket: r̄_lon(t), r̄_lat(t).
- New prior: hermite + chord_frame_to_residual(r̄_lon(t), r̄_lat(t), chord_dir).

Critical constraint: if this approach doesn't work, DO NOT modify any production code.
"""

import sys
import os
import math
import numpy as np
import torch
import yaml
from pathlib import Path

sys.path.insert(0, "/workspace")
os.chdir("/workspace")

from data.av2_dataset import Argoverse2Dataset
from data.normalization import (
    compute_hermite_prior, _hermite_numpy,
    RESIDUAL_SCALE, RESIDUAL_SCALE_LON, RESIDUAL_SCALE_LAT,
    residual_to_chord_frame, chord_frame_to_residual, compute_chord_dir,
    _HERMITE_TAN_THR, _HERMITE_TAN_C,
    denormalize, denormalize_residual, denormalize_residual_chord, unpack_chord,
)


# ============================================================================
# Bucketing scheme
# ============================================================================

# Bucket edges (degrees of absolute heading change)
BUCKET_EDGES = [0, 15, 45, 90, 120, 180]
BUCKET_NAMES = ["0-15", "15-45", "45-90", "90-120", "120-180"]


def get_bucket(abs_delta_deg):
    """Return bucket index for an absolute heading change in degrees."""
    for i, edge in enumerate(BUCKET_EDGES[:-1]):
        if BUCKET_EDGES[i] <= abs_delta_deg < BUCKET_EDGES[i + 1]:
            return i
    return len(BUCKET_NAMES) - 1  # last bucket


# ============================================================================
# Data collection
# ============================================================================

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_scenario_data(sample):
    """Extract scenario data needed for residual analysis."""
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


def compute_residual_metrics(future, prior, history_end, goal):
    """Compute residual = future - prior metrics."""
    residual = future - prior
    norm = np.linalg.norm(residual, axis=-1)
    peak = float(norm.max())
    mean = float(norm.mean())

    r_lon, r_lat = residual_to_chord_frame(residual, history_end, goal)
    if r_lon is None:
        r_lon = np.zeros_like(norm)
        r_lat = np.zeros_like(norm)

    peak_lon = float(np.abs(r_lon).max())
    peak_lat = float(np.abs(r_lat).max())
    mean_lon = float(np.abs(r_lon).mean())
    mean_lat = float(np.abs(r_lat).mean())
    excess_fraction = float((norm > 1.0).mean())

    return {
        "peak": peak, "mean": mean,
        "peak_lon": peak_lon, "peak_lat": peak_lat,
        "mean_lon": mean_lon, "mean_lat": mean_lat,
        "excess_fraction": excess_fraction,
    }


# ============================================================================
# Mean residual collection
# ============================================================================

def collect_residuals(dataset, n_samples=2000, seed=0):
    """Collect chord-frame residuals from training scenarios.

    For each scenario:
      - Compute (history_end, goal, h0, h1, future, prior_m)
      - Compute residual = future - prior_m
      - Project residual onto chord frame to get (r_lon(t), r_lat(t))
      - Record (|Δh| bucket, r_lon(t), r_lat(t), chord_len)

    Returns:
      bucket_residuals: dict bucket_idx -> list of (r_lon (T,), r_lat (T,), chord_len)
    """
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    bucket_residuals = {i: [] for i in range(len(BUCKET_NAMES))}
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

            residual = future - prior_m  # (T, 2)
            r_lon, r_lat = residual_to_chord_frame(residual, history_end, goal)
            if r_lon is None:
                skipped += 1
                continue

            bucket = get_bucket(abs_delta_deg)
            bucket_residuals[bucket].append((r_lon, r_lat, chord_len))
        except Exception as e:
            skipped += 1
            continue

    return bucket_residuals, skipped


def compute_bucket_mean_residuals(bucket_residuals, normalize_by_chord=True):
    """Compute mean residual curve per bucket.

    Args:
        bucket_residuals: dict bucket_idx -> list of (r_lon (T,), r_lat (T,), chord_len)
        normalize_by_chord: if True, normalize residual by chord length before averaging
            (so that scenarios with different chord lengths contribute comparably).
            The result is stored as ratio; at inference time, multiply by chord_len.

    Returns:
        bucket_means: dict bucket_idx -> (r_lon_mean (T,), r_lat_mean (T,), count)
                      in absolute meters (already multiplied back by chord_len if normalized).
    """
    bucket_means = {}
    for bidx, items in bucket_residuals.items():
        if not items:
            bucket_means[bidx] = None
            continue

        r_lons = np.stack([it[0] for it in items])  # (N, T)
        r_lats = np.stack([it[1] for it in items])  # (N, T)
        chord_lens = np.array([it[2] for it in items])  # (N,)

        if normalize_by_chord:
            # Normalize by chord length: residual_ratio = residual / chord_len
            r_lons_n = r_lons / chord_lens[:, None]
            r_lats_n = r_lats / chord_lens[:, None]
            r_lon_mean_ratio = r_lons_n.mean(axis=0)  # (T,)
            r_lat_mean_ratio = r_lats_n.mean(axis=0)
            # Store as ratio (so we can multiply by chord_len at inference)
            bucket_means[bidx] = (r_lon_mean_ratio, r_lat_mean_ratio, len(items), "ratio")
        else:
            r_lon_mean = r_lons.mean(axis=0)
            r_lat_mean = r_lats.mean(axis=0)
            bucket_means[bidx] = (r_lon_mean, r_lat_mean, len(items), "absolute")

    return bucket_means


# ============================================================================
# Residual-corrected prior
# ============================================================================

def prior_with_residual_correction(
    P0, P1, h0, h1, n,
    bucket_means,
    use_correction=True,
    normalize_by_chord=True,
    blend_factor=1.0,
):
    """Hermite prior + mean residual correction.

    Args:
        bucket_means: dict from compute_bucket_mean_residuals
        use_correction: if False, just returns plain Hermite (for ablation)
        normalize_by_chord: must match what was used in compute_bucket_mean_residuals
        blend_factor: 0 = no correction, 1 = full correction, 0.5 = half
    """
    base_prior = _hermite_numpy(P0, P1, h0, h1, n)
    if not use_correction:
        return base_prior

    # Determine bucket
    delta = h1 - h0
    delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
    abs_delta_deg = math.degrees(abs(delta))
    bucket = get_bucket(abs_delta_deg)

    entry = bucket_means.get(bucket)
    if entry is None:
        return base_prior

    r_lon_mean, r_lat_mean, count, mode = entry
    if count < 5:  # too few samples to trust the mean
        return base_prior

    chord = P1 - P0
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-3:
        return base_prior

    # Compute chord direction (matches compute_chord_dir)
    chord_dir = chord / max(chord_len, 1e-8)

    # Apply correction
    if mode == "ratio":
        r_lon_abs = r_lon_mean * chord_len
        r_lat_abs = r_lat_mean * chord_len
    else:
        r_lon_abs = r_lon_mean
        r_lat_abs = r_lat_mean

    r_lon_abs = r_lon_abs * blend_factor
    r_lat_abs = r_lat_abs * blend_factor

    # Convert back to xy
    residual_m = chord_frame_to_residual(r_lon_abs, r_lat_abs, chord_dir)
    return base_prior + residual_m.astype(np.float32)


# ============================================================================
# Experiment
# ============================================================================

def run_experiment(scenario_indices, n_train=2000, n_eval=300, seed=0):
    """Train on n_train scenarios, evaluate on n_eval held-out scenarios.

    Returns:
        results: dict form_name -> list of per-scenario metrics
        scenario_info: list of dicts
    """
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

    # --- Phase 1: collect residuals on training set ---
    print(f"\n[Phase 1] Collecting residuals from {n_train} training scenarios...")
    bucket_residuals, skipped = collect_residuals(dataset, n_samples=n_train, seed=seed)
    print(f"  Bucket counts:")
    for i, name in enumerate(BUCKET_NAMES):
        print(f"    {name}°: {len(bucket_residuals[i])} scenarios")
    print(f"  Skipped: {skipped}")

    # --- Phase 2: compute mean residuals ---
    print(f"\n[Phase 2] Computing bucket mean residuals (chord-normalized)...")
    bucket_means_norm = compute_bucket_mean_residuals(bucket_residuals, normalize_by_chord=True)
    for i, name in enumerate(BUCKET_NAMES):
        entry = bucket_means_norm[i]
        if entry is None:
            continue
        r_lon, r_lat, count, mode = entry
        print(f"  {name}° (n={count}): |r_lon| peak={np.abs(r_lon).max():.4f} mean={np.abs(r_lon).mean():.4f} (ratio); "
              f"|r_lat| peak={np.abs(r_lat).max():.4f} mean={np.abs(r_lat).mean():.4f} (ratio)")

    print(f"\n[Phase 2b] Computing bucket mean residuals (absolute)...")
    bucket_means_abs = compute_bucket_mean_residuals(bucket_residuals, normalize_by_chord=False)
    for i, name in enumerate(BUCKET_NAMES):
        entry = bucket_means_abs[i]
        if entry is None:
            continue
        r_lon, r_lat, count, mode = entry
        print(f"  {name}° (n={count}): |r_lon| peak={np.abs(r_lon).max():.4f}m mean={np.abs(r_lon).mean():.4f}m; "
              f"|r_lat| peak={np.abs(r_lat).max():.4f}m mean={np.abs(r_lat).mean():.4f}m")

    # --- Phase 3: evaluate on held-out scenarios ---
    # Use a different seed for evaluation scenarios
    np.random.seed(seed + 1)
    eval_pool_size = len(dataset)
    eval_indices = np.random.choice(eval_pool_size, size=min(n_eval, eval_pool_size), replace=False)
    # Add the 5 fixed test scenarios for direct comparison
    fixed = [2532, 4845, 7578, 1234, 5678]
    eval_indices = list(eval_indices) + [i for i in fixed if i not in eval_indices]

    print(f"\n[Phase 3] Evaluating on {len(eval_indices)} held-out scenarios...")

    # Define formulations to evaluate
    formulations = {
        "current (no correction)": lambda P0, P1, h0, h1, n:
            _hermite_numpy(P0, P1, h0, h1, n),
        "correction (chord-norm, blend=1.0)": lambda P0, P1, h0, h1, n:
            prior_with_residual_correction(P0, P1, h0, h1, n, bucket_means_norm, True, True, 1.0),
        "correction (chord-norm, blend=0.5)": lambda P0, P1, h0, h1, n:
            prior_with_residual_correction(P0, P1, h0, h1, n, bucket_means_norm, True, True, 0.5),
        "correction (absolute, blend=1.0)": lambda P0, P1, h0, h1, n:
            prior_with_residual_correction(P0, P1, h0, h1, n, bucket_means_abs, True, False, 1.0),
        "correction (absolute, blend=0.5)": lambda P0, P1, h0, h1, n:
            prior_with_residual_correction(P0, P1, h0, h1, n, bucket_means_abs, True, False, 0.5),
    }

    results = {name: [] for name in formulations}
    scenario_info = []

    for idx in eval_indices:
        try:
            sample = dataset[int(idx)]
            history_end, goal, h0, h1, future, prior_m = load_scenario_data(sample)
        except Exception as e:
            continue

        chord = goal - history_end
        chord_len = float(np.linalg.norm(chord))
        delta = h1 - h0
        delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
        abs_delta_deg = math.degrees(abs(delta))
        bucket = get_bucket(abs_delta_deg)

        scenario_info.append({
            "idx": int(idx), "chord_len": chord_len,
            "abs_delta_deg": abs_delta_deg, "bucket": bucket,
        })

        n_future = future.shape[0]
        for name, fn in formulations.items():
            try:
                prior = fn(history_end, goal, h0, h1, n_future)
                m = compute_residual_metrics(future, prior, history_end, goal)
                results[name].append(m)
            except Exception as e:
                results[name].append(None)

    return results, scenario_info, bucket_means_norm, bucket_means_abs


def aggregate_results(results):
    summary = {}
    for name, per_scenario in results.items():
        valid = [m for m in per_scenario if m is not None]
        if not valid:
            summary[name] = None
            continue
        summary[name] = {
            "n": len(valid),
            "peak_mean": float(np.mean([m["peak"] for m in valid])),
            "peak_p90": float(np.percentile([m["peak"] for m in valid], 90)),
            "mean_mean": float(np.mean([m["mean"] for m in valid])),
            "peak_lon_mean": float(np.mean([m["peak_lon"] for m in valid])),
            "peak_lat_mean": float(np.mean([m["peak_lat"] for m in valid])),
            "mean_lon_mean": float(np.mean([m["mean_lon"] for m in valid])),
            "mean_lat_mean": float(np.mean([m["mean_lat"] for m in valid])),
            "excess_fraction": float(np.mean([m["excess_fraction"] for m in valid])),
        }
    return summary


def print_summary(summary, baseline_name="current (no correction)"):
    if baseline_name not in summary or summary[baseline_name] is None:
        print("ERROR: baseline not available")
        return
    baseline = summary[baseline_name]
    print()
    print("=" * 110)
    print(f"{'Formulation':<48} {'peak(m)':>8} {'Δpeak%':>8} {'mean(m)':>8} {'Δmean%':>8} {'excess%':>8}")
    print("-" * 110)
    for name, s in summary.items():
        if s is None:
            continue
        d_peak = (s["peak_mean"] - baseline["peak_mean"]) / baseline["peak_mean"] * 100
        d_mean = (s["mean_mean"] - baseline["mean_mean"]) / baseline["mean_mean"] * 100
        marker = " ←" if name == baseline_name else ""
        print(f"{name:<48} {s['peak_mean']:>8.3f} {d_peak:>+8.1f} {s['mean_mean']:>8.3f} {d_mean:>+8.1f} {s['excess_fraction']*100:>7.1f}%{marker}")
    print("=" * 110)


def print_bucketed(results, scenario_info, baseline_name="current (no correction)"):
    """Print per-bucket comparison."""
    print()
    print("=" * 100)
    print("Per-bucket peak residual (mean across scenarios in bucket):")
    print("-" * 100)

    # Get formulation names
    form_names = list(results.keys())

    # Group by bucket
    by_bucket = {i: [] for i in range(len(BUCKET_NAMES))}
    for i, si in enumerate(scenario_info):
        by_bucket[si["bucket"]].append(i)

    print(f"{'Bucket':<12} {'n':>4} ", end="")
    for name in form_names:
        short = name.split("(")[0].strip()[:18]
        print(f"{short:>20}", end="")
    print()
    print("-" * 100)

    for bidx, name in enumerate(BUCKET_NAMES):
        idxs = by_bucket[bidx]
        if not idxs:
            continue
        print(f"{name}°{'':<8} {len(idxs):>4} ", end="")
        baseline_peaks = [results[baseline_name][i]["peak"] for i in idxs if results[baseline_name][i] is not None]
        b_val = sum(baseline_peaks) / len(baseline_peaks) if baseline_peaks else float('nan')
        for name in form_names:
            peaks = [results[name][i]["peak"] for i in idxs if results[name][i] is not None]
            if not peaks:
                print(f"{'--':>20}", end="")
            else:
                v = sum(peaks) / len(peaks)
                if b_val == b_val:
                    d = (v - b_val) / b_val * 100
                    print(f"{v:>9.3f} ({d:+5.1f}%)", end="")
                else:
                    print(f"{v:>20.3f}", end="")
        print()
    print("=" * 100)


if __name__ == "__main__":
    # Train on 2000 scenarios, evaluate on 305 (300 random + 5 fixed)
    results, scenario_info, bucket_means_norm, bucket_means_abs = run_experiment(
        scenario_indices=None, n_train=2000, n_eval=300, seed=0
    )

    # Per-scenario detail for the 5 fixed scenarios (last 5 in eval list)
    print()
    print("=" * 100)
    print("Per-scenario details (fixed test scenarios):")
    print("=" * 100)
    for i in range(len(scenario_info) - 5, len(scenario_info)):
        si = scenario_info[i]
        print(f"\nScenario {si['idx']}: chord={si['chord_len']:.2f}m, |Δh|={si['abs_delta_deg']:.1f}°, bucket={BUCKET_NAMES[si['bucket']]}°")
        baseline_m = results["current (no correction)"][i]
        baseline_peak = baseline_m["peak"] if baseline_m else float('nan')
        for name in results:
            m = results[name][i]
            if m is None:
                continue
            d = (m["peak"] - baseline_peak) / baseline_peak * 100 if baseline_peak == baseline_peak else 0
            print(f"  {name:<48} peak={m['peak']:.3f}m ({d:+.1f}%) mean={m['mean']:.3f}m")

    # Summary
    summary = aggregate_results(results)
    print_summary(summary)

    # Bucketed
    print_bucketed(results, scenario_info)
