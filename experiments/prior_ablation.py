"""Hermite prior ablation: test if heading-based curvature tuning reduces GT residual.

Strategy:
- Current prior: cubic Hermite with adaptive tangent scaling
  scale = dist * (1 + C * excess²), excess = max(|Δh| - 40°, 0), C = 0.4
- Candidate alternatives:
  A. Aggressive scaling: higher C, lower threshold
  B. Asymmetric scaling: separate C for left vs right turns
  C. Higher power: excess^3 instead of excess²
  D. Arc-biarc prior (two circular arcs with G1 continuity)
  E. Linear curvature (clothoid-like) prior
  F. Mid-point bias correction

Metric: GT residual = future - prior
  - peak (max over T)
  - mean
  - per-axis (lon, lat in chord frame)
  - aggregated over many scenarios

Critical: this script does NOT modify any production code.
If results show no improvement, we keep the current prior unchanged.
"""

import sys
import os
import math
import numpy as np
import torch
import yaml
from pathlib import Path

# Setup
sys.path.insert(0, "/workspace")
os.chdir("/workspace")

from data.av2_dataset import Argoverse2Dataset
from data.normalization import (
    compute_hermite_prior, _hermite_numpy,
    RESIDUAL_SCALE, RESIDUAL_SCALE_LON, RESIDUAL_SCALE_LAT,
    residual_to_chord_frame, compute_chord_dir, _HERMITE_TAN_THR, _HERMITE_TAN_C,
    denormalize, denormalize_residual, denormalize_residual_chord, unpack_chord,
)


# ============================================================================
# Candidate prior formulations
# ============================================================================

def prior_current(P0, P1, h0, h1, n):
    """Current production Hermite: scale = dist * (1 + 0.4 * excess²), thr=40°."""
    return _hermite_numpy(P0, P1, h0, h1, n)


def prior_aggressive(P0, P1, h0, h1, n, C=0.8, thr_deg=20.0):
    """Aggressive scaling: C=0.8, thr=20° — curvier for any non-trivial turn."""
    t = np.linspace(0, 1, n, dtype=np.float32)
    chord = P1 - P0
    dist = max(np.linalg.norm(chord), 1e-6)
    d0 = np.array([np.cos(h0), np.sin(h0)], dtype=np.float32)
    d1 = np.array([np.cos(h1), np.sin(h1)], dtype=np.float32)
    delta = h1 - h0
    delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
    excess = max(abs(delta) - math.radians(thr_deg), 0.0)
    scale = dist * (1.0 + C * excess * excess)
    m0 = scale * d0
    m1 = scale * d1
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    return h00[:, None]*P0 + h10[:, None]*m0 + h01[:, None]*P1 + h11[:, None]*m1


def prior_power3(P0, P1, h0, h1, n, C=0.4, thr_deg=40.0):
    """Power-3 scaling: excess^3 (sharper transition above threshold)."""
    t = np.linspace(0, 1, n, dtype=np.float32)
    chord = P1 - P0
    dist = max(np.linalg.norm(chord), 1e-6)
    d0 = np.array([np.cos(h0), np.sin(h0)], dtype=np.float32)
    d1 = np.array([np.cos(h1), np.sin(h1)], dtype=np.float32)
    delta = h1 - h0
    delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
    excess = max(abs(delta) - math.radians(thr_deg), 0.0)
    scale = dist * (1.0 + C * excess ** 3)
    m0 = scale * d0
    m1 = scale * d1
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    return h00[:, None]*P0 + h10[:, None]*m0 + h01[:, None]*P1 + h11[:, None]*m1


def prior_asymmetric(P0, P1, h0, h1, n, C_left=0.4, C_right=0.6, thr_deg=40.0):
    """Asymmetric scaling: different C for left vs right turns.

    Hypothesis: right turns (typically tighter in right-side traffic) need
    different scaling than left turns.
    """
    t = np.linspace(0, 1, n, dtype=np.float32)
    chord = P1 - P0
    dist = max(np.linalg.norm(chord), 1e-6)
    d0 = np.array([np.cos(h0), np.sin(h0)], dtype=np.float32)
    d1 = np.array([np.cos(h1), np.sin(h1)], dtype=np.float32)
    delta = h1 - h0
    delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
    # delta > 0 = left turn (CCW), delta < 0 = right turn (CW)
    excess = max(abs(delta) - math.radians(thr_deg), 0.0)
    C = C_left if delta > 0 else C_right
    scale = dist * (1.0 + C * excess * excess)
    m0 = scale * d0
    m1 = scale * d1
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    return h00[:, None]*P0 + h10[:, None]*m0 + h01[:, None]*P1 + h11[:, None]*m1


def prior_bezier_quintic(P0, P1, h0, h1, n):
    """Quintic Bezier with heading-based control points.

    Bezier quintic has 4 control points B1..B4 (between B0=P0 and B5=P1).
    We set B1, B4 from headings, and B2, B3 are placed geometrically to
    produce a curved path. This is more flexible than Hermite cubic.

    B1 = P0 + (chord_len/5) * d0
    B4 = P1 - (chord_len/5) * d1
    B2, B3 = intersection-based: place at 40% and 60% along the chord,
    with lateral offset proportional to heading change.
    """
    t = np.linspace(0, 1, n, dtype=np.float32)
    chord = P1 - P0
    dist = max(np.linalg.norm(chord), 1e-6)
    d0 = np.array([np.cos(h0), np.sin(h0)], dtype=np.float32)
    d1 = np.array([np.cos(h1), np.sin(h1)], dtype=np.float32)

    delta = h1 - h0
    delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))

    B0 = P0
    B5 = P1
    # End-tangent control points — use adaptive scale
    excess = max(abs(delta) - _HERMITE_TAN_THR, 0.0)
    s = dist * (1.0 + _HERMITE_TAN_C * excess * excess)
    B1 = P0 + (s / 5.0) * d0
    B4 = P1 - (s / 5.0) * d1

    # Mid control points: place at chord 1/3 and 2/3, with lateral offset
    # proportional to heading change.
    chord_hat = chord / dist
    perp = np.array([-chord_hat[1], chord_hat[0]], dtype=np.float32)

    # Offset = chord_len * tan(delta/2) / 3 — pushes the curve toward the
    # turning direction. Clamp to avoid degenerate cases.
    offset = np.tan(delta / 2) * dist / 3.0
    offset = np.clip(offset, -dist * 0.4, dist * 0.4)

    B2 = P0 + chord * (2/5) + offset * perp
    B3 = P0 + chord * (3/5) + offset * perp

    # Quintic Bezier
    pts = (
        (1 - t)[:, None] ** 5 * B0 +
        5 * (1 - t)[:, None] ** 4 * t[:, None] * B1 +
        10 * (1 - t)[:, None] ** 3 * t[:, None] ** 2 * B2 +
        10 * (1 - t)[:, None] ** 2 * t[:, None] ** 3 * B3 +
        5 * (1 - t)[:, None] * t[:, None] ** 4 * B4 +
        t[:, None] ** 5 * B5
    )
    return pts.astype(np.float32)


def prior_arc(P0, P1, h0, h1, n):
    """Single circular arc through P0, P1 with tangent h0 at P0.

    The arc passes through P0 and P1, and is tangent to h0 at P0.
    h1 is enforced only approximately (single arc can't enforce both
    headings unless they're geometrically consistent).

    Falls back to Hermite if arc is degenerate (zero curvature or
    inconsistent geometry).
    """
    chord = P1 - P0
    dist = max(np.linalg.norm(chord), 1e-6)
    d0 = np.array([np.cos(h0), np.sin(h0)], dtype=np.float32)

    # Compute arc: center is intersection of perpendicular bisector of chord
    # and the line through P0 perpendicular to d0.
    # Perpendicular bisector: passes through (P0+P1)/2, direction perpendicular to chord.
    mid = (P0 + P1) / 2.0
    chord_hat = chord / dist
    bisector_dir = np.array([-chord_hat[1], chord_hat[0]], dtype=np.float32)

    # Line through P0 perpendicular to d0 (normal to tangent = radial direction)
    n0 = np.array([-d0[1], d0[0]], dtype=np.float32)

    # Solve: mid + s * bisector_dir = P0 + t * n0
    # => s * bisector_dir - t * n0 = P0 - mid
    A = np.array([
        [bisector_dir[0], -n0[0]],
        [bisector_dir[1], -n0[1]],
    ])
    b = P0 - mid

    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-9:
        # Lines are parallel — no finite arc; fall back to Hermite.
        return _hermite_numpy(P0, P1, h0, h1, n)

    s = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
    center = mid + s * bisector_dir

    radius = np.linalg.norm(P0 - center)
    if radius < 1e-3 or radius > 1e4:
        # Degenerate (too large = essentially straight, or too small = numerical)
        return _hermite_numpy(P0, P1, h0, h1, n)

    # Angles
    theta0 = math.atan2(P0[1] - center[1], P0[0] - center[0])
    theta1 = math.atan2(P1[1] - center[1], P1[0] - center[0])

    # Determine arc direction (CCW or CW) — based on tangent direction at P0.
    # If d0 points CCW around center, go CCW; else CW.
    radial0 = (P0 - center) / radius
    tangent0_ccw = np.array([-radial0[1], radial0[0]], dtype=np.float32)
    sign = 1.0 if np.dot(tangent0_ccw, d0) > 0 else -1.0

    # Sweep angle from theta0 to theta1 in direction `sign`
    sweep = theta1 - theta0
    if sign > 0:
        # CCW: sweep should be positive
        while sweep < 0:
            sweep += 2 * math.pi
        while sweep > 2 * math.pi:
            sweep -= 2 * math.pi
    else:
        # CW: sweep should be negative
        while sweep > 0:
            sweep -= 2 * math.pi
        while sweep < -2 * math.pi:
            sweep += 2 * math.pi

    # Sanity: arc should not be more than ~2π (impossible geometry)
    if abs(sweep) > 1.9 * math.pi:
        return _hermite_numpy(P0, P1, h0, h1, n)

    thetas = theta0 + np.linspace(0, sweep, n, dtype=np.float32)
    xs = center[0] + radius * np.cos(thetas)
    ys = center[1] + radius * np.sin(thetas)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def prior_bezier_cubic_offset(P0, P1, h0, h1, n, offset_factor=0.4):
    """Cubic Bezier with offset-based curvature control.

    B0 = P0, B3 = P1.
    B1 = P0 + (chord/3) * d0 — matches start tangent.
    B2 = P1 - (chord/3) * d1 — matches end tangent.
    Then offset B1, B2 perpendicular to chord by amount proportional to
    heading change (more change = more offset = curvier path).
    """
    t = np.linspace(0, 1, n, dtype=np.float32)
    chord = P1 - P0
    dist = max(np.linalg.norm(chord), 1e-6)
    d0 = np.array([np.cos(h0), np.sin(h0)], dtype=np.float32)
    d1 = np.array([np.cos(h1), np.sin(h1)], dtype=np.float32)

    delta = h1 - h0
    delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))

    B0 = P0
    B3 = P1
    B1 = P0 + (dist / 3.0) * d0
    B2 = P1 - (dist / 3.0) * d1

    # Lateral offset based on heading change
    chord_hat = chord / dist
    perp = np.array([-chord_hat[1], chord_hat[0]], dtype=np.float32)
    # Sign of delta: positive = left turn (CCW) = curve to left = perp direction
    # The offset is applied symmetrically to B1 and B2 to make the curve bow.
    offset = offset_factor * dist * np.sin(delta) * 0.5
    offset = np.clip(offset, -dist * 0.5, dist * 0.5)
    B1 = B1 + offset * perp
    B2 = B2 + offset * perp

    pts = (
        (1 - t)[:, None] ** 3 * B0 +
        3 * (1 - t)[:, None] ** 2 * t[:, None] * B1 +
        3 * (1 - t)[:, None] * t[:, None] ** 2 * B2 +
        t[:, None] ** 3 * B3
    )
    return pts.astype(np.float32)


def prior_clothoid_approx(P0, P1, h0, h1, n, n_segs=8):
    """Approximate clothoid (linear curvature change) by stitching small arcs.

    Curvature k(s) goes linearly from k0 to k1, where k0, k1 are chosen so
    that the resulting curve passes through P1 and ends with heading h1.

    Approximation: discretize s into n_segs segments, each a circular arc
    with the average curvature over that segment. Iteratively solve for
    (k0, k1) by Newton's method on (P1, h1) constraint.

    Simplification: pick (k0, k1) so that k_avg = (k0+k1)/2 = arc curvature,
    and curvature variation dk = k1 - k0 controls how the curvature
    distributes. Solve for k_avg and dk using the 2D constraint (position).

    For now, just use single arc and adjust by heading mismatch.
    """
    # Start with single arc
    arc = prior_arc(P0, P1, h0, h1, n)

    # If single arc was returned, also compute end heading at P1
    if n < 2:
        return arc

    # Compute the actual end heading from the arc
    tangent_end = arc[-1] - arc[-2]
    actual_h1 = math.atan2(tangent_end[1], tangent_end[0])

    # Compare with desired h1 — if mismatch is large, try to fix by perturbing
    # the chord to introduce heading change distribution.
    h1_err = h1 - actual_h1
    h1_err = h1_err - 2 * math.pi * np.round(h1_err / (2 * math.pi))

    # If end heading already matches, we're done
    if abs(h1_err) < 0.05:
        return arc

    # Otherwise, fall back to Hermite (which enforces both headings exactly
    # but doesn't follow an arc shape)
    return _hermite_numpy(P0, P1, h0, h1, n)


# ============================================================================
# Residual metrics
# ============================================================================

def compute_residual_metrics(future, prior, history_end, goal):
    """Compute residual = future - prior metrics.

    All inputs are (T, 2) in local meters.

    Returns dict with:
      peak, mean (Euclidean)
      peak_lon, peak_lat, mean_lon, mean_lat (chord frame, abs)
      excess_fraction (fraction of |residual| > 1m)
    """
    residual = future - prior  # (T, 2)

    # Euclidean residual
    norm = np.linalg.norm(residual, axis=-1)  # (T,)
    peak = float(norm.max())
    mean = float(norm.mean())

    # Chord-frame decomposition
    r_lon, r_lat = residual_to_chord_frame(residual, history_end, goal)
    if r_lon is None:
        # Degenerate chord
        r_lon = np.zeros_like(norm)
        r_lat = np.zeros_like(norm)

    peak_lon = float(np.abs(r_lon).max())
    peak_lat = float(np.abs(r_lat).max())
    mean_lon = float(np.abs(r_lon).mean())
    mean_lat = float(np.abs(r_lat).mean())

    # Excess fraction: how many frames exceed 1m deviation
    excess_fraction = float((norm > 1.0).mean())

    return {
        "peak": peak, "mean": mean,
        "peak_lon": peak_lon, "peak_lat": peak_lat,
        "mean_lon": mean_lon, "mean_lat": mean_lat,
        "excess_fraction": excess_fraction,
    }


# ============================================================================
# Scenario loading
# ============================================================================

def load_scenario(sample, dataset):
    """Extract (history_end, goal, h0, h1, future_local, drivable_areas_local)
    from a dataset sample.

    Re-uses the dataset's own loading so we get exactly the same data the
    model sees. We reconstruct unnormalized values from the normalized
    sample fields (denormalize / denormalize_residual).
    """
    # Reconstruct from normalized sample fields.
    # goal_norm: normalized goal in local coords (1D, 2)
    goal = to_np(denormalize(sample["goal"].reshape(1, 2))).flatten()
    # history_norm: normalized history in local coords (T_h, 2) — last point is history_end
    history_end = to_np(denormalize(sample["history"][-1:].reshape(1, 2))).flatten()
    h0 = float(sample["start_heading"].item())
    h1 = float(sample["end_heading"].item())

    # Reconstruct future_local = prior + residual (all in meters)
    prior_norm = to_np(sample["prior"])  # (T, 2) normalized
    prior_m = to_np(denormalize(prior_norm))
    traj_norm = to_np(sample["trajectory"])  # (T, 2) normalized residual

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
    n_future = future.shape[0]

    # Get drivable areas for boundary-aware prior (optional)
    da = None
    scene_data = sample.get("scene_data")
    if scene_data is not None:
        da = scene_data.get("drivable_areas_local", None)

    return history_end, goal, h0, h1, future, n_future, da


def to_np(x):
    """Convert torch tensor or numpy array to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ============================================================================
# Main experiment
# ============================================================================

PRIOR_FORMULATIONS = {
    "current (C=0.4, thr=40°, power=2)": prior_current,
    "aggressive (C=0.8, thr=20°)": prior_aggressive,
    "power3 (C=0.4, thr=40°, power=3)": prior_power3,
    "asymmetric (left=0.4, right=0.6)": prior_asymmetric,
    "bezier_quintic": prior_bezier_quintic,
    "arc": prior_arc,
    "bezier_cubic_offset": prior_bezier_cubic_offset,
    "clothoid_approx": prior_clothoid_approx,
}


def run_experiment(scenario_indices, n_total=100):
    """Run experiment on a list of scenario indices.

    Returns:
        results: dict keyed by formulation name, each a list of per-scenario dicts
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

    # Build full index list
    if scenario_indices is None:
        # Use first n_total scenarios in the dataset
        all_indices = list(range(min(n_total, len(dataset))))
    else:
        all_indices = [i for i in scenario_indices if 0 <= i < len(dataset)]

    print(f"Running experiment on {len(all_indices)} scenarios...")

    results = {name: [] for name in PRIOR_FORMULATIONS}
    scenario_info = []

    for idx in all_indices:
        try:
            sample = dataset[idx]
        except Exception as e:
            print(f"  [skip {idx}] load error: {e}")
            continue

        try:
            history_end, goal, h0, h1, future, n_future, da = load_scenario(sample, dataset)
        except Exception as e:
            print(f"  [skip {idx}] extract error: {e}")
            continue

        # Compute chord length and heading change for reporting
        chord = goal - history_end
        chord_len = float(np.linalg.norm(chord))
        delta = h1 - h0
        delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
        abs_delta_deg = math.degrees(abs(delta))

        scenario_info.append({
            "idx": idx, "chord_len": chord_len, "abs_delta_deg": abs_delta_deg,
        })

        # Compute each prior formulation
        for name, fn in PRIOR_FORMULATIONS.items():
            try:
                prior = fn(history_end, goal, h0, h1, n_future)
                metrics = compute_residual_metrics(future, prior, history_end, goal)
                results[name].append(metrics)
            except Exception as e:
                print(f"  [skip {idx}/{name}] prior error: {e}")
                results[name].append(None)

    return results, scenario_info


def aggregate_results(results, scenario_info):
    """Aggregate per-scenario results into summary stats."""
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


def bucket_by_turn_angle(results, scenario_info):
    """Bucket results by absolute heading change.

    Buckets: 0-15° (straight), 15-45° (mild), 45-90° (moderate),
             90-135° (sharp), 135-180° (U-turn)
    """
    bucket_edges = [0, 15, 45, 90, 135, 180]
    bucket_names = ["0-15°", "15-45°", "45-90°", "90-135°", "135-180°"]

    by_bucket = {name: {form: [] for form in PRIOR_FORMULATIONS} for name in bucket_names}

    for si, scenario in enumerate(scenario_info):
        deg = scenario["abs_delta_deg"]
        for i, name in enumerate(bucket_names):
            if bucket_edges[i] <= deg < bucket_edges[i + 1]:
                for form in PRIOR_FORMULATIONS:
                    m = results[form][si]
                    if m is not None:
                        by_bucket[name][form].append(m)
                break

    return by_bucket, bucket_names


def print_summary(summary, baseline_name="current (C=0.4, thr=40°, power=2)"):
    """Print summary table comparing formulations against baseline."""
    if baseline_name not in summary or summary[baseline_name] is None:
        print("ERROR: baseline not available")
        return

    baseline = summary[baseline_name]
    print()
    print("=" * 100)
    print(f"{'Formulation':<45} {'peak(m)':>8} {'Δpeak%':>8} {'mean(m)':>8} {'Δmean%':>8} {'excess%':>8}")
    print("-" * 100)
    for name, s in summary.items():
        if s is None:
            continue
        d_peak = (s["peak_mean"] - baseline["peak_mean"]) / baseline["peak_mean"] * 100
        d_mean = (s["mean_mean"] - baseline["mean_mean"]) / baseline["mean_mean"] * 100
        marker = " ←" if name == baseline_name else ""
        print(f"{name:<45} {s['peak_mean']:>8.3f} {d_peak:>+8.1f} {s['mean_mean']:>8.3f} {d_mean:>+8.1f} {s['excess_fraction']*100:>7.1f}%{marker}")
    print("=" * 100)


def print_bucketed(by_bucket, bucket_names, baseline_name="current (C=0.4, thr=40°, power=2)"):
    """Print per-bucket comparison."""
    print()
    print("=" * 110)
    print(f"{'Bucket':<12} ", end="")
    for form in PRIOR_FORMULATIONS:
        short = form.split(" ")[0]
        print(f"{short:>14}", end="")
    print()
    print("-" * 110)
    for bn in bucket_names:
        bucket = by_bucket[bn]
        n = len(bucket[baseline_name]) if baseline_name in bucket else 0
        print(f"{bn:<12} (n={n:<3}) ", end="")
        for form in PRIOR_FORMULATIONS:
            ms = bucket[form]
            if not ms:
                print(f"{'--':>14}", end="")
            else:
                peak = float(np.mean([m["peak"] for m in ms]))
                print(f"{peak:>14.3f}", end="")
        print()
    print("=" * 110)


def print_bucketed_delta(by_bucket, bucket_names, baseline_name="current (C=0.4, thr=40°, power=2)"):
    """Print per-bucket delta vs baseline."""
    print()
    print("=" * 110)
    print(f"{'Bucket':<12} ", end="")
    for form in PRIOR_FORMULATIONS:
        short = form.split(" ")[0]
        print(f"{short:>14}", end="")
    print()
    print("-" * 110)
    for bn in bucket_names:
        bucket = by_bucket[bn]
        baseline_ms = bucket[baseline_name]
        if not baseline_ms:
            baseline_peak = float('nan')
        else:
            baseline_peak = float(np.mean([m["peak"] for m in baseline_ms]))
        n = len(baseline_ms)
        print(f"{bn:<12} (n={n:<3}) ", end="")
        for form in PRIOR_FORMULATIONS:
            ms = bucket[form]
            if not ms or baseline_peak != baseline_peak:
                print(f"{'--':>14}", end="")
            else:
                peak = float(np.mean([m["peak"] for m in ms]))
                delta_pct = (peak - baseline_peak) / baseline_peak * 100
                print(f"{delta_pct:>+13.1f}%", end="")
        print()
    print("=" * 110)


if __name__ == "__main__":
    # Test on a mix of scenarios: the 5 from prior experiments + 95 more random
    fixed_indices = [2532, 4845, 7578, 1234, 5678]
    # Add 95 more random scenarios for statistical power
    np.random.seed(42)
    random_indices = np.random.choice(11000, size=95, replace=False).tolist()
    all_indices = fixed_indices + random_indices

    results, scenario_info = run_experiment(all_indices, n_total=100)

    # Per-scenario detail for the 5 fixed scenarios
    print()
    print("=" * 80)
    print("Per-scenario details (fixed test scenarios):")
    print("=" * 80)
    for i, si in enumerate(scenario_info[:5]):
        print(f"\nScenario {si['idx']}: chord={si['chord_len']:.2f}m, |Δh|={si['abs_delta_deg']:.1f}°")
        baseline_peak = results["current (C=0.4, thr=40°, power=2)"][i]["peak"] if results["current (C=0.4, thr=40°, power=2)"][i] else float('nan')
        for form in PRIOR_FORMULATIONS:
            m = results[form][i]
            if m is None:
                continue
            d_peak = (m["peak"] - baseline_peak) / baseline_peak * 100 if baseline_peak == baseline_peak else 0
            print(f"  {form:<45} peak={m['peak']:.3f}m ({d_peak:+.1f}%) mean={m['mean']:.3f}m")

    # Aggregate summary
    summary = aggregate_results(results, scenario_info)
    print()
    print_summary(summary)

    # Bucketed by turn angle
    by_bucket, bucket_names = bucket_by_turn_angle(results, scenario_info)
    print()
    print("Peak residual by turn angle bucket (mean across scenarios):")
    print_bucketed(by_bucket, bucket_names)
    print()
    print("Peak residual delta vs baseline by turn angle bucket:")
    print_bucketed_delta(by_bucket, bucket_names)
