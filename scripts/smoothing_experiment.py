"""Experiment: try multiple smoothing schemes on the same raw trajectories
and visualize them side-by-side so the user can review which one preserves
the raw shape while still damping oscillations.

Inputs: /workspace/output/collision_video_20sc_v2/results.pt
  - generated_local: smoothed (current scheme — too aggressive)
  - generated_local_raw: raw model output (the baseline we want to stay close to)

Schemes implemented (all operate on the RAW future trajectory + history tail):
  M0: Raw (no smoothing) — baseline
  M1: Savitzky-Golay only (light, window=7, poly=2) — gentle low-pass
  M2: Savitzky-Golay only (medium, window=15, poly=2) — current window without kinematic
  M3: Savitzky-Golay only (heavy, window=25, poly=3) — strong low-pass
  M4: Moving-average (window=5) — simplest smoother
  M5: Gaussian smooth (sigma=2.0) — smooth, no poly fit
  M6: Current pipeline (SavGol w=15 + kinematic projection) — reference for "too aggressive"
  M7: SavGol w=11 + soft kinematic (relaxed a_max=6.0, jerk_max=8.0) — middle ground

Output: one comparison figure per scenario at
  /workspace/output/smoothing_experiment/scenario_{idx}_smoothing.png
with 2 columns × 4 rows = 8 panels (M0..M7).
"""
import sys
sys.path.insert(0, "/workspace")
import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splrep, splev

from data.av2_dataset import Argoverse2Dataset
from data.normalization import denormalize
from data.coordinate_utils import global_to_local
from viz.style import apply_dark_theme, save_figure, get_trajectory_colors, BG_COLOR, TEXT_COLOR
from viz.utils import to_numpy


DT = 0.1
N_CTX = 5  # history context frames for join-aware smoothers


# ---------------------------------------------------------------------------
# Smoothing schemes — each returns (T_fut, 2) smoothed future positions
# ---------------------------------------------------------------------------

def _enforce_join_continuity(joined, endpoint_blend=True, join_strength=0.5):
    """Soften the join velocity constraint AND keep the endpoint fixed.

    Only C1 (position + velocity) is enforced, and only PARTIALLY — the
    join velocity is blended toward history's last motion, not forced
    to match it exactly. Acceleration is not constrained.

    Args:
        joined: (T_joined, 2) — [hist_tail ; fut] with hist already FROZEN.
        endpoint_blend: if True, distribute endpoint correction across the
                        whole future so the endpoint stays at the smoothed
                        curve's end.
        join_strength: 0.0 = no join fix (pure SavGol); 1.0 = full C1
                       (v_join exactly matches history's continuation).
                       Default 0.5 = halfway blend.

    Returns:
        (T_joined, 2) with history unchanged and future re-integrated.
    """
    fut_start = N_CTX + 1
    T = joined.shape[0]
    if T <= fut_start or N_CTX < 2 or join_strength <= 0.0:
        return joined

    # History's last velocity & acceleration (frozen)
    v_hist_last = (joined[fut_start - 1] - joined[fut_start - 2]) / DT
    v_hist_prev = (joined[fut_start - 2] - joined[fut_start - 3]) / DT
    a_hist_last = (v_hist_last - v_hist_prev) / DT

    # Smoothed future's first velocity
    v_join_smoothed = (joined[fut_start] - joined[fut_start - 1]) / DT

    # Desired v_join (full continuation of history's motion)
    v_join_desired = v_hist_last + a_hist_last * DT

    # PARTIAL blend: target = (1-strength)*smoothed + strength*desired
    # Only `strength` fraction of the error is corrected.
    v_join_target = (1.0 - join_strength) * v_join_smoothed + join_strength * v_join_desired
    dv_join = v_join_smoothed - v_join_target  # = join_strength * (smoothed - desired)

    # Smoothed future positions & velocities
    fut_pos_smoothed = joined[fut_start:].copy()
    T_fut = fut_pos_smoothed.shape[0]
    fut_vel_smoothed = np.zeros_like(fut_pos_smoothed)
    fut_vel_smoothed[:-1] = np.diff(fut_pos_smoothed, axis=0) / DT
    fut_vel_smoothed[-1] = fut_vel_smoothed[-2]

    # Two smoothstep envelopes
    s = np.arange(T_fut, dtype=np.float64) / max(T_fut - 1, 1)
    env_head = 1.0 - s
    env_head = env_head * env_head * (3.0 - 2.0 * env_head)
    env_tail = s * s * (3.0 - 2.0 * s)

    # Head correction: fix v_join partially, decays to 0 at endpoint
    v_corr_head = dv_join[np.newaxis, :] * env_head[:, np.newaxis]
    fut_vel_corrected = fut_vel_smoothed - v_corr_head

    # Tail correction: preserve endpoint, rises from 0 at join
    if endpoint_blend:
        endpoint_corrected = joined[fut_start - 1] + fut_vel_corrected.sum(axis=0) * DT
        endpoint_target = fut_pos_smoothed[-1]
        endpoint_drift = endpoint_corrected - endpoint_target
        if np.linalg.norm(endpoint_drift) > 1e-6:
            env_tail_sum = env_tail.sum()
            if env_tail_sum > 1e-6:
                v_corr_tail = -endpoint_drift / (env_tail_sum * DT)
                fut_vel_corrected = fut_vel_corrected + v_corr_tail[np.newaxis, :] * env_tail[:, np.newaxis]

    # Re-integrate positions from corrected velocities
    new_fut = np.zeros_like(fut_pos_smoothed)
    new_fut[0] = joined[fut_start - 1] + fut_vel_corrected[0] * DT
    for i in range(1, T_fut):
        new_fut[i] = new_fut[i - 1] + fut_vel_corrected[i] * DT

    out = joined.copy()
    out[fut_start:] = new_fut
    return out


def _savgol_smooth(joined, window, poly):
    """Apply SavGol to joined [hist_tail; fut], preserve history exactly."""
    if joined.shape[0] < window:
        return joined.copy()
    out = np.zeros_like(joined)
    out[:, 0] = savgol_filter(joined[:, 0], window, poly, deriv=0)
    out[:, 1] = savgol_filter(joined[:, 1], window, poly, deriv=0)
    fut_start = N_CTX + 1
    # Preserve history exactly
    out[:fut_start] = joined[:fut_start]
    # Re-anchor future[0] to history[-1] (no position jump at join)
    offset = joined[fut_start - 1] - out[fut_start - 1]
    out[fut_start:] = out[fut_start:] + offset
    return out


def _moving_average(joined, window):
    """Centered moving average with edge handling via reflection."""
    out = np.zeros_like(joined)
    half = window // 2
    for d in range(2):
        x = joined[:, d]
        padded = np.pad(x, half, mode="reflect")
        smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
        out[:, d] = smoothed[:joined.shape[0]]
    fut_start = N_CTX + 1
    out[:fut_start] = joined[:fut_start]
    offset = joined[fut_start - 1] - out[fut_start - 1]
    out[fut_start:] = out[fut_start:] + offset
    return out


def _gaussian_smooth(joined, sigma):
    out = np.zeros_like(joined)
    for d in range(2):
        out[:, d] = gaussian_filter1d(joined[:, d], sigma=sigma, mode="nearest")
    fut_start = N_CTX + 1
    out[:fut_start] = joined[:fut_start]
    offset = joined[fut_start - 1] - out[fut_start - 1]
    out[fut_start:] = out[fut_start:] + offset
    return out


def _ema_smooth(joined, alpha):
    """Exponential moving average. alpha in (0, 1]; smaller = smoother.

    Forward-pass EMA: y[i] = alpha*x[i] + (1-alpha)*y[i-1].
    Implemented per-axis. Edge: y[0] = x[0].
    """
    out = np.zeros_like(joined)
    out[0] = joined[0]
    for i in range(1, joined.shape[0]):
        out[i] = alpha * joined[i] + (1.0 - alpha) * out[i - 1]
    fut_start = N_CTX + 1
    out[:fut_start] = joined[:fut_start]
    offset = joined[fut_start - 1] - out[fut_start - 1]
    out[fut_start:] = out[fut_start:] + offset
    return out


def _butterworth_lowpass(joined, cutoff_hz, order=2):
    """Zero-phase Butterworth low-pass filter (forward-backward filtfilt).

    cutoff_hz: cutoff frequency in Hz. Trajectory is 10Hz (dt=0.1), so
    Nyquist = 5 Hz. cutoff=1.0 Hz removes oscillations faster than 1 Hz.
    """
    fs = 1.0 / DT  # 10 Hz
    nyq = 0.5 * fs  # 5 Hz
    wn = cutoff_hz / nyq  # normalized
    wn = min(max(wn, 0.01), 0.99)
    b, a = butter(order, wn, btype="low", analog=False)
    out = np.zeros_like(joined)
    for d in range(2):
        out[:, d] = filtfilt(b, a, joined[:, d])
    fut_start = N_CTX + 1
    out[:fut_start] = joined[:fut_start]
    offset = joined[fut_start - 1] - out[fut_start - 1]
    out[fut_start:] = out[fut_start:] + offset
    return out


def _bspline_smooth(joined, n_coeffs=None, degree=3):
    """B-spline smoothing: fit a uniform B-spline with `n_coeffs` control
    points and sample back. Smaller n_coeffs = smoother.
    """
    if n_coeffs is None:
        # Default: ~half as many control points as samples → mild smoothing
        n_coeffs = max(joined.shape[0] // 2, 8)
    out = np.zeros_like(joined)
    n_pts = joined.shape[0]
    t = np.linspace(0, 1, n_pts)
    for d in range(2):
        s_factor = 0.1 if degree >= 3 else 0.05
        tck = splrep(t, joined[:, d], k=degree, s=n_pts * s_factor)
        out[:, d] = splev(t, tck)
    fut_start = N_CTX + 1
    out[:fut_start] = joined[:fut_start]
    offset = joined[fut_start - 1] - out[fut_start - 1]
    out[fut_start:] = out[fut_start:] + offset
    return out


def _soft_kinematic_project(joined, v_max=15.0, a_max=6.0, jerk_max=8.0, n_iters=3):
    """Lighter kinematic projection: relaxed limits, fewer iterations.
    Velocities are derived from smoothed positions, then clamped, then
    positions are re-integrated from clamped velocities.
    """
    fut_start = N_CTX + 1
    T = joined.shape[0]
    # Derive velocity from positions
    vel = np.zeros_like(joined)
    vel[:-1] = np.diff(joined, axis=0) / DT
    vel[-1] = vel[-2]
    # Preserve history velocity exactly
    vel[:fut_start - 1] = np.diff(joined[:fut_start], axis=0) / DT

    for _ in range(n_iters):
        # Forward pass: clamp acceleration (vel[i] - vel[i-1]) / dt <= a_max
        for i in range(fut_start, T):
            dv = vel[i] - vel[i - 1]
            dv_norm = np.linalg.norm(dv)
            max_dv = a_max * DT
            if dv_norm > max_dv:
                vel[i] = vel[i - 1] + dv * (max_dv / dv_norm)
            # Clamp speed
            speed = np.linalg.norm(vel[i])
            if speed > v_max:
                vel[i] = vel[i] * (v_max / speed)
        # Backward pass: clamp jerk
        for i in range(T - 2, fut_start - 1, -1):
            da = vel[i] - vel[i - 1]
            da_norm = np.linalg.norm(da)
            max_da = jerk_max * DT * DT
            if da_norm > max_da:
                vel[i] = vel[i - 1] + da * (max_da / da_norm)

    # Re-integrate positions from velocity
    pos = np.zeros_like(joined)
    pos[:fut_start] = joined[:fut_start]
    for i in range(fut_start - 1, T - 1):
        pos[i + 1] = pos[i] + vel[i] * DT
    return pos


def apply_schemes(raw_future, history):
    """Apply 12 schemes covering 6 smoothing methods × join_strength ≤ 0.25.

    Methods: SavGol, Gaussian, Moving-avg, EMA, Butterworth, B-spline.
    All smoothing methods are pure low-pass filters (no kinematic clamp).
    Join fix is at most 0.25 strength.
    """
    n_ctx = min(N_CTX, history.shape[0] - 1)
    hist_tail = history[-(n_ctx + 1):].copy()
    joined = np.vstack([hist_tail, raw_future]).copy()
    fut_start = n_ctx + 1

    schemes = {}

    # M0: Raw (no smoothing, no join fix)
    schemes["M0_raw"] = raw_future.copy()

    # M1: SavGol w=7, join 0.25
    s = _savgol_smooth(joined, window=7, poly=2)
    schemes["M1_savgol7_s0.25"] = _enforce_join_continuity(s, join_strength=0.25)[fut_start:].copy()

    # M2: SavGol w=7, no join fix
    s = _savgol_smooth(joined, window=7, poly=2)
    schemes["M2_savgol7_nojoin"] = s[fut_start:].copy()

    # M3: Gaussian σ=1.5, join 0.20
    s = _gaussian_smooth(joined, sigma=1.5)
    schemes["M3_gauss_s1.5_s0.20"] = _enforce_join_continuity(s, join_strength=0.20)[fut_start:].copy()

    # M4: Gaussian σ=1.0, no join fix
    s = _gaussian_smooth(joined, sigma=1.0)
    schemes["M4_gauss_s1.0_nojoin"] = s[fut_start:].copy()

    # M5: Moving-avg w=5, join 0.25
    s = _moving_average(joined, window=5)
    schemes["M5_mavg5_s0.25"] = _enforce_join_continuity(s, join_strength=0.25)[fut_start:].copy()

    # M6: Moving-avg w=7, no join fix
    s = _moving_average(joined, window=7)
    schemes["M6_mavg7_nojoin"] = s[fut_start:].copy()

    # M7: EMA α=0.4, join 0.25
    s = _ema_smooth(joined, alpha=0.4)
    schemes["M7_ema_a0.4_s0.25"] = _enforce_join_continuity(s, join_strength=0.25)[fut_start:].copy()

    # M8: EMA α=0.5, no join fix
    s = _ema_smooth(joined, alpha=0.5)
    schemes["M8_ema_a0.5_nojoin"] = s[fut_start:].copy()

    # M9: Butterworth cutoff=1.5 Hz, join 0.25
    s = _butterworth_lowpass(joined, cutoff_hz=1.5, order=2)
    schemes["M9_butter1.5_s0.25"] = _enforce_join_continuity(s, join_strength=0.25)[fut_start:].copy()

    # M10: Butterworth cutoff=2.0 Hz, no join fix
    s = _butterworth_lowpass(joined, cutoff_hz=2.0, order=2)
    schemes["M10_butter2.0_nojoin"] = s[fut_start:].copy()

    # M11: B-spline (degree=3, s=0.1), join 0.25
    s = _bspline_smooth(joined, degree=3)
    schemes["M11_bspline_s0.25"] = _enforce_join_continuity(s, join_strength=0.25)[fut_start:].copy()

    # M12: B-spline (degree=3), no join fix
    s = _bspline_smooth(joined, degree=3)
    schemes["M12_bspline_nojoin"] = s[fut_start:].copy()

    return schemes


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot_panel(ax, gen_smoothed, gt, prior, history, sampled_goals, title):
    n = len(gen_smoothed)
    colors = get_trajectory_colors(n)

    for i, traj in enumerate(gen_smoothed):
        ax.plot(traj[:, 0], traj[:, 1], "-", color=colors[i],
                alpha=0.55, linewidth=1.0)

    mean = gen_smoothed.mean(axis=0)
    ax.plot(mean[:, 0], mean[:, 1], "b-", linewidth=2.0, label="Mean")

    ax.plot(gt[:, 0], gt[:, 1], "k--", linewidth=1.8, label="GT")

    if prior is not None:
        ax.plot(prior[:, 0], prior[:, 1], color="cyan", linestyle="-",
                linewidth=2.2, alpha=0.85, label="Prior", zorder=5)

    if history is not None and len(history) > 0:
        ax.plot(history[:, 0], history[:, 1], color="orange", linestyle="-",
                linewidth=2.5, alpha=0.85, label="History", zorder=6)
        # Dashed connector from last history point to mean future start —
        # exposes any join discontinuity at t=50
        if gen_smoothed.ndim == 3 and gen_smoothed.shape[0] > 0:
            mean_start = gen_smoothed[:, 0, :].mean(axis=0)
            ax.plot([history[-1, 0], mean_start[0]],
                    [history[-1, 1], mean_start[1]],
                    color="white", linestyle=":", linewidth=1.2, alpha=0.8,
                    zorder=4)

    if sampled_goals is not None:
        ax.scatter(sampled_goals[:, 0], sampled_goals[:, 1], c="red",
                   alpha=0.85, s=60, marker="o", edgecolors="white",
                   linewidths=0.8, label="Sampled goals", zorder=7)

    ax.set_aspect("equal")
    ax.set_title(title, color=TEXT_COLOR, fontsize=10)


def main():
    results_path = "/workspace/output/collision_video_20sc_v2/results.pt"
    output_dir = Path("/workspace/output/smoothing_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Dataset size: {len(dataset)}")

    results = torch.load(results_path, weights_only=False)
    print(f"Loaded {len(results)} results")

    SCHEMES = [
        "M0_raw",
        "M1_savgol7_s0.25", "M2_savgol7_nojoin",
        "M3_gauss_s1.5_s0.20", "M4_gauss_s1.0_nojoin",
        "M5_mavg5_s0.25", "M6_mavg7_nojoin",
        "M7_ema_a0.4_s0.25", "M8_ema_a0.5_nojoin",
        "M9_butter1.5_s0.25", "M10_butter2.0_nojoin",
        "M11_bspline_s0.25", "M12_bspline_nojoin",
    ]

    for r in results:
        idx = r["scenario_idx"]
        gen_smoothed = np.array(r["generated_local"], dtype=np.float32)  # current M6
        gen_raw = np.array(r["generated_local_raw"], dtype=np.float32)
        gt_local = np.array(r["gt_local"], dtype=np.float32)
        goal_local = np.array(r["goal_local"], dtype=np.float32)
        ref_pos = np.array(r["ref_pos"], dtype=np.float32)
        ref_heading = float(r["ref_heading"])
        partner = r.get("partner")

        # Sampled goals / collision points
        sampled_goals = None
        if partner is not None and "sampled_collision_points" in partner:
            cp = []
            for sp in partner["sampled_collision_points"]:
                cp_g = np.asarray(sp["collision_point_global"], dtype=np.float32).reshape(2)
                cp_l, _ = global_to_local(cp_g.reshape(1, 2), np.zeros(1), ref_pos, ref_heading)
                cp.append(cp_l[0].astype(np.float32))
            sampled_goals = np.array(cp, dtype=np.float32) if cp else None
        else:
            sg = r.get("sampled_goals_local")
            if sg is not None:
                sampled_goals = np.array(sg, dtype=np.float32)

        # History
        sample = dataset[idx]
        history_m = to_numpy(denormalize(sample["history"]))

        # Prior (mean across samples)
        prior_mean = None
        if partner is not None and "ego_prior_global" in partner:
            epg = np.array(partner["ego_prior_global"], dtype=np.float32)
            N, T, _ = epg.shape
            flat, _ = global_to_local(epg.reshape(-1, 2), np.zeros(N * T), ref_pos, ref_heading)
            prior_local = flat.reshape(N, T, 2)
            prior_mean = prior_local.mean(axis=0)

        # Apply all schemes to each raw trajectory
        # gen_raw: (N, T, 2). For each traj, apply schemes, collect per-scheme.
        n_trajs = gen_raw.shape[0]
        per_scheme = {s: [] for s in SCHEMES}
        for i in range(n_trajs):
            schemes_i = apply_schemes(gen_raw[i], history_m)
            for s in SCHEMES:
                per_scheme[s].append(schemes_i[s])
        # Stack to (N, T, 2)
        for s in SCHEMES:
            per_scheme[s] = np.array(per_scheme[s], dtype=np.float32)

        # Plot 4×4 grid (13 schemes + 3 empty slots)
        n_schemes = len(SCHEMES)
        n_cols = 4
        n_rows = (n_schemes + n_cols - 1) // n_cols  # 4 rows for 13 schemes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 22))
        for i, s in enumerate(SCHEMES):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]
            apply_dark_theme(ax, fig)
            label = {
                "M0_raw": "M0: Raw (no smoothing)",
                "M1_savgol7_s0.25": "M1: SavGol w=7 + join 0.25",
                "M2_savgol7_nojoin": "M2: SavGol w=7, no join",
                "M3_gauss_s1.5_s0.20": "M3: Gauss σ=1.5 + join 0.20",
                "M4_gauss_s1.0_nojoin": "M4: Gauss σ=1.0, no join",
                "M5_mavg5_s0.25": "M5: Mavg w=5 + join 0.25",
                "M6_mavg7_nojoin": "M6: Mavg w=7, no join",
                "M7_ema_a0.4_s0.25": "M7: EMA α=0.4 + join 0.25",
                "M8_ema_a0.5_nojoin": "M8: EMA α=0.5, no join",
                "M9_butter1.5_s0.25": "M9: Butter 1.5Hz + join 0.25",
                "M10_butter2.0_nojoin": "M10: Butter 2.0Hz, no join",
                "M11_bspline_s0.25": "M11: B-spline + join 0.25",
                "M12_bspline_nojoin": "M12: B-spline, no join",
            }[s]
            _plot_panel(ax, per_scheme[s], gt_local, prior_mean, history_m,
                        sampled_goals, label)
            # Quantify divergence from raw
            diff = np.linalg.norm(per_scheme[s] - gen_raw, axis=-1)
            # Quantify join continuity: accelnorm at t=49 -> t=50
            # v_join = (future[0] - history[-1]) / dt
            # v_hist_last = (history[-1] - history[-2]) / dt
            # join_accel = ||v_join - v_hist_last|| / dt
            fut_arr = per_scheme[s]  # (N, T, 2)
            if fut_arr.ndim == 3 and history_m is not None and history_m.shape[0] >= 2:
                v_join = (fut_arr[:, 0, :] - history_m[-1]) / DT  # (N, 2)
                v_hist_last = (history_m[-1] - history_m[-2]) / DT  # (2,)
                join_accels = np.linalg.norm(v_join - v_hist_last, axis=-1) / DT
                join_max = join_accels.max()
                join_mean = join_accels.mean()
                join_str = f"join: max_a={join_max:.2f}, mean_a={join_mean:.2f} m/s²"
            else:
                join_str = "join: n/a"
            ax.text(0.02, 0.98,
                    f"vs raw: max={diff.max():.2f}m, mean={diff.mean():.2f}m\n{join_str}",
                    transform=ax.transAxes, color=TEXT_COLOR, fontsize=8,
                    verticalalignment="top",
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))

        # Hide unused subplots (3 empty cells in 4×4 grid for 13 schemes)
        for j in range(n_schemes, n_rows * n_cols):
            row, col = divmod(j, n_cols)
            axes[row, col].axis("off")

        # Shared legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            seen = set()
            unique = []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l)
                    unique.append((h, l))
            handles, labels = zip(*unique)
            fig.legend(handles, labels, loc="lower center",
                       bbox_to_anchor=(0.5, -0.01), ncol=min(7, len(handles)),
                       facecolor="white", edgecolor="gray",
                       labelcolor=TEXT_COLOR, fontsize=9)

        fig.suptitle(f"Scenario {idx} — Smoothing Scheme Comparison (N={n_trajs})",
                     color=TEXT_COLOR, fontsize=14, y=0.995)
        plt.tight_layout(rect=(0, 0.04, 1, 0.98))
        fig.subplots_adjust(hspace=0.20, wspace=0.10, bottom=0.06)

        save_figure(fig, output_dir / f"scenario_{idx}_smoothing.png")
        plt.close(fig)
        print(f"  scenario {idx}: saved ({n_trajs} trajs × {n_schemes} schemes)")

    print(f"\nDone. {len(results)} scenarios → {output_dir}")


if __name__ == "__main__":
    main()
