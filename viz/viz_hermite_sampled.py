"""Visualize Hermite prior with inference-like goal sampling.

For each scenario, sample 5 perturbed goals (anisotropic near GT),
then generate Hermite priors for each. Shows how priors vary
with sampled endpoints, simulating inference conditions.

Goal perturbation: anisotropic Gaussian in endpoint heading frame
  - sigma_lon = 5.0m (along heading)
  - sigma_lat = 2.0m (perpendicular to heading)
End heading: GT + small noise (N(0, 0.05 rad ~3°))
Start heading: unchanged (from history, known at inference time)
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.av2_dataset import Argoverse2Dataset
from data.normalization import (
    denormalize, compute_hermite_prior,
    unpack_chord, denormalize_residual_chord, chord_frame_to_residual,
    denormalize_residual,
)
from viz.style import apply_dark_theme, save_figure, TEXT_COLOR, COLORS, get_trajectory_colors

# Inference parameters — sigma scales with chord length
# sigma_lon = chord_len * 0.10 (clamped >= 0.5m)
# sigma_lat = chord_len * 0.04 (clamped >= 0.3m)
# At chord=50m: sigma_lon=5m, sigma_lat=2m (matches original fixed defaults)
SIGMA_LON_RATIO = 0.10
SIGMA_LAT_RATIO = 0.04
SIGMA_LON_MIN = 0.5   # meters
SIGMA_LAT_MIN = 0.3   # meters
HEADING_SIGMA = 0.05    # radians (~3°), noise on end_heading
N_GOALS_PER_SCENE = 5
N_SCENES = 20


def get_gt_trajectory(sample):
    use_chord = sample["use_chord_frame"].item()
    chord_dir = sample["chord_dir"].numpy()
    if use_chord == 1.0:
        r_lon_n, r_lat_n = unpack_chord(sample["trajectory"].numpy())
        r_lon, r_lat = denormalize_residual_chord(r_lon_n, r_lat_n)
        residual_m = chord_frame_to_residual(r_lon, r_lat, chord_dir)
    else:
        residual_m = denormalize_residual(sample["trajectory"].numpy())
    prior_m = denormalize(sample["prior"].numpy())
    return prior_m + residual_m


def sample_goal(goal_m, end_heading, history_end, rng):
    """Anisotropic goal sampling near GT, sigma scaled by chord length."""
    chord_len = max(np.linalg.norm(goal_m - history_end), 1e-6)
    sigma_lon = max(chord_len * SIGMA_LON_RATIO, SIGMA_LON_MIN)
    sigma_lat = max(chord_len * SIGMA_LAT_RATIO, SIGMA_LAT_MIN)
    dx_lon = rng.standard_normal() * sigma_lon
    dx_lat = rng.standard_normal() * sigma_lat
    cos_h = np.cos(end_heading)
    sin_h = np.sin(end_heading)
    perturbed = goal_m.copy()
    perturbed[0] += dx_lon * cos_h - dx_lat * sin_h
    perturbed[1] += dx_lon * sin_h + dx_lat * cos_h
    return perturbed


def sample_end_heading(gt_heading, rng):
    """Perturb GT end heading with small noise."""
    return gt_heading + rng.standard_normal() * HEADING_SIGMA


def plot_sampled_priors(data, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    apply_dark_theme(ax, fig)

    # Map background
    for da_pts in data["drivable_areas"]:
        if len(da_pts) >= 3:
            poly = MplPolygon(da_pts, closed=True, facecolor="#e8e8e8",
                              edgecolor="#999999", linewidth=0.5, alpha=0.5)
            ax.add_patch(poly)
    for lb_dict in data["lane_boundaries"]:
        for key, style, color, lw in [
            ("left", "-", "black", 1.0),
            ("right", "-", "black", 1.0),
            ("centerline", "--", "gray", 0.5),
        ]:
            pts = lb_dict.get(key)
            if pts is not None and len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], style, color=color,
                        linewidth=lw, alpha=0.5)

    # History
    ax.plot(data["history"][:, 0], data["history"][:, 1], "-",
            color="gray", linewidth=3, alpha=0.8, label="History", zorder=3)

    # GT trajectory
    ax.plot(data["gt"][:, 0], data["gt"][:, 1], "k-", linewidth=3,
            label="GT", alpha=0.9, zorder=4)

    # GT goal + heading arrow
    ax.plot(data["goal_gt"][0], data["goal_gt"][1], "r*", markersize=18,
            label="GT Goal", zorder=5)
    arrow_len = 5.0
    ax.annotate("", xy=(data["goal_gt"][0] + arrow_len * np.cos(data["end_heading_gt"]),
                        data["goal_gt"][1] + arrow_len * np.sin(data["end_heading_gt"])),
                xytext=(data["goal_gt"][0], data["goal_gt"][1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))

    # Start heading arrow
    ax.annotate("", xy=(data["history_end"][0] + arrow_len * np.cos(data["start_heading"]),
                        data["history_end"][1] + arrow_len * np.sin(data["start_heading"])),
                xytext=(data["history_end"][0], data["history_end"][1]),
                arrowprops=dict(arrowstyle="->", color="lime", lw=2))

    # Sampled priors + sampled goals
    traj_colors = get_trajectory_colors(N_GOALS_PER_SCENE)
    for i, (prior, goal, h_end) in enumerate(zip(
            data["sampled_priors"], data["sampled_goals"], data["sampled_end_headings"])):
        c = traj_colors[i % len(traj_colors)]
        ax.plot(prior[:, 0], prior[:, 1], "--", color=c, linewidth=1.8,
                alpha=0.8, zorder=3, label=f"Prior #{i+1}")
        ax.plot(goal[0], goal[1], "D", color=c, markersize=7, zorder=5)
        # Sampled end heading arrow
        ax.annotate("", xy=(goal[0] + 4.0 * np.cos(h_end),
                            goal[1] + 4.0 * np.sin(h_end)),
                    xytext=(goal[0], goal[1]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5, alpha=0.7))

    # Hermite with GT goal (reference)
    hermite_gt = compute_hermite_prior(
        data["history_end"], data["goal_gt"],
        data["start_heading"], data["end_heading_gt"], 60)
    h_rmse = np.sqrt(np.mean(np.linalg.norm(data["gt"] - hermite_gt, axis=-1) ** 2))
    ax.plot(hermite_gt[:, 0], hermite_gt[:, 1], ":", color=COLORS["prior"],
            linewidth=1.5, alpha=0.5, label=f"Hermite(GT) {h_rmse:.2f}m", zorder=3)

    delta = data["end_heading_gt"] - data["start_heading"]
    delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
    chord_len = np.linalg.norm(data["goal_gt"] - data["history_end"])
    s_lon = max(chord_len * SIGMA_LON_RATIO, SIGMA_LON_MIN)
    s_lat = max(chord_len * SIGMA_LAT_RATIO, SIGMA_LAT_MIN)
    ax.set_title(
        f"#{data['idx']}  Δh={np.degrees(delta):.0f}°  "
        f"chord={chord_len:.1f}m  σ_lon={s_lon:.1f}m σ_lat={s_lat:.1f}m",
        color=TEXT_COLOR, fontsize=13, pad=10)

    ax.legend(facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR,
              fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=4)
    ax.set_xlabel("X (m)", color=TEXT_COLOR)
    ax.set_ylabel("Y (m)", color=TEXT_COLOR)
    ax.set_aspect("equal")

    all_pts = np.vstack([data["gt"], data["history"]])
    center = all_pts.mean(axis=0)
    span = max(all_pts[:, 0].ptp(), all_pts[:, 1].ptp(), 30)
    margin = span * 0.3
    ax.set_xlim(center[0] - span/2 - margin, center[0] + span/2 + margin)
    ax.set_ylim(center[1] - span/2 - margin, center[1] + span/2 + margin)

    plt.tight_layout()
    save_figure(fig, save_path)


def main():
    data_dir = os.environ.get("DATA_DIR", "av2_dataset_1k/train/")
    output_dir = Path("viz_output/hermite_sampled")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Argoverse2Dataset(
        data_dir=data_dir,
        n_future=60, n_history=20, n_lanes=24, lane_feat_dim=46,
        n_neighbors=6, split="eval",
        return_scene_data=True, prior_type="hermite", residual_frame="chord",
    )

    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=N_SCENES, replace=False)

    for rank, idx in enumerate(indices):
        idx = int(idx)
        sample = dataset[idx]
        history_m = denormalize(sample["history"].numpy())
        history_end = history_m[-1]
        goal_m = denormalize(sample["goal"].numpy().reshape(1, 2)).flatten()
        start_h = sample["start_heading"].item()
        end_h_gt = sample["end_heading"].item()
        gt_m = get_gt_trajectory(sample)
        scene_data = sample.get("scene_data", {})

        # Sample 5 goals + end_headings
        sampled_goals = []
        sampled_end_headings = []
        sampled_priors = []
        for _ in range(N_GOALS_PER_SCENE):
            sg = sample_goal(goal_m, end_h_gt, history_end, rng)
            sh = sample_end_heading(end_h_gt, rng)
            sp = compute_hermite_prior(history_end, sg, start_h, sh, 60)
            sampled_goals.append(sg)
            sampled_end_headings.append(sh)
            sampled_priors.append(sp)

        data = {
            "idx": idx,
            "history": history_m,
            "gt": gt_m,
            "history_end": history_end,
            "goal_gt": goal_m,
            "start_heading": start_h,
            "end_heading_gt": end_h_gt,
            "sampled_goals": sampled_goals,
            "sampled_end_headings": sampled_end_headings,
            "sampled_priors": sampled_priors,
            "lane_boundaries": scene_data.get("lane_boundaries_local", []),
            "drivable_areas": scene_data.get("drivable_areas_local", []),
        }

        save_path = output_dir / f"sampled_{rank+1:02d}_idx{idx}.png"
        plot_sampled_priors(data, save_path)
        print(f"  #{rank+1} idx={idx} saved")

    print(f"\nSaved {N_SCENES} visualizations to {output_dir}/")


if __name__ == "__main__":
    main()
