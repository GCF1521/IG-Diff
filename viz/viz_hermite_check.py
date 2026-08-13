"""Visualize Hermite prior on 10 random AV2 scenarios.

Shows: map background (lane boundaries + drivable areas), GT trajectory,
Hermite prior, goal marker, start/end heading arrows.
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


def plot_hermite_check(data, idx, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    apply_dark_theme(ax, fig)

    # Drivable areas
    for da_pts in data["drivable_areas"]:
        if len(da_pts) >= 3:
            poly = MplPolygon(da_pts, closed=True, facecolor="#e8e8e8",
                              edgecolor="#999999", linewidth=0.5, alpha=0.5)
            ax.add_patch(poly)

    # Lane boundaries
    for lb_dict in data["lane_boundaries"]:
        is_int = lb_dict.get("is_intersection", False)
        for key, style, color, lw in [
            ("left", "-", "black", 1.0),
            ("right", "-", "black", 1.0),
            ("centerline", "--", "gray", 0.5),
        ]:
            pts = lb_dict.get(key)
            if pts is not None and len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], style, color=color,
                        linewidth=lw, alpha=0.5)

    history = data["history"]
    gt = data["gt"]
    hermite = data["hermite"]
    goal = data["goal"]
    start_h = data["start_heading"]
    end_h = data["end_heading"]

    # History
    ax.plot(history[:, 0], history[:, 1], "-", color="gray",
            linewidth=3, alpha=0.8, label="History", zorder=3)

    # GT trajectory
    ax.plot(gt[:, 0], gt[:, 1], "k-", linewidth=3, label="GT",
            alpha=0.9, zorder=4)

    # Hermite prior
    h_rmse = np.sqrt(np.mean(np.linalg.norm(gt - hermite, axis=-1) ** 2))
    ax.plot(hermite[:, 0], hermite[:, 1], "--", color=COLORS["prior"],
            linewidth=2, label=f"Hermite ({h_rmse:.2f}m)", zorder=3)

    # Goal
    ax.plot(goal[0], goal[1], "r*", markersize=18, label="Goal", zorder=5)

    # Start point
    ax.plot(gt[0, 0], gt[0, 1], "o", color="lime", markersize=10,
            label="Start", zorder=5)

    # Heading arrows
    arrow_len = 5.0
    # Start heading (from Hermite start = history_end)
    ax.annotate("", xy=(hermite[0, 0] + arrow_len * np.cos(start_h),
                        hermite[0, 1] + arrow_len * np.sin(start_h)),
                xytext=(hermite[0, 0], hermite[0, 1]),
                arrowprops=dict(arrowstyle="->", color="lime", lw=2))
    # End heading (from goal)
    ax.annotate("", xy=(goal[0] + arrow_len * np.cos(end_h),
                        goal[1] + arrow_len * np.sin(end_h)),
                xytext=(goal[0], goal[1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))

    delta = end_h - start_h
    delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
    chord_len = np.linalg.norm(goal - hermite[0])
    ax.set_title(
        f"#{idx}  Δheading={np.degrees(delta):.0f}°  chord={chord_len:.1f}m  "
        f"RMSE={h_rmse:.2f}m",
        color=TEXT_COLOR, fontsize=13, pad=10)

    ax.legend(facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR,
              fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=4)
    ax.set_xlabel("X (m)", color=TEXT_COLOR)
    ax.set_ylabel("Y (m)", color=TEXT_COLOR)
    ax.set_aspect("equal")

    # Auto-zoom
    all_pts = np.vstack([gt, history])
    center = all_pts.mean(axis=0)
    span = max(all_pts[:, 0].ptp(), all_pts[:, 1].ptp(), 30)
    margin = span * 0.3
    ax.set_xlim(center[0] - span / 2 - margin, center[0] + span / 2 + margin)
    ax.set_ylim(center[1] - span / 2 - margin, center[1] + span / 2 + margin)

    plt.tight_layout()
    save_figure(fig, save_path)


def main():
    data_dir = os.environ.get("DATA_DIR", "av2_dataset_1k/train/")
    output_dir = Path("viz_output/hermite_check")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Argoverse2Dataset(
        data_dir=data_dir,
        n_future=60, n_history=20, n_lanes=24, lane_feat_dim=46,
        n_neighbors=6, split="eval",
        return_scene_data=True, prior_type="hermite", residual_frame="chord",
    )

    # Random 10 scenarios
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=10, replace=False)

    for rank, idx in enumerate(indices):
        idx = int(idx)
        sample = dataset[idx]
        history_m = denormalize(sample["history"].numpy())
        history_end = history_m[-1]
        goal_m = denormalize(sample["goal"].numpy().reshape(1, 2)).flatten()
        start_h = sample["start_heading"].item()
        end_h = sample["end_heading"].item()

        gt_m = get_gt_trajectory(sample)
        hermite_m = compute_hermite_prior(history_end, goal_m, start_h, end_h, 60)

        scene_data = sample.get("scene_data", {})
        data = {
            "history": history_m,
            "gt": gt_m,
            "hermite": hermite_m,
            "goal": goal_m,
            "start_heading": start_h,
            "end_heading": end_h,
            "lane_boundaries": scene_data.get("lane_boundaries_local", []),
            "drivable_areas": scene_data.get("drivable_areas_local", []),
        }

        save_path = output_dir / f"hermite_{rank+1:02d}_idx{idx}.png"
        plot_hermite_check(data, idx, save_path)
        print(f"  #{rank+1} idx={idx} saved")

    print(f"\nSaved 10 visualizations to {output_dir}/")


if __name__ == "__main__":
    main()
