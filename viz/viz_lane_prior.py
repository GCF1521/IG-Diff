"""Visualize lane centerline prior vs Hermite prior for 10 scenarios.

Each plot shows:
  - Lane centerlines (gray dashed)
  - GT trajectory (black solid)
  - Hermite prior (purple dotted)
  - Lane centerline prior (blue dashed) — or "Hermite fallback" if no lane found
  - Start/end heading arrows
  - Lane boundaries
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.av2_dataset import Argoverse2Dataset
from data.normalization import (
    denormalize, compute_hermite_prior, compute_lane_prior,
    unpack_chord, denormalize_residual_chord, chord_frame_to_residual,
    denormalize_residual,
)
from viz.style import apply_dark_theme, save_figure, TEXT_COLOR, COLORS

N_SCENES = 10


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


def plot_scene(data, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    apply_dark_theme(ax, fig)

    # Lane boundaries & centerlines
    for lb_dict in data["lane_boundaries"]:
        for key, style, color, lw in [
            ("left", "-", "#888888", 0.8),
            ("right", "-", "#888888", 0.8),
            ("centerline", "--", "#bbbbbb", 0.5),
        ]:
            pts = lb_dict.get(key)
            if pts is not None and len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], style, color=color,
                        linewidth=lw, alpha=0.6)

    # Drivable areas
    for da_pts in data["drivable_areas"]:
        if len(da_pts) >= 3:
            poly = MplPolygon(da_pts, closed=True, facecolor="#f0f0f0",
                              edgecolor="#cccccc", linewidth=0.3, alpha=0.4)
            ax.add_patch(poly)

    # History
    ax.plot(data["history"][:, 0], data["history"][:, 1], "-",
            color="gray", linewidth=3, alpha=0.8, label="History", zorder=3)

    # GT trajectory
    ax.plot(data["gt"][:, 0], data["gt"][:, 1], "k-", linewidth=3,
            label="GT", alpha=0.9, zorder=4)

    # GT goal marker
    ax.plot(data["goal"][0], data["goal"][1], "r*", markersize=18,
            label="Goal", zorder=5)

    # Hermite prior
    hermite = data["hermite_prior"]
    h_rmse = np.sqrt(np.mean(np.linalg.norm(data["gt"] - hermite, axis=-1) ** 2))
    ax.plot(hermite[:, 0], hermite[:, 1], ":", color=COLORS["prior"],
            linewidth=2.0, alpha=0.8, label=f"Hermite (RMSE={h_rmse:.2f}m)", zorder=3)

    # Lane centerline prior
    lane = data["lane_prior"]
    l_rmse = np.sqrt(np.mean(np.linalg.norm(data["gt"] - lane, axis=-1) ** 2))
    is_fallback = data["lane_is_fallback"]
    lane_label = f"Lane fallback (RMSE={l_rmse:.2f}m)" if is_fallback else f"Lane CL (RMSE={l_rmse:.2f}m)"
    ax.plot(lane[:, 0], lane[:, 1], "--", color="steelblue",
            linewidth=2.0, alpha=0.9, label=lane_label, zorder=3)

    # Start heading arrow
    arrow_len = 5.0
    ax.annotate("", xy=(data["history_end"][0] + arrow_len * np.cos(data["start_heading"]),
                        data["history_end"][1] + arrow_len * np.sin(data["start_heading"])),
                xytext=(data["history_end"][0], data["history_end"][1]),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))

    # End heading arrow
    ax.annotate("", xy=(data["goal"][0] + arrow_len * np.cos(data["end_heading"]),
                        data["goal"][1] + arrow_len * np.sin(data["end_heading"])),
                xytext=(data["goal"][0], data["goal"][1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))

    # Title
    delta = data["end_heading"] - data["start_heading"]
    delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
    chord_len = np.linalg.norm(data["goal"] - data["history_end"])
    status = "FALLBACK" if is_fallback else "LANE"
    ax.set_title(
        f"#{data['idx']}  Δh={np.degrees(delta):.0f}°  chord={chord_len:.1f}m  [{status}]",
        color=TEXT_COLOR, fontsize=13, pad=10)

    ax.legend(facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR,
              fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=3)
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
    output_dir = Path("viz_output/lane_prior")
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
        end_h = sample["end_heading"].item()
        gt_m = get_gt_trajectory(sample)
        scene_data = sample.get("scene_data", {})
        lane_segments = scene_data.get("lane_segments", {})
        ref_pos = sample["ref_pos"].numpy()
        ref_heading = sample["ref_heading"].item()

        # Hermite prior (using GT goal + headings)
        hermite_prior = compute_hermite_prior(history_end, goal_m, start_h, end_h, 60)

        # Lane centerline prior (using GT goal + headings)
        lane_prior = compute_lane_prior(
            history_end, goal_m, lane_segments, ref_pos, ref_heading,
            start_h, end_h, 60,
        )

        # Check if lane prior is just a Hermite fallback
        lane_is_fallback = np.allclose(lane_prior, hermite_prior, atol=1e-3)

        data = {
            "idx": idx,
            "history": history_m,
            "gt": gt_m,
            "history_end": history_end,
            "goal": goal_m,
            "start_heading": start_h,
            "end_heading": end_h,
            "hermite_prior": hermite_prior,
            "lane_prior": lane_prior,
            "lane_is_fallback": lane_is_fallback,
            "lane_boundaries": scene_data.get("lane_boundaries_local", []),
            "drivable_areas": scene_data.get("drivable_areas_local", []),
        }

        save_path = output_dir / f"lane_prior_{rank+1:02d}_idx{idx}.png"
        plot_scene(data, save_path)

        h_rmse = np.sqrt(np.mean(np.linalg.norm(gt_m - hermite_prior, axis=-1) ** 2))
        l_rmse = np.sqrt(np.mean(np.linalg.norm(gt_m - lane_prior, axis=-1) ** 2))
        status = "FALLBACK" if lane_is_fallback else "LANE"
        print(f"  #{rank+1} idx={idx}  Hermite RMSE={h_rmse:.2f}m  Lane RMSE={l_rmse:.2f}m  [{status}]")

    print(f"\nSaved {N_SCENES} visualizations to {output_dir}/")


if __name__ == "__main__":
    main()
