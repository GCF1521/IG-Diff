"""Visualize GT, Hermite prior (no map), and Lane prior (with map) for 10 scenarios.

Shows ground truth trajectory, Hermite cubic prior, and lane centerline prior
with adaptive curvature correction, overlaid on the road map.
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
    denormalize_residual, _push_away_from_boundaries, _BOUNDARY_PUSH_DISTANCE,
)
from viz.style import apply_dark_theme, save_figure, TEXT_COLOR, COLORS


SELECTED_INDICES = [52, 396, 35, 423, 377, 41, 391, 86, 273, 492,
                   7, 103, 258, 444, 511]


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


def plot_comparison(data, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    apply_dark_theme(ax, fig)

    # Map: drivable areas
    for da_pts in data["drivable_areas"]:
        if len(da_pts) >= 3:
            poly = MplPolygon(da_pts, closed=True, facecolor="#e8e8e8",
                              edgecolor="#999999", linewidth=0.5, alpha=0.5)
            ax.add_patch(poly)

    # Map: lane boundaries and centerlines
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

    # Hermite prior (no map)
    h_rmse = np.sqrt(np.mean(np.linalg.norm(data["gt"] - data["hermite"], axis=-1) ** 2))
    h_max = np.max(np.linalg.norm(data["gt"] - data["hermite"], axis=-1))
    h_mean = np.mean(np.linalg.norm(data["gt"] - data["hermite"], axis=-1))
    ax.plot(data["hermite"][:, 0], data["hermite"][:, 1], "-",
            color="orange", linewidth=3.5, alpha=0.9,
            label=f"Hermite (no map) RMSE={h_rmse:.2f}m max={h_max:.2f}m avg={h_mean:.2f}m",
            zorder=5)

    # Lane prior (with map)
    if data["is_fallback"]:
        l_label = "Lane (with map) → fallback to Hermite"
    else:
        l_rmse = np.sqrt(np.mean(np.linalg.norm(data["gt"] - data["lane"], axis=-1) ** 2))
        l_max = np.max(np.linalg.norm(data["gt"] - data["lane"], axis=-1))
        l_mean = np.mean(np.linalg.norm(data["gt"] - data["lane"], axis=-1))
        l_label = f"Lane (with map) RMSE={l_rmse:.2f}m max={l_max:.2f}m avg={l_mean:.2f}m"
    ax.plot(data["lane"][:, 0], data["lane"][:, 1], "-",
            color="dodgerblue", linewidth=2.5, alpha=0.85,
            label=l_label, zorder=3)

    # GT goal marker
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

    delta_h = data["end_heading_gt"] - data["start_heading"]
    delta_h = delta_h - 2 * np.pi * np.round(delta_h / (2 * np.pi))
    chord_len = np.linalg.norm(data["goal_gt"] - data["history_end"])
    tag = "FALLBACK" if data["is_fallback"] else "LANE PRIOR"
    title = (f"#{data['idx']}  Δh={np.degrees(delta_h):.0f}°  "
             f"chord={chord_len:.1f}m  [{tag}]")
    if not data["is_fallback"]:
        title += f"  bound_viol={data['boundary_violation_frac']:.0%}"
        if data["min_boundary_dist"] < float("inf"):
            title += f"  min_d={data['min_boundary_dist']:.1f}m"
    if data.get("is_parking"):
        title += "  [PARKING]"
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, pad=10)

    ax.legend(facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR,
              fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=2)
    ax.set_xlabel("X (m)", color=TEXT_COLOR)
    ax.set_ylabel("Y (m)", color=TEXT_COLOR)
    ax.set_aspect("equal")

    all_pts = np.vstack([data["gt"], data["history"], data["hermite"], data["lane"]])
    center = all_pts.mean(axis=0)
    span = max(all_pts[:, 0].ptp(), all_pts[:, 1].ptp(), 30)
    margin = span * 0.3
    ax.set_xlim(center[0] - span/2 - margin, center[0] + span/2 + margin)
    ax.set_ylim(center[1] - span/2 - margin, center[1] + span/2 + margin)

    plt.tight_layout()
    save_figure(fig, save_path)


def main():
    data_dir = os.environ.get("DATA_DIR", "av2_dataset_1k/train/")
    output_dir = Path("viz_output/lane_vs_hermite")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Argoverse2Dataset(
        data_dir=data_dir,
        n_future=60, n_history=20, n_lanes=24, lane_feat_dim=46,
        n_neighbors=6, split="eval",
        return_scene_data=True, prior_type="hermite", residual_frame="chord",
    )

    for rank, idx in enumerate(SELECTED_INDICES):
        sample = dataset[idx]
        hist_pos = sample.get("history_pos", sample["history"][..., :2])
        history_m = denormalize(hist_pos.numpy())
        history_end = history_m[-1]
        goal_m = denormalize(sample["goal"].numpy().reshape(1, 2)).flatten()
        start_h = sample["start_heading"].item()
        end_h = sample["end_heading"].item()
        gt_m = get_gt_trajectory(sample)
        scene_data = sample.get("scene_data", {})

        hermite_m = compute_hermite_prior(history_end, goal_m, start_h, end_h, 60)
        lane_segments = scene_data.get("lane_segments", {})
        lane_boundaries_local = scene_data.get("lane_boundaries_local", [])
        drivable_areas_local = scene_data.get("drivable_areas_local", [])
        lane_m = compute_lane_prior(
            history_end, goal_m, lane_segments,
            sample["ref_pos"].numpy(), sample["ref_heading"].item(),
            start_h, end_h, 60,
            lane_boundaries_local=lane_boundaries_local,
            drivable_areas_local=drivable_areas_local,
        )
        is_fallback = np.allclose(lane_m, hermite_m, atol=0.01)

        # Compute boundary violation stats
        _, boundary_violation_frac = _push_away_from_boundaries(
            lane_m, lane_boundaries_local, _BOUNDARY_PUSH_DISTANCE)
        min_boundary_dist = float("inf")
        if lane_boundaries_local:
            from data.normalization import _point_to_segment_info
            for lb in lane_boundaries_local:
                for key in ("left", "right"):
                    pts = lb.get(key)
                    if pts is not None and len(pts) >= 2:
                        pts = np.asarray(pts, dtype=np.float32)
                        for j in range(len(pts) - 1):
                            for pi in range(0, len(lane_m), 5):
                                d, _, _ = _point_to_segment_info(lane_m[pi], pts[j], pts[j+1])
                                min_boundary_dist = min(min_boundary_dist, d)

        is_parking = sample.get("is_parking", None)
        if is_parking is not None:
            is_parking = is_parking.item() > 0.5

        data = {
            "idx": idx,
            "history": history_m,
            "gt": gt_m,
            "hermite": hermite_m,
            "lane": lane_m,
            "history_end": history_end,
            "goal_gt": goal_m,
            "start_heading": start_h,
            "end_heading_gt": end_h,
            "is_fallback": is_fallback,
            "lane_boundaries": scene_data.get("lane_boundaries_local", []),
            "drivable_areas": scene_data.get("drivable_areas_local", []),
            "boundary_violation_frac": boundary_violation_frac,
            "min_boundary_dist": min_boundary_dist,
            "is_parking": is_parking,
        }

        save_path = output_dir / f"compare_{rank+1:02d}_idx{idx}.png"
        plot_comparison(data, save_path)
        print(f"  #{rank+1} idx={idx} fallback={is_fallback} saved")

    print(f"\nSaved {len(SELECTED_INDICES)} visualizations to {output_dir}/")


if __name__ == "__main__":
    main()
