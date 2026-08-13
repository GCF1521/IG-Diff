"""Compare Lane prior vs Ground Truth with real AV2 map overlay.

Generates 10 visualizations: 3 straight, 4 turn, 3 U-turn.
Each shows the lane centerlines, drivable areas, GT trajectory, lane prior,
Hermite prior, and goal — all in local coordinates on the real map.
"""

import sys
import os
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


def classify_scenario(history_end, goal, start_heading, end_heading):
    delta = end_heading - start_heading
    delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
    abs_delta = abs(delta)
    chord_len = np.linalg.norm(goal - history_end)
    if abs_delta < 0.3:
        return "straight", abs_delta, chord_len
    elif abs_delta > 2.5:
        return "uturn", abs_delta, chord_len
    else:
        return "turn", abs_delta, chord_len


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


def find_diverse_scenarios(dataset, max_scan=1500):
    """Find diverse representative scenarios per modality, spaced by chord length."""
    candidates = {"straight": [], "turn": [], "uturn": []}
    needed = {"straight": 3, "turn": 4, "uturn": 3}

    for idx in range(min(len(dataset), max_scan)):
        sample = dataset[idx]
        history_m = denormalize(sample["history"].numpy())
        history_end = history_m[-1]
        goal_m = denormalize(sample["goal"].numpy().reshape(1, 2)).flatten()
        start_h = sample["start_heading"].item()
        end_h = sample["end_heading"].item()

        modal, abs_delta, chord_len = classify_scenario(
            history_end, goal_m, start_h, end_h)

        if len(candidates[modal]) >= needed[modal] * 3:
            continue

        scene_data = sample.get("scene_data", {})
        lane_segments = scene_data.get("lane_segments", {})
        if not lane_segments:
            continue

        gt_m = get_gt_trajectory(sample)
        hermite_m = compute_hermite_prior(history_end, goal_m, start_h, end_h, 60)
        lane_m = compute_lane_prior(
            history_end, goal_m, lane_segments,
            sample["ref_pos"].numpy(), sample["ref_heading"].item(),
            start_h, end_h, 60)

        is_fallback = np.allclose(lane_m, hermite_m, atol=0.01)
        h_rmse = np.sqrt(np.mean(np.linalg.norm(gt_m - hermite_m, axis=-1) ** 2))
        l_rmse = np.sqrt(np.mean(np.linalg.norm(gt_m - lane_m, axis=-1) ** 2))

        candidates[modal].append({
            "idx": idx,
            "gt": gt_m,
            "history": history_m,
            "goal": goal_m,
            "hermite": hermite_m,
            "lane": lane_m,
            "h_rmse": h_rmse,
            "l_rmse": l_rmse,
            "abs_delta": abs_delta,
            "chord_len": chord_len,
            "is_fallback": is_fallback,
            "lane_boundaries": scene_data.get("lane_boundaries_local", []),
            "drivable_areas": scene_data.get("drivable_areas_local", []),
        })

    # Select diverse subset for each modality
    results = {}
    for modal in ["straight", "turn", "uturn"]:
        pool = candidates[modal]
        n = needed[modal]
        n_non_fallback = min(n, 2) if modal != "straight" else 0  # straight: all fallback is fine

        # Split into non-fallback and fallback pools
        non_fb = [d for d in pool if not d["is_fallback"]]
        fb = [d for d in pool if d["is_fallback"]]

        # Sort: non-fallback by best improvement, fallback by h_rmse (show challenging cases)
        non_fb.sort(key=lambda d: d["l_rmse"] - d["h_rmse"])
        fb.sort(key=lambda d: -d["h_rmse"])

        selected = []
        # Pick non-fallback first (for turn/uturn)
        for d in non_fb:
            if len(selected) >= n_non_fallback:
                break
            too_similar = any(
                abs(d["chord_len"] - s["chord_len"]) < 5.0 for s in selected
            )
            if too_similar and len(non_fb) > n_non_fallback * 2:
                continue
            selected.append(d)

        # Fill remaining with fallback cases
        for d in fb:
            if len(selected) >= n:
                break
            too_similar = any(
                abs(d["chord_len"] - s["chord_len"]) < 5.0 for s in selected
            )
            if too_similar and len(fb) > (n - len(selected)) * 2:
                continue
            selected.append(d)

        results[modal] = selected

    return results


def plot_map_comparison(data, modal_name, fig_num, save_path):
    """Plot GT vs Lane prior vs Hermite prior with real map background."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    apply_dark_theme(ax, fig)

    # Draw drivable areas (light fill)
    for da_pts in data["drivable_areas"]:
        if len(da_pts) >= 3:
            poly = MplPolygon(da_pts, closed=True, facecolor="#e8e8e8",
                              edgecolor="#999999", linewidth=0.5, alpha=0.5)
            ax.add_patch(poly)

    # Draw lane boundaries (left/right edges + centerline)
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

    gt = data["gt"]
    history = data["history"]
    goal = data["goal"]
    hermite = data["hermite"]
    lane = data["lane"]

    # History
    ax.plot(history[:, 0], history[:, 1], "-", color="gray",
            linewidth=3, alpha=0.8, label="History", zorder=3)

    # GT trajectory
    ax.plot(gt[:, 0], gt[:, 1], "k-", linewidth=3, label="GT",
            alpha=0.9, zorder=4)

    # Hermite prior
    ax.plot(hermite[:, 0], hermite[:, 1], "--", color=COLORS["prior"],
            linewidth=2, label=f"Hermite ({data['h_rmse']:.2f}m)", zorder=3)

    # Lane prior
    if not data["is_fallback"]:
        ax.plot(lane[:, 0], lane[:, 1], "-", color="#00FFFF",
                linewidth=2, label=f"Lane ({data['l_rmse']:.2f}m)", zorder=3)
    else:
        ax.text(0.5, 0.95, "Lane prior → Hermite fallback",
                transform=ax.transAxes, color="gray", fontsize=10,
                ha="center", va="top")

    # Goal marker
    ax.plot(goal[0], goal[1], "r*", markersize=18, label="Goal", zorder=5)

    # Start marker
    ax.plot(gt[0, 0], gt[0, 1], "o", color="lime", markersize=10,
            label="Start", zorder=5)

    # Endpoint markers for lane and hermite
    ax.plot(hermite[-1, 0], hermite[-1, 1], "D", color="#FFD700",
            markersize=6, zorder=5)
    if not data["is_fallback"]:
        ax.plot(lane[-1, 0], lane[-1, 1], "D", color="#00FFFF",
                markersize=6, zorder=5)

    delta_deg = np.degrees(data["abs_delta"])
    is_fb = " (fallback)" if data["is_fallback"] else ""
    ax.set_title(
        f"#{fig_num} {modal_name.upper()} — "
        f"Δheading={delta_deg:.0f}°  chord={data['chord_len']:.1f}m"
        f"{is_fb}",
        color=TEXT_COLOR, fontsize=13, pad=10)

    ax.legend(facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR,
              fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=4)
    ax.set_xlabel("X (m)", color=TEXT_COLOR)
    ax.set_ylabel("Y (m)", color=TEXT_COLOR)
    ax.set_aspect("equal")

    # Auto-zoom to trajectory area with margin
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
    output_dir = Path("viz_output/prior_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Argoverse2Dataset(
        data_dir=data_dir,
        n_future=60,
        n_history=20,
        n_lanes=24,
        lane_feat_dim=46,
        n_neighbors=6,
        split="eval",
        return_scene_data=True,
        prior_type="hermite",
        residual_frame="chord",
    )

    print(f"Dataset: {len(dataset)} scenarios")
    print("Scanning for diverse scenarios...")

    results = find_diverse_scenarios(dataset, max_scan=1500)

    fig_num = 1
    total_saved = 0
    for modal in ["turn", "uturn", "straight"]:
        scenarios = results[modal]
        print(f"\n{modal.upper()}: {len(scenarios)} scenarios selected")
        for data in scenarios:
            fb_tag = " [fallback]" if data["is_fallback"] else ""
            imp = data["h_rmse"] - data["l_rmse"]
            print(f"  #{fig_num}: idx={data['idx']} "
                  f"Δh={np.degrees(data['abs_delta']):.0f}° "
                  f"chord={data['chord_len']:.1f}m "
                  f"H={data['h_rmse']:.2f}m L={data['l_rmse']:.2f}m "
                  f"Δ={imp:+.2f}m{fb_tag}")

            save_path = output_dir / f"prior_{fig_num:02d}_{modal}.png"
            plot_map_comparison(data, modal, fig_num, save_path)
            fig_num += 1
            total_saved += 1

    print(f"\nSaved {total_saved} visualizations to {output_dir}/")


if __name__ == "__main__":
    main()
