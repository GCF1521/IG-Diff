"""BEV scene visualization with map lanes, neighbors, and generated trajectories."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from viz.style import apply_dark_theme, save_figure, get_trajectory_colors, TEXT_COLOR


def plot_bev_scene(
    lane_boundaries: list,   # list of dicts: {left: (M,2), right: (M,2), centerline: (N,2), is_intersection: bool}
    drivable_areas: list,    # list of (P,2) polygons
    focal_traj: np.ndarray,  # (T, 2) local coords
    neighbor_trajs: list,    # list of (H, 2) arrays, local coords
    neighbor_positions: list, # list of (x,y,heading) current positions
    generated_trajs: list = None,  # list of (T, 2) arrays
    gt_traj: np.ndarray = None,
    goal: np.ndarray = None,
    title: str = "BEV Scene",
    save_path: str = None,
    extent: list = None,
    ped_crossings: list = None,  # list of dicts: {edge1: (2,2), edge2: (2,2)}
):
    """Render full BEV scene with lanes, agents, and trajectories."""
    fig, ax = plt.subplots(figsize=(12, 10))
    apply_dark_theme(ax, fig)

    # Drivable areas
    for da in drivable_areas:
        polygon = plt.Polygon(da, facecolor="#e8e8e8", edgecolor="#999999", alpha=0.5, linewidth=0.5)
        ax.add_patch(polygon)

    # Lane boundaries
    for lane in lane_boundaries:
        for side, color, lw in [("left", "black", 1.0), ("right", "black", 1.0)]:
            pts = lane[side]
            ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=lw, alpha=0.5)

        # Centerline (dashed)
        if "centerline" in lane:
            cl = lane["centerline"]
            ax.plot(cl[:, 0], cl[:, 1], "--", color="gray", linewidth=0.5, alpha=0.5)

        # Intersection marker
        if lane.get("is_intersection"):
            cl = lane["centerline"]
            ax.plot(cl[:, 0], cl[:, 1], "-", color="orange", linewidth=1.5, alpha=0.4)

    # Pedestrian crossings
    if ped_crossings:
        for pc in ped_crossings:
            e1 = pc["edge1"]
            e2 = pc["edge2"]
            ax.plot(e1[:, 0], e1[:, 1], "-", color="purple", linewidth=1.5, alpha=0.7)
            ax.plot(e2[:, 0], e2[:, 1], "-", color="purple", linewidth=1.5, alpha=0.7)
            poly_pts = np.array([e1[0], e1[1], e2[1], e2[0]])
            polygon = plt.Polygon(poly_pts, facecolor="purple", edgecolor="purple", alpha=0.15)
            ax.add_patch(polygon)

    # Neighbor agents
    for traj, pos in zip(neighbor_trajs, neighbor_positions):
        x, y, h = pos
        rect = patches.Rectangle(
            (x - 2.0, y - 1.0), 4.0, 2.0,
            angle=np.degrees(h), facecolor="orange", alpha=0.6,
            rotation_point="center",
        )
        ax.add_patch(rect)
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], "-", color="orange", alpha=0.4, linewidth=1)

    # Focal vehicle
    focal_pos = focal_traj[-1] if len(focal_traj) > 0 else np.array([0, 0])
    rect = patches.Rectangle(
        (focal_pos[0] - 2.0, focal_pos[1] - 1.0), 4.0, 2.0,
        angle=0, facecolor="cyan", edgecolor="blue", linewidth=2,
        rotation_point="center",
    )
    ax.add_patch(rect)

    # Generated trajectories — each with distinct color
    if generated_trajs:
        N = len(generated_trajs)
        traj_colors = get_trajectory_colors(N)
        for i, traj in enumerate(generated_trajs):
            ax.plot(traj[:, 0], traj[:, 1], "-", color=traj_colors[i],
                    alpha=0.5, linewidth=1.0)
        mean = np.mean(generated_trajs, axis=0)
        ax.plot(mean[:, 0], mean[:, 1], "b-", linewidth=2.5, label="Mean")

    # GT
    if gt_traj is not None:
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], "k--", linewidth=2, label="GT")

    # Goal
    if goal is not None:
        ax.plot(goal[0], goal[1], "r*", markersize=15, label="Goal")

    ax.set_aspect("equal")
    if extent:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        ax.set_xlim(-50, 50)
        ax.set_ylim(-10, 80)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=6,
              facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_title(title, color=TEXT_COLOR)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.1)

    if save_path:
        save_figure(fig, save_path)
    return fig