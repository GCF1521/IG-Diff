"""BEV scene animation: 11-second scenario playback as GIF or MP4.

Ego-centric view: the viewport follows the focal vehicle, so map elements
and neighbors appear to move relative to the ego as it drives.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from pathlib import Path

from viz.style import BG_COLOR, COLORS, TEXT_COLOR, get_trajectory_colors


def _rotate_points(pts, angle):
    """Rotate an (N, 2) array by angle (radians) around the origin."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def animate_bev_scene(
    history_m: np.ndarray,       # (H, 2) history trajectory in local meters
    gt_future_m: np.ndarray,     # (T, 2) GT future trajectory
    gen_trajs_m: list,           # list of (T, 2) generated trajectories
    lane_boundaries: list,       # list of dicts: {left, right, centerline, is_intersection}
    drivable_areas: list,        # list of (P, 2) polygons
    neighbor_trajs: list = None, # list of (H, 2) arrays — DEPRECATED, use neighbor_full_trajs
    neighbor_positions: list = None,  # list of (x, y, heading) at t=50
    neighbor_full_trajs: list = None,  # list of (H+T, 2) full trajectories (history + future)
    goal_m: np.ndarray = None,   # (2,) goal position
    sampled_goals: np.ndarray = None,  # (N, 2) sampled goals
    ped_crossings: list = None,
    fps: int = 10,
    save_path: str = None,
    view_radius: float = 40.0,
    title: str = "BEV Animation",
):
    """Animate the 11-second scenario (2s history + 6s future + buffer)."""
    n_history = len(history_m)
    n_future = len(gt_future_m)
    n_total = n_history + n_future

    gt_full = np.concatenate([history_m, gt_future_m], axis=0)

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Pre-compute per-trajectory colors
    N_gen = len(gen_trajs_m) if gen_trajs_m else 0
    traj_colors = get_trajectory_colors(N_gen) if N_gen > 0 else []

    def _get_heading(frame_idx):
        if frame_idx <= 0:
            frame_idx = 0
        if frame_idx >= n_total - 1:
            frame_idx = n_total - 2
        return np.arctan2(
            gt_full[frame_idx + 1, 1] - gt_full[frame_idx, 1],
            gt_full[frame_idx + 1, 0] - gt_full[frame_idx, 0],
        )

    def _transform_to_ego(points, ego_pos, ego_heading):
        shifted = points - ego_pos
        rotated = _rotate_points(shifted, -ego_heading)
        return rotated

    def draw_frame(frame_idx):
        ax.clear()
        ax.set_facecolor(BG_COLOR)

        ax.set_xlim(-view_radius * 0.6, view_radius * 0.6)
        ax.set_ylim(-view_radius, view_radius)
        ax.set_aspect("equal")
        ax.tick_params(colors=TEXT_COLOR)

        ego_pos = gt_full[min(frame_idx, n_total - 1)]
        ego_heading = _get_heading(min(frame_idx, n_total - 1))

        if frame_idx < n_history:
            phase = f"HISTORY (t={frame_idx - n_history:+.1f}s)"
        else:
            t_future = frame_idx - n_history
            phase = f"FUTURE (t={t_future * 0.1:.1f}s)"

        ax.set_title(f"{title} — {phase}", color=TEXT_COLOR, fontsize=12)

        # Drivable areas
        for da in drivable_areas:
            da_ego = _transform_to_ego(da, ego_pos, ego_heading)
            polygon = plt.Polygon(da_ego, facecolor="#e8e8e8", edgecolor="#999999",
                                  alpha=0.5, linewidth=0.5)
            ax.add_patch(polygon)

        # Lane boundaries
        for lane in lane_boundaries:
            for side, col, lw in [("left", "black", 1.0), ("right", "black", 1.0)]:
                pts = lane[side]
                if len(pts) > 0:
                    pts_ego = _transform_to_ego(pts, ego_pos, ego_heading)
                    ax.plot(pts_ego[:, 0], pts_ego[:, 1], "-", color=col, linewidth=lw, alpha=0.5)
            if "centerline" in lane:
                cl = lane["centerline"]
                if len(cl) > 0:
                    cl_ego = _transform_to_ego(cl, ego_pos, ego_heading)
                    ax.plot(cl_ego[:, 0], cl_ego[:, 1], "--", color="gray", linewidth=0.5, alpha=0.5)

        # Pedestrian crossings
        if ped_crossings:
            for pc in ped_crossings:
                e1, e2 = pc["edge1"], pc["edge2"]
                e1_ego = _transform_to_ego(e1, ego_pos, ego_heading)
                e2_ego = _transform_to_ego(e2, ego_pos, ego_heading)
                ax.plot(e1_ego[:, 0], e1_ego[:, 1], "-", color="purple", linewidth=1.5, alpha=0.7)
                ax.plot(e2_ego[:, 0], e2_ego[:, 1], "-", color="purple", linewidth=1.5, alpha=0.7)
                poly_pts = np.array([e1_ego[0], e1_ego[1], e2_ego[1], e2_ego[0]])
                polygon = plt.Polygon(poly_pts, facecolor="purple", alpha=0.15)
                ax.add_patch(polygon)

        # GT trajectory trail
        trail_end = min(frame_idx + 1, n_total)
        trail = gt_full[:trail_end]
        if len(trail) > 1:
            trail_ego = _transform_to_ego(trail, ego_pos, ego_heading)
            ax.plot(trail_ego[:, 0], trail_ego[:, 1], "--", color=COLORS["gt"],
                    linewidth=1.5, alpha=0.6, label="GT")

        # Focal vehicle
        rect = patches.Rectangle(
            (-2.0, -1.0), 4.0, 2.0,
            angle=0, facecolor=COLORS["focal"],
            edgecolor="blue", linewidth=2, rotation_point="center",
        )
        ax.add_patch(rect)

        # Generated trajectories (each with distinct color)
        if frame_idx >= n_history and gen_trajs_m:
            t_future = frame_idx - n_history
            for gi, gen_traj in enumerate(gen_trajs_m):
                if t_future < len(gen_traj):
                    gen_trail = gen_traj[:t_future + 1]
                    gen_trail_ego = _transform_to_ego(gen_trail, ego_pos, ego_heading)
                    ax.plot(gen_trail_ego[:, 0], gen_trail_ego[:, 1], "-",
                            color=traj_colors[gi], alpha=0.5, linewidth=1.0)
                    remaining = gen_traj[t_future:]
                    remaining_ego = _transform_to_ego(remaining, ego_pos, ego_heading)
                    ax.plot(remaining_ego[:, 0], remaining_ego[:, 1], "-",
                            color=traj_colors[gi], alpha=0.1, linewidth=0.5)

            # Mean generated
            gen_arr = np.array(gen_trajs_m)
            mean_gen = gen_arr.mean(axis=0)
            if t_future < len(mean_gen):
                mean_trail = mean_gen[:t_future + 1]
                mean_ego = _transform_to_ego(mean_trail, ego_pos, ego_heading)
                ax.plot(mean_ego[:, 0], mean_ego[:, 1], "-",
                        color="steelblue", linewidth=2.5, label="Generated")

        # Goal
        if goal_m is not None:
            goal_ego = _transform_to_ego(goal_m.reshape(1, 2), ego_pos, ego_heading)[0]
            ax.plot(goal_ego[0], goal_ego[1], "*", color=COLORS["goal"],
                    markersize=15, label="Goal")

        # Sampled goals
        if sampled_goals is not None:
            sg_ego = _transform_to_ego(sampled_goals, ego_pos, ego_heading)
            ax.scatter(sg_ego[:, 0], sg_ego[:, 1],
                       c="red", alpha=0.3, s=15)

        # Neighbor vehicles
        if neighbor_full_trajs is not None:
            for full_traj in neighbor_full_trajs:
                fi = min(frame_idx, len(full_traj) - 1)
                npos = full_traj[fi].reshape(1, 2)
                npos_ego = _transform_to_ego(npos, ego_pos, ego_heading)[0]

                fi_prev = max(fi - 1, 0)
                fi_next = min(fi + 1, len(full_traj) - 1)
                if fi_next > fi_prev:
                    n_heading = np.arctan2(
                        full_traj[fi_next, 1] - full_traj[fi_prev, 1],
                        full_traj[fi_next, 0] - full_traj[fi_prev, 0],
                    )
                else:
                    n_heading = 0.0
                h_rel = n_heading - ego_heading

                rect = patches.Rectangle(
                    (npos_ego[0] - 2.0, npos_ego[1] - 1.0), 4.0, 2.0,
                    angle=np.degrees(h_rel), facecolor=COLORS["neighbor"], alpha=0.6,
                    rotation_point="center",
                )
                ax.add_patch(rect)

                ntrail = full_traj[:fi + 1]
                if len(ntrail) > 1:
                    ntrail_ego = _transform_to_ego(ntrail, ego_pos, ego_heading)
                    ax.plot(ntrail_ego[:, 0], ntrail_ego[:, 1], "-", color=COLORS["neighbor"],
                            alpha=0.3, linewidth=1)
        elif neighbor_trajs is not None and neighbor_positions is not None:
            for traj, pos_info in zip(neighbor_trajs, neighbor_positions):
                x, y, h = pos_info
                npos = np.array([[x, y]])
                npos_ego = _transform_to_ego(npos, ego_pos, ego_heading)[0]
                h_rel = h - ego_heading

                rect = patches.Rectangle(
                    (npos_ego[0] - 2.0, npos_ego[1] - 1.0), 4.0, 2.0,
                    angle=np.degrees(h_rel), facecolor=COLORS["neighbor"], alpha=0.6,
                    rotation_point="center",
                )
                ax.add_patch(rect)
                if len(traj) > 1:
                    traj_ego = _transform_to_ego(traj, ego_pos, ego_heading)
                    ax.plot(traj_ego[:, 0], traj_ego[:, 1], "-", color=COLORS["neighbor"],
                            alpha=0.3, linewidth=1)

        ax.legend(loc="upper right", facecolor="white", edgecolor="gray",
                  labelcolor=TEXT_COLOR, fontsize=8)

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=n_total,
        interval=1000 // fps, blit=False,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix == ".gif":
            anim.save(str(save_path), writer="pillow", fps=fps)
        elif save_path.suffix == ".mp4":
            # Use mpeg4 codec — this ffmpeg build's h264 (libopenh264) is
            # broken ("Incorrect library version loaded"). mpeg4 is widely
            # supported and produces compatible .mp4 files.
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, codec="mpeg4",
                                  extra_args=["-pix_fmt", "yuv420p"])
            anim.save(str(save_path), writer=writer)
        else:
            save_path = save_path.with_suffix(".gif")
            anim.save(str(save_path), writer="pillow", fps=fps)
        plt.close(fig)
        print(f"  Saved animation: {save_path}")

    return fig, anim