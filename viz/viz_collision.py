"""Collision scene visualization in AV2 global coordinates.

Draws the full AV2 scene (drivable areas, lane boundaries, pedestrian crossings,
surrounding agents) in global coordinates — matching av2-api's
scenario_visualization style — then overlays ego + partner generated trajectories.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator

from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat

from viz.style import save_figure, TEXT_COLOR

# ---------- Colors (matching av2-api scenario_visualization) ----------
_DRIVABLE_AREA_COLOR = "#7A7A7A"
_LANE_SEGMENT_COLOR = "#E0E0E0"
_PED_CROSSING_COLOR = "#E0E0E0"

_DEFAULT_ACTOR_COLOR = "#D3E8EF"
_FOCAL_AGENT_COLOR = "#ECA25B"
_AV_COLOR = "#007672"

# Generated trajectory overlays
_EGO_GEN_COLOR = "#1E90FF"
_PARTNER_GEN_COLOR = "#FF4500"
_EGO_GT_COLOR = "#0B3D91"
_PARTNER_GT_COLOR = "#8B0000"
_COLLISION_POINT_COLOR = "#FF0000"

# Geometry
_VEHICLE_LENGTH = 4.0
_VEHICLE_WIDTH = 2.0
_CYCLIST_LENGTH = 2.0
_CYCLIST_WIDTH = 0.7
_PLOT_BOUNDS_BUFFER = 30.0
_BBOX_ZORDER = 100

_STATIC_OBJECT_TYPES = {ObjectType.STATIC, ObjectType.BACKGROUND,
                        ObjectType.CONSTRUCTION, ObjectType.RIDERLESS_BICYCLE}


def _plot_polylines(ax, polylines, *, style="-", line_width=1.0, alpha=1.0, color="r"):
    for polyline in polylines:
        ax.plot(polyline[:, 0], polyline[:, 1], style,
                linewidth=line_width, color=color, alpha=alpha)


def _plot_polygons(ax, polygons, *, alpha=1.0, color="r"):
    for polygon in polygons:
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)


def _plot_actor_bounding_box(ax, cur_location, heading, color,
                              bbox_size=(_VEHICLE_LENGTH, _VEHICLE_WIDTH),
                              edgecolor="black", linewidth=1.0, alpha=0.9):
    bbox_length, bbox_width = bbox_size
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)
    bbox = Rectangle(
        (pivot_x, pivot_y), bbox_length, bbox_width,
        angle=np.degrees(heading), facecolor=color,
        edgecolor=edgecolor, linewidth=linewidth, alpha=alpha,
        zorder=_BBOX_ZORDER,
    )
    ax.add_patch(bbox)


def _draw_static_map(ax, static_map, show_ped_xings=False):
    """Draw drivable areas, lane segments, and ped crossings — exact AV2 style."""
    if static_map is None:
        return
    # Drivable areas
    for da in static_map.vector_drivable_areas.values():
        try:
            _plot_polygons(ax, [da.xyz[:, :2]], alpha=0.5, color=_DRIVABLE_AREA_COLOR)
        except Exception:
            pass

    # Lane segments
    for ls in static_map.vector_lane_segments.values():
        try:
            _plot_polylines(ax,
                [ls.left_lane_boundary.xyz[:, :2], ls.right_lane_boundary.xyz[:, :2]],
                line_width=0.5, color=_LANE_SEGMENT_COLOR)
        except Exception:
            pass

    # Ped crossings
    if show_ped_xings:
        for pc in static_map.vector_pedestrian_crossings.values():
            try:
                _plot_polylines(ax,
                    [pc.edge1.xyz[:, :2], pc.edge2.xyz[:, :2]],
                    alpha=1.0, color=_PED_CROSSING_COLOR)
            except Exception:
                pass


def _draw_actors(ax, scenario, timestep, focal_track_id, partner_track_id):
    """Draw all actor tracks up to *timestep*, matching AV2 style.

    Ego (focal) drawn in orange, partner in red, AV in teal, others in blue-gray.
    Returns the focal agent bounding box extent for auto-zoom.
    """
    track_bounds = None
    for track in scenario.tracks:
        # Gather states up to timestep
        actor_timesteps = np.array(
            [s.timestep for s in track.object_states if s.timestep <= timestep])
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        traj = np.array([list(s.position) for s in track.object_states if s.timestep <= timestep])
        headings = np.array([s.heading for s in track.object_states if s.timestep <= timestep])

        is_focal = (track.track_id == focal_track_id)
        is_partner = (track.track_id == partner_track_id)

        # Choose color
        if is_focal:
            track_color = _FOCAL_AGENT_COLOR
        elif is_partner:
            track_color = _PARTNER_GEN_COLOR
        elif track.track_id == "AV":
            track_color = _AV_COLOR
        elif track.object_type in _STATIC_OBJECT_TYPES:
            continue
        else:
            track_color = _DEFAULT_ACTOR_COLOR

        # Draw trajectory history
        if is_focal:
            _plot_polylines(ax, [traj], color=track_color, line_width=2.5)
        elif is_partner:
            _plot_polylines(ax, [traj], color=track_color, line_width=2.5)
        elif track.track_id == "AV":
            _plot_polylines(ax, [traj], color=track_color, line_width=1.5, alpha=0.7)
        else:
            _plot_polylines(ax, [traj], color=track_color, line_width=0.8, alpha=0.5)

        # Draw bounding box at current position
        cur_pos = traj[-1]
        cur_heading = headings[-1]
        if track.object_type == ObjectType.VEHICLE or track.object_type == ObjectType.BUS:
            _plot_actor_bounding_box(ax, cur_pos, cur_heading, track_color)
        elif track.object_type in (ObjectType.CYCLIST, ObjectType.MOTORCYCLIST):
            _plot_actor_bounding_box(ax, cur_pos, cur_heading, track_color,
                                      bbox_size=(_CYCLIST_LENGTH, _CYCLIST_WIDTH))
        else:
            ax.plot(cur_pos[0], cur_pos[1], "o", color=track_color, markersize=4)

        # Track bounds from focal
        if is_focal:
            x_min, x_max = traj[:, 0].min(), traj[:, 0].max()
            y_min, y_max = traj[:, 1].min(), traj[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)

    return track_bounds


def plot_collision_scene(
    scenario,
    static_map,
    ego_trajs_global: np.ndarray,
    partner_trajs_global: np.ndarray,
    focal_track_id: str,
    partner_track_id: str,
    ego_gt_global: np.ndarray = None,
    partner_gt_global: np.ndarray = None,
    collision_point_global: np.ndarray = None,
    collision_rate: float = None,
    collision_mean_min_dist: float = None,
    collision_rate_strict: float = None,
    collision_threshold: float = 1.5,
    title: str = "Isomorphic Collision Guidance",
    save_path: str = None,
    timestep: int = 50,
    n_ego_show: int = 10,
    n_partner_show: int = 10,
    n_top_show: int = 5,
    collision_region: dict = None,
    ego_prior_global: np.ndarray = None,
    partner_prior_global: np.ndarray = None,
    show_fans: bool = True,
):
    """Render full AV2 scene with collision trajectories overlaid.

    Layout follows the AV2 official scenario_visualization:
    - grey drivable areas, light lane boundaries
    - coloured actor bounding boxes with trajectory tails
    - then overlay generated ego (blue) and partner (red) futures
    """
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 6)
    ax = fig.add_subplot(gs[0, :5])
    ax_info = fig.add_subplot(gs[0, 5])

    ax.set_facecolor("white")
    ax_info.set_facecolor("white")
    ax_info.axis("off")

    # 1) Static map
    _draw_static_map(ax, static_map)

    # 1.5) Collision region overlay: full V-type fans (ego + partner),
    #      their intersection ∩ drivable area (green), the geometric
    #      centroid of the intersection (red star), and the 2m endpoint-
    #      sampling disk (gray) centered on the centroid. Mirrors the
    #      reference viz in output/extrap_viz/scenario_*_fan.png.
    if collision_region is not None:
        try:
            mask = collision_region["mask"]
            gx = collision_region["gx"]
            gy = collision_region["gy"]
            ego_poly = collision_region.get("ego_poly")
            partner_poly = collision_region.get("partner_poly")
            centroid_global = collision_region.get("centroid_global")

            # Full ego fan (blue, semi-transparent) — skip when show_fans=False
            if show_fans and ego_poly is not None and len(ego_poly) >= 3:
                ax.fill(ego_poly[:, 0], ego_poly[:, 1],
                        color=_EGO_GEN_COLOR, alpha=0.15, zorder=3)
                ax.plot(ego_poly[:, 0], ego_poly[:, 1],
                        color=_EGO_GEN_COLOR, linewidth=1.0, alpha=0.6, zorder=3)
            # Full partner fan (red, semi-transparent) — skip when show_fans=False
            if show_fans and partner_poly is not None and len(partner_poly) >= 3:
                ax.fill(partner_poly[:, 0], partner_poly[:, 1],
                        color=_PARTNER_GEN_COLOR, alpha=0.15, zorder=3)
                ax.plot(partner_poly[:, 0], partner_poly[:, 1],
                        color=_PARTNER_GEN_COLOR, linewidth=1.0, alpha=0.6, zorder=3)

            # Intersection region (green)
            if mask.any():
                extent = (gx[0], gx[-1], gy[0], gy[-1])
                masked = np.where(mask, 1.0, np.nan)
                ax.imshow(masked, extent=extent, origin="lower",
                          cmap=plt.cm.RdYlGn, alpha=0.55, vmin=0, vmax=1,
                          aspect="auto", zorder=4)

            # Geometric centroid (red star)
            if centroid_global is not None:
                ax.plot(centroid_global[0], centroid_global[1], "*",
                        color="red", markersize=20,
                        markeredgecolor="black", markeredgewidth=1.5,
                        zorder=109, label="Centroid")
                # 2m endpoint-sampling disk (gray)
                sampling_circle = plt.Circle(
                    centroid_global, 2.0,
                    fill=True, facecolor="gray", alpha=0.35,
                    edgecolor="dimgray", linewidth=1.2, linestyle="--",
                    zorder=108)
                ax.add_patch(sampling_circle)
        except Exception:
            pass

    # 2) All actors up to timestep
    track_bounds = _draw_actors(ax, scenario, timestep, focal_track_id, partner_track_id)

    # 2.5) Pick top-K driving pairs (K = n_top_show, default 5).
    #      Only these top-K pairs are drawn — no thin "rest" lines.
    #      top1 = boldest, top2..K = medium.
    drive_ego_idx_top = []   # list of ego indices, ranked (best first)
    drive_partner_idx_top = []
    drive_pairs_full = []    # full tuples (ei, pi, end_dist, ct, min_dist, eligible)
    try:
        from viz.viz_collision_animation import _select_top_k_collision_pairs, _extract_other_vehicle_trajs
        drivable_polys = []
        if static_map is not None:
            for da in static_map.vector_drivable_areas.values():
                try:
                    drivable_polys.append(da.xyz[:, :2].astype(np.float64))
                except Exception:
                    pass
        other_trajs = _extract_other_vehicle_trajs(
            scenario, focal_track_id, partner_track_id, t_start=50, t_end=110)
        drive_pairs_full = _select_top_k_collision_pairs(
            ego_trajs_global, partner_trajs_global, drivable_polys,
            other_trajs, vehicle_collision_threshold=collision_threshold,
            k=n_top_show)
        for pair in drive_pairs_full:
            drive_ego_idx_top.append(pair[0])
            drive_partner_idx_top.append(pair[1])
    except Exception:
        drive_pairs_full = []

    # Backwards-compat singletons (top1) for downstream code
    drive_ego_idx = drive_ego_idx_top[0] if drive_ego_idx_top else 0
    drive_partner_idx = drive_partner_idx_top[0] if drive_partner_idx_top else 0
    top_pairs = drive_pairs_full

    # 3) Generated ego trajectories — only top-K (5) drawn.
    #    top1 boldest, top2..K medium. No thin "rest" lines.
    top_ego_set = set(drive_ego_idx_top)
    top1_ego = drive_ego_idx_top[0] if len(drive_ego_idx_top) >= 1 else None
    top25_ego = drive_ego_idx_top[1:5]
    # top2..top5 medium first (lower zorder)
    for i in top25_ego:
        if i < len(ego_trajs_global):
            traj = ego_trajs_global[i]
            ax.plot(traj[:, 0], traj[:, 1], "-", color=_EGO_GEN_COLOR,
                    alpha=0.75, linewidth=1.6, zorder=51)
    # top1 boldest
    if top1_ego is not None and top1_ego < len(ego_trajs_global):
        traj = ego_trajs_global[top1_ego]
        ax.plot(traj[:, 0], traj[:, 1], "-", color=_EGO_GEN_COLOR,
                alpha=0.95, linewidth=2.8, zorder=52)

    # 4) Generated partner trajectories — same tiered rendering (top5 only)
    top_partner_set = set(drive_partner_idx_top)
    top1_partner = drive_partner_idx_top[0] if len(drive_partner_idx_top) >= 1 else None
    top25_partner = drive_partner_idx_top[1:5]
    for j in top25_partner:
        if j < len(partner_trajs_global):
            traj = partner_trajs_global[j]
            ax.plot(traj[:, 0], traj[:, 1], "-", color=_PARTNER_GEN_COLOR,
                    alpha=0.75, linewidth=1.6, zorder=51)
    if top1_partner is not None and top1_partner < len(partner_trajs_global):
        traj = partner_trajs_global[top1_partner]
        ax.plot(traj[:, 0], traj[:, 1], "-", color=_PARTNER_GEN_COLOR,
                alpha=0.95, linewidth=2.8, zorder=52)

    # 5) GT
    if ego_gt_global is not None:
        ax.plot(ego_gt_global[:, 0], ego_gt_global[:, 1], "--", color=_EGO_GT_COLOR,
                linewidth=2.5, label="Ego GT", zorder=55)
    if partner_gt_global is not None:
        ax.plot(partner_gt_global[:, 0], partner_gt_global[:, 1], "--", color=_PARTNER_GT_COLOR,
                linewidth=2.5, label="Partner GT", zorder=55)

    # 5.5) Prior trajectories (Hermite cubic from start→goal) — thick black
    #      lines for both vehicles. These show the model's prior shape
    #      before diffusion refinement.
    if ego_prior_global is not None and len(ego_prior_global) > 1:
        ax.plot(ego_prior_global[:, 0], ego_prior_global[:, 1], "-",
                color="black", linewidth=3.0, alpha=0.85, zorder=56,
                label="Ego prior")
    if partner_prior_global is not None and len(partner_prior_global) > 1:
        ax.plot(partner_prior_global[:, 0], partner_prior_global[:, 1], "-",
                color="black", linewidth=3.0, alpha=0.85, zorder=56,
                label="Partner prior")

    # 6) Collision point (top-1 generated collision point — midpoint of
    #    the closest-frame pair between the top-1 ego and top-1 partner
    #    trajectories). Falls back to the caller-supplied collision_point_global
    #    if top-1 selection cannot be run (e.g., no drivable polys available).
    gen_collision_point = None
    if len(top_pairs) > 0:
        ei, pi, _, ct, _, _ = top_pairs[0]
        gen_collision_point = (
            ego_trajs_global[ei, ct] + partner_trajs_global[pi, ct]) / 2.0
    if gen_collision_point is None:
        gen_collision_point = collision_point_global
    if gen_collision_point is not None:
        ax.plot(gen_collision_point[0], gen_collision_point[1], "D",
                color=_COLLISION_POINT_COLOR, markersize=14,
                markeredgecolor="darkred", markeredgewidth=1.5,
                zorder=110, label="Generated Collision Point")
        circle = plt.Circle(gen_collision_point, collision_threshold,
                            fill=False, color=_COLLISION_POINT_COLOR,
                            linestyle="--", linewidth=1.5, alpha=0.5)
        ax.add_patch(circle)
        zone = plt.Circle(gen_collision_point, collision_threshold,
                          fill=True, facecolor=_COLLISION_POINT_COLOR, alpha=0.06)
        ax.add_patch(zone)

    # 7) Endpoint markers — top5 only. top1 largest, top2..5 medium.
    for i in top25_ego:
        if i < len(ego_trajs_global):
            traj = ego_trajs_global[i]
            ax.plot(traj[-1, 0], traj[-1, 1], "o", color=_EGO_GEN_COLOR,
                    markersize=6, alpha=0.85, zorder=61)
    if top1_ego is not None and top1_ego < len(ego_trajs_global):
        traj = ego_trajs_global[top1_ego]
        ax.plot(traj[-1, 0], traj[-1, 1], "o", color=_EGO_GEN_COLOR,
                markersize=9, alpha=0.95, zorder=62)
    for j in top25_partner:
        if j < len(partner_trajs_global):
            traj = partner_trajs_global[j]
            ax.plot(traj[-1, 0], traj[-1, 1], "o", color=_PARTNER_GEN_COLOR,
                    markersize=6, alpha=0.85, zorder=61)
    if top1_partner is not None and top1_partner < len(partner_trajs_global):
        traj = partner_trajs_global[top1_partner]
        ax.plot(traj[-1, 0], traj[-1, 1], "o", color=_PARTNER_GEN_COLOR,
                markersize=9, alpha=0.95, zorder=62)

    # 8) Auto extent — center on collision point (or focal trajectory)
    if gen_collision_point is not None:
        cx, cy = gen_collision_point
    elif track_bounds is not None:
        cx = (track_bounds[0] + track_bounds[1]) / 2
        cy = (track_bounds[2] + track_bounds[3]) / 2
    else:
        cx, cy = 0.0, 0.0

    # Gather all relevant points for extent
    all_pts = []
    if len(ego_trajs_global) > 0:
        all_pts.append(ego_trajs_global.reshape(-1, 2))
    if len(partner_trajs_global) > 0:
        all_pts.append(partner_trajs_global.reshape(-1, 2))
    if ego_gt_global is not None:
        all_pts.append(ego_gt_global.reshape(-1, 2))
    if partner_gt_global is not None:
        all_pts.append(partner_gt_global.reshape(-1, 2))
    if all_pts:
        pts = np.concatenate(all_pts, axis=0)
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        half = max(x_max - x_min, y_max - y_min) / 2 + _PLOT_BOUNDS_BUFFER
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
    else:
        half = 60.0

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Legend
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=_FOCAL_AGENT_COLOR, lw=2, label="Ego history"),
        mlines.Line2D([], [], color=_EGO_GEN_COLOR, lw=2, label="Ego gen"),
        mlines.Line2D([], [], color=_EGO_GT_COLOR, lw=2, ls="--", label="Ego GT"),
        mlines.Line2D([], [], color=_PARTNER_GEN_COLOR, lw=2, label="Partner history"),
        mlines.Line2D([], [], color=_PARTNER_GEN_COLOR, lw=2, label="Partner gen"),
        mlines.Line2D([], [], color=_PARTNER_GT_COLOR, lw=2, ls="--", label="Partner GT"),
        mlines.Line2D([], [], color="black", lw=3, label="Prior (ego+partner)"),
    ]
    if collision_region is not None:
        handles.append(mlines.Line2D([], [], color=_EGO_GEN_COLOR, alpha=0.4,
                                      marker="s", ls="None", markersize=10,
                                      label="Ego fan (V)"))
        handles.append(mlines.Line2D([], [], color=_PARTNER_GEN_COLOR, alpha=0.4,
                                      marker="s", ls="None", markersize=10,
                                      label="Partner fan (V)"))
        handles.append(mlines.Line2D([], [], color="green", alpha=0.5,
                                      marker="s", ls="None", markersize=10,
                                      label="Intersection"))
        handles.append(mlines.Line2D([], [], color="red", marker="*",
                                      ls="None", markersize=12,
                                      label="Centroid"))
        handles.append(mlines.Line2D([], [], color="gray", alpha=0.5,
                                      marker="o", ls="None", markersize=10,
                                      label="2m sampling disk"))
    if gen_collision_point is not None:
        handles.append(mlines.Line2D([], [], color=_COLLISION_POINT_COLOR, marker="D",
                                      ls="None", markersize=10, label="Generated Collision Pt"))
    ax.legend(handles=handles, loc="upper left", fontsize=7, facecolor="white",
              edgecolor="gray", ncol=2)

    # Title
    ttl = title
    if collision_rate is not None:
        ttl += f"  |  Collision Rate: {collision_rate:.1%}"
    ax.set_title(ttl, color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=10)

    # Info panel
    info = ["=== COLLISION INFO ===", ""]
    if collision_rate is not None:
        info += [f"Collision Rate", f"  (d<{collision_threshold}m): {collision_rate:.1%}"]
    if collision_rate_strict is not None:
        info += [f"  Strict(d<1.0m): {collision_rate_strict:.1%}"]
    if collision_mean_min_dist is not None:
        info += [f"Mean Min Dist:", f"  {collision_mean_min_dist:.2f}m"]
    info += [
        "", "=== MECHANISM ===", "",
        "Ego and Partner share",
        "an isomorphic model",
        "and converge to a",
        "common collision",
        "point (red diamond).",
        "Top-1 pair shown",
        "bold; pairs #2..K",
        "thinner.", "",
        "Isomorphic guidance:",
        "same architecture,",
        "different content,",
        "shared collision goal.",
        "",
        f"Ego samples:  {len(ego_trajs_global)}",
        f"Partner samples: {len(partner_trajs_global)}",
    ]
    if partner_track_id is not None:
        info += ["", f"Partner ID:", f"  {partner_track_id}"]

    ax_info.text(0.02, 0.98, "\n".join(info), transform=ax_info.transAxes,
                 fontsize=7.5, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                           edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_collision_heatmap(
    ego_trajs_global: np.ndarray,
    partner_trajs_global: np.ndarray,
    threshold: float = 1.5,
    title: str = "Collision Distance Heatmap",
    save_path: str = None,
):
    """Plot (N_ego, N_partner) min-distance heatmap."""
    dists = np.linalg.norm(
        ego_trajs_global[:, None, :, :] - partner_trajs_global[None, :, :, :], axis=-1
    )
    min_dists = dists.min(axis=-1)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    im = ax.imshow(min_dists, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Min Distance (m)")
    ax.contour(min_dists, levels=[threshold], colors=["red"], linewidths=2)
    frac = (min_dists < threshold).mean()
    ax.set_title(f"{title}  |  Collision Rate: {frac:.1%}", color=TEXT_COLOR)
    ax.set_xlabel("Partner Sample #")
    ax.set_ylabel("Ego Sample #")
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig
