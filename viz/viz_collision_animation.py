"""Animate ego + partner collision scenario as MP4 — AV2 scenario_visualization style.

Strictly follows av2-api's `visualize_scenario` (src/av2/datasets/motion_forecasting/
viz/scenario_visualization.py):

- BEV plot, focal agent (ego) trajectory bounds drive the view + 30m buffer
- History phase (t=0..49): each track's observed states plotted up to t
- Future phase (t=50..109): focal & partner replaced by the selected driving pair
  (best collision pair from 400 cross-pairs); OTHER tracks use their observed GT
  future states (AV2 provides full 110-ts observations for all tracks)
- Top-K collision pairs (ranked by min_dist ascending) overlaid: pair #1 is the
  driving pair (bold), pairs #2..K shown thinner
- cv2.VideoWriter + mp4v codec, 10 fps — identical pipeline to av2-api
- in-memory PNG buffer per frame, then encode
"""

import io
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from PIL import Image as img

from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
)
from av2.map.map_api import ArgoverseStaticMap

_OBS_DURATION_TIMESTEPS = 50
_PRED_DURATION_TIMESTEPS = 60
_DT_SEC = 0.1

_ESTIMATED_VEHICLE_LENGTH_M = 4.0
_ESTIMATED_VEHICLE_WIDTH_M = 2.0
_ESTIMATED_CYCLIST_LENGTH_M = 2.0
_ESTIMATED_CYCLIST_WIDTH_M = 0.7
_PLOT_BOUNDS_BUFFER_M = 30.0

# av2-api colors
_DRIVABLE_AREA_COLOR = "#7A7A7A"
_LANE_SEGMENT_COLOR = "#E0E0E0"
_DEFAULT_ACTOR_COLOR = "#D3E8EF"
_FOCAL_AGENT_COLOR = "#ECA25B"
_AV_COLOR = "#007672"
_BOUNDING_BOX_ZORDER = 100

# Generated trajectory overlay colors
_EGO_GEN_COLOR = "#1E90FF"
_PARTNER_GEN_COLOR = "#FF4500"
_COLLISION_POINT_COLOR = "#FF0000"

_STATIC_OBJECT_TYPES = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}


def _heading_from_traj(traj: np.ndarray, t: int) -> float:
    """Estimate heading at frame t from positions t and t+1 (radians)."""
    if t < 0:
        t = 0
    if t >= len(traj) - 1:
        t = len(traj) - 2
    return float(np.arctan2(traj[t + 1, 1] - traj[t, 1],
                            traj[t + 1, 0] - traj[t, 0]))


def _point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (kept for reference / single-point use).

    Vectorized workloads should use `_points_in_polys` which delegates to
    matplotlib's Path.contains_points — this scalar version is ~100x slower.
    """
    x, y = pt[0], pt[1]
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _points_in_polys(pts: np.ndarray, polys: List[np.ndarray]) -> np.ndarray:
    """Vectorized point-in-polygon (union) test.

    Args:
        pts: (M, 2) points.
        polys: list of (N_k, 2) polygons. A point is "inside" if it's inside
            at least one polygon.

    Returns:
        (M,) bool array — True if point i is inside the union of polys.
    """
    if len(polys) == 0:
        return np.ones(len(pts), dtype=bool)
    from matplotlib.path import Path
    out = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        p = np.asarray(poly)
        if len(p) < 3:
            continue
        # Close the polygon if it isn't already
        if not np.allclose(p[0], p[-1]):
            p = np.vstack([p, p[0]])
        path = Path(p, closed=True)
        out |= path.contains_points(pts)
    return out


def _traj_in_drivable_area(traj: np.ndarray,
                            drivable_polys: List[np.ndarray],
                            n_subsamples: int = 10) -> bool:
    """Check if every point of traj lies inside at least one drivable-area polygon.

    Subsamples between consecutive frames so that trajectories which jump
    across a drivable-area boundary between samples (but happen to land inside
    on both endpoints) are still caught. With n_subsamples=10, each pair of
    adjacent frames is split into 10 intermediate points that are all tested.

    Args:
        traj: (T, 2) trajectory.
        drivable_polys: list of (N, 2) polygons (global coords).
        n_subsamples: number of intermediate points to test between each pair
            of adjacent frames (0 = only the trajectory points themselves).

    Returns:
        True if all T points (and all subsamples) are inside the union of drivable areas.
    """
    mask = _traj_in_drivable_area_mask(traj, drivable_polys, n_subsamples)
    return bool(mask.all()) if mask is not None else True


def _traj_in_drivable_area_mask(traj: np.ndarray,
                                drivable_polys: List[np.ndarray],
                                n_subsamples: int = 10) -> Optional[np.ndarray]:
    """Per-point inside-mask for _traj_in_drivable_area.

    Returns:
        (M,) bool array — True for each test point (original frames +
        subsamples) inside the union of drivable polys. None if no polys
        are provided (treated as "all inside").
    """
    if len(drivable_polys) == 0:
        return None  # no map info available — don't penalize

    traj = np.asarray(traj, dtype=np.float64)
    # Build the full test-point set: original frames + subsamples between them
    if n_subsamples > 0 and len(traj) >= 2:
        # alphas: (n_subsamples, 1, 1) for broadcasting over (T-1, 2)
        alphas = (np.arange(1, n_subsamples + 1) / (n_subsamples + 1)).reshape(-1, 1, 1)
        # (n_subsamples, T-1, 2)
        subs = (1 - alphas) * traj[:-1][None, :, :] + alphas * traj[1:][None, :, :]
        subs = subs.reshape(-1, 2)
        all_pts = np.vstack([traj, subs])
    else:
        all_pts = traj

    return _points_in_polys(all_pts, drivable_polys)


def _compute_smoothness(traj: np.ndarray) -> float:
    """Trajectory smoothness — mean squared jerk (3rd derivative), lower = smoother.

    Args:
        traj: (T, 2) trajectory.

    Returns:
        mean(||a_{t+1} - a_t||^2) where a is acceleration (2nd difference).
    """
    if traj.shape[0] < 4:
        return 0.0
    # velocity (1st diff), acceleration (2nd diff), jerk (3rd diff)
    vel = np.diff(traj, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    return float((jerk ** 2).sum(axis=-1).mean())


def _compute_curvature(traj: np.ndarray) -> float:
    """Trajectory curvature — mean squared curvature, lower = straighter/smoother.

    Uses the discrete curvature formula k = |x'y'' - y'x''| / (x'^2 + y'^2)^1.5.
    """
    if traj.shape[0] < 3:
        return 0.0
    dx = np.gradient(traj[:, 0])
    dy = np.gradient(traj[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    denom = (dx ** 2 + dy ** 2) ** 1.5 + 1e-9
    k = num / denom
    return float((k ** 2).mean())


def _extract_other_vehicle_trajs(scenario: ArgoverseScenario,
                                  focal_track_id: str,
                                  partner_track_id: str,
                                  t_start: int = 50,
                                  t_end: int = 110,
                                  ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract (positions, headings) for all in-scene vehicles other than focal/partner.

    AV2 motion forecasting tracks carry observed states across all 110 timesteps
    they're present. We pull t ∈ [t_start, t_end) — the future phase — for every
    VEHICLE/BUS/MOTORCYCLIST track that is neither the focal (ego) nor the
    selected partner. These are the "surrounding vehicles" the partner must not
    collide with.

    Returns:
        list of (positions (T,2), headings (T,)) per other vehicle. T may differ
        per vehicle (some leave the scene early); each entry only covers the
        timesteps the vehicle is actually present in [t_start, t_end).
    """
    others = []
    valid_types = {ObjectType.VEHICLE, ObjectType.BUS, ObjectType.MOTORCYCLIST}
    for track in scenario.tracks:
        if track.track_id in (focal_track_id, partner_track_id):
            continue
        if track.object_type not in valid_types:
            continue
        states = [s for s in track.object_states if t_start <= s.timestep < t_end]
        if len(states) < 2:
            continue
        positions = np.array([list(s.position) for s in states], dtype=np.float64)
        headings = np.array([s.heading for s in states], dtype=np.float64)
        others.append((positions, headings))
    return others


def _traj_collides_with_vehicles(traj: np.ndarray,
                                  other_vehicle_trajs: List[Tuple[np.ndarray, np.ndarray]],
                                  threshold: float = 1.5,
                                  ) -> bool:
    """Check if traj comes within `threshold` meters of any other vehicle at
    any matching timestep.

    Both traj and each other-vehicle traj are indexed by future-frame index
    (0..59). We pair them frame-by-frame while both are present. A single
    frame where the inter-vehicle distance < threshold counts as a collision.

    The threshold is the same 1.5m used for ego-partner collision detection —
    conservative (vehicle half-diagonal ≈ 2.24m, so 1.5m flags any significant
    box overlap without being so loose that legitimate passing triggers it).
    """
    if not other_vehicle_trajs:
        return False
    T = traj.shape[0]
    for other_pos, _ in other_vehicle_trajs:
        n = min(T, other_pos.shape[0])
        if n < 1:
            continue
        diff = traj[:n] - other_pos[:n]
        d = np.sqrt((diff ** 2).sum(axis=-1))
        if (d < threshold).any():
            return True
    return False


def _select_top_k_collision_pairs(
    ego_trajs: np.ndarray,
    partner_trajs: np.ndarray,
    drivable_polys: List[np.ndarray],
    other_vehicle_trajs: List[Tuple[np.ndarray, np.ndarray]],
    vehicle_collision_threshold: float = 1.5,
    k: int = 5,
    w_end: float = 0.50,
    w_goal: float = 0.20,
    w_smooth: float = 0.15,
    w_curv: float = 0.15,
) -> List[Tuple[int, int, float, int, float, bool]]:
    """Select top-k (ego_idx, partner_idx) pairs by composite score.

    Scoring (lower = better):
    1. HARD CONSTRAINTS (disqualify — sent to bottom):
       a. EITHER ego or partner trajectory leaves the drivable area at ANY frame.
       b. EITHER ego or partner collides with any OTHER in-scene vehicle
          (any non-focal, non-partner vehicle) at ANY future frame. "Collides"
          = frame-distance < vehicle_collision_threshold.
       A pair is eligible only if both (a) and (b) pass.
    2. Among eligible pairs, composite score = weighted sum of:
       - end_dist:         ||ego_end - partner_end|| (terminal convergence;
                            weight 0.50, primary — picks pairs whose endpoints
                            actually meet at the shared goal/collision point)
       - goal_completion:  ||ego_end - mid|| + ||partner_end - mid|| where
                            mid = endpoint midpoint of the pair
                                                                  (weight 0.20)
       - smoothness:       mean jerk^2 of ego+partner     (weight 0.15)
       - curvature:        mean curvature^2 of ego+partner (weight 0.15)

    Returns:
        list of (ego_idx, partner_idx, end_dist, min_dist_t, min_dist, eligible)
        sorted by composite score ascending (best first).
        Disqualified pairs are appended at the end (sorted by min_dist).
    """
    n_ego = ego_trajs.shape[0]
    n_partner = partner_trajs.shape[0]
    raw = []  # (ei, pi, end_dist, t_min, md, eligible, goal_comp, smooth, curv, da_violations)
    for i in range(n_ego):
        # Precompute per-trajectory drivable-area membership for ego[i]
        # across all frames (with subsamples) so we can also report a
        # "violation count" for fallback ranking.
        ego_in_mask = _traj_in_drivable_area_mask(ego_trajs[i], drivable_polys)
        ego_in_da = bool(ego_in_mask.all())
        ego_da_violations = int((~ego_in_mask).sum())
        ego_hits_other = _traj_collides_with_vehicles(
            ego_trajs[i], other_vehicle_trajs, vehicle_collision_threshold)
        for j in range(n_partner):
            diff = ego_trajs[i] - partner_trajs[j]
            d = np.sqrt((diff ** 2).sum(axis=-1))
            end_dist = float(d[-1])
            t_min = int(np.argmin(d))
            md = float(d[t_min])

            partner_in_mask = _traj_in_drivable_area_mask(partner_trajs[j], drivable_polys)
            partner_in_da = bool(partner_in_mask.all())
            partner_da_violations = int((~partner_in_mask).sum())
            partner_hits_other = _traj_collides_with_vehicles(
                partner_trajs[j], other_vehicle_trajs, vehicle_collision_threshold)
            eligible = ego_in_da and partner_in_da and \
                       (not ego_hits_other) and (not partner_hits_other)

            # Goal completion: distance from each endpoint to the shared midpoint
            mid = (ego_trajs[i, -1] + partner_trajs[j, -1]) / 2.0
            goal_comp = float(
                np.linalg.norm(ego_trajs[i, -1] - mid) +
                np.linalg.norm(partner_trajs[j, -1] - mid))

            smooth = _compute_smoothness(ego_trajs[i]) + \
                     _compute_smoothness(partner_trajs[j])
            curv = _compute_curvature(ego_trajs[i]) + \
                   _compute_curvature(partner_trajs[j])

            da_violations = ego_da_violations + partner_da_violations
            raw.append((i, j, end_dist, t_min, md, eligible,
                        goal_comp, smooth, curv, da_violations))

    # Split eligible vs disqualified
    eligible_pairs = [r for r in raw if r[5]]
    disqualified = [r for r in raw if not r[5]]

    if len(eligible_pairs) == 0:
        # All disqualified — rank by (da_violations ASC, end_dist ASC).
        # This prefers pairs that stay closest to drivable areas, breaking
        # ties by terminal convergence. Avoids the prior behaviour of
        # picking a pair that exits the gray area for many frames just
        # because its endpoints happen to converge.
        raw.sort(key=lambda x: (x[9], x[2]))
        return [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in raw[:k]]

    # Min-max normalize each metric across eligible pairs
    def minmax(vals):
        arr = np.array(vals, dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-9:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    end_dists = minmax([r[2] for r in eligible_pairs])
    goal_comps = minmax([r[6] for r in eligible_pairs])
    smooths = minmax([r[7] for r in eligible_pairs])
    curvs = minmax([r[8] for r in eligible_pairs])

    W_END, W_GOAL, W_SMOOTH, W_CURV = w_end, w_goal, w_smooth, w_curv
    scored = []
    for idx, r in enumerate(eligible_pairs):
        score = (W_END * end_dists[idx] +
                 W_GOAL * goal_comps[idx] +
                 W_SMOOTH * smooths[idx] +
                 W_CURV * curvs[idx])
        scored.append((score, r))

    scored.sort(key=lambda x: x[0])

    # Build result: top-k eligible first, then disqualified (sorted by
    # da_violations ASC, end_dist ASC).
    # Diversity-aware greedy: prefer pairs whose ego_idx AND partner_idx
    # are both unused so far. Without this, the same ego traj (whose
    # endpoint happens to converge best) dominates every top pair —
    # resulting in only 2-3 unique trajectories drawn instead of 2k.
    # Falls back to next-best pair (regardless of diversity) once no
    # both-fresh pair remains, so we always return k pairs if available.
    result = []
    used_ego = set()
    used_partner = set()
    remaining = list(scored)  # copy, scored ascending
    while len(result) < k and remaining:
        # 1st pass: pick best pair with BOTH ego and partner unused
        picked = None
        for s_idx, (_, r) in enumerate(remaining):
            if r[0] not in used_ego and r[1] not in used_partner:
                picked = s_idx
                break
        # 2nd pass: pick best pair with EITHER ego or partner unused
        if picked is None:
            for s_idx, (_, r) in enumerate(remaining):
                if r[0] not in used_ego or r[1] not in used_partner:
                    picked = s_idx
                    break
        # 3rd pass: just take the best remaining
        if picked is None:
            picked = 0
        _, r = remaining.pop(picked)
        result.append((r[0], r[1], r[2], r[3], r[4], r[5]))
        used_ego.add(r[0])
        used_partner.add(r[1])

    if len(result) < k:
        disqualified.sort(key=lambda x: (x[9], x[2]))
        for r in disqualified:
            if len(result) >= k:
                break
            result.append((r[0], r[1], r[2], r[3], r[4], r[5]))
    return result


def _plot_polylines(polylines, *, style="-", line_width=1.0, alpha=1.0, color="r"):
    for polyline in polylines:
        plt.plot(polyline[:, 0], polyline[:, 1], style,
                 linewidth=line_width, color=color, alpha=alpha)


def _plot_polygons(polygons, *, alpha=1.0, color="r"):
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)


def _plot_actor_bounding_box(ax, cur_location, heading, color,
                              bbox_size=(_ESTIMATED_VEHICLE_LENGTH_M,
                                         _ESTIMATED_VEHICLE_WIDTH_M)):
    bbox_length, bbox_width = bbox_size
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)
    bbox = Rectangle(
        (pivot_x, pivot_y), bbox_length, bbox_width,
        angle=np.degrees(heading), color=color,
        zorder=_BOUNDING_BOX_ZORDER,
    )
    ax.add_patch(bbox)


def _plot_static_map_elements(static_map: ArgoverseStaticMap,
                              show_ped_xings: bool = False) -> None:
    """Plot drivable areas, lane segments, and ped crossings — av2-api style."""
    if static_map is None:
        return
    for drivable_area in static_map.vector_drivable_areas.values():
        try:
            _plot_polygons([drivable_area.xyz[:, :2]],
                            alpha=0.5, color=_DRIVABLE_AREA_COLOR)
        except Exception:
            pass

    for lane_segment in static_map.vector_lane_segments.values():
        try:
            _plot_polylines(
                [lane_segment.left_lane_boundary.xyz[:, :2],
                 lane_segment.right_lane_boundary.xyz[:, :2]],
                line_width=0.5, color=_LANE_SEGMENT_COLOR,
            )
        except Exception:
            pass

    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            try:
                _plot_polylines(
                    [ped_xing.edge1.xyz[:, :2], ped_xing.edge2.xyz[:, :2]],
                    alpha=1.0, color=_LANE_SEGMENT_COLOR,
                )
            except Exception:
                pass


def _track_states_up_to(track, t_end: int):
    """Return (positions, headings, timesteps) for a track up to t_end (inclusive).

    AV2 motion forecasting tracks have observed states at all 110 timesteps
    they're present (no hidden future). So this works for any t_end in [0, 109].
    """
    states = [s for s in track.object_states if s.timestep <= t_end]
    if not states:
        return (np.empty((0, 2)), np.empty((0,)), np.array([], dtype=int))
    positions = np.array([list(s.position) for s in states])
    headings = np.array([s.heading for s in states])
    timesteps = np.array([s.timestep for s in states], dtype=int)
    return (positions, headings, timesteps)


def _plot_actor_tracks_av2_style(
    ax,
    scenario: ArgoverseScenario,
    timestep: int,
    focal_track_id: str,
    partner_track_id: str,
    drive_ego_traj: np.ndarray,      # (T, 2) global — selected driving ego (pair #1)
    drive_partner_traj: np.ndarray,  # (T, 2) global — selected driving partner (pair #1)
    top_pairs: List[Tuple[int, int, float, int]],  # top-k collision pairs
    ego_trajs_global: np.ndarray,    # (N, T, 2) all generated ego trajs
    partner_trajs_global: np.ndarray,  # (N, T, 2) all generated partner trajs
    collision_point_global: Optional[np.ndarray],
    collision_threshold: float,
):
    """Plot all actor tracks up to *timestep* — av2-api `_plot_actor_tracks` style.

    History phase (timestep <= 49): all tracks use observed states up to t.
    Future phase (timestep >= 50):
      - focal & partner use the driving-pair generated trajectory
      - OTHER tracks use their observed GT future states (AV2 provides full
        110-ts observations for all tracks present)

    Top-K collision pairs overlaid: pair #1 bold (driving), pairs #2..K thin.

    Returns focal trajectory bounds (x_min, x_max, y_min, y_max) for view sizing.
    """
    track_bounds = None
    future_start_ts = _OBS_DURATION_TIMESTEPS  # 50

    for track in scenario.tracks:
        is_focal = (track.track_id == focal_track_id)
        is_partner = (track.track_id == partner_track_id)
        is_av = (track.track_id == "AV")

        # ---- Determine (positions, headings) up to current timestep ----
        if is_focal or is_partner:
            # Focal/partner: observed history + generated driving-pair future
            if timestep < future_start_ts:
                # History phase: use observed states up to timestep
                positions, headings, timesteps = _track_states_up_to(track, timestep)
                if positions.shape[0] < 1 or timesteps[-1] != timestep:
                    continue
            else:
                # Future phase: history + generated future[:fut_idx+1]
                positions_hist, headings_hist, _ = _track_states_up_to(
                    track, future_start_ts - 1)
                future = drive_ego_traj if is_focal else drive_partner_traj
                fut_idx = timestep - future_start_ts
                if fut_idx >= len(future):
                    fut_idx = len(future) - 1
                fut_pos = future[:fut_idx + 1]
                fut_hd = np.array([_heading_from_traj(future, max(0, k))
                                   for k in range(fut_idx + 1)])
                if positions_hist.shape[0] > 0:
                    positions = np.concatenate([positions_hist, fut_pos], axis=0)
                    headings = np.concatenate([headings_hist, fut_hd], axis=0)
                else:
                    positions = fut_pos
                    headings = fut_hd
        else:
            # Other tracks: use observed GT states up to timestep.
            # AV2 provides full 110-ts observations, so tracks present at
            # future timesteps will be plotted with their real future motion.
            positions, headings, timesteps = _track_states_up_to(track, timestep)
            if positions.shape[0] < 1:
                continue
            # av2-api rule: last observed ts must equal current timestep
            # (skips tracks that left the scene before this timestep)
            if timesteps[-1] != timestep:
                continue

        # ---- Choose color ----
        if is_focal:
            track_color = _FOCAL_AGENT_COLOR
        elif is_partner:
            track_color = _PARTNER_GEN_COLOR
        elif is_av:
            track_color = _AV_COLOR
        elif track.object_type in _STATIC_OBJECT_TYPES:
            continue
        else:
            track_color = _DEFAULT_ACTOR_COLOR

        # ---- Plot trajectory history polyline ----
        if is_focal:
            _plot_polylines([positions], color=track_color, line_width=2)
        elif is_partner:
            _plot_polylines([positions], color=track_color, line_width=2)
        elif is_av:
            _plot_polylines([positions], color=track_color, line_width=1.5, alpha=0.7)
        else:
            _plot_polylines([positions], color=track_color, line_width=0.8, alpha=0.5)

        # ---- Plot bounding box at current position ----
        cur_pos = positions[-1]
        cur_heading = headings[-1]
        if track.object_type == ObjectType.VEHICLE or track.object_type == ObjectType.BUS:
            _plot_actor_bounding_box(
                ax, cur_pos, cur_heading, track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )
        elif track.object_type in (ObjectType.CYCLIST, ObjectType.MOTORCYCLIST):
            _plot_actor_bounding_box(
                ax, cur_pos, cur_heading, track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        else:
            plt.plot(cur_pos[0], cur_pos[1], "o",
                     color=track_color, markersize=4)

        # ---- Track bounds from focal (for view sizing) ----
        if is_focal:
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)

    # ---- Top-K collision pairs as overlays (drawn during future phase) ----
    if timestep >= future_start_ts and len(top_pairs) > 0:
        for rank, (ei, pi, end_dist, ct, md, eligible) in enumerate(top_pairs):
            ego_traj = ego_trajs_global[ei]
            partner_traj = partner_trajs_global[pi]
            if rank == 0:
                # Driving pair (#1): bold
                plt.plot(ego_traj[:, 0], ego_traj[:, 1], "-",
                         color=_EGO_GEN_COLOR, alpha=0.85, linewidth=2.6, zorder=30)
                plt.plot(partner_traj[:, 0], partner_traj[:, 1], "-",
                         color=_PARTNER_GEN_COLOR, alpha=0.85, linewidth=2.6, zorder=30)
                # Endpoint markers
                plt.plot(ego_traj[-1, 0], ego_traj[-1, 1], "o",
                         color=_EGO_GEN_COLOR, markersize=8, zorder=31)
                plt.plot(partner_traj[-1, 0], partner_traj[-1, 1], "o",
                         color=_PARTNER_GEN_COLOR, markersize=8, zorder=31)
            else:
                # Pairs #2..K: medium visibility
                # Disqualified pairs (out of drivable area) shown dashed + grey-ish
                if eligible:
                    ego_c, partner_c = _EGO_GEN_COLOR, _PARTNER_GEN_COLOR
                    ls = "-"
                else:
                    ego_c = partner_c = "#999999"
                    ls = "--"
                plt.plot(ego_traj[:, 0], ego_traj[:, 1], ls,
                         color=ego_c, alpha=0.55, linewidth=1.4, zorder=20)
                plt.plot(partner_traj[:, 0], partner_traj[:, 1], ls,
                         color=partner_c, alpha=0.55, linewidth=1.4, zorder=20)
                # Endpoint markers (smaller)
                plt.plot(ego_traj[-1, 0], ego_traj[-1, 1], "o",
                         color=ego_c, markersize=4, alpha=0.6, zorder=21)
                plt.plot(partner_traj[-1, 0], partner_traj[-1, 1], "o",
                         color=partner_c, markersize=4, alpha=0.6, zorder=21)

    # ---- Collision point (top-1 generated collision point — midpoint of
    #      the closest-frame pair between top-1 ego and top-1 partner) ----
    if collision_point_global is not None:
        plt.plot(collision_point_global[0], collision_point_global[1], "D",
                 color=_COLLISION_POINT_COLOR, markersize=14,
                 markeredgecolor="darkred", markeredgewidth=1.0, zorder=110)
        circle = plt.Circle(collision_point_global, collision_threshold,
                            fill=False, color=_COLLISION_POINT_COLOR,
                            linestyle="--", linewidth=1.0, alpha=0.5)
        ax.add_patch(circle)

    return track_bounds


def animate_collision_scenario(
    scenario: ArgoverseScenario,
    static_map: ArgoverseStaticMap,
    ego_trajs_global: np.ndarray,        # (N, T, 2) generated ego trajectories
    partner_trajs_global: np.ndarray,    # (N, T, 2) generated partner trajectories
    focal_track_id: str,
    partner_track_id: str,
    ego_gt_global: Optional[np.ndarray] = None,
    partner_gt_global: Optional[np.ndarray] = None,
    collision_point_global: Optional[np.ndarray] = None,
    collision_threshold: float = 1.5,
    n_top_pairs: int = 5,
    fps: int = 10,
    save_path: str = None,
    title: str = "Collision Scenario Animation",
    view_buffer_m: float = _PLOT_BOUNDS_BUFFER_M,
    w_end: float = 0.50,
    w_goal: float = 0.20,
    w_smooth: float = 0.15,
    w_curv: float = 0.15,
):
    """Animate the collision scenario as an MP4 — av2-api `visualize_scenario` style.

    Pipeline mirrors av2-api:
    1. For each timestep in [0, 109]:
       - new matplotlib figure
       - plot static map (drivable areas, lane segments)
       - plot actor tracks up to current timestep:
         * history (t<50): observed states for all tracks
         * future (t>=50): focal & partner use driving-pair generated traj;
           other tracks use observed GT future states
       - top-K collision pairs overlaid (pair #1 = driving pair, bold)
       - view bounds = focal agent trajectory extent + 30m buffer
       - minimize margins, axes off
       - save frame to in-memory PNG buffer
    2. Encode all frames to MP4 with cv2.VideoWriter + mp4v codec at 10 fps.
    """
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vid_path = str(save_path.parents[0] / f"{save_path.stem}.mp4")

    # Gather drivable-area polygons (global coords) for eligibility check
    drivable_polys = []
    if static_map is not None:
        for da in static_map.vector_drivable_areas.values():
            try:
                drivable_polys.append(da.xyz[:, :2].astype(np.float64))
            except Exception:
                pass

    # Gather all OTHER in-scene vehicle trajectories (future phase) so we can
    # disqualify pairs where ego or partner collides with a surrounding vehicle.
    other_vehicle_trajs = _extract_other_vehicle_trajs(
        scenario, focal_track_id, partner_track_id, t_start=50, t_end=110)

    # Select top-K collision pairs by composite score
    # (drivable-area + inter-vehicle collision eligibility +
    #  end_dist + goal_completion + smoothness + curvature)
    top_pairs = _select_top_k_collision_pairs(
        ego_trajs_global, partner_trajs_global, drivable_polys,
        other_vehicle_trajs, vehicle_collision_threshold=collision_threshold,
        k=n_top_pairs,
        w_end=w_end, w_goal=w_goal, w_smooth=w_smooth, w_curv=w_curv)
    print(f"    Top-{n_top_pairs} collision pairs (composite score: "
          f"{w_end}*end_dist + {w_goal}*goal_completion + {w_smooth}*smoothness + {w_curv}*curvature, "
          f"min-max normalized; pairs leaving drivable area OR colliding with "
          f"surrounding vehicles disqualified). "
          f"Surrounding vehicles checked: {len(other_vehicle_trajs)}")
    for rank, (ei, pi, end_dist, ct, md, eligible) in enumerate(top_pairs):
        marker = "  <-- DRIVING PAIR" if rank == 0 else ""
        tag = "OK" if eligible else "DISQUALIFIED"
        print(f"      #{rank+1}: ego[{ei}] + partner[{pi}] "
              f"end_dist={end_dist:.3f}m  traj_min_dist={md:.3f}m at t={ct}  "
              f"[{tag}]{marker}")

    drive_ego_idx, drive_partner_idx = top_pairs[0][0], top_pairs[0][1]
    drive_ego_traj = ego_trajs_global[drive_ego_idx]      # (T, 2) global
    drive_partner_traj = partner_trajs_global[drive_partner_idx]

    # Collision point = midpoint of the closest-frame pair between the top-1
    # ego and top-1 partner trajectories (the actual "where they collide"
    # along the trajectory, not just an endpoint midpoint).
    # top_pairs[0] = (ego_idx, partner_idx, end_dist, t_min, min_dist, eligible)
    closest_t = top_pairs[0][3]
    gen_collision_point = (
        drive_ego_traj[closest_t] + drive_partner_traj[closest_t]) / 2.0
    collision_point_global = gen_collision_point

    # Pre-compute stable view bounds from FULL focal trajectory
    # (history + driving-pair future + partner traj + collision point)
    focal_track = None
    for t in scenario.tracks:
        if t.track_id == focal_track_id:
            focal_track = t
            break
    if focal_track is not None:
        hist_pos, _, _ = _track_states_up_to(focal_track, _OBS_DURATION_TIMESTEPS - 1)
        all_focal_pos = np.concatenate([hist_pos, drive_ego_traj], axis=0) \
            if hist_pos.shape[0] > 0 else drive_ego_traj
        all_pts = np.concatenate([all_focal_pos, drive_partner_traj], axis=0)
        if collision_point_global is not None:
            all_pts = np.concatenate(
                [all_pts, np.asarray(collision_point_global).reshape(1, 2)], axis=0)
        x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
        y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
        plot_bounds = (x_min, x_max, y_min, y_max)
    else:
        plot_bounds = (0.0, 60.0, 0.0, 60.0)

    n_total = _OBS_DURATION_TIMESTEPS + _PRED_DURATION_TIMESTEPS  # 110
    frames = []

    for timestep in range(n_total):
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # 1) Static map
        _plot_static_map_elements(static_map, show_ped_xings=False)

        # 2) Actor tracks + collision overlays
        _plot_actor_tracks_av2_style(
            ax, scenario, timestep,
            focal_track_id, partner_track_id,
            drive_ego_traj, drive_partner_traj,
            top_pairs, ego_trajs_global, partner_trajs_global,
            collision_point_global, collision_threshold,
        )

        # 3) View bounds — focal trajectory + buffer (stable, computed once)
        plt.xlim(
            plot_bounds[0] - view_buffer_m,
            plot_bounds[1] + view_buffer_m,
        )
        plt.ylim(
            plot_bounds[2] - view_buffer_m,
            plot_bounds[3] + view_buffer_m,
        )
        plt.gca().set_aspect("equal", adjustable="box")

        # 4) Phase label in title
        if timestep < _OBS_DURATION_TIMESTEPS:
            phase = f"HISTORY (t={timestep * _DT_SEC - 5.0:+.1f}s)"
        else:
            t_future = (timestep - _OBS_DURATION_TIMESTEPS) * _DT_SEC
            phase = f"FUTURE (t={t_future:+.1f}s)"
        plt.title(f"{title} — {phase}", fontsize=12)

        # 5) Minimize plot margins, axes invisible — av2-api style
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())

        # 6) Save frame to in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frame = img.open(buf)
        frames.append(frame)

    # 7) Encode frames to MP4 — cv2.VideoWriter + mp4v codec (av2-api style)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(vid_path, fourcc, fps=fps, frameSize=frames[0].size)
    for i in range(len(frames)):
        frame_temp = frames[i].copy()
        video.write(cv2.cvtColor(np.array(frame_temp), cv2.COLOR_RGB2BGR))
    video.release()

    print(f"    Saved collision animation: {vid_path}")
