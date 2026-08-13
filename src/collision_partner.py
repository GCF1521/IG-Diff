"""Isomorphic collision partner trajectory generation.

Uses the same diffusion model (same weights, same ConditionEncoder structure)
to generate a second vehicle's trajectory, with both vehicles sharing the
same collision point as their goal endpoint. This is the "isomorphic guidance"
mechanism: same architecture, different content, shared collision goal.
"""

import math
import numpy as np
import torch
from typing import Dict, Optional, List

from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
from av2.map.map_api import ArgoverseStaticMap

from data.coordinate_utils import global_to_local, local_to_global, velocity_global_to_local
from data.normalization import (
    normalize, denormalize, denormalize_residual, denormalize_residual_chord,
    normalize_residual, normalize_residual_chord,
    compute_hermite_prior, compute_prior,
    compute_chord_dir, unpack_chord, pack_chord,
    residual_to_chord_frame, chord_frame_to_residual,
    CHORD_EPSILON, DT, VELOCITY_SCALE, ACCELERATION_SCALE,
)
from data.map_encoding import encode_map_tokens
from data.neighbor_encoding import encode_neighbor_tokens
from model.diffusion import DiffusionProcess
from model.tf_cross_denoiser import TFCrossDenoiser
from model.dps_guidance import full_dps_sample
from src.drivable_check import build_paths, sample_goal_and_prior_in_drivable, all_points_in_drivable


# ---------------------------------------------------------------------------
# 0. End-heading computation (lane-aligned + fallback chain)
# ---------------------------------------------------------------------------

def _lane_tangent_at_point(
    lane_segments: Dict,
    query_pt_global: np.ndarray,
    max_search_radius: float = 30.0,
) -> Optional[float]:
    """Find the tangent direction (radians, global) of the lane centerline
    nearest to query_pt_global.

    Iterates all lane_segments, projects query onto each centerline, picks the
    nearest valid projection within max_search_radius. Returns the local tangent
    at the projection point (oriented along the lane's natural direction).

    Returns None if no lane is close enough.
    """
    q = np.asarray(query_pt_global, dtype=np.float64).reshape(2)
    best_dist = float("inf")
    best_tangent = None

    for ls_id, ls in lane_segments.items():
        cl = ls.get("centerline")
        if not cl or len(cl) < 2:
            continue
        pts = np.array([(p["x"], p["y"]) for p in cl], dtype=np.float64)
        if len(pts) < 2:
            continue

        # For each segment, project q onto the segment and find min distance.
        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            ab = b - a
            ab_len_sq = float(ab[0] ** 2 + ab[1] ** 2)
            if ab_len_sq < 1e-9:
                continue
            t = float(np.dot(q - a, ab) / ab_len_sq)
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
            d = float(np.linalg.norm(q - proj))
            if d < best_dist:
                best_dist = d
                # Tangent at projection: use the segment direction (forward).
                # If t > 0.5 use the next segment's direction for continuity.
                if t > 0.5 and i + 1 < len(pts) - 1:
                    nxt = pts[i + 2] - pts[i + 1]
                    if nxt[0] ** 2 + nxt[1] ** 2 > 1e-9:
                        best_tangent = float(np.arctan2(nxt[1], nxt[0]))
                        continue
                best_tangent = float(np.arctan2(ab[1], ab[0]))

    if best_tangent is None or best_dist > max_search_radius:
        return None
    return best_tangent


def _chord_heading(history_end_local: np.ndarray, goal_local: np.ndarray) -> float:
    """Heading from history_end to goal (local frame)."""
    dx = goal_local[0] - history_end_local[0]
    dy = goal_local[1] - history_end_local[1]
    if dx * dx + dy * dy < 1e-12:
        return 0.0
    return float(np.arctan2(dy, dx))


def _select_partner_end_heading(
    history_end_local: np.ndarray,
    goal_local: np.ndarray,
    start_heading_local: float,
    partner_ref_heading: float,
    lane_segments: Dict,
    collision_point_global: np.ndarray,
    drivable_paths_local,
    n_future: int,
    prior_type: str = "hermite",
    max_dist_to_lane: float = 30.0,
) -> float:
    """Pick an end_heading (local frame) that yields a Hermite prior fully inside
    drivable areas. Tries a chain of physically-motivated candidates:

      1. Lane-aligned tangent at collision point (preferred — partner must
         arrive along the road).
      2. Reverse of lane tangent (partner approaches from the opposite lane).
      3. Chord direction (atan2(goal - history_end)) — current behavior.
      4. Start heading (straight-line continuation).
      5. Sweep ±180° in 30° steps — last-resort geometric search.

    For each candidate, computes the Hermite prior and checks that every
    waypoint lies inside drivable_areas_local. Returns the first candidate
    that passes; falls back to candidate 3 (chord) if none passes.

    Args:
        All inputs in local frame except collision_point_global, lane_segments
        (which are in global frame).
        drivable_paths_local: list of matplotlib Path objects in partner-local
            frame, or [] to disable the constraint (returns candidate 1).
    """
    # Candidate 1 & 2: lane tangent (forward / reverse)
    lane_tangent_global = _lane_tangent_at_point(
        lane_segments, collision_point_global, max_search_radius=max_dist_to_lane
    )

    candidates = []  # list of (heading_local, tag)
    if lane_tangent_global is not None:
        candidates.append((lane_tangent_global - partner_ref_heading, "lane_fwd"))
        candidates.append((lane_tangent_global + np.pi - partner_ref_heading, "lane_rev"))

    # Candidate 3: chord direction
    candidates.append((_chord_heading(history_end_local, goal_local), "chord"))

    # Candidate 4: start heading (straight continuation)
    candidates.append((float(start_heading_local), "start_h"))

    # Candidate 5: sweep ±pi around chord in 30° steps
    chord_h = _chord_heading(history_end_local, goal_local)
    for delta_deg in [30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]:
        delta = np.deg2rad(delta_deg)
        candidates.append((chord_h + delta, f"sweep_{delta_deg}"))

    # If no drivable constraint, just return lane_fwd (or chord as fallback)
    if not drivable_paths_local:
        return candidates[0][0] if candidates else _chord_heading(history_end_local, goal_local)

    # Try each candidate; return first whose prior is fully inside drivable areas
    for h_local, tag in candidates:
        if prior_type == "hermite":
            try:
                prior = compute_hermite_prior(
                    history_end_local, goal_local,
                    float(start_heading_local), float(h_local), n_future
                )
            except Exception:
                continue
        else:
            prior = compute_prior(history_end_local, goal_local, n_future)
        if all_points_in_drivable(prior, drivable_paths_local):
            return float(h_local)

    # None satisfied the constraint — return chord direction (current behavior)
    return _chord_heading(history_end_local, goal_local)


# ---------------------------------------------------------------------------
# 1. Partner selection
# ---------------------------------------------------------------------------

def select_collision_partner(
    scenario,
    focal_track,
    n_future: int = 60,
    min_future_states: int = 10,
    min_dist_threshold: float = 50.0,
) -> Optional[Dict]:
    """Select the non-focal track whose future trajectory comes closest to the focal's.

    Returns dict with partner track info and the collision point (global coords),
    or None if no suitable partner is found.
    """
    # Focal future positions (global)
    focal_future = []
    for s in focal_track.object_states:
        if 50 <= s.timestep < 50 + n_future:
            focal_future.append(np.array(s.position, dtype=np.float64))
    if len(focal_future) < min_future_states:
        return None
    focal_future = np.array(focal_future)  # (T_f, 2)

    best_partner = None
    best_min_dist = float("inf")
    best_collision_point = None
    best_collision_step = None

    _PARKING_SPEED_THRESH = 0.833   # m/s — 3 kph
    _PARKING_MIN_CONSECUTIVE = 30   # steps — 3s at 10Hz

    VALID_TYPES = {"vehicle", "bus", "motorcyclist"}
    for track in scenario.tracks:
        if track.track_id == focal_track.track_id:
            continue
        if track.category.value < 2:  # Skip FRAGMENT
            continue
        if track.object_type.value not in VALID_TYPES:
            continue

        # --- Ensure candidate exists at t=50 (same prediction start as ego) ---
        has_t50 = any(s.timestep == 50 for s in track.object_states)
        if not has_t50:
            continue

        # --- Skip parking vehicles ---
        speeds = []
        for s in track.object_states:
            if 30 <= s.timestep < 50 + n_future:
                vx, vy = s.velocity
                speeds.append(math.sqrt(vx * vx + vy * vy))
        if speeds:
            max_consec = 0
            cur = 0
            for sp in speeds:
                if sp < _PARKING_SPEED_THRESH:
                    cur += 1
                    max_consec = max(max_consec, cur)
                else:
                    cur = 0
            if max_consec >= _PARKING_MIN_CONSECUTIVE:
                continue

        cand_future = []
        for s in track.object_states:
            if 50 <= s.timestep < 50 + n_future:
                cand_future.append(np.array(s.position, dtype=np.float64))
        if len(cand_future) < min_future_states:
            continue
        cand_future = np.array(cand_future)  # (T_c, 2)

        # Pairwise distances between focal and candidate at aligned timesteps
        T_align = min(len(focal_future), len(cand_future))
        dists = np.linalg.norm(focal_future[:T_align] - cand_future[:T_align], axis=-1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist < best_min_dist:
            best_min_dist = min_dist
            best_partner = track
            # Collision point = ego's final position (shared goal for both ego
            # and partner — this is the "isomorphic guidance" requirement:
            # both vehicles share the same endpoint so their trajectories
            # converge to a single collision point).
            best_collision_point = focal_future[-1].copy()
            best_collision_step = min_idx  # relative to t=50

    if best_partner is None or best_min_dist > min_dist_threshold:
        return None

    return {
        "track": best_partner,
        "min_dist": best_min_dist,
        "collision_point_global": best_collision_point.astype(np.float32),
        "collision_timestep": 50 + best_collision_step,
        "collision_step": best_collision_step,  # relative to t=50, for heading lookup
    }


# ---------------------------------------------------------------------------
# 1b. Fan-based collision region & arrival-heading computation
# ---------------------------------------------------------------------------

def _build_fan_vertices(p0, heading, speed, t_horizon=6.0, a_lat=3.0,
                         n_steps=40, r_turn_override=None):
    """Trumpet-shaped fan polygon: apex at p0, extends forward v*t_horizon,
    lateral span grows from 0 to ±r_turn via circular arcs (centers at
    p0 ± r_turn * n_dir). r_turn = v²/a_lat by default, or override."""
    r_long = speed * t_horizon
    if r_turn_override is not None:
        r_turn = float(r_turn_override)
    else:
        r_turn = (speed ** 2) / a_lat if speed > 1e-3 else 1.0
    t_dir = np.array([np.cos(heading), np.sin(heading)])
    n_dir = np.array([-np.sin(heading), np.cos(heading)])
    ds = np.linspace(0, r_long, n_steps + 1)
    ds_clip = np.minimum(ds, r_turn)
    y_lat = r_turn - np.sqrt(np.maximum(r_turn ** 2 - ds_clip ** 2, 0))
    left = [p0 + d * t_dir + y * n_dir for d, y in zip(ds, y_lat)]
    right = [p0 + d * t_dir - y * n_dir for d, y in zip(ds[::-1], y_lat[::-1])]
    return np.array(left + right)


def _points_in_polygon(pts, poly):
    from matplotlib.path import Path as MplPath
    p = np.asarray(poly)
    if not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[0]])
    return MplPath(p, closed=True).contains_points(pts)


def _compute_collision_region_grid(ego_poly, partner_poly, drivable_polys,
                                     grid_res=0.5, padding=10.0):
    all_polys = [ego_poly, partner_poly] + list(drivable_polys)
    all_pts = np.vstack(all_polys)
    xmin, xmax = all_pts[:, 0].min() - padding, all_pts[:, 0].max() + padding
    ymin, ymax = all_pts[:, 1].min() - padding, all_pts[:, 1].max() + padding
    nx = int((xmax - xmin) / grid_res) + 1
    ny = int((ymax - ymin) / grid_res) + 1
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(gx, gy)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    in_e = _points_in_polygon(pts, ego_poly).reshape(ny, nx)
    in_p = _points_in_polygon(pts, partner_poly).reshape(ny, nx)
    in_da = np.zeros((ny, nx), dtype=bool)
    for dp in drivable_polys:
        if len(dp) < 3:
            continue
        in_da |= _points_in_polygon(pts, dp).reshape(ny, nx)
    return (in_e & in_p & in_da), gx, gy


def _arrival_heading(p0, start_heading, collision_pt):
    """Arc-tangent arrival heading at collision_pt for a vehicle starting at
    p0 with start_heading. Project to (d, y) local; r = (d²+y²)/(2|y|);
    θ = atan2(d, r-|y|); arrival = start_heading ± θ (sign from y)."""
    d_vec = np.asarray(collision_pt) - np.asarray(p0)
    t_dir = np.array([np.cos(start_heading), np.sin(start_heading)])
    n_dir = np.array([-np.sin(start_heading), np.cos(start_heading)])
    d = float(np.dot(d_vec, t_dir))
    y = float(np.dot(d_vec, n_dir))
    if abs(y) < 1e-3:
        return float(start_heading)
    r = (d * d + y * y) / (2.0 * abs(y))
    theta = float(np.arctan2(d, r - abs(y)))
    return float(start_heading + (theta if y > 0 else -theta))


def compute_collision_region_adaptive(
    ego_p0, ego_h0, ego_speed,
    part_p0, part_h0, part_speed,
    drivable_polys,
    t_horizon_start=6.0, t_horizon_max=15.0, t_horizon_step=1.0,
    a_lat=3.0, grid_res=0.5, padding=10.0,
):
    """Try 6s fan first; if zero area, extend t_horizon forward (keep r_turn
    fixed — fan gets longer, NOT wider). Returns (ego_poly, partner_poly,
    inside_mask, grid_x, grid_y, t_used)."""
    e_r = (ego_speed ** 2) / a_lat if ego_speed > 1e-3 else 1.0
    p_r = (part_speed ** 2) / a_lat if part_speed > 1e-3 else 1.0
    t = t_horizon_start
    best = None
    while t <= t_horizon_max + 1e-6:
        ep = _build_fan_vertices(ego_p0, ego_h0, ego_speed, t_horizon=t,
                                  a_lat=a_lat, r_turn_override=e_r)
        pp = _build_fan_vertices(part_p0, part_h0, part_speed, t_horizon=t,
                                  a_lat=a_lat, r_turn_override=p_r)
        mask, gx, gy = _compute_collision_region_grid(
            ep, pp, drivable_polys, grid_res=grid_res, padding=padding)
        area = float(mask.sum()) * (grid_res ** 2)
        if area > 1e-3:
            return ep, pp, mask, gx, gy, t
        if best is None or area > best[5]:
            best = (ep, pp, mask, gx, gy, area, t)
        t += t_horizon_step
    return best[0], best[1], best[2], best[3], best[4], t_horizon_max


def sample_collision_points_from_region(
    ego_p0, ego_h0, ego_speed,
    part_p0, part_h0, part_speed,
    drivable_polys,
    n_points=5, seed=42,
    t_horizon_start=6.0, t_horizon_max=15.0, t_horizon_step=1.0,
    grid_res=0.5,
):
    """Compute the collision region (fan-intersection ∩ drivable) and pick
    its GEOMETRIC CENTROID as the single collision point. Endpoints for
    both vehicles are then sampled inside a circle of radius
    `endpoint_sampling_radius` (default 2m) centered on the centroid —
    see _sample_goal_in_region.

    Returns: (list with one dict, region_constraint) where the dict is
    {collision_point_global (=centroid), ego_arr_h, part_arr_h,
    t_horizon_used, region_area_m2}, and region_constraint is a dict
    {gx, gy, mask, grid_res, ego_poly, partner_poly, centroid_global}
    in global coords. The two fan polygons are included so the viz can
    draw the full V-type regions, not just the intersection.
    Returns ([], None) if region is empty.
    """
    ego_poly, partner_poly, mask, gx, gy, t_used = compute_collision_region_adaptive(
        ego_p0, ego_h0, ego_speed, part_p0, part_h0, part_speed,
        drivable_polys, t_horizon_start=t_horizon_start,
        t_horizon_max=t_horizon_max, t_horizon_step=t_horizon_step,
        grid_res=grid_res,
    )
    if not mask.any():
        return [], None
    region_area = float(mask.sum()) * (grid_res ** 2)

    # Geometric centroid of the inside region
    XX, YY = np.meshgrid(gx, gy)
    centroid = np.array([XX[mask].mean(), YY[mask].mean()], dtype=np.float64)

    # Region constraint: grid + fan polygons + centroid. The fan polygons
    # are needed by the viz to draw the full V-type regions; the centroid
    # is the center of the endpoint-sampling disk.
    region_constraint = {
        "gx": gx, "gy": gy, "mask": mask, "grid_res": grid_res,
        "ego_poly": ego_poly, "partner_poly": partner_poly,
        "centroid_global": centroid,
    }

    e_arr = _arrival_heading(ego_p0, ego_h0, centroid)
    p_arr = _arrival_heading(part_p0, part_h0, centroid)
    out = [{
        "collision_point_global": centroid,
        "ego_arrival_heading": e_arr,
        "partner_arrival_heading": p_arr,
        "t_horizon_used": float(t_used),
        "region_area_m2": region_area,
    }]
    return out, region_constraint


def _point_in_region_global(pt, region_constraint):
    """O(1) grid-lookup test: is pt inside the collision region?

    The region is stored as a (mask, gx, gy) grid in global coords. We find
    the nearest grid cell and check the mask. Points outside the grid
    extent return False.
    """
    if region_constraint is None:
        return True  # No constraint → allow (backward-compat)
    gx = region_constraint["gx"]
    gy = region_constraint["gy"]
    mask = region_constraint["mask"]
    # Find nearest grid index
    ix = int(np.searchsorted(gx, pt[0]) - 1)
    iy = int(np.searchsorted(gy, pt[1]) - 1)
    if ix < 0 or ix >= len(gx) or iy < 0 or iy >= len(gy):
        return False
    return bool(mask[iy, ix])


def _sample_goal_in_region(
    goal_m, goal_sigma_lon, goal_sigma_lat, gt_endpoint_heading,
    region_paths, max_attempts=100, rng=None,
    sampling_radius=2.0, centroid_local=None,
):
    """Sample a goal perturbation whose endpoint lies inside a disk of
    radius `sampling_radius` (default 2m) centered on `centroid_local`
    (the collision-region centroid expressed in this vehicle's local
    frame). The disk constraint replaces the previous convex-hull-of-
    region constraint — the user wants endpoints clustered tightly
    around the centroid, not spread across the whole region.

    Falls back to the original goal_m if no acceptable sample found.
    """
    if rng is None:
        rng = np.random.default_rng()
    if centroid_local is None:
        # No centroid supplied — fall back to centering the disk on goal_m
        centroid_local = goal_m.copy()
    for _ in range(max_attempts):
        # Uniform sample inside a disk of radius sampling_radius
        # (sqrt for uniform area sampling; theta uniform in [0, 2π))
        r = sampling_radius * math.sqrt(rng.random())
        theta = rng.uniform(0.0, 2.0 * math.pi)
        cand = centroid_local.copy()
        cand[0] += r * math.cos(theta)
        cand[1] += r * math.sin(theta)
        return cand
    return goal_m.copy()


def _centroid_local_from_global(region_constraint, ref_pos_np, ref_heading_val):
    """Project the collision-region centroid (global) into a vehicle's local
    frame. Used by _build_collision_sample_batch to center the 2m endpoint-
    sampling disk on the centroid in local coords."""
    if region_constraint is None or "centroid_global" not in region_constraint:
        return None
    centroid_global = np.asarray(region_constraint["centroid_global"], dtype=np.float64).reshape(1, 2)
    centroid_local, _ = global_to_local(
        centroid_global, np.zeros(1), ref_pos_np, ref_heading_val,
    )
    return centroid_local[0].astype(np.float64)


# ---------------------------------------------------------------------------
# 2. Partner condition encoding
# ---------------------------------------------------------------------------

def _extract_track_states(track, t_start, t_end):
    """Extract position/heading/velocity for a track in [t_start, t_end)."""
    positions, headings, velocities = [], [], []
    for s in track.object_states:
        if t_start <= s.timestep < t_end:
            positions.append(np.array(s.position, dtype=np.float64))
            headings.append(float(s.heading))
            velocities.append(np.array(s.velocity, dtype=np.float64))
    return positions, headings, velocities


def _compute_travel_heading(positions: list, box_heading: float,
                             n_frames: int = 3) -> float:
    """Estimate travel-direction heading from the last `n_frames` positions.

    Mirrors the ego-side logic in av2_dataset.py:_load_item but generalized to
    use the last N frames (default 3) for robustness against single-frame noise.

    Algorithm:
    1. Start with the box heading (from data).
    2. If the last 2..N positions span enough distance (>=0.2m), compute the
       atan2 heading from the earliest to the latest of those positions.
    3. If the travel heading diverges from the box heading by <45°, override
       with travel heading (trust positions). Otherwise keep box heading
       (side-slip / low-speed / mid-turn snap).

    Args:
        positions: list of (2,) arrays, ordered by timestep ascending. The
            last `n_frames` are used (or fewer if not enough history).
        box_heading: the dataset heading at the prediction start (fallback).
        n_frames: number of trailing frames to use for the travel estimate.

    Returns:
        Estimated start heading (radians, global frame).
    """
    start_heading = float(box_heading)
    if len(positions) < 2:
        return start_heading
    k = min(n_frames, len(positions))
    p_start = positions[-k]
    p_end = positions[-1]
    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    if dx * dx + dy * dy < 0.04:  # <0.2 m apart — too slow, trust box heading
        return start_heading
    pos_heading = float(np.arctan2(dy, dx))
    diff = pos_heading - start_heading
    diff = diff - 2 * np.pi * np.round(diff / (2 * np.pi))
    if abs(diff) < np.pi / 4:  # <45° divergence — trust positions
        return pos_heading
    return start_heading


def encode_partner_conditions(
    partner_track,
    focal_track,
    scenario,
    lane_segments: Dict,
    collision_point_global: np.ndarray,
    cfg_data: Dict,
    drivable_areas_global: Optional[List[np.ndarray]] = None,
    partner_end_heading_global: Optional[float] = None,
) -> Dict:
    """Encode all conditions for the partner agent in its own reference frame.

    Mirrors Argoverse2Dataset._load_item() but centered on the partner.
    Start heading is computed from the last few history positions (travel
    direction), falling back to the box heading when the two diverge.
    End heading is computed geometrically as atan2(goal - history_end),
    unless partner_end_heading_global is provided (e.g., from the fan
    arc-tangent arrival-heading calculation), in which case that overrides.
    """
    n_future = cfg_data.get("n_future", 60)
    n_history = cfg_data.get("n_history", 20)
    history_start = cfg_data.get("history_start", 30)
    n_lanes = cfg_data.get("n_lanes", 24)
    lane_feat_dim = cfg_data.get("lane_feat_dim", 46)
    n_neighbors = cfg_data.get("n_neighbors", 6)
    prior_type = cfg_data.get("prior_type", "hermite")
    residual_frame = cfg_data.get("residual_frame", "chord")

    # --- Partner reference frame (t=50) ---
    # Partner MUST have a state at t=50 to use as reference — same frame as ego
    t50_state = None
    for s in partner_track.object_states:
        if s.timestep == 50:
            t50_state = s
            break
    if t50_state is None:
        return None  # Can't align with ego's timeline without t=50 state

    partner_ref_pos = np.array(t50_state.position, dtype=np.float64)
    partner_ref_heading = float(t50_state.heading)

    # --- Partner history (t=30..49) ---
    hist_positions, hist_headings, hist_velocities = _extract_track_states(partner_track, history_start, 50)
    if len(hist_positions) < n_history:
        # Pad by repeating last position
        while len(hist_positions) < n_history:
            hist_positions.append(hist_positions[-1] if hist_positions else np.zeros(2))
            hist_headings.append(hist_headings[-1] if hist_headings else 0.0)
            hist_velocities.append(hist_velocities[-1] if hist_velocities else np.zeros(2))

    hist_positions = np.array(hist_positions[:n_history])
    hist_headings = np.array(hist_headings[:n_history])
    hist_velocities = np.array(hist_velocities[:n_history])
    history_local, _ = global_to_local(hist_positions, hist_headings, partner_ref_pos, partner_ref_heading)
    history_norm = normalize(history_local)
    # Velocity and acceleration in partner-local frame (rotation only)
    history_velocities_local = velocity_global_to_local(hist_velocities, partner_ref_heading).astype(np.float32)
    history_acc_local = np.zeros_like(history_velocities_local)
    history_acc_local[1:] = (history_velocities_local[1:] - history_velocities_local[:-1]) / DT
    history_acc_local[0] = history_acc_local[1] if n_history > 1 else 0.0
    history_velocity_norm = history_velocities_local / VELOCITY_SCALE
    history_acc_norm = history_acc_local / ACCELERATION_SCALE
    history_full_norm = np.concatenate(
        [history_norm, history_velocity_norm, history_acc_norm], axis=-1
    ).astype(np.float32)

    # --- Partner future (t=50..109) ---
    fut_positions, fut_headings, _ = _extract_track_states(partner_track, 50, 50 + n_future)
    if len(fut_positions) < n_future:
        while len(fut_positions) < n_future:
            fut_positions.append(fut_positions[-1] if fut_positions else np.zeros(2))
            fut_headings.append(fut_headings[-1] if fut_headings else 0.0)

    fut_positions = np.array(fut_positions[:n_future])
    fut_headings = np.array(fut_headings[:n_future])
    future_local, _ = global_to_local(fut_positions, fut_headings, partner_ref_pos, partner_ref_heading)

    # --- Goal = collision point in partner's local frame ---
    goal_local, _ = global_to_local(
        collision_point_global.reshape(1, 2), np.zeros(1), partner_ref_pos, partner_ref_heading
    )
    goal_local = goal_local[0].astype(np.float32)
    goal_norm = normalize(goal_local.reshape(1, 2)).flatten()

    # --- Drivable areas in partner's local frame (needed for end_heading selection) ---
    drivable_areas_local = []
    if drivable_areas_global:
        for da_pts in drivable_areas_global:
            try:
                da_arr = np.asarray(da_pts, dtype=np.float64)
                if len(da_arr) < 3:
                    continue
                da_local, _ = global_to_local(
                    da_arr, np.zeros(len(da_arr)), partner_ref_pos, partner_ref_heading
                )
                drivable_areas_local.append(da_local.astype(np.float32))
            except Exception:
                pass
    da_paths_local = build_paths(drivable_areas_local) if drivable_areas_local else []

    # --- Headings ---
    # start_heading: travel-direction estimate from the last few history frames.
    # The dataset box heading at t=50 can diverge from actual motion direction
    # (side-slip, low-speed, mid-turn snap, annotation noise). We mirror the
    # ego-side logic in av2_dataset.py:_load_item but use the last 3 frames for
    # robustness. The ref frame still uses the box heading (so history_local
    # coordinates are unchanged); only the Hermite start tangent is overridden.
    start_heading_global = _compute_travel_heading(
        hist_positions.tolist() if isinstance(hist_positions, np.ndarray)
        else hist_positions,
        partner_ref_heading,
        n_frames=3,
    )
    start_heading = start_heading_global - partner_ref_heading

    # end_heading: lane-aligned at collision point, with fallback chain.
    # The collision point lies on a road (ego's real future endpoint), so the
    # lane tangent there is the physically-correct arrival direction. If no
    # nearby lane is found, or the resulting Hermite prior leaves the drivable
    # area, fall back through chord / start_heading / sweep candidates.
    # If partner_end_heading_global is provided (e.g., from fan arc-tangent),
    # skip the lane-fallback chain and use it directly.
    history_end = history_local[-1]
    if partner_end_heading_global is not None:
        end_heading = float(partner_end_heading_global) - partner_ref_heading
    else:
        end_heading = _select_partner_end_heading(
            history_end_local=history_end,
            goal_local=goal_local,
            start_heading_local=start_heading,
            partner_ref_heading=partner_ref_heading,
            lane_segments=lane_segments,
            collision_point_global=collision_point_global,
            drivable_paths_local=da_paths_local,
            n_future=n_future,
            prior_type=prior_type,
        )
    end_heading_global = end_heading + partner_ref_heading

    # --- Prior ---
    if prior_type == "hermite":
        prior_local = compute_hermite_prior(
            history_end, goal_local, start_heading, end_heading, n_future
        )
    else:
        prior_local = compute_prior(history_end, goal_local, n_future)

    # --- Residual ---
    residual_local = future_local - prior_local

    # Chord-frame residual
    chord_dir, chord_len, chord_valid = compute_chord_dir(history_end, goal_local)
    use_chord = chord_valid and chord_len > CHORD_EPSILON and residual_frame == "chord"

    if use_chord:
        r_lon, r_lat = residual_to_chord_frame(residual_local, history_end, goal_local)
        if r_lon is None:
            use_chord = False
            residual_norm = normalize_residual(residual_local)
            chord_dir = np.zeros(2, dtype=np.float32)
        else:
            r_lon_n, r_lat_n = normalize_residual_chord(r_lon, r_lat)
            residual_norm = pack_chord(r_lon_n, r_lat_n)
    else:
        residual_norm = normalize_residual(residual_local)

    prior_norm = normalize(prior_local)

    # --- Map tokens (re-encode in partner's frame) ---
    map_tokens = np.zeros((n_lanes, lane_feat_dim), dtype=np.float32)
    map_mask = np.zeros(n_lanes, dtype=np.float32)
    if lane_segments:
        map_tokens, map_mask, _ = encode_map_tokens(
            lane_segments, partner_ref_pos, partner_ref_heading, n_lanes
        )

    # --- Neighbor tokens (re-encode with focal as a neighbor) ---
    tracks_dicts = []
    for track in scenario.tracks:
        if track.track_id == partner_track.track_id:
            continue  # Skip partner itself
        states = []
        for s in track.object_states:
            states.append({
                "timestep": s.timestep,
                "position": s.position,
                "heading": s.heading,
                "velocity": s.velocity,
                "observed": s.observed,
            })
        # Remap categories: focal (3) → SCORED (2) so it's included as neighbor
        cat = track.category.value
        if track.track_id == focal_track.track_id:
            cat = 2  # SCORED — include focal as neighbor
        tracks_dicts.append({
            "track_id": track.track_id,
            "object_type": track.object_type.value,
            "object_category": cat,
            "object_states": states,
        })

    neighbor_tokens = np.zeros((n_neighbors, n_history + 1, 6), dtype=np.float32)
    neighbor_mask = np.zeros(n_neighbors, dtype=np.float32)
    if tracks_dicts:
        neighbor_tokens, neighbor_mask, _, _, _ = encode_neighbor_tokens(
            tracks_dicts, partner_ref_pos, partner_ref_heading,
            n_neighbors, n_history, n_future, history_start,
        )

    def _to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    return {
        "trajectory": _to_tensor(residual_norm),
        "history": _to_tensor(history_full_norm),     # (T_hist, 6) [x, y, vx, vy, ax, ay] normalized
        "history_pos": _to_tensor(history_norm),      # (T_hist, 2) positions only
        "history_velocity": _to_tensor(history_velocity_norm),  # (T_hist, 2) normalized
        "history_acceleration": _to_tensor(history_acc_norm),   # (T_hist, 2) normalized
        "goal": _to_tensor(goal_norm),
        "prior": _to_tensor(prior_norm),
        "map_tokens": _to_tensor(map_tokens),
        "map_mask": _to_tensor(map_mask),
        "neighbor_tokens": _to_tensor(neighbor_tokens),
        "neighbor_mask": _to_tensor(neighbor_mask),
        "ref_pos": _to_tensor(partner_ref_pos.astype(np.float32)),
        "ref_heading": torch.tensor(partner_ref_heading, dtype=torch.float32),
        "start_heading": torch.tensor(start_heading, dtype=torch.float32),
        "end_heading": torch.tensor(end_heading, dtype=torch.float32),
        "start_heading_global": torch.tensor(start_heading_global, dtype=torch.float32),
        "end_heading_global": torch.tensor(end_heading_global, dtype=torch.float32),
        "use_chord_frame": torch.tensor(float(use_chord), dtype=torch.float32),
        "chord_dir": _to_tensor(chord_dir.astype(np.float32)) if chord_dir is not None else torch.zeros(2),
        # GT for comparison
        "future_local": future_local.astype(np.float32),
        # Drivable areas in partner-local coords (for inference-time goal/prior sampling)
        "drivable_areas_local": drivable_areas_local,
    }


# ---------------------------------------------------------------------------
# 3. Collision rate computation
# ---------------------------------------------------------------------------

def compute_collision_rate(
    ego_trajs_global: np.ndarray,
    partner_trajs_global: np.ndarray,
    threshold: float = 1.5,
    threshold_strict: float = 1.0,
) -> Dict:
    """Compute collision rate between ego and partner trajectory sets.

    Args:
        ego_trajs_global: (N_ego, T, 2) in global coords
        partner_trajs_global: (N_partner, T, 2) in global coords
        threshold: collision distance threshold (meters)
        threshold_strict: strict collision distance (meters)

    Returns:
        Dict with collision_rate, mean_min_dist, collision_rate_strict
    """
    if ego_trajs_global.size == 0 or partner_trajs_global.size == 0:
        return {
            "collision_rate": 0.0,
            "mean_min_dist": float("nan"),
            "collision_rate_strict": 0.0,
        }

    # Align timestep dimension (use shorter horizon — both should be n_future
    # in normal use, but defensive truncation avoids broadcast errors).
    T = min(ego_trajs_global.shape[1], partner_trajs_global.shape[1])
    ego = ego_trajs_global[:, :T, :]
    partner = partner_trajs_global[:, :T, :]

    # (N_ego, N_partner, T)
    dists = np.linalg.norm(
        ego[:, None, :, :] - partner[None, :, :, :], axis=-1
    )
    min_dists = dists.min(axis=-1)  # (N_ego, N_partner)

    return {
        "collision_rate": float((min_dists < threshold).mean()),
        "mean_min_dist": float(min_dists.mean()),
        "collision_rate_strict": float((min_dists < threshold_strict).mean()),
        "min_dist_matrix": min_dists,  # (N_ego, N_partner) for analysis
    }


# ---------------------------------------------------------------------------
# 4. Partner trajectory generation (orchestrator)
# ---------------------------------------------------------------------------

def _find_focal_track(scenario):
    """Find the focal track from the scenario (same logic as _load_item)."""
    focal_id = getattr(scenario, "focal_track_id", None)
    if focal_id:
        for track in scenario.tracks:
            if track.track_id == focal_id:
                return track
    for track in scenario.tracks:
        if track.category.value == 3:
            return track
    return scenario.tracks[0] if scenario.tracks else None


def _load_static_map(scenario, map_dir, parquet_path: str = None):
    """Load the static map for a scenario.

    Handles the case where scenario.map_id is None (common in AV2 1k subset)
    by deriving the log_id from the parquet file path instead.
    """
    from pathlib import Path
    map_dir = Path(map_dir) if isinstance(map_dir, str) else map_dir

    # Determine log_id: prefer map_id, fallback to scenario_id or parquet stem
    log_id = str(scenario.map_id) if scenario.map_id is not None else None
    if log_id is None or log_id == "None":
        log_id = getattr(scenario, "scenario_id", None)
    if (log_id is None or log_id == "None") and parquet_path is not None:
        log_id = Path(parquet_path).stem.replace("scenario_", "")

    if log_id is None or log_id == "None":
        return None

    static_map = None
    # 1) Official layout: map_dir/{log_id}/log_map_archive_{log_id}.json
    map_path = map_dir / log_id / f"log_map_archive_{log_id}.json"
    if map_path.is_file():
        try:
            static_map = ArgoverseStaticMap.from_json(map_path)
        except Exception:
            pass

    # 2) Flat layout: search by log_id in map_dir
    if static_map is None:
        for mc in map_dir.glob(f"**/log_map_archive_{log_id}*.json"):
            try:
                static_map = ArgoverseStaticMap.from_json(mc)
                break
            except Exception:
                continue

    # 3) Same directory as parquet file
    if static_map is None and parquet_path is not None:
        parquet_parent = Path(parquet_path).parent
        map_path = parquet_parent / f"log_map_archive_{log_id}.json"
        if map_path.is_file():
            try:
                static_map = ArgoverseStaticMap.from_json(map_path)
            except Exception:
                pass

    return static_map


def _build_lane_segments(static_map):
    """Build lane_segments dict from static map (same format as _load_item)."""
    from data.av2_dataset import _compute_centerline_from_boundaries
    lane_segments = {}
    if static_map is None:
        return lane_segments
    for ls_id, ls in static_map.vector_lane_segments.items():
        real_centerline = None
        try:
            left_xyz = ls.left_lane_boundary.xyz
            right_xyz = ls.right_lane_boundary.xyz
            if left_xyz is not None and right_xyz is not None and len(left_xyz) >= 2 and len(right_xyz) >= 2:
                cl_xyz = _compute_centerline_from_boundaries(left_xyz, right_xyz)
                if cl_xyz is not None and len(cl_xyz) >= 2:
                    real_centerline = [{"x": p[0], "y": p[1], "z": p[2]} for p in cl_xyz.tolist()]
        except Exception:
            pass

        lane_segments[str(ls_id)] = {
            "id": ls.id,
            "left_lane_boundary": [{"x": p[0], "y": p[1], "z": p[2]} for p in ls.left_lane_boundary.xyz.tolist()],
            "right_lane_boundary": [{"x": p[0], "y": p[1], "z": p[2]} for p in ls.right_lane_boundary.xyz.tolist()],
            "centerline": real_centerline,
            "lane_type": ls.lane_type.value,
            "is_intersection": ls.is_intersection,
            "has_predecessor": len(ls.predecessors) > 0,
            "has_successor": len(ls.successors) > 0,
            "predecessors": [str(p) for p in ls.predecessors],
            "successors": [str(s) for s in ls.successors],
            "left_neighbor_id": str(ls.left_neighbor_id) if ls.left_neighbor_id is not None else None,
            "right_neighbor_id": str(ls.right_neighbor_id) if ls.right_neighbor_id is not None else None,
            "left_mark_type": ls.left_mark_type.value if hasattr(ls, "left_mark_type") else "NONE",
            "right_mark_type": ls.right_mark_type.value if hasattr(ls, "right_mark_type") else "NONE",
        }
    return lane_segments


def _reconstruct_trajectory(residual_norm, prior_norm, use_chord_frame, chord_dir):
    """Reconstruct full trajectory from normalized residual + prior."""
    prior = denormalize(prior_norm)
    if use_chord_frame is not None and use_chord_frame.item() == 1.0 and chord_dir is not None:
        r_lon_n, r_lat_n = unpack_chord(residual_norm)
        r_lon, r_lat = denormalize_residual_chord(r_lon_n, r_lat_n)
        residual = chord_frame_to_residual(r_lon, r_lat, chord_dir)
    else:
        residual = denormalize_residual(residual_norm)
    trajectory = prior + residual
    return normalize(trajectory)


def _build_collision_sample_batch(
    sampled_point, ego_sample, partner_sample, cfg_data, args, device,
    n_per_point, region_constraint=None,
):
    """Build a per-vehicle (goal, prior, conditions) batch of size n_per_point
    for a single sampled collision point.

    Each call perturbs the goal around the collision point in vehicle-local
    coords (small anisotropic sigma). The endpoint is constrained to lie
    INSIDE the collision region (the fan-intersection ∩ drivable polygon),
    reprojected into each vehicle's local frame. This guarantees the goal
    never leaves the collision region (and therefore never leaves the
    drivable area, since the region is already intersected with drivable).

    Returns: dict with batched tensors ready for full_dps_sample, plus CPU
    bookkeeping (priors_for_reconstruct, use_chord_flags,
    chord_dirs_for_reconstruct) and the vehicle's ref_pos/ref_heading.
    """
    out = {}
    for tag, sample in (("ego", ego_sample), ("partner", partner_sample)):
        arrival_h = sampled_point[f"{tag}_arrival_heading"]
        ref_pos_np = sample["ref_pos"].numpy()
        ref_heading_val = sample["ref_heading"].item()
        # Collision point in this vehicle's local frame
        cp_global = sampled_point["collision_point_global"]
        cp_local, _ = global_to_local(
            cp_global.reshape(1, 2), np.zeros(1),
            ref_pos_np, ref_heading_val,
        )
        goal_m = cp_local[0].astype(np.float64)
        # Use positions-only slice (history is now 6-dim [x,y,vx,vy,ax,ay])
        hist_pos = sample.get("history_pos", sample["history"][..., :2])
        history_end_m = denormalize(hist_pos[-1:].reshape(1, 2)).numpy().flatten()
        start_heading_m = sample["start_heading"].item()
        end_heading_m = float(arrival_h) - ref_heading_val

        prior_type = cfg_data.get("prior_type", "hermite")

        # Anisotropic sigma around the collision point
        chord_len = max(np.linalg.norm(goal_m - history_end_m), 1e-6)
        goal_sigma_lon = args.goal_sigma_lon if args.goal_sigma_lon > 0 else max(chord_len * 0.10, 0.5)
        goal_sigma_lat = args.goal_sigma_lat if args.goal_sigma_lat > 0 else max(chord_len * 0.04, 0.3)
        gt_endpoint_heading = end_heading_m

        da_local_list = sample.get("drivable_areas_local", None)
        da_paths = build_paths(da_local_list) if da_local_list else []

        # Both vehicles use the SAME global collision point (the region
        # centroid). goal_m is already that point projected into this
        # vehicle's local frame, so no per-vehicle perturbation — ego and
        # partner share one strict endpoint. Diversity across the N
        # trajectories comes only from diffusion sampling noise, not from
        # goal perturbation.
        def _compute_perturbed_prior(history_end, g_m, _eh=end_heading_m,
                                      _sh=start_heading_m, _pt=prior_type,
                                      _da=da_local_list):
            if _pt == "hermite":
                return compute_hermite_prior(
                    history_end, g_m, _sh, _eh, cfg_data["n_future"],
                    drivable_areas_local=_da,
                )
            return compute_prior(history_end, g_m, cfg_data["n_future"])

        perturbed_goals_m = []
        perturbed_priors_m = []
        priors_for_reconstruct = []
        chord_dirs_for_reconstruct = []
        use_chord_flags = []
        use_chord_frame_cpu = sample.get("use_chord_frame")
        chord_dir_cpu = sample.get("chord_dir")

        for _ in range(n_per_point):
            perturbed_goal_m = goal_m.copy()
            perturbed_prior_m = _compute_perturbed_prior(
                history_end_m, perturbed_goal_m)
            perturbed_goals_m.append(perturbed_goal_m)
            perturbed_priors_m.append(perturbed_prior_m)
            priors_for_reconstruct.append(
                torch.tensor(normalize(perturbed_prior_m), dtype=torch.float32)
            )
            if use_chord_frame_cpu is not None and use_chord_frame_cpu.item() == 1.0:
                pert_chord_dir, _, pert_chord_valid = compute_chord_dir(
                    history_end_m, perturbed_goal_m)
                if pert_chord_valid:
                    chord_dirs_for_reconstruct.append(
                        torch.tensor(pert_chord_dir, dtype=torch.float32))
                    use_chord_flags.append(torch.tensor(1.0, dtype=torch.float32))
                else:
                    chord_dirs_for_reconstruct.append(torch.zeros(2, dtype=torch.float32))
                    use_chord_flags.append(torch.tensor(0.0, dtype=torch.float32))
            else:
                chord_dirs_for_reconstruct.append(
                    chord_dir_cpu if chord_dir_cpu is not None else torch.zeros(2))
                use_chord_flags.append(
                    use_chord_frame_cpu if use_chord_frame_cpu is not None else torch.tensor(0.0))

        N = n_per_point
        batch_goal = torch.stack([
            torch.tensor(normalize(g.reshape(1, 2)), dtype=torch.float32).flatten()
            for g in perturbed_goals_m
        ]).to(device)
        batch_prior = torch.stack([
            torch.tensor(normalize(p), dtype=torch.float32)
            for p in perturbed_priors_m
        ]).to(device)
        batch_conditions = {
            "goal": batch_goal,
            "map_tokens": sample["map_tokens"].unsqueeze(0).expand(N, -1, -1).to(device),
            "map_mask": sample["map_mask"].unsqueeze(0).expand(N, -1).to(device),
            "neighbor_tokens": sample["neighbor_tokens"].unsqueeze(0).expand(N, -1, -1, -1).to(device),
            "neighbor_mask": sample["neighbor_mask"].unsqueeze(0).expand(N, -1).to(device),
            "history": sample["history"].unsqueeze(0).expand(N, -1, -1).to(device),
        }
        use_chord_val = use_chord_flags[0] if use_chord_flags else torch.tensor(0.0)
        if use_chord_val.item() == 1.0:
            batch_chord_dir = torch.stack(chord_dirs_for_reconstruct).float().mean(0).unsqueeze(0).to(device)
            batch_use_chord = use_chord_val.unsqueeze(0).to(device)
        else:
            batch_chord_dir = None
            batch_use_chord = None

        out[tag] = {
            "batch_conditions": batch_conditions,
            "batch_prior": batch_prior,
            "batch_chord_dir": batch_chord_dir,
            "batch_use_chord": batch_use_chord,
            "priors_for_reconstruct": priors_for_reconstruct,
            "use_chord_flags": use_chord_flags,
            "chord_dirs_for_reconstruct": chord_dirs_for_reconstruct,
            "ref_pos": ref_pos_np,
            "ref_heading": ref_heading_val,
            "sampled_point_local": goal_m,
            "arrival_heading_global": float(arrival_h),
        }
    return out


def generate_collision_pair_trajectories(
    idx: int,
    sample: Dict,
    dataset,
    model: TFCrossDenoiser,
    diffusion: DiffusionProcess,
    device: str,
    cfg: Dict,
    args,
    n_points: int = 5,
    n_per_point: int = 3,
) -> Optional[Dict]:
    """Sample n_points collision points from the fan-intersection region,
    then for each point generate n_per_point ego + n_per_point partner
    trajectories, all converging to that collision point (with each
    vehicle's arrival heading computed from the fan arc-tangent).

    Returns dict with ego_trajs_global, partner_trajs_global (lists of
    (T, 2) arrays), collision_point_global (top-1), all_collision_points
    (list of dicts), and bookkeeping for visualization.
    """
    cfg_data = cfg["data"]
    n_future = cfg_data["n_future"]

    phys_idx = dataset._valid_indices[idx] if dataset._valid_indices is not None else idx
    parquet_path = dataset.scenario_files[phys_idx]
    scenario = load_argoverse_scenario_parquet(parquet_path)
    focal_track = _find_focal_track(scenario)
    if focal_track is None:
        return None

    partner_info = select_collision_partner(scenario, focal_track, n_future=n_future)
    if partner_info is None:
        return None
    partner_track = partner_info["track"]

    # Lane segments + static_map + drivable polys
    lane_segments = sample.get("scene_data", {}).get("lane_segments", {})
    static_map = _load_static_map(scenario, dataset.map_dir, parquet_path)
    if not lane_segments and static_map is not None:
        lane_segments = _build_lane_segments(static_map)
    drivable_areas_global = sample.get("scene_data", {}).get("drivable_areas_global", None)
    if not drivable_areas_global and static_map is not None:
        drivable_areas_global = []
        for da_id, da in static_map.vector_drivable_areas.items():
            try:
                da_pts = da.xyz[:, :2]
                if len(da_pts) >= 3:
                    drivable_areas_global.append(da_pts.astype(np.float32))
            except Exception:
                pass

    # t=50 state for both vehicles
    e_t50 = next((s for s in focal_track.object_states if s.timestep == 50), None)
    p_t50 = next((s for s in partner_track.object_states if s.timestep == 50), None)
    if e_t50 is None or p_t50 is None:
        return None
    ego_p0 = np.array(e_t50.position, dtype=np.float64)
    part_p0 = np.array(p_t50.position, dtype=np.float64)
    # start_heading from pure last-3-frame atan2 — no dataset heading
    e_pos, _, _ = _extract_track_states(focal_track, 30, 50)
    p_pos, _, _ = _extract_track_states(partner_track, 30, 50)
    if len(e_pos) < 3 or len(p_pos) < 3:
        return None
    def _h3(positions):
        ps = positions[-3]
        pe = positions[-1]
        return float(np.arctan2(pe[1] - ps[1], pe[0] - ps[0]))
    ego_h0 = _h3(e_pos)
    part_h0 = _h3(p_pos)
    ego_speed = float(np.hypot(*e_t50.velocity))
    part_speed = float(np.hypot(*p_t50.velocity))
    if ego_speed < 2.0 or part_speed < 2.0:
        return None  # parked

    # Sample collision points from fan region
    sampled_points, region_constraint = sample_collision_points_from_region(
        ego_p0, ego_h0, ego_speed,
        part_p0, part_h0, part_speed,
        drivable_areas_global if drivable_areas_global else [],
        n_points=n_points, seed=42,
    )
    if not sampled_points:
        return None

    # Encode ego + partner conditions once (these don't change per point —
    # only goal/end_heading change, which we override per-point)
    ego_sample = sample  # already encoded by dataset._load_item
    partner_sample = encode_partner_conditions(
        partner_track, focal_track, scenario, lane_segments,
        sampled_points[0]["collision_point_global"], cfg_data,
        drivable_areas_global=drivable_areas_global,
        partner_end_heading_global=sampled_points[0]["partner_arrival_heading"],
    )
    if partner_sample is None:
        return None

    # Generate per-point batches. With the new centroid-only scheme,
    # sampled_points has exactly 1 entry — generate n_points * n_per_point
    # trajectories for it (so total sample budget is unchanged: 5 * 3 = 15).
    n_trajs_per_collision_point = max(1, n_points * n_per_point)

    ego_trajs_global_all = []
    partner_trajs_global_all = []
    ego_trajs_local_all = []
    partner_trajs_local_all = []
    ego_prior_global_all = []   # prior (Hermite) in global coords, per traj
    partner_prior_global_all = []
    per_point_meta = []
    for sp in sampled_points:
        # Override end_heading for both vehicles
        ego_end_h_global = sp["ego_arrival_heading"]
        # We need to inject this end_heading into ego_sample's prior computation.
        # Build a shallow copy with overridden end_heading (so _build_collision_sample_batch picks it up)
        # Actually, _build_collision_sample_batch recomputes end_heading_m from
        # arrival_h - ref_heading, so we just need to pass the sampled_point dict.
        batch = _build_collision_sample_batch(
            sp, ego_sample, partner_sample, cfg_data, args, device,
            n_per_point=n_trajs_per_collision_point,
            region_constraint=region_constraint,
        )

        # Run DDIM for ego
        ego_result = full_dps_sample(
            diffusion, model, batch["ego"]["batch_conditions"],
            traj_len=n_future,
            n_inference_steps=cfg["diffusion"]["inference_steps"],
            cfg_weight=args.cfg_weight,
            dps_eta=args.dps_eta,
            prior_norm=batch["ego"]["batch_prior"],
            use_cfg=True,
            use_dps=args.dps_eta > 0,
            device=str(device),
            save_intermediates=False,
            dynamic_threshold=args.dynamic_threshold,
            use_chord_frame=batch["ego"]["batch_use_chord"],
            chord_dir=batch["ego"]["batch_chord_dir"],
            spacing=cfg["diffusion"].get("inference_spacing", "linear"),
        )
        # Run DDIM for partner
        partner_result = full_dps_sample(
            diffusion, model, batch["partner"]["batch_conditions"],
            traj_len=n_future,
            n_inference_steps=cfg["diffusion"]["inference_steps"],
            cfg_weight=args.cfg_weight,
            dps_eta=args.dps_eta,
            prior_norm=batch["partner"]["batch_prior"],
            use_cfg=True,
            use_dps=args.dps_eta > 0,
            device=str(device),
            save_intermediates=False,
            dynamic_threshold=args.dynamic_threshold,
            use_chord_frame=batch["partner"]["batch_use_chord"],
            chord_dir=batch["partner"]["batch_chord_dir"],
            spacing=cfg["diffusion"].get("inference_spacing", "linear"),
        )

        # Reconstruct + to global
        ego_result_cpu = ego_result.cpu()
        partner_result_cpu = partner_result.cpu()
        e_ref_pos = batch["ego"]["ref_pos"]
        e_ref_h = batch["ego"]["ref_heading"]
        p_ref_pos = batch["partner"]["ref_pos"]
        p_ref_h = batch["partner"]["ref_heading"]

        for i in range(n_trajs_per_collision_point):
            e_prior_norm = batch["ego"]["priors_for_reconstruct"][i].numpy()
            e_prior_denorm = denormalize(e_prior_norm)
            e_prior_global, _ = local_to_global(
                e_prior_denorm, np.zeros(e_prior_denorm.shape[0]),
                e_ref_pos, e_ref_h)
            ego_prior_global_all.append(e_prior_global)

            e_traj_norm = _reconstruct_trajectory(
                ego_result_cpu[i].unsqueeze(0),
                batch["ego"]["priors_for_reconstruct"][i],
                batch["ego"]["use_chord_flags"][i],
                batch["ego"]["chord_dirs_for_reconstruct"][i],
            ).squeeze(0)
            e_traj_denorm = denormalize(e_traj_norm).numpy()
            e_global, _ = local_to_global(e_traj_denorm, np.zeros(e_traj_denorm.shape[0]),
                                            e_ref_pos, e_ref_h)
            ego_trajs_local_all.append(e_traj_denorm)
            ego_trajs_global_all.append(e_global)

            p_prior_norm = batch["partner"]["priors_for_reconstruct"][i].numpy()
            p_prior_denorm = denormalize(p_prior_norm)
            p_prior_global, _ = local_to_global(
                p_prior_denorm, np.zeros(p_prior_denorm.shape[0]),
                p_ref_pos, p_ref_h)
            partner_prior_global_all.append(p_prior_global)

            p_traj_norm = _reconstruct_trajectory(
                partner_result_cpu[i].unsqueeze(0),
                batch["partner"]["priors_for_reconstruct"][i],
                batch["partner"]["use_chord_flags"][i],
                batch["partner"]["chord_dirs_for_reconstruct"][i],
            ).squeeze(0)
            p_traj_denorm = denormalize(p_traj_norm).numpy()
            p_global, _ = local_to_global(p_traj_denorm, np.zeros(p_traj_denorm.shape[0]),
                                            p_ref_pos, p_ref_h)
            partner_trajs_local_all.append(p_traj_denorm)
            partner_trajs_global_all.append(p_global)

        per_point_meta.append(sp)

    # Snapshot the raw (unsmoothed) trajectories for the smoothed-vs-raw
    # comparison figure. Stored in the return dict as
    # ego_generated_local_raw / ego_generated_global_raw (and partner_*).
    ego_trajs_local_raw = [t.copy() for t in ego_trajs_local_all]
    partner_trajs_local_raw = [t.copy() for t in partner_trajs_local_all]
    ego_trajs_global_raw = [g.copy() for g in ego_trajs_global_all]
    partner_trajs_global_raw = [g.copy() for g in partner_trajs_global_all]

    # Kinematic smoothing/projection with history context.
    # New pipeline: smooth the full 11s (history+future) with the history
    # tail anchored, so the future's first-frame position AND velocity
    # match the observed t=49→t=50 motion — no jump at t=50 in video.
    # Velocity is the primary variable; positions are integrated from
    # velocity, so pos and vel are always self-consistent.
    if args.smoothing or args.kinematic_projection:
        from model.smoothing import (
            smooth_trajectory_batch,
            smooth_and_project_batch_with_history,
        )
        ego_np = np.array(ego_trajs_local_all, dtype=np.float32)
        part_np = np.array(partner_trajs_local_all, dtype=np.float32)
        # History in local meters (denormalized). Both ego and partner have
        # their OWN history (different reference frames).
        # sample["history"] is now 6-dim — slice to positions for smoothing.
        ego_hist_pos = ego_sample.get("history_pos", ego_sample["history"][..., :2])
        part_hist_pos = partner_sample.get("history_pos", partner_sample["history"][..., :2])
        ego_history_m = denormalize(ego_hist_pos).numpy().astype(np.float32)
        part_history_m = denormalize(part_hist_pos).numpy().astype(np.float32)
        if args.kinematic_projection:
            ego_np_s, ego_vel_s = smooth_and_project_batch_with_history(
                ego_np, history=ego_history_m,
                savgol_window=args.smoothing_window,
                savgol_polyorder=args.smoothing_polyorder,
                v_max=args.kinematic_v_max,
                a_max=args.kinematic_a_max,
                jerk_max=args.kinematic_jerk_max,
                kappa_max=args.kinematic_kappa_max,
                kinematic_iters=args.kinematic_iters,
                smooth_method=args.smoothing_method,
                join_strength=args.join_strength,
            )
            part_np_s, part_vel_s = smooth_and_project_batch_with_history(
                part_np, history=part_history_m,
                savgol_window=args.smoothing_window,
                savgol_polyorder=args.smoothing_polyorder,
                v_max=args.kinematic_v_max,
                a_max=args.kinematic_a_max,
                jerk_max=args.kinematic_jerk_max,
                kappa_max=args.kinematic_kappa_max,
                kinematic_iters=args.kinematic_iters,
                smooth_method=args.smoothing_method,
                join_strength=args.join_strength,
            )
        else:
            ego_np_s = smooth_trajectory_batch(
                ego_np,
                window_length=args.smoothing_window,
                polyorder=args.smoothing_polyorder,
                preserve_endpoints=True,
            )
            part_np_s = smooth_trajectory_batch(
                part_np,
                window_length=args.smoothing_window,
                polyorder=args.smoothing_polyorder,
                preserve_endpoints=True,
            )
            ego_vel_s = np.zeros_like(ego_np_s)
            ego_vel_s[:, :-1] = np.diff(ego_np_s, axis=1) / 0.1
            ego_vel_s[:, -1] = ego_vel_s[:, -2]
            part_vel_s = np.zeros_like(part_np_s)
            part_vel_s[:, :-1] = np.diff(part_np_s, axis=1) / 0.1
            part_vel_s[:, -1] = part_vel_s[:, -2]
        ego_trajs_local_all = [ego_np_s[i] for i in range(ego_np_s.shape[0])]
        partner_trajs_local_all = [part_np_s[i] for i in range(part_np_s.shape[0])]
        ego_trajs_global_all = []
        partner_trajs_global_all = []
        for traj_np in ego_trajs_local_all:
            t_g, _ = local_to_global(traj_np, np.zeros(traj_np.shape[0]), e_ref_pos, e_ref_h)
            ego_trajs_global_all.append(t_g)
        for traj_np in partner_trajs_local_all:
            t_g, _ = local_to_global(traj_np, np.zeros(traj_np.shape[0]), p_ref_pos, p_ref_h)
            partner_trajs_global_all.append(t_g)

    # Top-5 selection: rank all (ego_i, partner_j) pairs by min distance,
    # pick 5 best. But the user wants "选出top5" — interpret as the top 5
    # collision points (which we already have only n_points=5 of). So no
    # further pruning; just return all and let viz pick top-3.
    # If user asked for top-5 PAIRS specifically, we'd compute collision_rate
    # per pair. For now, return everything.

    # Compute collision rate
    ego_arr = np.array(ego_trajs_global_all, dtype=np.float32)
    part_arr = np.array(partner_trajs_global_all, dtype=np.float32)
    collision_metrics = compute_collision_rate(
        ego_arr, part_arr, threshold=args.collision_threshold,
    )

    # Partner GT for viz
    partner_future_local = partner_sample["future_local"]
    partner_gt_global, _ = local_to_global(
        partner_future_local, np.zeros(partner_future_local.shape[0]),
        p_ref_pos, p_ref_h,
    )

    return {
        "ego_generated_global": ego_trajs_global_all,
        "ego_generated_local": ego_trajs_local_all,
        "partner_generated_global": partner_trajs_global_all,
        "partner_generated_local": partner_trajs_local_all,
        "ego_generated_local_raw": ego_trajs_local_raw,
        "ego_generated_global_raw": ego_trajs_global_raw,
        "partner_generated_local_raw": partner_trajs_local_raw,
        "partner_generated_global_raw": partner_trajs_global_raw,
        "ego_prior_global": ego_prior_global_all,
        "partner_prior_global": partner_prior_global_all,
        "partner_gt_global": partner_gt_global,
        "partner_ref_pos": p_ref_pos,
        "partner_ref_heading": p_ref_h,
        "collision_point_global": sampled_points[0]["collision_point_global"],
        "sampled_collision_points": sampled_points,
        "collision_rate": collision_metrics["collision_rate"],
        "collision_mean_min_dist": collision_metrics["mean_min_dist"],
        "collision_rate_strict": collision_metrics["collision_rate_strict"],
        "partner_min_dist_gt": partner_info["min_dist"],
        "partner_collision_timestep": partner_info["collision_timestep"],
        "partner_track_id": partner_info["track"].track_id,
        "scenario": scenario,
        "static_map": static_map,
        "focal_track_id": focal_track.track_id,
        "region_area_m2": sampled_points[0].get("region_area_m2", 0.0),
        "t_horizon_used": sampled_points[0].get("t_horizon_used", 6.0),
        "collision_region": region_constraint,
    }


def generate_partner_trajectories(
    idx: int,
    sample: Dict,
    dataset,
    model: TFCrossDenoiser,
    diffusion: DiffusionProcess,
    device: str,
    cfg: Dict,
    args,
) -> Optional[Dict]:
    """Generate collision partner trajectories and compute collision rate.

    Returns a dict with partner trajectory data and collision metrics,
    or None if no suitable partner is found.
    """
    # Get physical index
    phys_idx = dataset._valid_indices[idx] if dataset._valid_indices is not None else idx
    parquet_path = dataset.scenario_files[phys_idx]

    # Load raw scenario
    scenario = load_argoverse_scenario_parquet(parquet_path)
    focal_track = _find_focal_track(scenario)
    if focal_track is None:
        return None

    # Select collision partner
    partner_info = select_collision_partner(
        scenario, focal_track, n_future=cfg["data"]["n_future"]
    )
    if partner_info is None:
        return None

    partner_track = partner_info["track"]
    collision_point_global = partner_info["collision_point_global"]

    # Get lane_segments: prefer from scene_data, fallback to recomputing
    lane_segments = sample.get("scene_data", {}).get("lane_segments", {})
    if not lane_segments:
        static_map = _load_static_map(scenario, dataset.map_dir, parquet_path)
        lane_segments = _build_lane_segments(static_map)
    else:
        # Still need static_map for visualization
        static_map = _load_static_map(scenario, dataset.map_dir, parquet_path)

    # Get drivable_areas_global from scene_data; if missing, recompute from static_map
    drivable_areas_global = sample.get("scene_data", {}).get("drivable_areas_global", None)
    if not drivable_areas_global and static_map is not None:
        drivable_areas_global = []
        for da_id, da in static_map.vector_drivable_areas.items():
            try:
                da_pts = da.xyz[:, :2]
                if len(da_pts) >= 3:
                    drivable_areas_global.append(da_pts.astype(np.float32))
            except Exception:
                pass

    # Encode partner conditions — partner uses its own end heading
    cfg_data = cfg["data"]
    partner_sample = encode_partner_conditions(
        partner_track, focal_track, scenario, lane_segments,
        collision_point_global, cfg_data,
        drivable_areas_global=drivable_areas_global,
    )
    if partner_sample is None:
        return None

    # Build conditions dict for the model
    conditions = {
        "goal": partner_sample["goal"].unsqueeze(0).to(device),
        "map_tokens": partner_sample["map_tokens"].unsqueeze(0).to(device),
        "map_mask": partner_sample["map_mask"].unsqueeze(0).to(device),
        "neighbor_tokens": partner_sample["neighbor_tokens"].unsqueeze(0).to(device),
        "neighbor_mask": partner_sample["neighbor_mask"].unsqueeze(0).to(device),
        "history": partner_sample["history"].unsqueeze(0).to(device),
    }

    prior_norm = partner_sample["prior"].unsqueeze(0).to(device)
    prior_cpu = partner_sample["prior"]
    use_chord_frame = partner_sample.get("use_chord_frame")
    chord_dir_cpu = partner_sample.get("chord_dir")

    # Batch-style generation: precompute all perturbed goals/priors,
    # then run one batched DDIM call (much faster than per-sample loop)

    # Compute goal perturbation params from partner's conditions
    goal_m = denormalize(partner_sample["goal"].reshape(1, 2)).numpy().flatten()
    partner_hist_pos = partner_sample.get("history_pos", partner_sample["history"][..., :2])
    history_end_m = denormalize(partner_hist_pos[-1:].reshape(1, 2)).numpy().flatten()
    start_heading_m = partner_sample["start_heading"].item()
    end_heading_m = partner_sample["end_heading"].item()
    prior_type = cfg_data.get("prior_type", "hermite")

    chord_len = max(np.linalg.norm(goal_m - history_end_m), 1e-6)
    goal_sigma_lon = args.goal_sigma_lon if args.goal_sigma_lon > 0 else max(chord_len * 0.10, 0.5)
    goal_sigma_lat = args.goal_sigma_lat if args.goal_sigma_lat > 0 else max(chord_len * 0.04, 0.3)
    gt_endpoint_heading = end_heading_m

    # Drivable-area paths in partner-local frame (for goal/prior sampling constraints)
    da_local_list = partner_sample.get("drivable_areas_local", None)
    da_paths = build_paths(da_local_list) if da_local_list else []

    def _compute_perturbed_prior(history_end, g_m):
        if prior_type == "hermite":
            return compute_hermite_prior(
                history_end, g_m, start_heading_m, end_heading_m,
                cfg_data["n_future"]
            )
        return compute_prior(history_end, g_m, cfg_data["n_future"])

    # 1) Precompute all perturbed goals and priors
    perturbed_goals_m = []
    perturbed_priors_m = []
    priors_for_reconstruct = []
    chord_dirs_for_reconstruct = []
    use_chord_flags = []

    for s_idx in range(args.n_samples):
        # Drivable-constrained anisotropic sampling:
        #   - Goal endpoint must lie inside a drivable area.
        #   - Prior trajectory must have every waypoint inside drivable areas.
        perturbed_goal_m, perturbed_prior_m = sample_goal_and_prior_in_drivable(
            goal_m=goal_m,
            history_end_m=history_end_m,
            goal_sigma_lon=goal_sigma_lon,
            goal_sigma_lat=goal_sigma_lat,
            gt_endpoint_heading=gt_endpoint_heading,
            paths=da_paths,
            compute_prior_fn=_compute_perturbed_prior,
            max_attempts=100,
        )

        perturbed_goals_m.append(perturbed_goal_m)
        perturbed_priors_m.append(perturbed_prior_m)
        priors_for_reconstruct.append(
            torch.tensor(normalize(perturbed_prior_m), dtype=torch.float32)
        )

        if use_chord_frame is not None and use_chord_frame.item() == 1.0:
            pert_chord_dir, _, pert_chord_valid = compute_chord_dir(history_end_m, perturbed_goal_m)
            if pert_chord_valid:
                chord_dirs_for_reconstruct.append(torch.tensor(pert_chord_dir, dtype=torch.float32))
                use_chord_flags.append(torch.tensor(1.0, dtype=torch.float32))
            else:
                chord_dirs_for_reconstruct.append(torch.zeros(2, dtype=torch.float32))
                use_chord_flags.append(torch.tensor(0.0, dtype=torch.float32))
        else:
            chord_dirs_for_reconstruct.append(chord_dir_cpu if chord_dir_cpu is not None else torch.zeros(2))
            use_chord_flags.append(use_chord_frame if use_chord_frame is not None else torch.tensor(0.0))

    # 2) Batch all conditions into single tensors
    N = args.n_samples
    batch_goal = torch.stack([
        torch.tensor(normalize(g.reshape(1, 2)), dtype=torch.float32).flatten()
        for g in perturbed_goals_m
    ]).to(device)  # (N, 2)
    batch_prior = torch.stack([
        torch.tensor(normalize(p), dtype=torch.float32)
        for p in perturbed_priors_m
    ]).to(device)  # (N, T, 2)

    batch_conditions = {
        "goal": batch_goal,  # (N, 2)
        "map_tokens": partner_sample["map_tokens"].unsqueeze(0).expand(N, -1, -1).to(device),
        "map_mask": partner_sample["map_mask"].unsqueeze(0).expand(N, -1).to(device),
        "neighbor_tokens": partner_sample["neighbor_tokens"].unsqueeze(0).expand(N, -1, -1, -1).to(device),
        "neighbor_mask": partner_sample["neighbor_mask"].unsqueeze(0).expand(N, -1).to(device),
        "history": partner_sample["history"].unsqueeze(0).expand(N, -1, -1).to(device),
    }

    # 3) Harmonize chord-frame params for batch
    # Use the most common (usually all same) chord flag
    use_chord_val = use_chord_flags[0] if use_chord_flags else torch.tensor(0.0)
    if use_chord_val.item() == 1.0:
        # Average chord_dir across samples (they're very similar)
        batch_chord_dir = torch.stack(chord_dirs_for_reconstruct).float().mean(0).unsqueeze(0).to(device)
        batch_use_chord = use_chord_val.unsqueeze(0).to(device)
    else:
        batch_chord_dir = None
        batch_use_chord = None

    # 4) Single batched DDIM call
    batch_result = full_dps_sample(
        diffusion, model, batch_conditions,
        traj_len=cfg_data["n_future"],
        n_inference_steps=cfg["diffusion"]["inference_steps"],
        cfg_weight=args.cfg_weight,
        dps_eta=args.dps_eta,
        prior_norm=batch_prior,
        use_cfg=True,
        use_dps=args.dps_eta > 0,
        device=str(device),
        save_intermediates=False,
        dynamic_threshold=args.dynamic_threshold,
        use_chord_frame=batch_use_chord,
        chord_dir=batch_chord_dir,
        spacing=cfg["diffusion"].get("inference_spacing", "linear"),
    )  # (N, T, 2)

    # 5) Reconstruct per-sample trajectories (move to CPU first)
    batch_result_cpu = batch_result.cpu()
    partner_ref_pos = partner_sample["ref_pos"].numpy()
    partner_ref_heading = partner_sample["ref_heading"].item()

    partner_generated_global = []
    partner_gen_local_list = []
    for i in range(N):
        traj_norm = _reconstruct_trajectory(
            batch_result_cpu[i].unsqueeze(0), priors_for_reconstruct[i],
            use_chord_flags[i], chord_dirs_for_reconstruct[i],
        ).squeeze(0)
        traj_denorm = denormalize(traj_norm)
        traj_np = traj_denorm.numpy()
        traj_global, _ = local_to_global(traj_np, np.zeros(traj_np.shape[0]),
                                          partner_ref_pos, partner_ref_heading)
        partner_generated_global.append(traj_global)
        partner_gen_local_list.append(traj_np)

    # Partner GT trajectory
    partner_future_local = partner_sample["future_local"]
    partner_gt_global, _ = local_to_global(
        partner_future_local, np.zeros(partner_future_local.shape[0]),
        partner_ref_pos, partner_ref_heading
    )

    # Kinematic projection for partner if requested
    if args.smoothing or args.kinematic_projection:
        from model.smoothing import smooth_trajectory_batch, smooth_and_project_batch
        gen_np = np.array(partner_gen_local_list, dtype=np.float32)
        if args.kinematic_projection:
            gen_np_smoothed = smooth_and_project_batch(
                gen_np,
                savgol_window=args.smoothing_window,
                savgol_polyorder=args.smoothing_polyorder,
                v_max=args.kinematic_v_max,
                a_max=args.kinematic_a_max,
                jerk_max=args.kinematic_jerk_max,
                kappa_max=args.kinematic_kappa_max,
                kinematic_iters=args.kinematic_iters,
            )
        else:
            gen_np_smoothed = smooth_trajectory_batch(
                gen_np,
                window_length=args.smoothing_window,
                polyorder=args.smoothing_polyorder,
                preserve_endpoints=True,
            )
        partner_gen_local_list = [gen_np_smoothed[i] for i in range(gen_np_smoothed.shape[0])]
        partner_generated_global = []
        for traj_np in partner_gen_local_list:
            traj_global, _ = local_to_global(traj_np, np.zeros(traj_np.shape[0]),
                                              partner_ref_pos, partner_ref_heading)
            partner_generated_global.append(traj_global)

    return {
        "partner_generated_global": partner_generated_global,
        "partner_generated_local": partner_gen_local_list,
        "partner_gt_global": partner_gt_global,
        "partner_ref_pos": partner_ref_pos,
        "partner_ref_heading": partner_ref_heading,
        "collision_point_global": collision_point_global,
        "partner_min_dist_gt": partner_info["min_dist"],
        "partner_collision_timestep": partner_info["collision_timestep"],
        "partner_track_id": partner_info["track"].track_id,
        "scenario": scenario,
        "static_map": static_map,
        "focal_track_id": focal_track.track_id,
    }
