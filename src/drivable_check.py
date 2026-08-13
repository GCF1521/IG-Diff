"""Drivable-area containment checks for goal/prior sampling.

Shared by inference.py (ego) and collision_partner.py (partner) to enforce:
  1. Sampled goal endpoint must lie inside a drivable area.
  2. Sampled prior trajectory must have every waypoint inside drivable areas.
"""

from typing import List, Optional

import numpy as np
from matplotlib.path import Path


def build_paths(drivable_areas: List[np.ndarray]) -> List[Path]:
    """Build matplotlib Path list from polygon arrays.

    Args:
        drivable_areas: list of (P, 2) polygon arrays (local or global coords).

    Returns:
        List of Path objects (empty polygons filtered out).
    """
    paths = []
    if not drivable_areas:
        return paths
    for da in drivable_areas:
        da_arr = np.asarray(da)
        if len(da_arr) >= 3:
            paths.append(Path(da_arr))
    return paths


def point_in_drivable(pt: np.ndarray, paths: List[Path]) -> bool:
    """Return True if pt is inside any drivable-area path."""
    if not paths:
        return True  # No map → allow
    return any(p.contains_point(pt) for p in paths)


def all_points_in_drivable(pts: np.ndarray, paths: List[Path]) -> bool:
    """Return True if every row of pts is inside at least one drivable-area path.

    Args:
        pts: (T, 2) array of trajectory waypoints.
        paths: list of Path objects.

    Returns:
        True if all waypoints are inside drivable areas (or paths is empty).
    """
    if not paths:
        return True
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    for pt in pts:
        if not any(p.contains_point(pt) for p in paths):
            return False
    return True


def sample_goal_in_drivable(
    goal_m: np.ndarray,
    history_end_m: np.ndarray,
    goal_sigma_lon: float,
    goal_sigma_lat: float,
    gt_endpoint_heading: float,
    paths: List[Path],
    max_attempts: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample a goal perturbation whose endpoint lies inside drivable areas.

    Perturbs goal_m anisotropically along the endpoint heading. Returns a
    drivable point; falls back to the original goal_m if no acceptable
    sample is found within max_attempts.
    """
    if rng is None:
        rng = np.random.default_rng()

    cos_h, sin_h = np.cos(gt_endpoint_heading), np.sin(gt_endpoint_heading)
    for _ in range(max_attempts):
        dx_lon = rng.standard_normal() * goal_sigma_lon
        dx_lat = rng.standard_normal() * goal_sigma_lat
        cand = goal_m.copy()
        cand[0] += dx_lon * cos_h - dx_lat * sin_h
        cand[1] += dx_lon * sin_h + dx_lat * cos_h
        if point_in_drivable(cand, paths):
            return cand
    # Fail-soft: return the original GT goal (assumed drivable)
    return goal_m.copy()


def sample_goal_and_prior_in_drivable(
    goal_m: np.ndarray,
    history_end_m: np.ndarray,
    goal_sigma_lon: float,
    goal_sigma_lat: float,
    gt_endpoint_heading: float,
    paths: List[Path],
    compute_prior_fn,
    max_attempts: int = 50,
    rng: Optional[np.random.Generator] = None,
):
    """Sample (goal, prior) such that goal is drivable AND every prior waypoint is drivable.

    Args:
        compute_prior_fn: callable(history_end_m, goal_m) -> prior (T, 2) in meters.

    Returns:
        (perturbed_goal_m, perturbed_prior_m). Falls back to (goal_m, original_prior)
        if no acceptable sample is found.
    """
    if rng is None:
        rng = np.random.default_rng()

    cos_h, sin_h = np.cos(gt_endpoint_heading), np.sin(gt_endpoint_heading)
    fallback_goal = goal_m.copy()
    fallback_prior = compute_prior_fn(history_end_m, fallback_goal)

    if not paths:
        # No map → no constraint, just do one anisotropic sample
        dx_lon = rng.standard_normal() * goal_sigma_lon
        dx_lat = rng.standard_normal() * goal_sigma_lat
        cand = goal_m.copy()
        cand[0] += dx_lon * cos_h - dx_lat * sin_h
        cand[1] += dx_lon * sin_h + dx_lat * cos_h
        return cand, compute_prior_fn(history_end_m, cand)

    # Phase 1: try with original sigma (max_attempts)
    # Phase 2: if all fail, expand sigma 2x and try again (max_attempts)
    # Phase 3: final fallback — return GT goal/prior only if GT prior is fully
    #          drivable; otherwise return the best-effort sample (least off-road).
    phases = [
        (goal_sigma_lon, goal_sigma_lat, max_attempts),
        (goal_sigma_lon * 2.0, goal_sigma_lat * 2.0, max_attempts),
    ]

    best_effort_goal = None
    best_effort_prior = None
    best_effort_off_frac = float("inf")

    for sigma_lon, sigma_lat, attempts in phases:
        for _ in range(attempts):
            dx_lon = rng.standard_normal() * sigma_lon
            dx_lat = rng.standard_normal() * sigma_lat
            cand = goal_m.copy()
            cand[0] += dx_lon * cos_h - dx_lat * sin_h
            cand[1] += dx_lon * sin_h + dx_lat * cos_h
            if not point_in_drivable(cand, paths):
                continue
            prior = compute_prior_fn(history_end_m, cand)
            if all_points_in_drivable(prior, paths):
                return cand, prior
            # Track best-effort (lowest off-road fraction) for fallback
            outside = sum(1 for pt in prior if not any(p.contains_point(pt) for p in paths))
            off_frac = outside / len(prior)
            if off_frac < best_effort_off_frac:
                best_effort_off_frac = off_frac
                best_effort_goal = cand
                best_effort_prior = prior

    # Final fallback: prefer GT if its prior is fully drivable
    if all_points_in_drivable(fallback_prior, paths):
        return fallback_goal, fallback_prior
    # Otherwise use the best-effort sample (least off-road)
    if best_effort_goal is not None:
        return best_effort_goal, best_effort_prior
    # Last resort: GT goal/prior even if not drivable
    return fallback_goal, fallback_prior
