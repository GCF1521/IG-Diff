"""Diagnose: at which phase do we lose candidates?"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.av2_dataset import Argoverse2Dataset
from data.normalization import (
    denormalize, compute_hermite_prior, compute_lane_prior,
    _LANE_MAX_ENDPOINT_OFFSET, _LANE_MAX_START_DIST,
    _LANE_MAX_ANGLE,
    _build_lane_graph, _lane_graph_search, _assemble_path,
    _push_away_from_boundaries, _BOUNDARY_PUSH_DISTANCE,
    _LANE_DEGRADE_BOUNDARY_FRAC, _LANE_DEGRADE_MAX_LATERAL_ACCEL,
    _LANE_DEGRADE_MIN_DELTA_H, _compute_curvature,
    _PARKING_SPEED_THRESH, _PARKING_MIN_CONSECUTIVE_STEPS,
)
from data.coordinate_utils import global_to_local
from data.map_encoding import interpolate_polyline


def main():
    data_dir = os.environ.get("DATA_DIR", "av2_dataset_1k/train/")
    dataset = Argoverse2Dataset(
        data_dir=data_dir,
        n_future=60, n_history=20, n_lanes=24, lane_feat_dim=46,
        n_neighbors=6, split="eval",
        return_scene_data=True, prior_type="hermite", residual_frame="chord",
    )

    n_scan = min(len(dataset), 300)
    phase1_pass = 0  # Has CLs with heading match near start
    phase2_pass = 0  # Lane graph search found at least one path
    phase3_pass = 0  # Assembled path endpoint within offset limit
    phase4_pass = 0  # Boundary push OK (violation frac < threshold)
    phase5_pass = 0  # No degradation triggered
    fallback_total = 0
    degrade_level1 = 0  # Blended with Hermite
    degrade_level2 = 0  # Full fallback to Hermite
    parking_count = 0

    for idx in range(n_scan):
        sample = dataset[idx]
        hist_pos = sample.get("history_pos", sample["history"][..., :2])
        history_m = denormalize(hist_pos.numpy())
        history_end = history_m[-1]
        goal_m = denormalize(sample["goal"].numpy().reshape(1, 2)).flatten()
        start_h = sample["start_heading"].item()
        end_h = sample["end_heading"].item()

        # Parking detection
        is_parking_flag = sample.get("is_parking", None)
        if is_parking_flag is not None and is_parking_flag.item() > 0.5:
            parking_count += 1

        scene_data = sample.get("scene_data", {})
        lane_segments = scene_data.get("lane_segments", {})
        lane_boundaries_local = scene_data.get("lane_boundaries_local", [])
        ref_pos = sample["ref_pos"].numpy()
        ref_heading = sample["ref_heading"].item()

        # Phase 1: transform centerlines and check heading match
        centerlines_local = {}
        for lane_id, lane in lane_segments.items():
            cl = lane.get("centerline")
            if cl is None or len(cl) < 2:
                continue
            cl_pts = np.array([(p["x"], p["y"]) for p in cl], dtype=np.float64)
            cl_local, _ = global_to_local(cl_pts, np.zeros(len(cl_pts)), ref_pos, ref_heading)
            centerlines_local[lane_id] = cl_local.astype(np.float32)

        if not centerlines_local:
            fallback_total += 1
            continue

        has_cl_match = any(
            abs(start_h - np.arctan2((cl[min(1,len(cl)-1)] - cl[0])[1],
                                      (cl[min(1,len(cl)-1)] - cl[0])[0])) < _LANE_MAX_ANGLE
            or abs(start_h - np.arctan2((cl[::-1][min(1,len(cl)-1)] - cl[::-1][0])[1],
                                         (cl[::-1][min(1,len(cl)-1)] - cl[::-1][0])[0])) < _LANE_MAX_ANGLE
            for cl in centerlines_local.values() if len(cl) >= 2
        )
        if has_cl_match:
            phase1_pass += 1

        # Phase 2: lane graph search
        adj = _build_lane_graph(lane_segments)
        start_lane_ids = []
        for lane_id, cl in centerlines_local.items():
            start_dists = np.linalg.norm(cl - history_end, axis=-1)
            if start_dists.min() > _LANE_MAX_START_DIST:
                continue
            for cl_try in [cl, cl[::-1]]:
                if len(cl_try) < 2:
                    continue
                tang0 = cl_try[1] - cl_try[0]
                if np.linalg.norm(tang0) <= 0.3:
                    continue
                h0 = start_h - np.arctan2(tang0[1], tang0[0])
                h0 = h0 - 2 * np.pi * np.round(h0 / (2 * np.pi))
                if abs(h0) <= _LANE_MAX_ANGLE:
                    start_lane_ids.append(lane_id)
                    break

        if not start_lane_ids:
            fallback_total += 1
            continue

        paths = _lane_graph_search(start_lane_ids, goal_m, lane_segments, ref_pos, ref_heading, adj)
        if not paths:
            fallback_total += 1
            continue
        phase2_pass += 1

        # Phase 3: assemble and check endpoint offset
        best_path = None
        best_score = float("inf")
        for path_ids in paths:
            assembled = _assemble_path(path_ids, lane_segments, ref_pos, ref_heading)
            if assembled is None or len(assembled) < 2:
                continue
            start_d = np.linalg.norm(assembled[0] - history_end)
            if start_d > _LANE_MAX_START_DIST:
                continue
            end_d = np.linalg.norm(assembled[-1] - goal_m)
            score = start_d + end_d
            if score < best_score:
                best_score = score
                best_path = assembled

        if best_path is not None:
            resampled = interpolate_polyline(best_path.astype(np.float32), 60)
            resampled = resampled + (history_end - resampled[0])
            endpoint_offset = np.linalg.norm(goal_m - resampled[-1])
            if endpoint_offset <= _LANE_MAX_ENDPOINT_OFFSET:
                phase3_pass += 1
            else:
                fallback_total += 1
                continue
        else:
            fallback_total += 1
            continue

        # Phase 4: boundary push check
        drivable_areas_local = scene_data.get("drivable_areas_local", [])
        lane_m = compute_lane_prior(
            history_end, goal_m, lane_segments, ref_pos, ref_heading,
            start_h, end_h, 60,
            lane_boundaries_local=lane_boundaries_local,
            drivable_areas_local=drivable_areas_local,
        )
        hermite_m = compute_hermite_prior(history_end, goal_m, start_h, end_h, 60)
        is_fallback = np.allclose(lane_m, hermite_m, atol=0.01)

        if is_fallback:
            fallback_total += 1
            continue

        _, boundary_violation_frac = _push_away_from_boundaries(
            lane_m, lane_boundaries_local, _BOUNDARY_PUSH_DISTANCE)
        if boundary_violation_frac < _LANE_DEGRADE_BOUNDARY_FRAC:
            phase4_pass += 1

        # Phase 5: degradation check
        delta_h = end_h - start_h
        delta_h = delta_h - 2 * np.pi * np.round(delta_h / (2 * np.pi))
        chord_len = np.linalg.norm(goal_m - history_end)
        v_avg = chord_len / (60 * 0.1)
        kappa_lane = _compute_curvature(lane_m)
        max_lateral_accel = v_avg * v_avg * np.max(np.abs(kappa_lane))

        cond_boundary = boundary_violation_frac > _LANE_DEGRADE_BOUNDARY_FRAC
        cond_kinematic = max_lateral_accel > _LANE_DEGRADE_MAX_LATERAL_ACCEL

        if cond_boundary and abs(delta_h) < _LANE_DEGRADE_MIN_DELTA_H:
            degrade_level2 += 1
        elif cond_kinematic and abs(delta_h) < _LANE_DEGRADE_MIN_DELTA_H:
            degrade_level2 += 1
        elif cond_boundary or cond_kinematic:
            degrade_level1 += 1
        else:
            phase5_pass += 1

    print(f"Phase 1 (heading match near start): {phase1_pass}/{n_scan}")
    print(f"Phase 2 (lane graph found path):    {phase2_pass}/{n_scan}")
    print(f"Phase 3 (endpoint offset OK):       {phase3_pass}/{n_scan}")
    print(f"Phase 4 (boundary violation < {int(_LANE_DEGRADE_BOUNDARY_FRAC*100)}%):   {phase4_pass}/{n_scan}")
    print(f"Phase 5 (no degradation):           {phase5_pass}/{n_scan}")
    print(f"Degradation Level 1 (blended):      {degrade_level1}/{n_scan}")
    print(f"Degradation Level 2 (fallback):     {degrade_level2}/{n_scan}")
    print(f"Would fallback:                     {fallback_total}/{n_scan} = {fallback_total/n_scan*100:.1f}%")
    print(f"Parking scenarios:                  {parking_count}/{n_scan} = {parking_count/n_scan*100:.1f}%")


if __name__ == "__main__":
    main()
