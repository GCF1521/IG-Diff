"""Diagnose fallback reasons for lane prior."""
import sys, os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.av2_dataset import Argoverse2Dataset
from data.normalization import (
    denormalize, compute_hermite_prior, compute_lane_prior,
    unpack_chord, denormalize_residual_chord, chord_frame_to_residual,
    denormalize_residual,
)


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


def main():
    data_dir = os.environ.get("DATA_DIR", "av2_dataset_1k/train/")
    dataset = Argoverse2Dataset(
        data_dir=data_dir,
        n_future=60, n_history=20, n_lanes=24, lane_feat_dim=46,
        n_neighbors=6, split="eval",
        return_scene_data=True, prior_type="hermite", residual_frame="chord",
    )

    n_scan = min(len(dataset), 1500)
    reasons = Counter()
    n_fallback = 0
    n_success = 0

    for idx in range(n_scan):
        sample = dataset[idx]
        history_m = denormalize(sample["history"].numpy())
        history_end = history_m[-1]
        goal_m = denormalize(sample["goal"].numpy().reshape(1, 2)).flatten()
        start_h = sample["start_heading"].item()
        end_h = sample["end_heading"].item()
        hermite_m = compute_hermite_prior(history_end, goal_m, start_h, end_h, 60)

        scene_data = sample.get("scene_data", {})
        lane_segments = scene_data.get("lane_segments", {})
        lane_m = compute_lane_prior(
            history_end, goal_m, lane_segments,
            sample["ref_pos"].numpy(), sample["ref_heading"].item(),
            start_h, end_h, 60,
        )

        is_fallback = np.allclose(lane_m, hermite_m, atol=0.01)
        if is_fallback:
            n_fallback += 1
            # Diagnose why
            if not lane_segments:
                reasons["no_lane_segments"] += 1
                continue

            # Check if any centerline points are near history_end
            from data.coordinate_utils import global_to_local
            ref_pos = sample["ref_pos"].numpy()
            ref_heading = sample["ref_heading"].item()
            has_nearby_cl = False
            any_cl = False
            for lane_id, lane in lane_segments.items():
                cl = lane.get("centerline")
                if cl is None or len(cl) < 2:
                    continue
                any_cl = True
                cl_pts = np.array([(p["x"], p["y"]) for p in cl], dtype=np.float64)
                cl_local, _ = global_to_local(cl_pts, np.zeros(len(cl_pts)), ref_pos, ref_heading)
                min_d = np.linalg.norm(cl_local - history_end, axis=-1).min()
                if min_d < 10.0:
                    has_nearby_cl = True
                    break

            if not any_cl:
                reasons["no_centerlines"] += 1
            elif not has_nearby_cl:
                reasons["no_nearby_centerline"] += 1
            else:
                reasons["centerline_found_but_filtered"] += 1
        else:
            n_success += 1

    print(f"Fallback: {n_fallback}/{n_scan} = {n_fallback/n_scan*100:.1f}%")
    print(f"Success+: {n_success}/{n_scan} = {n_success/n_scan*100:.1f}%")
    print("\nFallback reasons:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count} ({count/n_fallback*100:.1f}%)")


if __name__ == "__main__":
    main()
