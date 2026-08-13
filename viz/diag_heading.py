"""Deeper diagnosis: why are centerlines with nearby points filtered?"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.av2_dataset import Argoverse2Dataset
from data.normalization import denormalize, compute_hermite_prior
from data.coordinate_utils import global_to_local


def main():
    data_dir = os.environ.get("DATA_DIR", "av2_dataset_1k/train/")
    dataset = Argoverse2Dataset(
        data_dir=data_dir,
        n_future=60, n_history=20, n_lanes=24, lane_feat_dim=46,
        n_neighbors=6, split="eval",
        return_scene_data=True, prior_type="hermite", residual_frame="chord",
    )

    n_scan = min(len(dataset), 300)
    # Collect heading diffs for nearby CLs
    heading_diffs = []

    for idx in range(n_scan):
        sample = dataset[idx]
        history_m = denormalize(sample["history"].numpy())
        history_end = history_m[-1]
        goal_m = denormalize(sample["goal"].numpy().reshape(1, 2)).flatten()
        start_h = sample["start_heading"].item()

        scene_data = sample.get("scene_data", {})
        lane_segments = scene_data.get("lane_segments", {})
        ref_pos = sample["ref_pos"].numpy()
        ref_heading = sample["ref_heading"].item()

        best_nearby_heading_diff = None

        for lane_id, lane in lane_segments.items():
            cl = lane.get("centerline")
            if cl is None or len(cl) < 2:
                continue
            cl_pts = np.array([(p["x"], p["y"]) for p in cl], dtype=np.float64)
            cl_local, _ = global_to_local(cl_pts, np.zeros(len(cl_pts)), ref_pos, ref_heading)
            cl_local = cl_local.astype(np.float32)

            min_d = np.linalg.norm(cl_local - history_end, axis=-1).min()
            if min_d > 10.0:
                continue

            # Check both forward and reversed
            for cl_try in [cl_local, cl_local[::-1]]:
                targ = cl_try[1] - cl_try[0]
                tlen = np.linalg.norm(targ)
                if tlen < 0.3:
                    continue
                cl_heading = np.arctan2(targ[1], targ[0])
                h_diff = start_h - cl_heading
                h_diff = h_diff - 2 * np.pi * np.round(h_diff / (2 * np.pi))
                abs_diff = abs(h_diff)

                if best_nearby_heading_diff is None or abs_diff < best_nearby_heading_diff:
                    best_nearby_heading_diff = abs_diff

        if best_nearby_heading_diff is not None:
            heading_diffs.append(best_nearby_heading_diff)

    heading_diffs = np.array(heading_diffs)
    print(f"Scenarios with nearby CLs: {len(heading_diffs)}/{n_scan}")
    print(f"Heading diff (deg) to best nearby CL direction:")
    print(f"  mean: {np.degrees(np.mean(heading_diffs)):.1f}")
    print(f"  median: {np.degrees(np.median(heading_diffs)):.1f}")
    print(f"  <30°: {(heading_diffs < np.radians(30)).sum()} ({(heading_diffs < np.radians(30)).mean()*100:.1f}%)")
    print(f"  <45°: {(heading_diffs < np.radians(45)).sum()} ({(heading_diffs < np.radians(45)).mean()*100:.1f}%)")
    print(f"  <90°: {(heading_diffs < np.radians(90)).sum()} ({(heading_diffs < np.radians(90)).mean()*100:.1f}%)")
    print(f"  >=90°: {(heading_diffs >= np.radians(90)).sum()} ({(heading_diffs >= np.radians(90)).mean()*100:.1f}%)")

    # Also check: what if we relax heading threshold to 90 degrees?
    # How many extra scenarios would get a lane prior?


if __name__ == "__main__":
    main()
