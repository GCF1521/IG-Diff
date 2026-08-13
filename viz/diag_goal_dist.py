"""Diagnose: how far is the nearest centerline point to goal for fallback cases?"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.av2_dataset import Argoverse2Dataset
from data.normalization import denormalize, compute_hermite_prior, compute_lane_prior
from data.coordinate_utils import global_to_local


def main():
    data_dir = os.environ.get("DATA_DIR", "av2_dataset_1k/train/")
    dataset = Argoverse2Dataset(
        data_dir=data_dir,
        n_future=60, n_history=20, n_lanes=24, lane_feat_dim=46,
        n_neighbors=6, split="eval",
        return_scene_data=True, prior_type="hermite", residual_frame="chord",
    )

    n_scan = min(len(dataset), 500)
    # For each fallback, measure distance from goal to nearest centerline point
    goal_to_cl_dists = []

    for idx in range(n_scan):
        sample = dataset[idx]
        history_m = denormalize(sample["history"].numpy())
        history_end = history_m[-1]
        goal_m = denormalize(sample["goal"].numpy().reshape(1, 2)).flatten()
        start_h = sample["start_heading"].item()
        end_h = sample["end_heading"].item()
        ref_pos = sample["ref_pos"].numpy()
        ref_heading = sample["ref_heading"].item()
        hermite_m = compute_hermite_prior(history_end, goal_m, start_h, end_h, 60)

        scene_data = sample.get("scene_data", {})
        lane_segments = scene_data.get("lane_segments", {})
        lane_m = compute_lane_prior(
            history_end, goal_m, lane_segments,
            ref_pos, ref_heading, start_h, end_h, 60,
        )

        is_fallback = np.allclose(lane_m, hermite_m, atol=0.01)
        if not is_fallback:
            continue

        # Measure distance from goal to any centerline point
        min_goal_cl = float("inf")
        for lane_id, lane in lane_segments.items():
            cl = lane.get("centerline")
            if cl is None or len(cl) < 2:
                continue
            cl_pts = np.array([(p["x"], p["y"]) for p in cl], dtype=np.float64)
            cl_local, _ = global_to_local(cl_pts, np.zeros(len(cl_pts)), ref_pos, ref_heading)
            min_goal_cl = min(min_goal_cl, np.linalg.norm(cl_local - goal_m, axis=-1).min())

        goal_to_cl_dists.append(min_goal_cl)

    if not goal_to_cl_dists:
        print("No fallback cases found")
        return

    dists = np.array(goal_to_cl_dists)
    print(f"Fallback cases: {len(dists)}/{n_scan}")
    print(f"Goal to nearest CL point (m):")
    print(f"  mean: {dists.mean():.1f}")
    print(f"  median: {np.median(dists):.1f}")
    print(f"  <5m: {(dists < 5).sum()} ({(dists < 5).mean()*100:.1f}%)")
    print(f"  <10m: {(dists < 10).sum()} ({(dists < 10).mean()*100:.1f}%)")
    print(f"  <20m: {(dists < 20).sum()} ({(dists < 20).mean()*100:.1f}%)")
    print(f"  >=20m: {(dists >= 20).sum()} ({(dists >= 20).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
