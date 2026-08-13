"""Statistics for lane prior: fallback rate and RMSE comparison."""
import sys, os
import numpy as np

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
        n_future=60,
        n_history=20,
        n_lanes=24,
        lane_feat_dim=46,
        n_neighbors=6,
        split="eval",
        return_scene_data=True,
        prior_type="hermite",
        residual_frame="chord",
    )

    n_scan = min(len(dataset), 1500)
    n_hermite_better = 0
    n_lane_better = 0
    n_equal = 0
    n_fallback = 0
    h_rmses = []
    l_rmses = []

    for idx in range(n_scan):
        sample = dataset[idx]
        hist_pos = sample.get("history_pos", sample["history"][..., :2])
        history_m = denormalize(hist_pos.numpy())
        history_end = history_m[-1]
        goal_m = denormalize(sample["goal"].numpy().reshape(1, 2)).flatten()
        start_h = sample["start_heading"].item()
        end_h = sample["end_heading"].item()

        gt_m = get_gt_trajectory(sample)
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

        h_rmse = np.sqrt(np.mean(np.linalg.norm(gt_m - hermite_m, axis=-1) ** 2))
        l_rmse = np.sqrt(np.mean(np.linalg.norm(gt_m - lane_m, axis=-1) ** 2))

        h_rmses.append(h_rmse)
        l_rmses.append(l_rmse)

        diff = h_rmse - l_rmse
        if diff > 0.05:
            n_lane_better += 1
        elif diff < -0.05:
            n_hermite_better += 1
        else:
            n_equal += 1

    print(f"Scanned: {n_scan}")
    print(f"Fallback rate: {n_fallback}/{n_scan} = {n_fallback/n_scan*100:.1f}%")
    print(f"Lane better: {n_lane_better}  |  Hermite better: {n_hermite_better}  |  Equal: {n_equal}")
    print(f"Hermite RMSE: {np.mean(h_rmses):.3f}m (median {np.median(h_rmses):.3f})")
    print(f"Lane RMSE:    {np.mean(l_rmses):.3f}m (median {np.median(l_rmses):.3f})")
    improvement = (np.mean(h_rmses) - np.mean(l_rmses)) / np.mean(h_rmses) * 100
    print(f"Improvement:  {improvement:+.2f}%")


if __name__ == "__main__":
    main()
