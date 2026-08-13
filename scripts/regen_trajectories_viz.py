"""Regenerate scenario_*_trajectories.png from saved results.pt with enhanced viz.

For the v2 run (collision_video_20sc_v2), results.pt contains BOTH:
  - generated_local: smoothed trajectories (after kinematic_projection)
  - generated_local_raw: raw trajectories (before kinematic_projection)

So this script emits a top/bottom comparison figure (smoothed on top, raw on
bottom) when raw is present. Otherwise it falls back to the single-panel plot.
"""
import sys
sys.path.insert(0, "/workspace")
import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from data.av2_dataset import Argoverse2Dataset
from data.normalization import denormalize
from data.coordinate_utils import global_to_local
from viz.viz_trajectory import plot_trajectories, plot_trajectories_comparison
from viz.style import save_figure
from viz.utils import to_numpy


def load_dataset(cfg):
    dataset = Argoverse2Dataset(
        data_dir=cfg["data"]["train_dir"],
        map_dir=cfg["data"].get("map_dir"),
        n_future=cfg["data"]["n_future"],
        n_history=cfg["data"]["n_history"],
        n_lanes=cfg["data"]["n_lanes"],
        lane_feat_dim=cfg["data"]["lane_feat_dim"],
        n_neighbors=cfg["data"]["n_neighbors"],
        split="eval",
        return_scene_data=True,
        prior_type=cfg["data"].get("prior_type", "hermite"),
        residual_frame=cfg["data"].get("residual_frame", "chord"),
        filter_parking=cfg["data"].get("filter_parking", False),
    )
    dataset.preload_cache()
    return dataset


def build_sampled_goals_local(r, ref_pos, ref_heading):
    """In collision_pair mode, replace stale sampled_goals_local with the
    actual collision points (in ego-local frame). Otherwise use the saved
    sampled_goals_local as-is.
    """
    partner = r.get("partner")
    if partner is not None and "sampled_collision_points" in partner:
        cp_local_list = []
        for sp in partner["sampled_collision_points"]:
            cp_g = np.asarray(sp["collision_point_global"], dtype=np.float32).reshape(2)
            cp_local, _ = global_to_local(
                cp_g.reshape(1, 2), np.zeros(1), ref_pos, ref_heading,
            )
            cp_local_list.append(cp_local[0].astype(np.float32))
        return np.array(cp_local_list, dtype=np.float32) if cp_local_list else None
    sg = r.get("sampled_goals_local")
    if sg is not None:
        return np.array(sg, dtype=np.float32)
    return None


def build_prior_mean(partner, ref_pos, ref_heading):
    """Convert ego_prior_global (per-sample) to local frame and average."""
    if partner is None or "ego_prior_global" not in partner:
        return None
    ego_prior_global = np.array(partner["ego_prior_global"], dtype=np.float32)
    N, T, _ = ego_prior_global.shape
    flat, _ = global_to_local(
        ego_prior_global.reshape(-1, 2),
        np.zeros(N * T),
        ref_pos, ref_heading,
    )
    prior_local = flat.reshape(N, T, 2)
    return prior_local.mean(axis=0)


def main():
    results_path = "/workspace/output/collision_video_20sc_v2/results.pt"
    output_dir = Path("/workspace/output/collision_video_20sc_v2/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("/workspace/config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    dataset = load_dataset(cfg)
    print(f"Dataset size: {len(dataset)}")

    results = torch.load(results_path, weights_only=False)
    print(f"Loaded {len(results)} results")

    for r in results:
        idx = r["scenario_idx"]
        gen_local = np.array(r["generated_local"], dtype=np.float32)
        gt_local = np.array(r["gt_local"], dtype=np.float32)
        goal_local = np.array(r["goal_local"], dtype=np.float32)
        ref_pos = np.array(r["ref_pos"], dtype=np.float32)
        ref_heading = float(r["ref_heading"])
        partner = r.get("partner")

        sampled_goals_local = build_sampled_goals_local(r, ref_pos, ref_heading)
        prior_mean = build_prior_mean(partner, ref_pos, ref_heading)

        # History from dataset (denormalize)
        sample = dataset[idx]
        history_m = to_numpy(denormalize(sample["history"]))

        # Raw (pre-smoothing) trajectories — may be missing in old runs
        gen_local_raw = None
        if "generated_local_raw" in r and r["generated_local_raw"] is not None:
            gen_local_raw = np.array(r["generated_local_raw"], dtype=np.float32)

        n_samples = len(gen_local)
        if gen_local_raw is not None and len(gen_local_raw) > 0:
            fig = plot_trajectories_comparison(
                gen_local, gen_local_raw,
                gt=gt_local, goal=goal_local, prior=prior_mean,
                title=f"Scenario {idx} — Trajectories (smoothed vs raw, N={n_samples})",
                sampled_goals=sampled_goals_local,
                history=history_m,
            )
            mode = "comparison"
        else:
            fig = plot_trajectories(
                gen_local,
                gt=gt_local, goal=goal_local, prior=prior_mean,
                title=f"Scenario {idx} — Trajectories (N={n_samples})",
                sampled_goals=sampled_goals_local,
                history=history_m,
            )
            mode = "single"
        save_figure(fig, output_dir / f"scenario_{idx}_trajectories.png")
        plt.close(fig)
        print(f"  scenario {idx}: saved ({mode}, prior={'yes' if prior_mean is not None else 'no'}, "
              f"history={history_m.shape}, gen={gen_local.shape}, raw={'yes' if gen_local_raw is not None else 'no'})")

    print(f"\nDone. Regenerated trajectories.png for {len(results)} scenarios in {output_dir}")


if __name__ == "__main__":
    main()
