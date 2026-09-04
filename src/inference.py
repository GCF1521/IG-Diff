"""Inference script for goal-conditioned trajectory generation.

Diffusion model predicts residual (normalized). Reconstruction:
  trajectory_local = prior + denormalize_residual(residual)
  trajectory_global = local_to_global(trajectory_local, ref_pos, ref_heading)

Supports optional visualization: trajectories, endpoints, ellipses, BEV scene,
denoising process, attention heatmaps, score heatmaps.
"""

import argparse
import os
import yaml
import torch
import numpy as np
import torch.multiprocessing as mp
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from data.av2_dataset import Argoverse2Dataset
from data.normalization import denormalize, denormalize_residual, normalize, compute_prior, compute_hermite_prior, compute_lane_prior
from data.normalization import (
    compute_chord_dir, unpack_chord, denormalize_residual_chord,
    chord_frame_to_residual, CHORD_EPSILON,
)
from data.coordinate_utils import local_to_global, global_to_local
from model.diffusion import DiffusionProcess
from model.tf_cross_denoiser import TFCrossDenoiser
from model.dps_guidance import full_dps_sample
from viz.style import save_figure
from viz.viz_trajectory import plot_trajectories, plot_endpoint_distribution, plot_confidence_ellipses
from viz.viz_denoising import plot_denoising_steps
from viz.viz_scene import plot_bev_scene
from viz.viz_animation import animate_bev_scene
from viz.viz_score_heatmap import compute_score_grid, plot_score_heatmap
from viz.viz_attention import plot_self_attention_heatmap, plot_cross_attention_map
from viz.viz_collision import plot_collision_scene, plot_collision_heatmap
from viz.utils import to_numpy
from src.drivable_check import build_paths, sample_goal_and_prior_in_drivable


def _collision_goal_local(args, partner_data, ref_pos, ref_heading, sample):
    """Return the goal endpoint to display in trajectory figures.

    In --collision_pair mode the strict shared endpoint is the sampled
    collision point (region centroid) — both vehicles converge to this
    single global point. Displaying the ego GT endpoint as "Goal" in this
    mode would be misleading, since the model never saw GT as its goal.
    Project the collision point into ego local frame instead.

    Otherwise (no collision_pair, or no partner found), fall back to the
    dataset-provided goal (ego GT endpoint for non-collision inference).
    """
    if (
        getattr(args, "collision_pair", False)
        and partner_data is not None
        and partner_data.get("collision_point_global") is not None
    ):
        cp_global = np.asarray(
            partner_data["collision_point_global"], dtype=np.float64
        ).reshape(1, 2)
        cp_local, _ = global_to_local(
            cp_global, np.zeros(1),
            np.asarray(ref_pos, dtype=np.float64), float(ref_heading),
        )
        return cp_local[0].astype(np.float32)
    return to_numpy(denormalize(sample["goal"].reshape(1, 2)).flatten())


def build_model(cfg: dict, device: str) -> TFCrossDenoiser:
    return TFCrossDenoiser(
        traj_len=cfg["data"]["n_future"],
        history_len=cfg["data"]["n_history"],
        n_lanes=cfg["data"]["n_lanes"],
        lane_feat_dim=cfg["data"]["lane_feat_dim"],
        n_neighbors=cfg["data"]["n_neighbors"],
        neighbor_hist_len=cfg["data"]["n_history"],
        neighbor_feat_dim=cfg["data"]["neighbor_feat_dim"],
        dim=cfg["model"]["dim"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        ffn_dim=cfg["model"]["ffn_dim"],
    ).to(device)


def reconstruct_trajectory(residual_norm, prior_norm,
                           use_chord_frame=None, chord_dir=None):
    """Reconstruct full trajectory from normalized residual and prior.

    residual_norm: (N, T, 2) — normalized residual predicted by diffusion
    prior_norm: (T, 2) or (1, T, 2) — normalized prior (Hermite spline or linear)
    use_chord_frame: optional scalar tensor — 1.0 if residual is in chord frame
    chord_dir: optional (2,) tensor — unit chord direction for chord-frame reconstruction
    Returns: (N, T, 2) — normalized full trajectory
    """
    if prior_norm.dim() == 3 and prior_norm.shape[0] == 1:
        prior_norm = prior_norm.squeeze(0)
    prior = denormalize(prior_norm)

    if use_chord_frame is not None and use_chord_frame.item() == 1.0 and chord_dir is not None:
        r_lon_n, r_lat_n = unpack_chord(residual_norm)
        r_lon, r_lat = denormalize_residual_chord(r_lon_n, r_lat_n)
        residual = chord_frame_to_residual(r_lon, r_lat, chord_dir)
    else:
        residual = denormalize_residual(residual_norm)

    trajectory = prior + residual
    return normalize(trajectory)


def generate_visualizations(
    idx, trajectories_m, gt_m, goal_m, prior_m, history_m,
    scene_data, intermediates, attn_dict,
    output_dir, cfg_weight, n_inference_steps,
    sampled_goals=None, animation_format=None,
    use_chord_frame=None, chord_dir=None,
    trajectories_raw_m=None,
):
    """Generate all visualization figures for one scenario."""
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(trajectories_m)
    gt_np = np.array(gt_m, dtype=np.float32) if not isinstance(gt_m, np.ndarray) else gt_m
    gen_np = np.array(trajectories_m, dtype=np.float32) if not isinstance(trajectories_m, np.ndarray) else trajectories_m
    goal_np = np.array(goal_m, dtype=np.float32) if not isinstance(goal_m, np.ndarray) else goal_m
    prior_np = np.array(prior_m, dtype=np.float32) if not isinstance(prior_m, np.ndarray) else prior_m

    # 1. Trajectory plot — if raw (pre-smoothing) trajectories are available,
    #    emit a top/bottom comparison figure; otherwise the standalone plot.
    if trajectories_raw_m is not None:
        from viz.viz_trajectory import plot_trajectories_comparison
        gen_raw_np = np.array(trajectories_raw_m, dtype=np.float32)
        fig = plot_trajectories_comparison(
            gen_np, gen_raw_np,
            gt=gt_np, goal=goal_np, prior=prior_np,
            title=f"Scenario {idx} — Trajectories (smoothed vs raw, N={n_samples})",
            sampled_goals=sampled_goals,
            history=history_m,
        )
    else:
        fig = plot_trajectories(
            gen_np, gt=gt_np, goal=goal_np, prior=prior_np,
            title=f"Scenario {idx} — Trajectories (N={n_samples})",
            sampled_goals=sampled_goals,
            history=history_m,
        )
    save_figure(fig, figures_dir / f"scenario_{idx}_trajectories.png")

    # 2. Endpoint distribution
    fig = plot_endpoint_distribution(gen_np, goal=goal_np, title=f"Scenario {idx} — Endpoints",
                                     sampled_goals=sampled_goals)
    save_figure(fig, figures_dir / f"scenario_{idx}_endpoints.png")

    # 3. Confidence ellipses
    fig = plot_confidence_ellipses(gen_np, gt=gt_np, goal=goal_np, title=f"Scenario {idx} — Confidence")
    save_figure(fig, figures_dir / f"scenario_{idx}_ellipses.png")

    # 4. BEV scene
    if scene_data is not None:
        # Convert neighbor_positions from list[dict] to list[tuple] for plot_bev_scene
        raw_positions = scene_data.get("neighbor_positions_local", [])
        neighbor_positions = [
            (p["x"], p["y"], p["heading"]) if isinstance(p, dict) else p
            for p in raw_positions
        ]
        fig = plot_bev_scene(
            lane_boundaries=scene_data.get("lane_boundaries_local", []),
            drivable_areas=scene_data.get("drivable_areas_local", []),
            focal_traj=history_m,
            neighbor_trajs=scene_data.get("neighbor_trajs_local", []),
            neighbor_positions=neighbor_positions,
            generated_trajs=[gen_np[i] for i in range(gen_np.shape[0])],
            gt_traj=gt_np,
            goal=goal_np,
            title=f"Scenario {idx} — BEV Scene",
            ped_crossings=scene_data.get("ped_crossings_local", []),
        )
        save_figure(fig, figures_dir / f"scenario_{idx}_scene.png")

        # 4b. BEV animation
        if animation_format is not None:
            history_full = np.array(history_m, dtype=np.float32) if not isinstance(history_m, np.ndarray) else history_m
            suffix = ".gif" if animation_format == "gif" else ".mp4"
            animate_bev_scene(
                history_m=history_full,
                gt_future_m=gt_np,
                gen_trajs_m=[gen_np[i] for i in range(gen_np.shape[0])],
                lane_boundaries=scene_data.get("lane_boundaries_local", []),
                drivable_areas=scene_data.get("drivable_areas_local", []),
                neighbor_trajs=scene_data.get("neighbor_trajs_local", []),
                neighbor_positions=neighbor_positions,
                neighbor_full_trajs=scene_data.get("neighbor_full_trajs_local", None),
                goal_m=goal_np,
                sampled_goals=sampled_goals,
                ped_crossings=scene_data.get("ped_crossings_local", []),
                fps=10,
                save_path=str(figures_dir / f"scenario_{idx}_animation{suffix}"),
                title=f"Scenario {idx} — BEV Animation",
            )

    # 5. Denoising process
    if intermediates is not None:
        residuals_list = intermediates.get("residuals", [])
        timesteps_list = intermediates.get("timesteps", [])
        if residuals_list:
            # Reconstruct full trajectories at each step
            traj_steps = []
            for res_norm in residuals_list:
                res_norm_squeezed = res_norm.squeeze(0)  # (T, 2)
                if use_chord_frame is not None and use_chord_frame.item() == 1.0 and chord_dir is not None:
                    r_lon_n, r_lat_n = unpack_chord(res_norm_squeezed)
                    r_lon, r_lat = denormalize_residual_chord(r_lon_n, r_lat_n)
                    residual = chord_frame_to_residual(r_lon, r_lat, chord_dir)
                else:
                    residual = denormalize_residual(res_norm_squeezed)
                full = prior_np + residual.numpy() if hasattr(residual, 'numpy') else prior_np + np.array(residual)
                traj_steps.append(full.astype(np.float32))
            fig = plot_denoising_steps(
                traj_steps, gt=gt_np, goal=goal_np,
                step_labels=[f"t={t}" for t in timesteps_list],
                title=f"Scenario {idx} — Denoising Process",
            )
            save_figure(fig, figures_dir / f"scenario_{idx}_denoising.png")

    # 6. Attention heatmaps
    if attn_dict is not None:
        for layer_name, layer_attn in attn_dict.items():
            if "self_attn" in layer_attn and layer_attn["self_attn"] is not None:
                fig = plot_self_attention_heatmap(
                    layer_attn["self_attn"],
                    layer_idx=int(layer_name.replace("layer", "")),
                )
                save_figure(fig, figures_dir / f"scenario_{idx}_self_attn_{layer_name}.png")

            if "cross_attn" in layer_attn and layer_attn["cross_attn"] is not None:
                # Compute condition section lengths for labels
                n_history = history_m.shape[0] if isinstance(history_m, np.ndarray) else len(history_m)
                cond_labels = [
                    ("goal", 1),
                    ("history", n_history),
                    ("map", cfg_data.get("n_lanes", 24)),
                    ("neighbors", cfg_data.get("n_neighbors", 6) * (n_history + 1)),
                ]
                fig = plot_cross_attention_map(
                    layer_attn["cross_attn"],
                    condition_labels=cond_labels,
                    layer_idx=int(layer_name.replace("layer", "")),
                )
                save_figure(fig, figures_dir / f"scenario_{idx}_cross_attn_{layer_name}.png")

    # 7. Score heatmap
    try:
        # Simple score: negative distance to goal
        goal_for_score = goal_np
        def simple_score(traj, endpoint):
            return -np.linalg.norm(endpoint - goal_for_score)

        score_grid, map_extent = compute_score_grid(
            score_fn=simple_score,
            grid_res=2.0,
            goal=goal_np,
        )
        fig = plot_score_heatmap(
            score_grid, map_extent,
            title=f"Scenario {idx} — Score Heatmap",
        )
        save_figure(fig, figures_dir / f"scenario_{idx}_score_heatmap.png")
    except Exception:
        pass  # Score heatmap is optional


# Global cfg for visualization callbacks
cfg_data = {}


def inference_worker(gpu_id, gpu_ids, args, cfg, output_dir, result_queue):
    """Worker function: load model on one GPU, process assigned scenarios."""
    global cfg_data
    cfg_data = cfg["data"]

    device = torch.device(f"cuda:{gpu_id}")
    n_gpus = len(gpu_ids)

    # Load model on this GPU
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(cfg, device)
    if args.use_raw_model and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diffusion = DiffusionProcess(n_steps=cfg["diffusion"]["n_steps"]).to(device)

    # Dataset
    data_dir = args.data_dir or cfg["data"]["train_dir"]
    viz_enabled = args.viz_scenarios > 0

    dataset = Argoverse2Dataset(
        data_dir=data_dir,
        map_dir=cfg["data"].get("map_dir"),
        n_future=cfg["data"]["n_future"],
        n_history=cfg["data"]["n_history"],
        n_lanes=cfg["data"]["n_lanes"],
        lane_feat_dim=cfg["data"]["lane_feat_dim"],
        n_neighbors=cfg["data"]["n_neighbors"],
        split="eval",
        return_scene_data=viz_enabled or args.collision_partner or args.collision_pair,
        prior_type=cfg["data"].get("prior_type", "hermite"),
        residual_frame=cfg["data"].get("residual_frame", "chord"),
        filter_parking=cfg["data"].get("filter_parking", False),
    )

    # Preload cache and apply parking filter
    # dataset.preload_cache()
    dataset.preload_cache(cache_path="output/dataset_cache_viz.pt")

    # Resolve scenario indices to evaluate
    if args.scenario_indices:
        all_indices = [int(x.strip()) for x in args.scenario_indices.split(",") if x.strip() != ""]
        # Filter by GPU round-robin so multi-GPU still works
        my_indices = [i for k, i in enumerate(all_indices) if k % n_gpus == gpu_id]
        my_indices = [i for i in my_indices if 0 <= i < len(dataset)]
    else:
        n_scenarios = min(len(dataset), args.n_scenarios)
        # Distribute scenarios round-robin across GPUs
        my_indices = list(range(gpu_id, n_scenarios, n_gpus))

    local_results = []

    for viz_counter, idx in enumerate(tqdm(my_indices, desc=f"GPU {gpu_id}", disable=False, position=gpu_id, leave=True)):
        sample = dataset[idx]

        conditions = {
            "goal": sample["goal"].unsqueeze(0).to(device),
            "map_tokens": sample["map_tokens"].unsqueeze(0).to(device),
            "map_mask": sample["map_mask"].unsqueeze(0).to(device),
            "neighbor_tokens": sample["neighbor_tokens"].unsqueeze(0).to(device),
            "neighbor_mask": sample["neighbor_mask"].unsqueeze(0).to(device),
            "history": sample["history"].unsqueeze(0).to(device),
        }

        prior_norm = sample["prior"].unsqueeze(0).to(device)
        prior_cpu = sample["prior"]
        use_chord_frame = sample.get("use_chord_frame", None)
        chord_dir_cpu = sample.get("chord_dir", None)  # (2,) tensor or None

        # GT goal and history endpoint in meters (for goal sampling).
        # sample["history_pos"] is (T_hist, 2) positions-only — use this
        # for any code that needs positions in meters (smoothing, goal sampling).
        history_pos = sample.get("history_pos", sample["history"][..., :2])
        goal_m = to_numpy(denormalize(sample["goal"].reshape(1, 2)).flatten())
        history_end_m = to_numpy(denormalize(history_pos[-1:].reshape(1, 2))).flatten()
        start_heading_m = sample["start_heading"].item()
        end_heading_m = sample["end_heading"].item()
        prior_type = cfg["data"].get("prior_type", "hermite")

        need_intermediates = viz_enabled and viz_counter < args.viz_scenarios and args.viz_denoising
        need_attention = viz_enabled and viz_counter < args.viz_scenarios and args.viz_attention

        # Generate multiple trajectories
        trajectories = []
        priors_for_reconstruct = []
        use_chord_flags = []          # per-sample flag for reconstruction
        chord_dirs_for_reconstruct = []  # per-sample chord_dir for reconstruction
        sampled_goals_list = []
        intermediates = None
        attn_dict = None

        # Compute GT endpoint heading for anisotropic goal sampling
        # Use the end_heading from dataset (from last 2 future points)
        gt_endpoint_heading = end_heading_m

        # Determine effective goal sigma values (adaptive: scale by chord length)
        # CLI > config (0.0) > auto. 0.0 triggers auto mode below.
        chord_len = max(np.linalg.norm(goal_m - history_end_m), 1e-6)
        goal_sigma_lon = args.goal_sigma_lon if args.goal_sigma_lon > 0 else max(chord_len * 0.10, 0.5)
        goal_sigma_lat = args.goal_sigma_lat if args.goal_sigma_lat > 0 else max(chord_len * 0.04, 0.3)
        goal_sampling_active = True
        if gpu_id == gpu_ids[0]:
            auto_lon = "auto" if args.goal_sigma_lon <= 0 else f"fixed={args.goal_sigma_lon}"
            auto_lat = "auto" if args.goal_sigma_lat <= 0 else f"fixed={args.goal_sigma_lat}"
            print(f"  [Scenario {idx}] chord_len={chord_len:.2f}m "
                  f"goal_sigma_lon={goal_sigma_lon:.3f}m ({auto_lon}) "
                  f"goal_sigma_lat={goal_sigma_lat:.3f}m ({auto_lat})")

        # Pre-build drivable-area paths once for all samples in this scenario
        da_local_list = sample.get("scene_data", {}).get("drivable_areas_local", None)
        da_paths = build_paths(da_local_list) if da_local_list else []

        def _compute_perturbed_prior(history_end, g_m):
            """Recompute prior for a perturbed goal (mirrors original branching)."""
            if prior_type == "lane":
                lane_segs = sample.get("scene_data", {}).get("lane_segments", {})
                ref_pos_np = to_numpy(sample["ref_pos"])
                ref_heading_val = sample["ref_heading"].item()
                if lane_segs:
                    return compute_lane_prior(
                        history_end, g_m, lane_segs,
                        ref_pos_np, ref_heading_val,
                        start_heading_m, end_heading_m, cfg["data"]["n_future"])
                return compute_hermite_prior(history_end, g_m, start_heading_m, end_heading_m, cfg["data"]["n_future"],
                                              drivable_areas_local=da_local_list)
            elif prior_type == "hermite":
                return compute_hermite_prior(history_end, g_m, start_heading_m, end_heading_m, cfg["data"]["n_future"],
                                              drivable_areas_local=da_local_list)
            else:
                return compute_prior(history_end, g_m, cfg["data"]["n_future"])

        for s_idx in range(args.n_samples):
            if goal_sampling_active:
                # Drivable-constrained anisotropic sampling:
                #   - Goal endpoint must lie inside a drivable area.
                #   - Prior trajectory must have every waypoint inside drivable areas.
                # Falls back to the unperturbed GT goal/prior if no acceptable
                # sample is found within max_attempts (rare with reasonable sigma).
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
                perturbed_goal_norm = torch.tensor(
                    normalize(perturbed_goal_m.reshape(1, 2)), dtype=torch.float32
                ).flatten().to(device)
                perturbed_prior_norm = torch.tensor(
                    normalize(perturbed_prior_m), dtype=torch.float32
                ).unsqueeze(0).to(device)
                cur_conditions = {**conditions, "goal": perturbed_goal_norm.unsqueeze(0)}
                cur_prior_norm = perturbed_prior_norm
                priors_for_reconstruct.append(perturbed_prior_norm.squeeze(0).cpu())
                sampled_goals_list.append(perturbed_goal_m.copy())

                # Recompute chord_dir for perturbed goal
                if use_chord_frame is not None and use_chord_frame.item() == 1.0:
                    pert_chord_dir, pert_chord_len, pert_chord_valid = compute_chord_dir(
                        history_end_m, perturbed_goal_m)
                    if pert_chord_valid:
                        chord_dirs_for_reconstruct.append(
                            torch.tensor(pert_chord_dir, dtype=torch.float32))
                        use_chord_flags.append(torch.tensor(1.0, dtype=torch.float32))
                    else:
                        chord_dirs_for_reconstruct.append(torch.zeros(2, dtype=torch.float32))
                        use_chord_flags.append(torch.tensor(0.0, dtype=torch.float32))
                else:
                    chord_dirs_for_reconstruct.append(None)
                    use_chord_flags.append(None if use_chord_frame is None else torch.tensor(0.0, dtype=torch.float32))
            else:
                cur_conditions = conditions
                cur_prior_norm = prior_norm
                priors_for_reconstruct.append(prior_cpu)
                sampled_goals_list.append(goal_m.copy())
                chord_dirs_for_reconstruct.append(chord_dir_cpu)
                use_chord_flags.append(use_chord_frame)

            cur_use_chord = use_chord_flags[-1]
            # Determine DPS step size: prefer --dps_eta; fall back to legacy
            # prior_adherence_dps_eta; finally to deviation_dps_eta when only
            # scheme-A deviation is active.
            effective_dps_eta = args.dps_eta
            if effective_dps_eta <= 0:
                effective_dps_eta = args.prior_adherence_dps_eta
            if effective_dps_eta <= 0 and args.deviation_weight > 0:
                effective_dps_eta = args.deviation_dps_eta
            any_dps_active = (args.dps_eta > 0) or (args.prior_adherence_weight > 0) or (args.deviation_weight > 0)
            result = full_dps_sample(
                diffusion, model, cur_conditions,
                traj_len=cfg["data"]["n_future"],
                n_inference_steps=cfg["diffusion"]["inference_steps"],
                cfg_weight=args.cfg_weight,
                dps_eta=effective_dps_eta,
                prior_norm=cur_prior_norm,
                use_cfg=True,
                use_dps=any_dps_active,
                device=str(device),
                save_intermediates=(need_intermediates and s_idx == 0),
                dynamic_threshold=args.dynamic_threshold,
                use_chord_frame=cur_use_chord.unsqueeze(0).to(device) if cur_use_chord is not None else None,
                chord_dir=chord_dirs_for_reconstruct[-1].unsqueeze(0).to(device) if chord_dirs_for_reconstruct[-1] is not None else None,
                spacing=cfg["diffusion"].get("inference_spacing", "linear"),
                prior_adherence_weight=args.prior_adherence_weight,
                prior_margin=5,
                deviation_weight=args.deviation_weight,
                deviation_thr_lon=args.deviation_thr_lon,
                deviation_thr_lat=args.deviation_thr_lat,
                deviation_t_max=args.deviation_t_max,
            )
            if need_intermediates and s_idx == 0 and isinstance(result, tuple):
                residual_norm, intermediates = result
                trajectories.append(residual_norm.squeeze(0).cpu())
            else:
                trajectories.append(result.squeeze(0).cpu())

        # Extract attention for first sample
        if need_attention:
            with torch.no_grad():
                t_vis = torch.tensor([diffusion.n_steps // 2], device=device)
                x_vis = torch.randn(1, cfg["data"]["n_future"], 2, device=device)
                _, attn_dict = model(
                    noisy_traj=x_vis, t=t_vis,
                    **conditions,
                    return_attn=True,
                )

        # Reconstruct full trajectories with per-sample priors and chord dirs
        residuals_norm = torch.stack(trajectories)
        trajectories_norm = []
        for i, res_norm in enumerate(residuals_norm):
            prior_i = priors_for_reconstruct[i]
            chord_dir_i = chord_dirs_for_reconstruct[i]
            use_chord_i = use_chord_flags[i]

            traj_norm = reconstruct_trajectory(
                res_norm.unsqueeze(0), prior_i,
                use_chord_frame=use_chord_i,
                chord_dir=chord_dir_i,
            ).squeeze(0)
            trajectories_norm.append(traj_norm)
        trajectories_norm = torch.stack(trajectories_norm)

        ref_pos = to_numpy(sample["ref_pos"])
        ref_heading = sample["ref_heading"].item()

        # GT trajectory
        gt_residual_norm = sample["trajectory"]
        gt_norm = reconstruct_trajectory(
            gt_residual_norm.unsqueeze(0), prior_cpu,
            use_chord_frame=use_chord_frame,
            chord_dir=chord_dir_cpu,
        ).squeeze(0)
        gt_denorm = denormalize(gt_norm)
        gt_np = to_numpy(gt_denorm)
        gt_global, _ = local_to_global(gt_np, np.zeros(gt_np.shape[0]), ref_pos, ref_heading)

        generated_global = []
        gen_local_list = []
        for traj_norm in trajectories_norm:
            traj_denorm = denormalize(traj_norm)
            traj_np = to_numpy(traj_denorm)
            traj_global, _ = local_to_global(traj_np, np.zeros(traj_np.shape[0]), ref_pos, ref_heading)
            generated_global.append(traj_global)
            gen_local_list.append(traj_np)

        # Snapshot the raw (unsmoothed) trajectories before any smoothing or
        # collision_pair override — used for the smoothed-vs-raw comparison
        # figure. Stored in result_entry as generated_local_raw /
        # generated_global_raw.
        gen_local_raw_list = [t.copy() for t in gen_local_list]
        generated_global_raw = [g.copy() for g in generated_global]

        # Denormalize history once — needed for smoothing (history context)
        # and for downstream visualization/scoring. Local meters, shape (T_hist, 2).
        # sample["history"] is now (T_hist, 6) — slice to positions for smoothing.
        history_m = to_numpy(denormalize(history_pos))

        # Apply smoothing and/or kinematic projection if requested.
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
            gen_np_for_smooth = np.array(gen_local_list, dtype=np.float32)
            # history_m is in local meters, shape (T_hist, 2). It is the
            # same for all N generated futures (same vehicle).
            history_for_smooth = np.asarray(history_m, dtype=np.float32)
            if args.kinematic_projection:
                gen_np_smoothed, gen_vel_smoothed = \
                    smooth_and_project_batch_with_history(
                        gen_np_for_smooth,
                        history=history_for_smooth,
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
                # SavGol-only fallback: use legacy batch (no history context).
                # Velocity is derived from smoothed positions.
                gen_np_smoothed = smooth_trajectory_batch(
                    gen_np_for_smooth,
                    window_length=args.smoothing_window,
                    polyorder=args.smoothing_polyorder,
                    preserve_endpoints=True,
                )
                gen_vel_smoothed = np.zeros_like(gen_np_smoothed)
                gen_vel_smoothed[:, :-1] = np.diff(gen_np_smoothed, axis=1) / 0.1
                gen_vel_smoothed[:, -1] = gen_vel_smoothed[:, -2]
            gen_local_list = [gen_np_smoothed[i] for i in range(gen_np_smoothed.shape[0])]
            gen_vel_local_list = [gen_vel_smoothed[i] for i in range(gen_vel_smoothed.shape[0])]
            generated_global = []
            for traj_np in gen_local_list:
                traj_global, _ = local_to_global(traj_np, np.zeros(traj_np.shape[0]), ref_pos, ref_heading)
                generated_global.append(traj_global)

        prior_m = to_numpy(denormalize(prior_cpu))
        # history_m was denormalized earlier (before smoothing)

        # Generate collision partner trajectories if requested
        partner_data = None
        if args.collision_pair:
            # New approach: 5 fan-sampled collision points × n_per_point
            # trajectories per vehicle, both converging to the sampled point
            # with arc-tangent arrival headings.
            from src.collision_partner import generate_collision_pair_trajectories
            pair_data = generate_collision_pair_trajectories(
                idx, sample, dataset, model, diffusion, device, cfg, args,
                n_points=args.collision_pair_points,
                n_per_point=args.collision_pair_per_point,
            )
            if pair_data is not None:
                # Override the ego trajectories with the paired ones
                generated_global = pair_data["ego_generated_global"]
                gen_local_list = pair_data["ego_generated_local"]
                # Override raw with pair's pre-smoothing trajectories so the
                # comparison figure shows pair-raw vs pair-smoothed (not the
                # unrelated single-vehicle raw).
                gen_local_raw_list = pair_data.get("ego_generated_local_raw") or gen_local_raw_list
                generated_global_raw = pair_data.get("ego_generated_global_raw") or generated_global_raw
                # In collision_pair mode the trajectories are generated to
                # converge at sampled collision points (NOT GT-goal
                # perturbations), so the goal-sampling scatter points are
                # meaningless here — replace them with the actual collision
                # points in ego-local frame so the viz reflects reality.
                sampled_collision_points = pair_data.get("sampled_collision_points", [])
                if sampled_collision_points:
                    cp_local_list = []
                    for sp in sampled_collision_points:
                        cp_g = np.asarray(sp["collision_point_global"], dtype=np.float32).reshape(2)
                        cp_local, _ = global_to_local(
                            cp_g.reshape(1, 2), np.zeros(1),
                            ref_pos, ref_heading,
                        )
                        cp_local_list.append(cp_local[0].astype(np.float32))
                    sampled_goals_list = cp_local_list
                else:
                    sampled_goals_list = None
                # Re-derive ref_pos/ref_heading from ego's original sample
                # (already same as pair_data uses internally)
                # Build partner_data dict in the shape the viz expects
                partner_data = {
                    "partner_generated_global": pair_data["partner_generated_global"],
                    "partner_generated_local": pair_data["partner_generated_local"],
                    "ego_generated_local_raw": pair_data.get("ego_generated_local_raw"),
                    "ego_generated_global_raw": pair_data.get("ego_generated_global_raw"),
                    "partner_generated_local_raw": pair_data.get("partner_generated_local_raw"),
                    "partner_generated_global_raw": pair_data.get("partner_generated_global_raw"),
                    "ego_prior_global": pair_data.get("ego_prior_global"),
                    "partner_prior_global": pair_data.get("partner_prior_global"),
                    "partner_gt_global": pair_data["partner_gt_global"],
                    "partner_ref_pos": pair_data["partner_ref_pos"],
                    "partner_ref_heading": pair_data["partner_ref_heading"],
                    "collision_point_global": pair_data["collision_point_global"],
                    "collision_rate": pair_data["collision_rate"],
                    "collision_mean_min_dist": pair_data["collision_mean_min_dist"],
                    "collision_rate_strict": pair_data["collision_rate_strict"],
                    "partner_min_dist_gt": pair_data["partner_min_dist_gt"],
                    "partner_collision_timestep": pair_data["partner_collision_timestep"],
                    "partner_track_id": pair_data["partner_track_id"],
                    "scenario": pair_data["scenario"],
                    "static_map": pair_data["static_map"],
                    "focal_track_id": pair_data["focal_track_id"],
                    "sampled_collision_points": pair_data["sampled_collision_points"],
                    "region_area_m2": pair_data["region_area_m2"],
                    "t_horizon_used": pair_data["t_horizon_used"],
                    "collision_region": pair_data.get("collision_region"),
                }
        elif args.collision_partner:
            from src.collision_partner import generate_partner_trajectories, compute_collision_rate
            partner_data = generate_partner_trajectories(
                idx, sample, dataset, model, diffusion, device, cfg, args
            )
            if partner_data is not None:
                # Compute collision rate between ego and partner trajectories
                ego_trajs_global = np.array(generated_global, dtype=np.float32)
                partner_trajs_global = np.array(partner_data["partner_generated_global"], dtype=np.float32)
                collision_metrics = compute_collision_rate(
                    ego_trajs_global, partner_trajs_global,
                    threshold=args.collision_threshold,
                )
                partner_data["collision_rate"] = collision_metrics["collision_rate"]
                partner_data["collision_mean_min_dist"] = collision_metrics["mean_min_dist"]
                partner_data["collision_rate_strict"] = collision_metrics["collision_rate_strict"]

        result_entry = {
            "scenario_idx": idx,
            "gt": gt_global,
            "gt_local": gt_np,
            "generated": generated_global,
            "generated_local": gen_local_list,
            # Raw (unsmoothed) trajectories for the smoothed-vs-raw comparison
            # figure. In collision_pair mode these are the pre-smoothing pair
            # trajectories; otherwise they are the pre-smoothing single-vehicle
            # trajectories. Equal to gen_local_list when no smoothing/projection
            # was applied.
            "generated_local_raw": gen_local_raw_list,
            "generated_global_raw": generated_global_raw,
            "goal_local": _collision_goal_local(
                args, partner_data, ref_pos, ref_heading, sample,
            ),
            "sampled_goals_local": sampled_goals_list,
            "ref_pos": ref_pos,
            "ref_heading": ref_heading,
            "drivable_areas_local": [
                da.tolist() if isinstance(da, np.ndarray) else da
                for da in sample.get("scene_data", {}).get("drivable_areas_local", [])
            ],
            "partner": partner_data,
            "is_parking": sample.get("is_parking", torch.tensor(0.0)).item(),
        }
        local_results.append(result_entry)

        # Generate visualizations for selected scenarios
        if viz_enabled and viz_counter < args.viz_scenarios:
            scene_data = sample.get("scene_data", None)
            gen_np = np.array(gen_local_list, dtype=np.float32)
            gen_raw_np = (
                np.array(gen_local_raw_list, dtype=np.float32)
                if gen_local_raw_list is not None and len(gen_local_raw_list) > 0
                else None
            )
            goal_m_viz = _collision_goal_local(
                args, partner_data, ref_pos, ref_heading, sample,
            )
            # In collision_pair mode sampled_goals_list holds the same single
            # collision point already shown as goal_m_viz — skip the duplicate
            # scatter to keep the figure clean.
            show_sampled_goals = (
                goal_sampling_active
                and not (args.collision_pair and partner_data is not None)
            )
            sampled_goals_np = (
                np.array(sampled_goals_list, dtype=np.float32)
                if show_sampled_goals else None
            )

            generate_visualizations(
                idx=idx,
                trajectories_m=gen_np,
                gt_m=gt_np,
                goal_m=goal_m_viz,
                prior_m=prior_m,
                history_m=history_m,
                scene_data=scene_data,
                intermediates=intermediates,
                attn_dict=attn_dict,
                output_dir=str(output_dir),
                cfg_weight=args.cfg_weight,
                n_inference_steps=cfg["diffusion"]["inference_steps"],
                sampled_goals=sampled_goals_np,
                animation_format=args.animation,
                use_chord_frame=use_chord_frame,
                chord_dir=chord_dir_cpu,
                trajectories_raw_m=gen_raw_np,
            )
            print(f"  [GPU {gpu_id}] Generated visualizations for scenario {idx}")

        # Generate collision visualizations
        if (args.collision_partner or args.collision_pair) and partner_data is not None:
            figures_dir = Path(output_dir) / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            ego_g = np.array(generated_global, dtype=np.float32)
            partner_g = np.array(partner_data["partner_generated_global"], dtype=np.float32)

            plot_collision_scene(
                scenario=partner_data.get("scenario"),
                static_map=partner_data.get("static_map"),
                ego_trajs_global=ego_g,
                partner_trajs_global=partner_g,
                focal_track_id=partner_data.get("focal_track_id"),
                partner_track_id=partner_data.get("partner_track_id"),
                ego_gt_global=gt_global,
                partner_gt_global=partner_data.get("partner_gt_global"),
                collision_point_global=partner_data.get("collision_point_global"),
                collision_rate=partner_data.get("collision_rate"),
                collision_mean_min_dist=partner_data.get("collision_mean_min_dist"),
                collision_rate_strict=partner_data.get("collision_rate_strict"),
                collision_threshold=args.collision_threshold,
                title=f"Scenario {idx} — Isomorphic Collision Guidance",
                save_path=str(figures_dir / f"scenario_{idx}_collision.png"),
                n_ego_show=len(ego_g),
                n_partner_show=len(partner_g),
                n_top_show=5,
                collision_region=partner_data.get("collision_region"),
                show_fans=False,
                ego_prior_global=partner_data.get("ego_prior_global")[0]
                    if partner_data.get("ego_prior_global") else None,
                partner_prior_global=partner_data.get("partner_prior_global")[0]
                    if partner_data.get("partner_prior_global") else None,
            )

            plot_collision_heatmap(
                ego_trajs_global=ego_g,
                partner_trajs_global=partner_g,
                threshold=args.collision_threshold,
                title=f"Scenario {idx} — Distance Matrix",
                save_path=str(figures_dir / f"scenario_{idx}_collision_heatmap.png"),
            ) if not args.skip_collision_heatmap else None
            print(f"  [GPU {gpu_id}] Generated collision visualizations for scenario {idx}")

            # Generate collision animation MP4 (ego + partner driving on best pair)
            if args.collision_animation:
                from viz.viz_collision_animation import animate_collision_scenario
                ego_gt_g = np.array(gt_global, dtype=np.float32) if gt_global is not None else None
                partner_gt_g_arr = partner_data.get("partner_gt_global")
                partner_gt_g = np.array(partner_gt_g_arr, dtype=np.float32) if partner_gt_g_arr is not None else None
                anim_path = figures_dir / f"scenario_{idx}_collision.mp4"
                animate_collision_scenario(
                    scenario=partner_data.get("scenario"),
                    static_map=partner_data.get("static_map"),
                    ego_trajs_global=ego_g,
                    partner_trajs_global=partner_g,
                    focal_track_id=partner_data.get("focal_track_id"),
                    partner_track_id=partner_data.get("partner_track_id"),
                    ego_gt_global=ego_gt_g,
                    partner_gt_global=partner_gt_g,
                    collision_point_global=None,
                    collision_threshold=args.collision_threshold,
                    n_top_pairs=args.collision_anim_background,
                    fps=args.collision_anim_fps,
                    save_path=str(anim_path),
                    title=f"Scenario {idx} — Collision Animation",
                    w_end=args.topk_w_end,
                    w_goal=args.topk_w_goal,
                    w_smooth=args.topk_w_smooth,
                    w_curv=args.topk_w_curv,
                )
                print(f"  [GPU {gpu_id}] Generated collision animation for scenario {idx}")

    result_queue.put(local_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--n_scenarios", type=int, default=10, help="Max number of scenarios to evaluate")
    parser.add_argument("--scenario_indices", type=str, default=None,
                        help="Comma-separated list of specific dataset indices to evaluate, e.g. '4845,7578,2532'. Overrides n_scenarios if set.")
    parser.add_argument("--cfg_weight", type=float, default=2.0)
    parser.add_argument("--dps_eta", type=float, default=0.0, help="DPS guidance strength (0=disabled)")
    parser.add_argument("--dynamic_threshold", type=float, default=1.5, help="Imagen-style dynamic threshold for x0_est (0=disabled, default=1.5)")
    parser.add_argument("--prior_adherence_weight", type=float, default=0.0,
                        help="DPS weight for pulling trajectory mid-section toward Hermite prior (0=disabled, try 0.5-2.0)")
    parser.add_argument("--prior_adherence_dps_eta", type=float, default=0.3,
                        help="DPS step size when only prior_adherence is active (default=0.3)")
    parser.add_argument("--deviation_weight", type=float, default=0.0,
                        help="Scheme-A DPS weight for hinge-style mid-section deviation penalty. "
                             "Only excess beyond (thr_lon, thr_lat) is penalized. 0=disabled, try 1.0-3.0.")
    parser.add_argument("--deviation_thr_lon", type=float, default=1.0,
                        help="Longitudinal deviation threshold in meters (default=1.0)")
    parser.add_argument("--deviation_thr_lat", type=float, default=0.5,
                        help="Lateral deviation threshold in meters (default=0.5)")
    parser.add_argument("--deviation_t_max", type=int, default=300,
                        help="Only apply deviation guidance when t < this value (default=300, "
                             "i.e. last 70%% of denoising. Set to 1000 to always apply.)")
    parser.add_argument("--deviation_dps_eta", type=float, default=0.5,
                        help="DPS step size when only deviation guidance is active (default=0.5)")
    parser.add_argument("--use_raw_model", action="store_true",
                        help="Load model_state_dict instead of ema_state_dict (v8 raw model has better mid-section adherence)")
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU IDs to use, e.g. '0,1,2,3'")
    parser.add_argument("--output_dir", type=str, default="output/samples")
    # Visualization flags
    parser.add_argument("--viz_scenarios", type=int, default=0, help="Number of scenarios to generate visualizations for (0=none)")
    parser.add_argument("--viz_denoising", action="store_true", help="Generate denoising process visualizations")
    parser.add_argument("--viz_attention", action="store_true", help="Generate attention heatmaps")
    parser.add_argument("--viz_score_heatmap", action="store_true", help="Generate score heatmaps")
    parser.add_argument("--smoothing", action="store_true", help="Apply Savitzky-Golay smoothing to generated trajectories")
    parser.add_argument("--smoothing_window", type=int, default=7, help="Smoothing window length (odd, default=7)")
    parser.add_argument("--smoothing_polyorder", type=int, default=3, help="Smoothing polynomial order (default=3)")
    parser.add_argument("--smoothing_method", type=str, default="savgol", choices=["savgol", "mavg"],
                        help="Smoothing filter: 'savgol' (Savitzky-Golay, default) or 'mavg' (centered moving average)")
    parser.add_argument("--join_strength", type=float, default=0.0,
                        help="Soft join velocity blend strength (0.0=none, 0.1=gentle, 1.0=full C1). Blends future's first velocity toward history's continuation.")
    parser.add_argument("--kinematic_projection", action="store_true", help="Apply kinematic projection (SavGol + physics constraints)")
    parser.add_argument("--kinematic_v_max", type=float, default=15.0, help="Max speed in m/s (default=15.0)")
    parser.add_argument("--kinematic_a_max", type=float, default=3.0, help="Max acceleration in m/s² (default=3.0)")
    parser.add_argument("--kinematic_jerk_max", type=float, default=4.0, help="Max jerk in m/s³ (default=4.0)")
    parser.add_argument("--kinematic_kappa_max", type=float, default=0.1, help="Max curvature in 1/m, min turning radius (default=0.1)")
    parser.add_argument("--kinematic_iters", type=int, default=5, help="Kinematic projection iterations (default=5)")
    parser.add_argument("--goal_sigma_lon", type=float, default=None,
                        help="Goal perturbation sigma along endpoint heading (meters). "
                             "None=use config (0.0=auto: max(chord*0.10, 0.5))")
    parser.add_argument("--goal_sigma_lat", type=float, default=None,
                        help="Goal perturbation sigma perpendicular to heading (meters). "
                             "None=use config (0.0=auto: max(chord*0.04, 0.3))")
    parser.add_argument("--animation", type=str, default=None, help="Save animation for viz scenarios: 'gif' or 'mp4' (default=None=disabled)")
    parser.add_argument("--collision_partner", action="store_true",
                        help="Generate collision partner trajectories and compute collision_rate")
    parser.add_argument("--collision_pair", action="store_true",
                        help="Generate ego + partner as paired trajectories converging to "
                             "fan-sampled collision points (5 points × n_per_point per vehicle). "
                             "Supersedes --collision_partner when both are set.")
    parser.add_argument("--collision_pair_points", type=int, default=5,
                        help="Number of collision points sampled from the fan region (default=5)")
    parser.add_argument("--collision_pair_per_point", type=int, default=3,
                        help="Trajectories per collision point per vehicle (default=3)")
    parser.add_argument("--collision_threshold", type=float, default=1.5,
                        help="Distance threshold for collision detection (meters, default=1.5)")
    parser.add_argument("--collision_animation", action="store_true",
                        help="Generate MP4 animation per scenario: ego + partner driving on best "
                             "collision pair, with 5 background trajectories. "
                             "Requires --collision_partner.")
    parser.add_argument("--collision_anim_background", type=int, default=5,
                        help="Number of background trajectories (lowest minFDE) to draw in animation (default=5)")
    parser.add_argument("--collision_anim_fps", type=int, default=10,
                        help="Animation FPS (default=10)")
    parser.add_argument("--topk_w_end", type=float, default=0.50,
                        help="Top-K composite score weight: end_dist (default=0.50)")
    parser.add_argument("--topk_w_goal", type=float, default=0.20,
                        help="Top-K composite score weight: goal_completion (default=0.20)")
    parser.add_argument("--topk_w_smooth", type=float, default=0.15,
                        help="Top-K composite score weight: smoothness (default=0.15)")
    parser.add_argument("--topk_w_curv", type=float, default=0.15,
                        help="Top-K composite score weight: curvature (default=0.15)")
    parser.add_argument("--no_timestamp", action="store_true",
                        help="Use --output_dir as-is without appending a timestamp subdirectory")
    parser.add_argument("--skip_collision_heatmap", action="store_true",
                        help="Skip collision heatmap PNG (distance matrix) — keep only collision.png + animation.mp4")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides config; config overrides hardcoded defaults.
    if args.goal_sigma_lon is None:
        args.goal_sigma_lon = cfg.get("inference", {}).get("goal_sigma_lon", 0.0)
    if args.goal_sigma_lat is None:
        args.goal_sigma_lat = cfg.get("inference", {}).get("goal_sigma_lat", 0.0)

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    n_gpus = len(gpu_ids)
    visible_str = ",".join(str(g) for g in gpu_ids)

    # Set CUDA_VISIBLE_DEVICES for all workers
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_str

    # Re-index: after setting CUDA_VISIBLE_DEVICES, GPUs are 0..n_gpus-1
    worker_gpu_ids = list(range(n_gpus))

    # Beijing time (UTC+8)
    from zoneinfo import ZoneInfo
    beijing_tz = ZoneInfo("Asia/Shanghai")
    timestamp = datetime.now(tz=beijing_tz).strftime("%Y%m%d_%H%M%S")

    if args.no_timestamp:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Using GPUs: {gpu_ids} ({n_gpus} GPU(s))")

    # Launch workers via multiprocessing
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    processes = []

    print(f"Launching {n_gpus} inference worker(s)...")
    for gpu_id in worker_gpu_ids:
        p = mp.Process(
            target=inference_worker,
            args=(gpu_id, worker_gpu_ids, args, cfg, output_dir, result_queue),
        )
        p.start()
        processes.append(p)

    # Collect results from all workers
    all_results = []
    for _ in range(n_gpus):
        all_results.extend(result_queue.get())

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Sort by scenario index for consistent ordering
    all_results.sort(key=lambda r: r["scenario_idx"])

    torch.save(all_results, output_dir / "results.pt")
    print(f"Saved {len(all_results)} scenarios to {output_dir / 'results.pt'}")


if __name__ == "__main__":
    main()
