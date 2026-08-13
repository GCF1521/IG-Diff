"""Training script for goal-conditioned trajectory generation.

Single-stage training with all conditions (goal, map, neighbors) from the start.
Diffusion model learns residual = trajectory - prior (linear interp from history_end to goal).
Uses endpoint-weighted loss to emphasize goal accuracy.
Supports multi-GPU training via DistributedDataParallel (torchrun).
Integrates auxiliary losses (smoothness, velocity, jerk, curvature) and
enhanced TensorBoard logging.
"""

import argparse
import math
import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from zoneinfo import ZoneInfo
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from data.av2_dataset import Argoverse2Dataset
from data.normalization import denormalize, denormalize_residual
from model.diffusion import DiffusionProcess
from model.tf_cross_denoiser import TFCrossDenoiser
from model.dps_guidance import full_dps_sample
from model.auxiliary_losses import compute_auxiliary_losses
from viz.viz_training import render_sample_trajectory, plot_noise_comparison, plot_residual_distribution, log_training_images
from viz.viz_trajectory import plot_trajectories


class NoOpWriter:
    """No-op SummaryWriter for non-main DDP ranks."""

    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_images(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass

    def close(self):
        pass


def get_available_gpus(min_free_mb=1000):
    """Return list of GPU IDs with at least min_free_mb free memory.

    Respects CUDA_VISIBLE_DEVICES: only considers GPUs that are visible
    to the current process.
    """
    allowed = None
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        try:
            allowed = set(int(x.strip()) for x in cvd.split(",") if x.strip())
        except ValueError:
            pass

    available = []
    try:
        result = os.popen(
            "nvidia-smi --query-gpu=index,memory.free --format=csv,nounits,noheader"
        ).read()
        for line in result.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) >= 2:
                idx = int(parts[0].strip())
                free_mb = int(parts[1].strip())
                if allowed is not None and idx not in allowed:
                    continue
                if free_mb >= min_free_mb:
                    available.append(idx)
    except Exception:
        pass
    if not available:
        available = list(range(torch.cuda.device_count()))
    return available


def setup_ddp():
    """Initialize DDP process group. Returns (rank, world_size).

    When launched via torchrun, RANK/WORLD_SIZE/LOCAL_RANK env vars are set
    automatically. CUDA_VISIBLE_DEVICES should already be configured by main()
    to only include free GPUs. Falls back to single-GPU mode otherwise.
    Uses NCCL by default; falls back to gloo if NCCL fails (e.g. small /dev/shm).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % world_size))

        # DDP timeout: default 5 min is too tight when rank 0 is busy with
        # checkpoint writes (~200MB) or EMA updates at epoch boundaries.
        # 30 min gives plenty of headroom for slow I/O or GC pauses.
        import datetime
        ddp_timeout = datetime.timedelta(minutes=30)

        # Prefer NCCL over Gloo for GPU training. NCCL uses CUDA IPC / NVLink
        # and does not depend on /dev/shm (the previous code forced Gloo when
        # /dev/shm < 256MB, but NCCL works fine without SHM — it just falls
        # back to other transports). NCCL is far more reliable than Gloo for
        # long-running multi-GPU training: Gloo's TCP transport in Docker
        # containers accumulates TIME_WAIT sockets and eventually fails with
        # "Connection closed by peer" during backward allreduce.
        # Set DDP_FORCE_GLOO=1 to force Gloo (for debugging only).
        force_gloo = os.environ.get("DDP_FORCE_GLOO") == "1"

        if not force_gloo:
            try:
                dist.init_process_group("nccl", rank=rank, world_size=world_size,
                                        timeout=ddp_timeout)
                if rank == 0:
                    print("  DDP backend: NCCL (timeout=30min)")
            except (RuntimeError, Exception) as e:
                if rank == 0:
                    print(f"  NCCL init failed ({type(e).__name__}: {e}); falling back to gloo")
                dist.init_process_group("gloo", rank=rank, world_size=world_size,
                                        timeout=ddp_timeout)
                if rank == 0:
                    print("  DDP backend: gloo (NCCL unavailable, timeout=30min)")
        else:
            dist.init_process_group("gloo", rank=rank, world_size=world_size,
                                    timeout=ddp_timeout)
            if rank == 0:
                print("  DDP backend: gloo (forced by DDP_FORCE_GLOO=1, timeout=30min)")

        torch.cuda.set_device(local_rank)
        return rank, world_size
    else:
        return 0, 1


def cleanup_ddp():
    """Destroy DDP process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def build_model(cfg: dict, device: str) -> nn.Module:
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
        dropout=cfg["model"]["dropout"],
    ).to(device)


def train(cfg: dict, device: str, resume_from: str = None):
    """Train residual prediction with all conditions from the start."""
    # DDP setup
    rank, world_size = setup_ddp()
    is_main = (rank == 0)

    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", rank % world_size))
        device = f"cuda:{local_rank}"

    # Visualization config
    viz_cfg = cfg.get("visualization", {})
    train_sample_interval = viz_cfg.get("train_sample_interval", 5)
    train_noise_interval = viz_cfg.get("train_noise_interval", 500)
    train_residual_interval = viz_cfg.get("train_residual_interval", 500)
    inference_sample_steps = viz_cfg.get("inference_sample_steps", 20)

    # Logging config
    log_cfg = cfg.get("logging", {})
    grad_norm_interval = log_cfg.get("grad_norm_interval", 100)
    condition_stats_interval = log_cfg.get("condition_stats_interval", 500)
    param_norm_interval = log_cfg.get("param_norm_interval", 500)
    ema_compare_interval = log_cfg.get("ema_compare_interval", 1)

    # Auxiliary loss config
    aux_cfg = cfg["training"].get("auxiliary_losses", {})

    # Data
    dataset = Argoverse2Dataset(
        data_dir=cfg["data"]["train_dir"],
        map_dir=cfg["data"].get("map_dir"),
        n_future=cfg["data"]["n_future"],
        n_history=cfg["data"]["n_history"],
        n_lanes=cfg["data"]["n_lanes"],
        lane_feat_dim=cfg["data"]["lane_feat_dim"],
        n_neighbors=cfg["data"]["n_neighbors"],
        split="train",
        drop_goal_p=cfg["training"]["drop_goal_p"],
        drop_map_p=cfg["training"]["drop_map_p"],
        drop_neighbor_p=cfg["training"]["drop_neighbor_p"],
        prior_type=cfg["data"].get("prior_type", "hermite"),
        residual_frame=cfg["data"].get("residual_frame", "chord"),
        filter_parking=cfg["data"].get("filter_parking", False),
    )

    # Pre-load all samples into memory so forked DataLoader workers
    # inherit the full cache via copy-on-write (no disk I/O at training time)
    if is_main:
        print("Preloading dataset into memory...")
    dataset.preload_cache()

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    num_workers = cfg["data"].get("num_workers", 4)
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(sampler is None),
        persistent_workers=(num_workers > 0),
    )

    # Model
    model = build_model(cfg, device)
    start_epoch = 0
    resume_ckpt = None
    if resume_from:
        resume_ckpt = torch.load(resume_from, map_location=device)
        # Prefer ema_state_dict for resume (it's what inference uses and typically better)
        src_state = resume_ckpt.get("ema_state_dict", resume_ckpt["model_state_dict"])
        model.load_state_dict(src_state)
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        if is_main:
            print(f"Resumed from {resume_from}, starting epoch {start_epoch}")

    # DDP wrap
    if world_size > 1:
        # broadcast_buffers=False: model has only one buffer (sin_pos, a fixed
        # sinusoidal PE) that is identical across ranks by construction.
        # Disabling the per-forward _sync_buffers() avoids the Gloo
        # "Connection closed by peer" crash that previously occurred at epoch
        # boundaries when rank 0 was busy with checkpoint/EMA work and other
        # ranks timed out waiting for the broadcast.
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
        )

    model_without_ddp = model.module if world_size > 1 else model

    diffusion = DiffusionProcess(n_steps=cfg["diffusion"]["n_steps"]).to(device)

    # EMA (operates on unwrapped model)
    ema_model = build_model(cfg, device)
    if resume_ckpt is not None and "ema_state_dict" in resume_ckpt:
        ema_model.load_state_dict(resume_ckpt["ema_state_dict"])
        if is_main:
            print("  Restored EMA state from checkpoint")
    else:
        ema_model.load_state_dict(model_without_ddp.state_dict())
    ema_decay = cfg["training"]["ema_decay"]

    # AMP scaler
    use_amp = cfg["training"].get("use_amp", True)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    # BF16 doesn't need loss scaling; only enable scaler for FP16
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    if is_main and use_amp:
        print(f"  Mixed precision: {amp_dtype}")

    # Optimizer with per-group weight_decay & LR
    # 1. output_head: higher weight_decay to prevent gradient monopoly
    # 2. cross_attn: NORMAL LR + HIGH weight_decay (not high LR — high LR caused
    #    norm inflation → activation saturation → gradient collapse)
    # 3. adaLN_mlp: inverted depth scaling — deeper layers get HIGHER LR
    #    (shallow layers naturally receive more gradient, need restraint)
    # 4. traj_embed: higher LR to break freeze
    # 5. other: base LR
    output_head_decay = cfg["training"].get("output_head_weight_decay", 0.1)
    output_head_lr_mult = cfg["training"].get("output_head_lr_mult", 1.0)
    base_weight_decay = cfg["training"]["weight_decay"]
    cross_attn_weight_decay = cfg["training"].get("cross_attn_weight_decay", 0.1)
    adaLN_lr_mult = cfg["training"].get("adaLN_lr_mult", 10.0)
    adaLN_layer_decay = cfg["training"].get("adaLN_layer_decay", 0.9)
    traj_embed_lr_mult = cfg["training"].get("traj_embed_lr_mult", 5.0)
    output_head_params = []
    cross_attn_params = []
    adaLN_per_layer = {}  # layer_idx -> [params]
    traj_embed_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "output_head" in name:
            output_head_params.append(param)
        elif "cross_attn" in name and "adaLN" not in name:
            cross_attn_params.append(param)
        elif "adaLN_mlp" in name:
            parts = name.split(".")
            try:
                block_idx = int(parts[parts.index("blocks") + 1])
            except (ValueError, IndexError):
                block_idx = 0
            adaLN_per_layer.setdefault(block_idx, []).append(param)
        elif "traj_embed" in name:
            traj_embed_params.append(param)
        else:
            other_params.append(param)
    base_lr = cfg["training"]["lr"]
    n_layers = max(adaLN_per_layer.keys()) + 1 if adaLN_per_layer else 1
    param_groups = [
        {"params": other_params, "weight_decay": base_weight_decay, "lr": base_lr},
    ]
    # Output head: high weight decay to limit gradient monopoly
    if len(output_head_params) > 0:
        param_groups.append({
            "params": output_head_params,
            "weight_decay": output_head_decay,
            "lr": base_lr * output_head_lr_mult,
        })
        if is_main:
            print(f"  Output head LR x{output_head_lr_mult}, WD {output_head_decay} ({len(output_head_params)} params)")
    # Cross-attention: normal LR + high weight decay to prevent norm inflation
    if len(cross_attn_params) > 0:
        param_groups.append({
            "params": cross_attn_params,
            "weight_decay": cross_attn_weight_decay,
            "lr": base_lr,
        })
        if is_main:
            print(f"  Cross-attn LR x1.0, WD {cross_attn_weight_decay} ({len(cross_attn_params)} params)")
    # AdaLN: inverted depth scaling — deeper layers get higher LR
    if adaLN_lr_mult != 1.0 and len(adaLN_per_layer) > 0:
        for layer_idx in sorted(adaLN_per_layer.keys()):
            # Inverted: shallow (low idx) gets lower mult, deep (high idx) gets higher
            # layer 0 -> mult * decay^(n-1), layer n-1 -> mult * decay^0 = mult
            depth_ratio = (n_layers - 1 - layer_idx) / max(n_layers - 1, 1)
            layer_mult = adaLN_lr_mult * (adaLN_layer_decay ** depth_ratio)
            param_groups.append({
                "params": adaLN_per_layer[layer_idx],
                "weight_decay": base_weight_decay,
                "lr": base_lr * layer_mult,
            })
            if is_main:
                print(f"  AdaLN layer {layer_idx} LR x{layer_mult:.1f} ({len(adaLN_per_layer[layer_idx])} params)")
    # Traj embed: higher LR to break freeze
    if len(traj_embed_params) > 0 and traj_embed_lr_mult != 1.0:
        param_groups.append({
            "params": traj_embed_params,
            "weight_decay": base_weight_decay,
            "lr": base_lr * traj_embed_lr_mult,
        })
        if is_main:
            print(f"  Traj embed LR x{traj_embed_lr_mult} ({len(traj_embed_params)} params)")
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr)

    warmup_steps = cfg["training"].get("warmup_steps", 0)
    total_steps = cfg["training"]["epochs"] * len(loader)

    # When resuming, the cosine schedule is recomputed against the new
    # total_steps. This means at the resume point, LR will jump back up
    # from the (previously near-zero) end of the old cosine to whatever
    # the new cosine says at that step — a "warm restart" that gives the
    # model fresh learning signal to keep improving past the old endpoint.
    # If the resume epoch is past the new total (e.g. config reduced
    # epochs), clamp to the end so LR stays at eta_min.
    if warmup_steps > 0:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            eff_step = min(step, total_steps - 1)
            progress = (eff_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(1e-6 / cfg["training"]["lr"], 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    if resume_ckpt is not None and "optimizer_state_dict" in resume_ckpt:
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            if is_main:
                print("  Restored optimizer state from checkpoint")
        except Exception as e:
            if is_main:
                print(f"  Warning: could not restore optimizer state: {e}")

    # TensorBoard — only rank 0 writes
    if is_main:
        timestamp = datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        log_dir = f"output/logs/train/{timestamp}"
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs: {log_dir}")
        print(f"  Launch: tensorboard --logdir=output/logs/train")
    else:
        writer = NoOpWriter()
        timestamp = datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")

    # Broadcast timestamp so all ranks use the same checkpoint dir
    if world_size > 1:
        # Split timestamp into chunks that fit in int64 (7 bytes per long)
        ts_bytes = timestamp.encode("utf-8") if is_main else b"\x00" * 20
        chunk_size = 7
        n_chunks = (len(ts_bytes) + chunk_size - 1) // chunk_size
        # Pad to fixed length
        ts_bytes = ts_bytes.ljust(n_chunks * chunk_size, b"\x00")
        ts_chunks = torch.zeros(n_chunks, dtype=torch.long, device=device)
        if is_main:
            for i in range(n_chunks):
                ts_chunks[i] = int.from_bytes(ts_bytes[i * chunk_size:(i + 1) * chunk_size], "big")
        dist.broadcast(ts_chunks, src=0)
        if not is_main:
            decoded = b""
            for i in range(n_chunks):
                decoded += ts_chunks[i].item().to_bytes(chunk_size, "big")
            timestamp = decoded.decode("utf-8").rstrip("\x00")

    # Training loop
    global_step = 0
    # best_loss now tracks EMA loss (same metric as early stopping) so that
    # best.pt reflects the model with best validation signal, not a noisy
    # single-epoch denoising_loss.
    best_loss = float("inf")
    # Restore best_loss from checkpoint so resumed training keeps the best.pt
    # threshold (otherwise every resume overwrites best.pt with a worse model).
    if resume_ckpt is not None and "loss" in resume_ckpt:
        best_loss = resume_ckpt["loss"]
        if is_main:
            print(f"  Restored best_loss (EMA)={best_loss:.4f} from checkpoint")
    n_epochs = cfg["training"]["epochs"]

    # Dynamic lambda_ep: linearly decay from lambda_endpoint to 1.0 over training
    lambda_ep_start = cfg["training"].get("lambda_endpoint", 2.0)
    lambda_ep_end = cfg["training"].get("lambda_endpoint_end", 1.0)

    # Variables for EMA comparison
    last_batch_cache = {}
    epoch_denoising_loss = 0.0
    best_ema_loss = float("inf")
    patience_counter = 0
    patience = cfg["training"].get("early_stop_patience", 30)
    ema_loss_window = []  # rolling window for smoother early stopping
    smooth_ema_loss = float("inf")  # default for epochs where EMA not computed

    for epoch in range(start_epoch, n_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_denoising_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}", disable=not is_main):
            # Diffusion target is normalized residual
            residual_norm = batch["trajectory"].to(device)
            timesteps = torch.randint(0, diffusion.n_steps, (residual_norm.shape[0],), device=device)
            noise = torch.randn_like(residual_norm)

            x_t = diffusion.add_noise(residual_norm, timesteps, noise)

            with autocast(enabled=use_amp, dtype=amp_dtype):
                noise_pred = model(
                    noisy_traj=x_t,
                    t=timesteps,
                    goal=batch["goal"].to(device),
                    map_tokens=batch["map_tokens"].to(device),
                    map_mask=batch["map_mask"].to(device),
                    neighbor_tokens=batch["neighbor_tokens"].to(device),
                    neighbor_mask=batch["neighbor_mask"].to(device),
                    history=batch["history"].to(device),
                )

                # Dynamic lambda_ep: decay from start→end over training (shape priority)
                progress = global_step / max(1, total_steps)
                lambda_ep = lambda_ep_start + (lambda_ep_end - lambda_ep_start) * progress

                # Endpoint-weighted loss on residual (main denoising loss)
                main_loss = diffusion.endpoint_weighted_loss(noise, noise_pred, lambda_ep=lambda_ep)

                # Log base_loss and endpoint_loss separately
                base_loss = nn.functional.mse_loss(noise[:, :-1], noise_pred[:, :-1])
                ep_loss = nn.functional.mse_loss(noise[:, -1:], noise_pred[:, -1:])

            # Auxiliary losses on denoised x_0 estimate (only for t in [min_t, max_t])
            aux_loss_val = torch.tensor(0.0, device=device)
            apply_aux_loss = False
            aux_losses_dict = None
            if aux_cfg.get("enabled", False):
                aux_interval = aux_cfg.get("interval", 1)
                should_compute_aux = (global_step % aux_interval == 0)

                if should_compute_aux:
                    prior_norm_batch = batch["prior"].to(device)
                    # History positions and velocities for join continuity
                    # losses. The history tensor is (B, T_hist, 6) with
                    # [x, y, vx, vy, ax, ay] normalized; split out positions
                    # and velocity at the last frame for the aux losses.
                    history_full = batch["history"].to(device)
                    history_pos_batch = history_full[..., :2]  # (B, T_hist, 2) normalized
                    # Velocity at the last history frame (B, 2) normalized
                    history_vel_last = history_full[:, -1, 2:4]
                    # Chord-frame bookkeeping
                    use_chord_batch = batch.get("use_chord_frame")
                    if use_chord_batch is not None:
                        use_chord_batch = use_chord_batch.to(device)
                    chord_dir_batch = batch.get("chord_dir")
                    if chord_dir_batch is not None:
                        chord_dir_batch = chord_dir_batch.to(device)

                    aux_min_t = aux_cfg.get("min_t", 0)
                    aux_max_t = aux_cfg.get("max_t", 200)

                    # Dynamic aux scaling: ramp from 0→1 over first 10% of training.
                    # Was 50% (progress*2.0) — too slow, delayed the divergence
                    # signal until epoch ~200. Shorter ramp gets aux loss to full
                    # strength early so its trajectory is visible sooner.
                    aux_ramp = min(1.0, progress * 10.0)

                    aux_losses_dict = compute_auxiliary_losses(
                        diffusion, x_t, noise_pred, timesteps, prior_norm_batch,
                        lambda_smooth=aux_cfg.get("lambda_smooth", 0.05) * aux_ramp,
                        lambda_vel=aux_cfg.get("lambda_vel", 0.05) * aux_ramp,
                        lambda_endpoint=aux_cfg.get("lambda_endpoint", 0.1) * aux_ramp,
                        lambda_curv=aux_cfg.get("lambda_curv", 0.005) * aux_ramp,
                        lambda_prior=aux_cfg.get("lambda_prior", 0.3) * aux_ramp,
                        lambda_join=aux_cfg.get("lambda_join", 0.5) * aux_ramp,
                        prior_margin=aux_cfg.get("prior_margin", 5),
                        min_t=aux_min_t,
                        max_t=aux_max_t,
                        gt_residual=residual_norm,
                        history_pos_norm=history_pos_batch,
                        history_velocity_norm=history_vel_last,
                        use_chord_frame=use_chord_batch,
                        chord_dir=chord_dir_batch,
                    )
                    aux_loss_val = aux_losses_dict["total"].detach()

                    apply_aux_loss = aux_losses_dict["n_valid"] > 0

                    # Always log aux metrics (even when dropped) for monitoring
                    writer.add_scalar("loss/aux_smoothness", aux_losses_dict["smoothness"].item(), global_step)
                    writer.add_scalar("loss/aux_velocity_consistency", aux_losses_dict["velocity_consistency"].item(), global_step)
                    writer.add_scalar("loss/aux_endpoint_consistency", aux_losses_dict["endpoint_consistency"].item(), global_step)
                    writer.add_scalar("loss/aux_curvature", aux_losses_dict["curvature"].item(), global_step)
                    writer.add_scalar("loss/aux_prior_adherence", aux_losses_dict["prior_adherence"].item(), global_step)
                    writer.add_scalar("loss/aux_join_continuity", aux_losses_dict["join_continuity"].item(), global_step)
                    writer.add_scalar("loss/aux_total", aux_losses_dict["total"].item(), global_step)
                    writer.add_scalar("loss/aux_n_valid", aux_losses_dict["n_valid"], global_step)

            # Backward pass: single combined backward on main + aux loss.
            # Previously this used a two-stage backward (aux backward with
            # retain_graph=True, clone grads, zero, main backward, add back)
            # to clip aux gradients separately. The two-stage approach broke
            # DDP's gradient bucket state machine under Gloo and was a
            # contributing cause of the "Connection closed by peer" crashes.
            # Aux lambdas are already small (max 0.5), so direct summation
            # gives equivalent effect without the DDP hazard.

            if apply_aux_loss:
                total_loss = main_loss + aux_losses_dict["total"]

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)

                # Log combined grad norm before final clip
                if grad_norm_interval > 0 and global_step % grad_norm_interval == 0:
                    combined_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf")
                    )
                    writer.add_scalar("grad_norm/aux_before_clip", combined_grad_norm.item(), global_step)

                # Final clip on combined gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            else:
                scaler.scale(main_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Skip optimizer step if gradients contain NaN/Inf (gradient explosion)
            has_bad_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_bad_grad = True
                    break
            if has_bad_grad:
                optimizer.zero_grad()
                if is_main and global_step % 100 == 0:
                    print(f"  WARNING: NaN/Inf gradient detected at step {global_step}, skipping optimizer step")
                continue

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # EMA update (on unwrapped model)
            with torch.no_grad():
                for p_ema, p_model in zip(ema_model.parameters(), model_without_ddp.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p_model.data, alpha=1 - ema_decay)

            epoch_loss += main_loss.item() + aux_loss_val.item()
            epoch_denoising_loss += main_loss.item()
            global_step += 1

            writer.add_scalar("loss/residual_denoising", main_loss.item(), global_step)
            writer.add_scalar("loss/base", base_loss.item(), global_step)
            writer.add_scalar("loss/endpoint", ep_loss.item(), global_step)
            writer.add_scalar("loss/total", main_loss.item() + aux_loss_val.item(), global_step)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("loss/lambda_ep", lambda_ep, global_step)

            # --- Enhanced TensorBoard logging ---

            # Gradient norms
            if grad_norm_interval > 0 and global_step % grad_norm_interval == 0:
                total_grad_norm = 0.0
                per_group_norms = {}
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_grad_norm += param_norm ** 2
                        for group in ["output_head", "adaLN_mlp", "self_attn", "cross_attn", "cond_encoder", "traj_embed", "timestep_embedder"]:
                            if group in name:
                                per_group_norms.setdefault(group, 0.0)
                                per_group_norms[group] += param_norm ** 2
                total_grad_norm = total_grad_norm ** 0.5
                writer.add_scalar("grad_norm/total", total_grad_norm, global_step)
                for group, norm in per_group_norms.items():
                    writer.add_scalar(f"grad_norm/{group}", norm ** 0.5, global_step)

            # Condition statistics
            if condition_stats_interval > 0 and global_step % condition_stats_interval == 0:
                writer.add_histogram("conditions/goal_x", batch["goal"][:, 0], global_step)
                writer.add_histogram("conditions/goal_y", batch["goal"][:, 1], global_step)
                writer.add_histogram("conditions/map_valid_count", batch["map_mask"].sum(dim=1), global_step)
                writer.add_histogram("conditions/neighbor_valid_count", batch["neighbor_mask"].sum(dim=1), global_step)
                residual_mag = residual_norm.norm(dim=-1)
                writer.add_histogram("residual/magnitude_per_frame", residual_mag.flatten(), global_step)

            # Parameter norms
            if param_norm_interval > 0 and global_step % param_norm_interval == 0:
                group_norms = {}
                for name, p in model_without_ddp.named_parameters():
                    pn = p.data.norm().item()
                    # Replace dots so TensorBoard treats them as flat names, not nested groups
                    flat_name = name.replace(".", "/")
                    writer.add_scalar(f"param_norm/{flat_name}", pn, global_step)
                    # Also aggregate by module group
                    for group in ["output_head", "adaLN_mlp", "self_attn", "cross_attn",
                                  "cond_encoder", "traj_embed", "timestep_embedder"]:
                        if group in name:
                            group_norms.setdefault(group, []).append(pn)
                for group, norms in group_norms.items():
                    writer.add_scalar(f"param_norm_group/{group}", sum(norms) / len(norms), global_step)

            # Save last batch for EMA comparison
            last_batch_cache = {
                "x_t": x_t.detach(),
                "timesteps": timesteps.detach(),
                "noise": noise.detach(),
                "goal": batch["goal"].to(device).detach(),
                "map_tokens": batch["map_tokens"].to(device).detach(),
                "map_mask": batch["map_mask"].to(device).detach(),
                "neighbor_tokens": batch["neighbor_tokens"].to(device).detach(),
                "neighbor_mask": batch["neighbor_mask"].to(device).detach(),
                "history": batch["history"].to(device).detach(),
            }

            # Noise comparison visualization
            if train_noise_interval > 0 and global_step % train_noise_interval == 0:
                fig = plot_noise_comparison(noise, noise_pred, timestep=timesteps[0].item())
                log_training_images(writer, {"training/noise_comparison": fig}, global_step)

            # Residual distribution visualization
            if train_residual_interval > 0 and global_step % train_residual_interval == 0:
                fig = plot_residual_distribution(residual_norm)
                log_training_images(writer, {"training/residual_distribution": fig}, global_step)

        avg_loss = epoch_loss / len(loader)
        avg_denoising_loss = epoch_denoising_loss / len(loader)
        writer.add_scalar("loss/epoch_avg", avg_loss, epoch)
        writer.add_scalar("loss/epoch_avg_denoising", avg_denoising_loss, epoch)
        if is_main:
            print(f"Epoch {epoch+1}: avg_denoising={avg_denoising_loss:.4f}, avg_total={avg_loss:.4f}")

        # EMA vs raw model comparison
        if ema_compare_interval > 0 and (epoch + 1) % ema_compare_interval == 0 and last_batch_cache:
            ema_model.eval()
            with torch.no_grad():
                ema_noise_pred = ema_model(
                    noisy_traj=last_batch_cache["x_t"],
                    t=last_batch_cache["timesteps"],
                    goal=last_batch_cache["goal"],
                    map_tokens=last_batch_cache["map_tokens"],
                    map_mask=last_batch_cache["map_mask"],
                    neighbor_tokens=last_batch_cache["neighbor_tokens"],
                    neighbor_mask=last_batch_cache["neighbor_mask"],
                    history=last_batch_cache["history"],
                )
                ema_loss = diffusion.endpoint_weighted_loss(last_batch_cache["noise"], ema_noise_pred, lambda_ep=lambda_ep)
            writer.add_scalar("loss/ema_denoising", ema_loss.item(), global_step)
            ema_model.train()

            # Early stopping based on EMA loss (rolling window of 5 to reduce noise)
            ema_loss_window.append(ema_loss.item())
            if len(ema_loss_window) > 5:
                ema_loss_window.pop(0)
            smooth_ema_loss = sum(ema_loss_window) / len(ema_loss_window)
            if smooth_ema_loss < best_ema_loss:
                best_ema_loss = smooth_ema_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if is_main and patience_counter > 0:
                print(f"  EMA loss no improvement for {patience_counter} epoch(s) (best={best_ema_loss:.4f}, current={smooth_ema_loss:.4f})")
            if patience_counter >= patience:
                if is_main:
                    print(f"Early stopping: EMA loss has not improved for {patience} epochs. Stopping at epoch {epoch+1}.")
                break

        # Sample trajectory visualization (use raw model, not EMA)
        # EMA needs thousands of steps to converge (half-life ~69 steps
        # at decay=0.99, still too slow early on). Raw model shows actual progress.
        #
        # Only rank 0 does the visualization work — other ranks skip it.
        # This is safe because render_sample_trajectory uses model_without_ddp
        # (no DDP communication) and writer is a NoOpWriter on non-main ranks.
        # Without this guard, all ranks wasted time generating samples that
        # only rank 0 could log.
        if train_sample_interval > 0 and (epoch + 1) % train_sample_interval == 0:
            if is_main:
                try:
                    fig = render_sample_trajectory(
                        model_without_ddp, diffusion, batch, batch["prior"][0], device,
                        n_inference_steps=inference_sample_steps,
                        cfg_weight=cfg["inference"]["cfg_weight"],
                        use_chord_frame=batch.get("use_chord_frame"),
                        chord_dir=batch.get("chord_dir"),
                    )
                    log_training_images(writer, {"training/sample_trajectory": fig}, global_step)
                    print(f"  Logged sample trajectory visualization")
                except Exception as e:
                    print(f"  Warning: sample trajectory visualization failed: {e}")

        # Save checkpoint — only rank 0
        if is_main:
            ckpt_dir = Path(f"output/checkpoints/train/{timestamp}")
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Save best.pt based on smooth EMA loss (same metric as early
            # stopping). Was: avg_denoising_loss (single-epoch noisy value).
            # Using EMA loss aligns best.pt with the actual validation signal.
            if smooth_ema_loss < best_loss:
                best_loss = smooth_ema_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model_without_ddp.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": smooth_ema_loss,
                    "config": cfg,
                }, ckpt_dir / "best.pt")
                print(f"  Saved best checkpoint (ema_loss={smooth_ema_loss:.4f})")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model_without_ddp.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": cfg,
            }, ckpt_dir / "latest.pt")

    writer.close()
    cleanup_ddp()
    return str(Path(f"output/checkpoints/train/{timestamp}") / "best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU IDs to use, e.g. '0,1,2,3'")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    n_gpus = len(gpu_ids)
    visible_str = ",".join(str(g) for g in gpu_ids)

    # If not already in DDP mode, relaunch via torchrun with specified GPUs
    if "RANK" not in os.environ:
        print(f"Using GPUs: {gpu_ids}")
        print(f"Launching DDP training with {n_gpus} GPU(s)...")
        new_env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": visible_str,
            # NCCL needs /dev/shm for its shared-memory transport, but this
            # container only has 64MB of /dev/shm (Docker default). NCCL
            # init fails with "Error while creating shared memory segment
            # /dev/shm/nccl-XXX (size ~9.6MB)" when 6 ranks each try to
            # allocate a segment. NCCL_SHM_DISABLE=1 forces NCCL to use
            # NET/IPC transports instead, which works without /dev/shm.
            # This is slower than SHM but far more reliable than the Gloo
            # backend (which suffers "Connection closed by peer" crashes
            # during long allreduce operations).
            "NCCL_SHM_DISABLE": "1",
            # Use simple socket transport for intra-container communication
            "NCCL_SOCKET_IFNAME": "lo",
        }
        os.execvpe(
            "torchrun",
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "-m", "src.train",
                "--config", args.config,
                "--gpus", args.gpus,
                *(["--resume", args.resume] if args.resume else []),
            ],
            new_env,
        )

    # Inside DDP worker process
    print(f"Training goal-conditioned trajectory model (residual prediction)")
    print(f"  Model: TFCrossDenoiser (adaLN-Z + Cross-Attention)")
    print(f"  Epochs: {cfg['training']['epochs']}, Batch: {cfg['training']['batch_size']}")
    print(f"  Endpoint loss weight: {cfg['training'].get('lambda_endpoint', 2.0)} → {cfg['training'].get('lambda_endpoint_end', 1.0)} (dynamic decay)")
    print(f"  GPUs: {gpu_ids}")
    print(f"  Auxiliary losses: {cfg['training'].get('auxiliary_losses', {}).get('enabled', False)}")
    print(f"  Multi-GPU mode: DDP with {n_gpus} GPU(s)")

    ckpt_path = train(cfg, "cuda:0", args.resume)
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"Training complete. Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
