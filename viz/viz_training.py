"""Training-time visualization: sample generation, noise comparison, residual distribution."""

import torch
import numpy as np
import matplotlib.pyplot as plt

from viz.style import COLORS, apply_dark_theme, save_figure, TEXT_COLOR
from viz.utils import to_numpy
from data.normalization import denormalize, denormalize_residual, normalize


def render_sample_trajectory(
    model,
    diffusion,
    batch,
    prior_norm,
    device,
    n_inference_steps: int = 20,
    cfg_weight: float = 2.0,
    use_chord_frame=None,
    chord_dir=None,
):
    """Generate one sample trajectory with EMA model for training monitoring.

    Returns a matplotlib figure showing generated vs GT trajectory.
    """
    from model.dps_guidance import full_dps_sample
    from data.normalization import unpack_chord, denormalize_residual_chord, chord_frame_to_residual

    model.eval()
    with torch.no_grad():
        conditions = {
            "goal": batch["goal"][:1].to(device),
            "map_tokens": batch["map_tokens"][:1].to(device),
            "map_mask": batch["map_mask"][:1].to(device),
            "neighbor_tokens": batch["neighbor_tokens"][:1].to(device),
            "neighbor_mask": batch["neighbor_mask"][:1].to(device),
            "history": batch["history"][:1].to(device),
        }

        use_cf = use_chord_frame[:1].to(device) if use_chord_frame is not None else None
        cd = chord_dir[:1].to(device) if chord_dir is not None else None

        residual_norm = full_dps_sample(
            diffusion, model, conditions,
            traj_len=batch["trajectory"].shape[1],
            n_inference_steps=n_inference_steps,
            cfg_weight=cfg_weight,
            use_cfg=True,
            use_dps=False,
            device=device,
            dynamic_threshold=1.5,
            use_chord_frame=use_cf,
            chord_dir=cd,
        )

    # Reconstruct: prior + denormalized residual (chord-frame aware)
    residual_norm_cpu = residual_norm.squeeze(0).cpu()
    prior = denormalize(prior_norm.cpu())
    if use_chord_frame is not None and use_chord_frame[0].item() == 1.0 and chord_dir is not None:
        r_lon_n, r_lat_n = unpack_chord(residual_norm_cpu)
        r_lon, r_lat = denormalize_residual_chord(r_lon_n, r_lat_n)
        residual = chord_frame_to_residual(r_lon, r_lat, chord_dir[0])
    else:
        residual = denormalize_residual(residual_norm_cpu)
    gen_traj = prior + residual
    gen_np = to_numpy(gen_traj)

    # GT reconstruction
    gt_residual_norm = batch["trajectory"][0]
    if use_chord_frame is not None and use_chord_frame[0].item() == 1.0 and chord_dir is not None:
        gt_lon_n, gt_lat_n = unpack_chord(gt_residual_norm)
        gt_lon, gt_lat = denormalize_residual_chord(gt_lon_n, gt_lat_n)
        gt_residual = chord_frame_to_residual(gt_lon, gt_lat, chord_dir[0])
    else:
        gt_residual = denormalize_residual(gt_residual_norm)
    gt_traj = prior + gt_residual
    gt_np = to_numpy(gt_traj)

    # Goal
    goal = to_numpy(denormalize(batch["goal"][0].cpu().reshape(1, 2)).flatten())

    fig, ax = plt.subplots(figsize=(10, 8))
    apply_dark_theme(ax, fig)

    ax.plot(gen_np[:, 0], gen_np[:, 1], "-", color=COLORS["mean"], linewidth=2, label="Generated")
    ax.plot(gt_np[:, 0], gt_np[:, 1], "--", color=COLORS["gt"], linewidth=1.5, alpha=0.8, label="GT")
    ax.plot(prior[:, 0], prior[:, 1], ":", color=COLORS["prior"], linewidth=1, alpha=0.6, label="Prior")
    ax.plot(goal[0], goal[1], "*", color=COLORS["goal"], markersize=15, label="Goal")
    ax.plot(gt_np[0, 0], gt_np[0, 1], "o", color=COLORS["start"], markersize=8, label="Start")
    ax.set_aspect("equal")
    ax.legend(facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR)
    ax.set_title("Training Sample", color=TEXT_COLOR)
    plt.tight_layout()

    return fig


def plot_noise_comparison(noise, noise_pred, timestep=None, sample_idx=0):
    """Visualize true noise vs predicted noise vs residual for one sample.

    Three-panel layout: True Noise | Predicted Noise | Difference
    """
    n = noise[sample_idx].detach().cpu()
    p = noise_pred[sample_idx].detach().cpu()

    n_np = np.array(n.tolist(), dtype=np.float32)
    p_np = np.array(p.tolist(), dtype=np.float32)
    diff = n_np - p_np

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    apply_dark_theme(axes[0], fig)
    apply_dark_theme(axes[1], fig)
    apply_dark_theme(axes[2], fig)

    for ax, data, title in [
        (axes[0], n_np, "True Noise"),
        (axes[1], p_np, "Predicted Noise"),
        (axes[2], diff, "Difference"),
    ]:
        ax.plot(data[:, 0], label="x", color="steelblue")
        ax.plot(data[:, 1], label="y", color="coral")
        ax.set_title(title, color=TEXT_COLOR)
        ax.legend(facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR, fontsize=8)
        ax.grid(True, alpha=0.2)
        if timestep is not None:
            ax.set_xlabel(f"timestep (diffusion t={timestep})", color=TEXT_COLOR)

    fig.suptitle("Noise Prediction Comparison", color=TEXT_COLOR)
    plt.tight_layout()
    return fig


def plot_residual_distribution(residual_norm, max_samples=256):
    """Histogram of residual magnitudes across the batch.

    Shows what the model needs to correct beyond the linear prior.
    """
    r = residual_norm[:max_samples].detach().cpu()
    r_np = np.array(r.tolist(), dtype=np.float32)
    magnitudes = np.linalg.norm(r_np.reshape(-1, 2), axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    apply_dark_theme(axes[0], fig)
    apply_dark_theme(axes[1], fig)

    # Overall magnitude distribution
    axes[0].hist(magnitudes, bins=50, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[0].axvline(magnitudes.mean(), color="coral", linestyle="--", label=f"mean={magnitudes.mean():.3f}")
    axes[0].set_title("Residual Magnitude Distribution", color=TEXT_COLOR)
    axes[0].set_xlabel("|residual|", color=TEXT_COLOR)
    axes[0].set_ylabel("count", color=TEXT_COLOR)
    axes[0].legend(facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR, fontsize=8)

    # Magnitude over timesteps
    mag_per_step = np.linalg.norm(r_np, axis=-1)  # (B, T)
    mean_mag = mag_per_step.mean(axis=0)
    std_mag = mag_per_step.std(axis=0)
    t = np.arange(len(mean_mag))
    axes[1].plot(t, mean_mag, color="steelblue", linewidth=1.5)
    axes[1].fill_between(t, mean_mag - std_mag, mean_mag + std_mag, alpha=0.2, color="steelblue")
    axes[1].set_title("Residual Magnitude vs Timestep", color=TEXT_COLOR)
    axes[1].set_xlabel("timestep", color=TEXT_COLOR)
    axes[1].set_ylabel("|residual|", color=TEXT_COLOR)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    return fig


def log_training_images(writer, fig_dict, global_step):
    """Log multiple matplotlib figures to TensorBoard.

    Args:
        writer: SummaryWriter instance
        fig_dict: {tag: fig} mapping
        global_step: training step number
    """
    for tag, fig in fig_dict.items():
        writer.add_figure(tag, fig, global_step)
        plt.close(fig)
