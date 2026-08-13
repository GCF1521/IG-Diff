"""Diffusion denoising process visualization: step-by-step subplot grid."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from viz.style import apply_dark_theme, save_figure, TEXT_COLOR


def plot_denoising_steps(
    saved_steps,  # dict {timestep: (T, 2)} or list of (T, 2) arrays
    gt: np.ndarray = None,
    goal: np.ndarray = None,
    step_labels: list = None,
    title: str = "Denoising Process",
    save_path: str = None,
):
    """Plot denoising process as a subplot grid showing key steps."""
    if isinstance(saved_steps, list):
        if step_labels is not None:
            saved_dict = {str(label): saved_steps[i] for i, label in enumerate(step_labels)}
        else:
            saved_dict = {str(i): saved_steps[i] for i in range(len(saved_steps))}
    else:
        saved_dict = saved_steps

    steps = sorted(saved_dict.keys())
    n = len(steps)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, t in enumerate(steps):
        traj = saved_dict[t]
        axes[i].plot(traj[:, 0], traj[:, 1], "b-o", markersize=2, linewidth=1.5)
        if gt is not None:
            axes[i].plot(gt[:, 0], gt[:, 1], "g--", linewidth=1, alpha=0.5)
        if goal is not None:
            axes[i].plot(goal[0], goal[1], "r*", markersize=10)
        axes[i].set_title(f"t={t}", color=TEXT_COLOR)
        axes[i].set_aspect("equal")
        axes[i].grid(True, alpha=0.3)
        apply_dark_theme(axes[i], fig)

    plt.suptitle(title, color=TEXT_COLOR)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_trajectory_cloud_convergence(
    stages: dict,  # {n_steps: list of (T,2) trajectories}
    gt: np.ndarray = None,
    title: str = "Trajectory Cloud Convergence",
    save_path: str = None,
):
    """Show trajectory samples at different denoising stages."""
    stages_list = sorted(stages.keys())
    n = len(stages_list)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i, n_steps in enumerate(stages_list):
        samples = stages[n_steps]
        for s in samples:
            axes[i].plot(s[:, 0], s[:, 1], "-", color="steelblue", alpha=0.2, linewidth=0.8)
        mean = np.mean(samples, axis=0)
        axes[i].plot(mean[:, 0], mean[:, 1], "b-", linewidth=2)
        if gt is not None:
            axes[i].plot(gt[:, 0], gt[:, 1], "g--", linewidth=2, alpha=0.7)
        axes[i].set_title(f"{n_steps} denoising steps", color=TEXT_COLOR)
        axes[i].set_aspect("equal")
        apply_dark_theme(axes[i], fig)

    plt.suptitle(title, color=TEXT_COLOR)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def animate_denoising(
    frames_data: list,  # list of (T, 2) trajectories at each step
    gt: np.ndarray = None,
    output_path: str = "output/figures/denoising.gif",
    interval: int = 100,
):
    """Create animated GIF of denoising process."""
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def update(frame):
        ax.clear()
        traj = frames_data[frame]
        ax.plot(traj[:, 0], traj[:, 1], "b-o", markersize=3, linewidth=2)
        if gt is not None:
            ax.plot(gt[:, 0], gt[:, 1], "g--", linewidth=2, alpha=0.5, label="GT")
        ax.set_title(f"Step: {frame + 1}/{len(frames_data)}", color=TEXT_COLOR)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-5, 80)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        apply_dark_theme(ax, fig)

    anim = FuncAnimation(fig, update, frames=len(frames_data), interval=interval)
    anim.save(output_path, writer="pillow")
    plt.close(fig)
    print(f"Animation saved to {output_path}")