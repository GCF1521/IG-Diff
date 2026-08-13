"""Trajectory visualization: multiple generated trajectories, confidence bands, endpoint distribution."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
import torch

from viz.style import apply_dark_theme, save_figure, get_trajectory_colors, BG_COLOR, TEXT_COLOR


def _plot_trajectories_on_ax(
    ax, generated, gt, goal=None, prior=None,
    sampled_goals=None, history=None, title=None,
):
    """Draw a single trajectories panel onto the given axes.

    Shared between the standalone figure (plot_trajectories) and the
    top/bottom comparison figure (plot_trajectories_comparison).
    """
    N = len(generated)
    traj_colors = get_trajectory_colors(N)

    # All trajectories — each with a distinct color
    for i, traj in enumerate(generated):
        ax.plot(traj[:, 0], traj[:, 1], "-", color=traj_colors[i],
                alpha=0.5, linewidth=1.0)

    # Mean trajectory
    mean = generated.mean(axis=0)
    std = generated.std(axis=0)
    ax.plot(mean[:, 0], mean[:, 1], "b-", linewidth=2.5, label="Mean")

    # Confidence band
    ax.fill_between(
        mean[:, 0],
        mean[:, 1] - 2 * std[:, 1],
        mean[:, 1] + 2 * std[:, 1],
        alpha=0.15, color="steelblue", label="95% CI",
    )

    # Ground truth
    ax.plot(gt[:, 0], gt[:, 1], "k--", linewidth=2, label="GT")

    # Prior trajectory — prominent solid line so user can verify fit
    if prior is not None:
        ax.plot(prior[:, 0], prior[:, 1], color="cyan", linestyle="-",
                linewidth=2.5, alpha=0.9, label="Prior", zorder=5)

    # History — thick orange line; connector dashed line to mean future start
    if history is not None and len(history) > 0:
        ax.plot(history[:, 0], history[:, 1], color="orange", linestyle="-",
                linewidth=3.0, alpha=0.9, label="History", zorder=6)
        ax.plot(history[0, 0], history[0, 1], "o", color="orange",
                markersize=8, zorder=6)
        # Dashed connector from last history point to mean future start
        ax.plot([history[-1, 0], mean[0, 0]],
                [history[-1, 1], mean[0, 1]],
                color="white", linestyle=":", linewidth=1.2, alpha=0.7,
                zorder=4)

    # Goal point
    if goal is not None:
        ax.plot(goal[0], goal[1], "r*", markersize=15, label="Goal")

    # Sampled goal endpoints (from goal sampling or collision points)
    if sampled_goals is not None:
        ax.scatter(sampled_goals[:, 0], sampled_goals[:, 1], c="red",
                   alpha=0.85, s=80, marker="o", edgecolors="white",
                   linewidths=1.0, label="Sampled goals", zorder=7)

    # Start point (use history[0] if available, else gt[0])
    if history is not None and len(history) > 0:
        ax.plot(history[0, 0], history[0, 1], "go", markersize=8, label="Start")
    else:
        ax.plot(gt[0, 0], gt[0, 1], "go", markersize=8, label="Start")

    # Temporal markers on mean
    for i in range(0, len(mean), 10):
        ax.plot(mean[i, 0], mean[i, 1], "ko", markersize=3)

    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title, color=TEXT_COLOR, fontsize=11)


def plot_trajectories(
    generated: np.ndarray,  # (N, T, 2)
    gt: np.ndarray,         # (T, 2)
    goal: np.ndarray = None,  # (2,)
    prior: np.ndarray = None,  # (T, 2)
    title: str = "Generated Trajectories",
    save_path: str = None,
    sampled_goals: np.ndarray = None,  # (N, 2) sampled goal endpoints
    history: np.ndarray = None,  # (T_hist, 2) history positions for join-continuity check
):
    """Plot multiple generated trajectories with mean, confidence band, and GT.

    The prior is drawn as a prominent solid cyan line so the user can compare
    how well the smoothed generated trajectories fit the prior. When history
    is provided, it is drawn as a thick orange line and a dashed connector
    bridges history -> future, exposing any join discontinuity.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    apply_dark_theme(ax, fig)
    _plot_trajectories_on_ax(
        ax, generated, gt, goal=goal, prior=prior,
        sampled_goals=sampled_goals, history=history, title=title,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=7,
              facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR, fontsize=8)
    plt.tight_layout()
    # Extra space below for legend
    fig.subplots_adjust(bottom=0.15)

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_trajectories_comparison(
    generated_smoothed: np.ndarray,  # (N, T, 2) — after kinematic projection
    generated_raw: np.ndarray,       # (N, T, 2) — before kinematic projection
    gt: np.ndarray,                  # (T, 2)
    goal: np.ndarray = None,
    prior: np.ndarray = None,
    title: str = "Trajectories (Smoothed vs Raw)",
    save_path: str = None,
    sampled_goals: np.ndarray = None,
    history: np.ndarray = None,
):
    """Two-panel figure: top = smoothed, bottom = raw. Same overlays on both.

    Lets the user visually compare how kinematic smoothing pulls the raw
    trajectories toward the prior and damps the oscillations around it.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 14))
    apply_dark_theme(ax_top, fig)
    apply_dark_theme(ax_bot, fig)

    n_s = len(generated_smoothed)
    n_r = len(generated_raw)
    _plot_trajectories_on_ax(
        ax_top, generated_smoothed, gt, goal=goal, prior=prior,
        sampled_goals=sampled_goals, history=history,
        title=f"Smoothed  (N={n_s})",
    )
    _plot_trajectories_on_ax(
        ax_bot, generated_raw, gt, goal=goal, prior=prior,
        sampled_goals=sampled_goals, history=history,
        title=f"Raw  (N={n_r})",
    )

    # Shared legend below the bottom panel (Mean / 95% CI / GT / Prior /
    # History / Goal / Sampled goals / Start). Use a single proxy legend so
    # we don't repeat entries.
    handles, labels = ax_bot.get_legend_handles_labels()
    if handles:
        seen = set()
        unique = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique.append((h, l))
        handles, labels = zip(*unique)
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=min(8, len(handles)),
                   facecolor="white", edgecolor="gray",
                   labelcolor=TEXT_COLOR, fontsize=9)

    fig.suptitle(title, color=TEXT_COLOR, fontsize=13, y=0.995)
    plt.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.subplots_adjust(hspace=0.18, bottom=0.10)

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_endpoint_distribution(
    generated: np.ndarray,  # (N, T, 2)
    goal: np.ndarray = None,
    title: str = "Endpoint Distribution",
    save_path: str = None,
    sampled_goals: np.ndarray = None,  # (N, 2) sampled goal endpoints
):
    """Scatter plot of generated endpoints."""
    fig, ax = plt.subplots(figsize=(8, 6))
    apply_dark_theme(ax, fig)

    N = len(generated)
    traj_colors = get_trajectory_colors(N)
    endpoints = generated[:, -1, :]
    for i, ep in enumerate(endpoints):
        ax.scatter(ep[0], ep[1], c=[traj_colors[i]], s=30, alpha=0.6)

    if goal is not None:
        ax.plot(goal[0], goal[1], "r*", markersize=15, label="Goal")

    # Sampled goal endpoints
    if sampled_goals is not None:
        ax.scatter(sampled_goals[:, 0], sampled_goals[:, 1], c="red",
                   alpha=0.4, s=20, label="Sampled goals")

    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4,
              facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_title(title, color=TEXT_COLOR)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_confidence_ellipses(
    generated: np.ndarray,
    gt: np.ndarray,
    goal: np.ndarray = None,
    n_std: float = 2.0,
    title: str = "Confidence Ellipses",
    save_path: str = None,
):
    """Plot confidence ellipses at every 5th timestep."""
    fig, ax = plt.subplots(figsize=(10, 8))
    apply_dark_theme(ax, fig)

    # GT trajectory
    ax.plot(gt[:, 0], gt[:, 1], "k--", linewidth=1, alpha=0.7, label="GT")

    # Mean trajectory
    mean = generated.mean(axis=0)
    ax.plot(mean[:, 0], mean[:, 1], "b-", linewidth=1.5, label="Mean")

    # Goal point
    if goal is not None:
        ax.plot(goal[0], goal[1], "r*", markersize=15, label="Goal")

    # Confidence ellipses every 5 frames
    for t in range(0, generated.shape[1], 5):
        cov = np.cov(generated[:, t, :].T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=mean[t], width=width, height=height, angle=angle,
                          facecolor="red", alpha=0.15, edgecolor="red", linewidth=1)
        ax.add_patch(ellipse)

    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4,
              facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_title(title, color=TEXT_COLOR)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    if save_path:
        save_figure(fig, save_path)
    return fig


if __name__ == "__main__":
    # Demo with random data
    N, T = 20, 60
    gen = np.random.randn(N, T, 2) * 0.5 + np.linspace(0, 30, T).reshape(1, T, 1) * np.array([1, 0]).reshape(1, 1, 2)
    gt = np.linspace(0, 30, T).reshape(T, 1) * np.array([1, 0]).reshape(1, 2)
    goal = gt[-1]

    plot_trajectories(gen, gt, goal, save_path="output/figures/demo_trajectories.png")
    plot_endpoint_distribution(gen, goal, save_path="output/figures/demo_endpoints.png")
    plot_confidence_ellipses(gen, gt, save_path="output/figures/demo_ellipses.png")
    print("Demo figures saved to output/figures/")