"""Score heatmap visualization overlaid on BEV map."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from viz.style import apply_dark_theme, save_figure, TEXT_COLOR


def compute_score_grid(
    score_fn,         # DrivingScore callable: (traj) -> dict with 'score'
    x_range: tuple = (-50, 50),
    y_range: tuple = (-10, 80),
    grid_res: float = 1.0,
    T: int = 60,
    goal: np.ndarray = None,
) -> np.ndarray:
    """Evaluate driving score at every point in a BEV grid."""
    xs = np.arange(x_range[0], x_range[1], grid_res)
    ys = np.arange(y_range[0], y_range[1], grid_res)
    xx, yy = np.meshgrid(xs, ys)

    score_grid = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            endpoint = np.array([xx[i, j], yy[i, j]])
            traj = np.linspace([0, 0], endpoint, T)
            score_grid[i, j] = score_fn(traj, endpoint)

    return score_grid, [x_range[0], x_range[1], y_range[0], y_range[1]]


def plot_score_heatmap(
    score_grid: np.ndarray,
    map_extent: list,
    sigma: float = 5,
    cmap: str = "RdYlGn",
    alpha: float = 0.6,
    title: str = "Driving Score Heatmap",
    save_path: str = None,
    lane_boundaries: list = None,
):
    """Overlay score heatmap on BEV map."""
    fig, ax = plt.subplots(figsize=(12, 10))
    apply_dark_theme(ax, fig)

    smoothed = gaussian_filter(score_grid, sigma=sigma)
    vmin, vmax = smoothed.min(), smoothed.max()
    if vmax > vmin:
        normalized = (smoothed - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(smoothed)

    im = ax.imshow(
        normalized, extent=map_extent, origin="lower",
        cmap=cmap, alpha=alpha, interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="Score", shrink=0.8)

    if lane_boundaries:
        for lane in lane_boundaries:
            for side in ["left", "right"]:
                pts = lane[side]
                ax.plot(pts[:, 0], pts[:, 1], "-", color="black", linewidth=0.8, alpha=0.5)

    ax.set_aspect("equal")
    ax.set_title(title, color=TEXT_COLOR)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_sub_scores(
    sub_scores: dict,  # {'p_collision': grid, 'p_boundary': grid, ...}
    map_extent: list,
    sigma: float = 5,
    save_path: str = None,
):
    """Plot 4 sub-score heatmaps in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for ax, (name, grid) in zip(axes.flat, sub_scores.items()):
        smoothed = gaussian_filter(grid, sigma=sigma)
        vmin, vmax = smoothed.min(), smoothed.max()
        normalized = (smoothed - vmin) / (vmax - vmin + 1e-8)

        ax.imshow(normalized, extent=map_extent, origin="lower", cmap="hot", alpha=0.6, interpolation="bilinear")
        ax.set_title(name, color=TEXT_COLOR)
        apply_dark_theme(ax, fig)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig