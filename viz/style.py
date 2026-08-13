"""Shared visualization style constants and utilities."""

import matplotlib.pyplot as plt

BG_COLOR = "white"
PANEL_COLOR = "#f5f5f5"
TEXT_COLOR = "black"

COLORS = {
    "gt": "black",
    "generated": "lime",
    "mean": "steelblue",
    "goal": "red",
    "start": "green",
    "neighbor": "orange",
    "focal": "cyan",
    "prior": "purple",
    "lane": "black",
    "centerline": "gray",
}


def get_trajectory_colors(n: int):
    """Return list of n distinct colors for trajectory plotting."""
    cmap = plt.cm.get_cmap("tab20" if n <= 20 else "hsv", max(n, 1))
    return [cmap(i) for i in range(n)]


def apply_dark_theme(ax, fig=None):
    """Apply white background theme to axes and figure."""
    if ax is not None:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(TEXT_COLOR)
    if fig is not None:
        fig.patch.set_facecolor(BG_COLOR)


# Keep alias for backward compat
apply_light_theme = apply_dark_theme


def save_figure(fig, save_path, dpi=150):
    """Save figure with white background and close."""
    fig.savefig(save_path, dpi=dpi, facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)