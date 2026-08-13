"""Attention visualization: self-attention and cross-attention heatmaps."""

import numpy as np
import matplotlib.pyplot as plt

from viz.style import apply_dark_theme, TEXT_COLOR
from viz.utils import to_numpy


def plot_self_attention_heatmap(attn_weights, layer_idx=0, sample_idx=0):
    """Visualize self-attention weights as heatmaps (one per head)."""
    if hasattr(attn_weights, "detach"):
        attn = to_numpy(attn_weights)
    else:
        attn = np.array(attn_weights, dtype=np.float32)

    if attn.ndim == 4:
        attn = attn[sample_idx]

    n_heads = attn.shape[0]
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_rows):
        for j in range(n_cols):
            head_idx = i * n_cols + j
            ax = axes[i, j]
            if head_idx < n_heads:
                im = ax.imshow(attn[head_idx], cmap="hot", interpolation="nearest", aspect="auto")
                ax.set_title(f"Head {head_idx}", color=TEXT_COLOR, fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            apply_dark_theme(ax)

    fig.suptitle(f"Self-Attention (Layer {layer_idx})", color=TEXT_COLOR)
    apply_dark_theme(None, fig)
    plt.tight_layout()
    return fig


def plot_cross_attention_map(attn_weights, condition_labels=None, layer_idx=0, sample_idx=0):
    """Visualize cross-attention weights: trajectory positions -> conditions."""
    if hasattr(attn_weights, "detach"):
        attn = to_numpy(attn_weights)
    else:
        attn = np.array(attn_weights, dtype=np.float32)

    if attn.ndim == 4:
        attn = attn[sample_idx]

    n_heads = attn.shape[0]
    T = attn.shape[1]

    avg_attn = attn.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]})
    apply_dark_theme(axes[0], fig)
    apply_dark_theme(axes[1], fig)

    im = axes[0].imshow(avg_attn, cmap="hot", interpolation="nearest", aspect="auto")
    axes[0].set_xlabel("Condition tokens", color=TEXT_COLOR)
    axes[0].set_ylabel("Trajectory timesteps", color=TEXT_COLOR)
    axes[0].set_title(f"Cross-Attention Avg (Layer {layer_idx})", color=TEXT_COLOR)
    plt.colorbar(im, ax=axes[0], fraction=0.02, pad=0.04)

    if condition_labels is not None:
        cumlen = 0
        for label, length in condition_labels:
            if cumlen > 0:
                axes[0].axvline(cumlen - 0.5, color="black", linewidth=0.5, alpha=0.5)
            axes[0].text(cumlen + length / 2, -2, label, color=TEXT_COLOR,
                         ha="center", va="bottom", fontsize=7, rotation=45)
            cumlen += length

    if condition_labels is not None:
        cumlen = 0
        for label, length in condition_labels:
            section_attn = avg_attn[:, cumlen:cumlen + length].mean(axis=1)
            axes[1].plot(section_attn, label=label, linewidth=1.5)
            cumlen += length
        axes[1].legend(facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR, fontsize=8)
    axes[1].set_xlabel("Trajectory timestep", color=TEXT_COLOR)
    axes[1].set_ylabel("Avg attention weight", color=TEXT_COLOR)
    axes[1].set_title("Attention per Condition Section", color=TEXT_COLOR)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    return fig