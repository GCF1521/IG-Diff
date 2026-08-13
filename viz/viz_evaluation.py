"""Evaluation visualization: metric distributions, per-scenario breakdown, model comparison."""

import numpy as np
import matplotlib.pyplot as plt

from viz.style import apply_dark_theme, save_figure, TEXT_COLOR


def plot_metric_distributions(metrics: dict, save_path: str = None):
    """Violin/box plots showing the distribution of each metric across scenarios."""
    names = list(metrics.keys())
    values = [metrics[n] for n in names]

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 6))
    if len(names) == 1:
        axes = [axes]

    for ax, name, vals in zip(axes, names, values):
        vals = np.array(vals)
        bp = ax.boxplot(vals, patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][0].set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("coral")
            median.set_linewidth(2)
        ax.set_title(name, color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("value", color=TEXT_COLOR)
        apply_dark_theme(ax, fig)

        ax.text(0.5, 0.95, f"mean={vals.mean():.3f}\nstd={vals.std():.3f}",
                transform=ax.transAxes, color=TEXT_COLOR, fontsize=8,
                ha="center", va="top")

    fig.suptitle("Metric Distributions Across Scenarios", color=TEXT_COLOR)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_per_scenario_breakdown(metrics: dict, n_show: int = 20, save_path: str = None):
    """Bar chart of minADE/minFDE per scenario, sorted by difficulty."""
    if "minADE" not in metrics:
        return None

    ade_values = np.array(metrics["minADE"])
    fde_values = np.array(metrics["minFDE"]) if "minFDE" in metrics else np.zeros_like(ade_values)
    n = min(n_show, len(ade_values))

    sort_idx = np.argsort(ade_values)[:n]

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_dark_theme(ax, fig)

    x = np.arange(n)
    width = 0.35
    ax.bar(x - width / 2, ade_values[sort_idx], width, label="minADE", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, fde_values[sort_idx], width, label="minFDE", color="coral", alpha=0.8)

    ax.set_xlabel("Scenario (sorted by difficulty)", color=TEXT_COLOR)
    ax.set_ylabel("Error (m)", color=TEXT_COLOR)
    ax.set_title(f"Per-Scenario Breakdown (top {n})", color=TEXT_COLOR)
    ax.legend(facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_metric_comparison(results: dict, save_path: str = None):
    """Radar chart comparing model variants."""
    variants = list(results.keys())
    metric_names = list(results[variants[0]].keys())
    n_metrics = len(metric_names)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    apply_dark_theme(ax, fig)

    colors = ["steelblue", "coral", "lime", "orange", "purple"]
    for i, variant in enumerate(variants):
        values = [results[variant][m] for m in metric_names]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=variant, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, color=TEXT_COLOR, fontsize=9)
    ax.set_title("Model Comparison", color=TEXT_COLOR, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor="white", edgecolor="gray", labelcolor=TEXT_COLOR)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig