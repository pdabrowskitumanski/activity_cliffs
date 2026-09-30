"""
Plotting module.

This module provides functions for visualizing various aspects of activity cliffs
analysis, including similarity distributions, correlation plots, distance matrices,
and statistical distributions.
"""

import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

from src.embeddings import BIT_EMBEDDING_METHODS, CONTINUOUS_EMBEDDING_METHODS
from src.fitting import (
    evaluate_scaled_distance_fit_pdf,
    fit_generalized_gaussian,
    fit_kww_decay,
    fit_normal,
    generalized_gaussian,
    kww_decay,
    sigmoid,
)

# ============================================================================
# Constants
# ============================================================================

# Similarity bins for binned log-Lipschitz analysis
SIMILARITY_BINS = [
    (0.0, 0.5, "<0.5"),
    (0.5, 0.7, "0.5-0.7"),
    (0.7, 0.8, "0.7-0.8"),
    (0.8, 0.9, "0.8-0.9"),
    (0.9, 1.0, ">0.9"),
]

# ============================================================================
# Helper Functions
# ============================================================================

def _format_axes(ax: plt.Axes, grid_alpha: float = 0.3, grid_axis: str | None = None) -> None:
    """
    Apply common formatting to axes (remove top/right spines, add grid).

    Args:
        ax: Matplotlib axes to format.
        grid_alpha: Transparency of grid lines.
        grid_axis: Which axis to show grid on ('x', 'y', or None for both).
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis is None:
        ax.grid(True, alpha=grid_alpha)
    else:
        ax.grid(True, alpha=grid_alpha, axis=grid_axis)


def _save_figure(fig: plt.Figure, output_path: Path | str | None, close: bool = True) -> None:
    """
    Save figure to file if output_path is provided.

    Args:
        fig: Matplotlib figure to save.
        output_path: Path to save the figure. If None, figure is not saved.
        close: Whether to close the figure after saving.
    """
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        if close:
            plt.close(fig)


def _setup_multi_panel_figure(
    n_items: int, n_cols: int | None = None, figsize: tuple[int, int] | None = None
) -> tuple[plt.Figure, np.ndarray]:
    """
    Set up a multi-panel figure with automatic layout.

    Args:
        n_items: Number of items to plot.
        n_cols: Number of columns. If None, uses min(3, n_items).
        figsize: Figure size. If None, auto-calculates based on n_cols and n_rows.

    Returns:
        Tuple of (figure, axes array).
    """
    if n_cols is None:
        n_cols = min(3, n_items)
    n_rows = math.ceil(n_items / n_cols)

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_items == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    return fig, axes


def _hide_unused_axes(axes: np.ndarray, n_used: int) -> None:
    """
    Hide unused axes in a multi-panel figure.

    Args:
        axes: Array of axes.
        n_used: Number of axes that are actually used.
    """
    for idx in range(n_used, len(axes)):
        axes[idx].set_visible(False)


def plot_activity_cliff_fractions(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    title: str = "Activity Cliff Analysis",
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot activity cliff fractions and number of pairs as a function of similarity.

    Creates a two-panel plot:
        - Left panel: Activity cliff fraction vs compound similarity threshold
        - Right panel: Number of pairs vs compound similarity threshold

    Args:
        df: DataFrame with columns 'compound_similarity_threshold', 
            'activity_cliff_fraction', and 'n_pairs'.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Overall title of the plot.
        figsize: Figure size as (width, height).

    Returns:
        matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left panel: Activity cliff fraction
    ax1.plot(
        df["compound_similarity_threshold"],
        df["activity_cliff_fraction"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="#2563eb",
    )
    ax1.set_xlabel("Compound Similarity Threshold", fontsize=12)
    ax1.set_ylabel("Activity Cliff Fraction", fontsize=12)
    ax1.set_title("Activity Cliff Fraction", fontsize=12)
    ax1.set_xlim(0, 1)
    _format_axes(ax1)

    # Right panel: Number of pairs
    ax2.plot(
        df["compound_similarity_threshold"],
        df["n_pairs"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="#dc2626",
    )
    ax2.set_xlabel("Compound Similarity Threshold", fontsize=12)
    ax2.set_ylabel("Number of Pairs", fontsize=12)
    ax2.set_title("Pairs Above Similarity Threshold", fontsize=12)
    ax2.set_xlim(0, 1)
    _format_axes(ax2)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, output_path, close=False)

    return fig


def plot_cliff_fraction_vs_pair_rank(
    df: pd.DataFrame,
    sigmoid_fit: dict[str, float] | None = None,
    output_path: Path | str | None = None,
    title: str = "Cliff Fraction vs Pair Rank",
    dataset_name: str | None = None,
    embedding_method: str | None = None,
    distance_metric: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot cliff fraction vs pair rank percentage with optional sigmoid fit.

    Args:
        df: DataFrame with columns 'percent_rank' and 'cliff_fraction'.
        sigmoid_fit: Dictionary with sigmoid fit parameters (c, a, k, x0).
            If provided, the sigmoid curve will be plotted.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Overall title of the plot.
        dataset_name: Name of the dataset for subtitle.
        embedding_method: Embedding method for subtitle.
        distance_metric: Distance metric for subtitle.
        figsize: Figure size as (width, height).

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot data points
    ax.plot(
        df["percent_rank"],
        df["cliff_fraction"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2563eb",
        label="Data",
    )

    # Plot sigmoid fit if provided
    if sigmoid_fit and not any(np.isnan([sigmoid_fit.get("c", np.nan),
                                          sigmoid_fit.get("a", np.nan),
                                          sigmoid_fit.get("k", np.nan),
                                          sigmoid_fit.get("x0", np.nan)])):
        x_fit = np.linspace(0, 100, 200)
        y_fit = sigmoid(
            x_fit,
            sigmoid_fit["c"],
            sigmoid_fit["a"],
            sigmoid_fit["k"],
            sigmoid_fit["x0"],
        )
        ax.plot(
            x_fit,
            y_fit,
            linewidth=2,
            linestyle="--",
            color="#dc2626",
            label="Sigmoid fit",
            alpha=0.8,
        )

    ax.set_xlabel("Top n% Most Similar Pairs", fontsize=12)
    ax.set_ylabel("Fraction of Cliffs", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(df["cliff_fraction"].max() * 1.1, 0.1))
    _format_axes(ax)

    # Add legend if sigmoid fit is plotted
    if sigmoid_fit and not any(np.isnan([sigmoid_fit.get("c", np.nan),
                                         sigmoid_fit.get("a", np.nan),
                                         sigmoid_fit.get("k", np.nan),
                                         sigmoid_fit.get("x0", np.nan)])):
        ax.legend(loc="best")

    # Add dataset info to title if available
    if dataset_name or (embedding_method and distance_metric):
        subtitle_parts = []
        if dataset_name:
            subtitle_parts.append(dataset_name)
        if embedding_method and distance_metric:
            subtitle_parts.append(f"({embedding_method}, {distance_metric})")
        if subtitle_parts:
            ax.text(
                0.5, 0.98,
                " - ".join(subtitle_parts),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                style="italic",
            )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_cumulative_cliff_fraction_Fn(
    df: pd.DataFrame,
    power_law_fit: dict,
    output_path: Path | str | None = None,
    title: str = "Cumulative cliff fraction F(n)",
    dataset_name: str | None = None,
    embedding_method: str | None = None,
    distance_metric: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot F(n) = (cliff pairs in top n% most similar) / (total cliff pairs), with a · n^s fit.

    Args:
        df: DataFrame with ``percent_rank`` and ``F_n`` columns.
        power_law_fit: Result of ``fit_power_law_cumulative_cliff_fraction`` (keys ``a``, ``s``, ``r_squared``).
        output_path: Where to save the figure.
        title: Plot title.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(
        df["percent_rank"],
        df["F_n"],
        marker="o",
        linewidth=2,
        markersize=7,
        color="#0d9488",
        label=r"$F(n)$ data",
    )
    if power_law_fit.get("success") and power_law_fit.get("a") is not None and power_law_fit.get("s") is not None:
        a = float(power_law_fit["a"])
        s = float(power_law_fit["s"])
        r2 = power_law_fit.get("r_squared")
        x_fit = np.linspace(float(df["percent_rank"].min()), float(df["percent_rank"].max()), 200)
        y_fit = a * np.power(np.maximum(x_fit, 1e-9), s)
        r2_s = f"{r2:.4f}" if r2 is not None and np.isfinite(r2) else "N/A"
        ax.plot(
            x_fit,
            y_fit,
            "--",
            color="#dc2626",
            linewidth=2,
            label=rf"Fit: $a \cdot n^s$ ($s={s:.4g}$, $R^2={r2_s}$)",
            alpha=0.9,
        )
        ax.legend(loc="best")
    ax.set_xlabel(r"Top $n$% most similar pairs", fontsize=12)
    ax.set_ylabel(
        r"$F(n) = \frac{\mathrm{cliff\ pairs\ in\ top\ }n\%}{\mathrm{total\ cliff\ pairs}}$",
        fontsize=11,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, min(1.05, max(float(df["F_n"].max()) * 1.1, 0.05)))
    ax.grid(True, alpha=0.3)
    _format_axes(ax)

    if dataset_name or (embedding_method and distance_metric):
        subtitle_parts = []
        if dataset_name:
            subtitle_parts.append(dataset_name)
        if embedding_method and distance_metric:
            subtitle_parts.append(f"({embedding_method}, {distance_metric})")
        if subtitle_parts:
            ax.text(
                0.5,
                0.98,
                " - ".join(subtitle_parts),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                style="italic",
            )

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_similarity_vs_log_ratio(
    results: dict[str, dict[str, float]],
    output_path: Path | str | None = None,
    title: str = "Similarity vs Log Ratio",
    figsize: tuple[int, int] = (15, 10),
    alpha: float = 0.3,
    similarity_thresholds: list[float] | None = None,
) -> plt.Figure:
    """
    Plot the correlation between compound similarity and log ratio of activities.

    Creates a 2x3 grid of panels, each showing pairs with similarity >= threshold.

    Args:
        results: Dictionary with pair data containing 'similarity' and 'log_ratio'.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Overall title of the plot.
        figsize: Figure size as (width, height).
        alpha: Transparency of scatter points.
        similarity_thresholds: List of 6 similarity thresholds for each panel.
            Default is [0.4, 0.5, 0.6, 0.7, 0.8, 0.9].

    Returns:
        matplotlib Figure object.
    """
    if similarity_thresholds is None:
        similarity_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Extract data from results
    similarities = np.array([data["similarity"] for data in results.values()])
    log_ratios = np.array([data["log_ratio"] for data in results.values()])

    # Find global max for consistent y-axis
    global_max_log_ratio = log_ratios.max() if len(log_ratios) > 0 else 1

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for ax, threshold in zip(axes, similarity_thresholds):
        # Filter pairs with similarity >= threshold
        mask = similarities >= threshold
        filtered_similarities = similarities[mask]
        filtered_log_ratios = log_ratios[mask]

        n_pairs = len(filtered_similarities)

        ax.scatter(
            filtered_similarities,
            filtered_log_ratios,
            alpha=alpha,
            s=10,
            color="#2563eb",
            edgecolors="none",
        )

        ax.set_xlabel("Compound Similarity", fontsize=10)
        ax.set_ylabel("Log Ratio", fontsize=10)
        ax.set_title(f"Similarity ≥ {threshold} (n={n_pairs:,})", fontsize=11)

        ax.set_xlim(threshold, 1)
        ax.set_ylim(0, global_max_log_ratio * 1.05)

        _format_axes(ax)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, output_path, close=False)

    return fig


def plot_conditional_probability(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    title: str = "Conditional Probability Analysis",
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot conditional probability analysis results.

    Creates a 2x2 grid:
        - Top left: Conditional probability vs similarity interval
        - Top right: Mean and median log_ratio vs similarity interval
        - Bottom left: Standard deviation of log_ratio vs similarity interval
        - Bottom right: Tail probabilities vs similarity interval

    Args:
        df: DataFrame from calculate_conditional_probability with columns:
            interval_lower, interval_upper, n_pairs, conditional_probability,
            mean_log_ratio, median_log_ratio, std_log_ratio, tail_prob_X.X columns.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Overall title of the plot.
        figsize: Figure size as (width, height).

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Calculate interval midpoints for x-axis
    midpoints = (df["interval_lower"] + df["interval_upper"]) / 2

    # Color palette
    colors = {
        "primary": "#2563eb",
        "secondary": "#dc2626",
        "tertiary": "#16a34a",
        "quaternary": "#9333ea",
    }

    # Top left: Conditional probability
    ax1 = axes[0, 0]
    ax1.bar(
        midpoints,
        df["conditional_probability"],
        width=df["interval_upper"] - df["interval_lower"],
        color=colors["primary"],
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.set_xlabel("Similarity Interval Midpoint", fontsize=11)
    ax1.set_ylabel("Conditional Probability", fontsize=11)
    ax1.set_title("P(Activity Cliff | Similarity Interval)", fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Top right: Mean and median log_ratio
    ax2 = axes[0, 1]
    ax2.plot(
        midpoints,
        df["mean_log_ratio"],
        marker="o",
        linewidth=2,
        markersize=6,
        color=colors["primary"],
        label="Mean",
    )
    ax2.plot(
        midpoints,
        df["median_log_ratio"],
        marker="s",
        linewidth=2,
        markersize=6,
        color=colors["secondary"],
        label="Median",
    )
    ax2.set_xlabel("Similarity Interval Midpoint", fontsize=11)
    ax2.set_ylabel("Log Ratio", fontsize=11)
    ax2.set_title("Mean & Median Log Ratio by Similarity", fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Bottom left: Standard deviation
    ax3 = axes[1, 0]
    ax3.bar(
        midpoints,
        df["std_log_ratio"],
        width=df["interval_upper"] - df["interval_lower"],
        color=colors["tertiary"],
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    ax3.set_xlabel("Similarity Interval Midpoint", fontsize=11)
    ax3.set_ylabel("Standard Deviation", fontsize=11)
    ax3.set_title("Std Dev of Log Ratio by Similarity", fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Bottom right: Tail probabilities
    ax4 = axes[1, 1]
    tail_cols = [col for col in df.columns if col.startswith("tail_prob_")]
    tail_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(tail_cols)))
    
    for col, color in zip(tail_cols, tail_colors):
        threshold = col.replace("tail_prob_", "")
        ax4.plot(
            midpoints,
            df[col],
            marker="o",
            linewidth=1.5,
            markersize=4,
            color=color,
            label=f"≥ {threshold}",
        )
    
    ax4.set_xlabel("Similarity Interval Midpoint", fontsize=11)
    ax4.set_ylabel("Tail Probability", fontsize=11)
    ax4.set_title("Tail Probabilities P(log_ratio ≥ t)", fontsize=12)
    ax4.set_xlim(0, 1)
    ax4.legend(loc="upper right", fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, output_path, close=False)

    return fig


def plot_lipschitz(
    lipschitz_data: dict,
    output_path: Path | str | None = None,
    title: str = "Lipschitz Statistics",
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot Lipschitz statistics histogram with key percentiles marked.

    Creates a two-panel plot:
        - Left: Histogram of Lipschitz values
        - Right: Cumulative distribution with percentile markers

    Args:
        lipschitz_data: Dictionary from calculate_lipschitz_statistics with:
            - 'lipschitz_values': array of values (optional, for CDF)
            - 'statistics': dict with max, percentile_90, percentile_95, etc.
            - 'histogram': dict with bin_edges and counts
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Overall title of the plot.
        figsize: Figure size as (width, height).

    Returns:
        matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    stats = lipschitz_data["statistics"]
    histogram = lipschitz_data["histogram"]
    bin_edges = np.array(histogram["bin_edges"])
    counts = np.array(histogram["counts"])

    # Calculate bin centers and widths
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Left panel: Histogram
    ax1.bar(
        bin_centers,
        counts,
        width=bin_widths,
        color="#2563eb",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )

    # Add percentile lines
    percentile_colors = {
        "percentile_90": ("#f59e0b", "90th"),
        "percentile_95": ("#ef4444", "95th"),
        "percentile_99": ("#7c3aed", "99th"),
    }
    
    for key, (color, label) in percentile_colors.items():
        val = stats[key]
        ax1.axvline(val, color=color, linestyle="--", linewidth=2, label=f"{label}: {val:.2f}")

    ax1.set_xlabel("Lipschitz Value (log_ratio / (1 - similarity))", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution of Lipschitz Values", fontsize=12)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right panel: Cumulative distribution or statistics summary
    # If lipschitz_values array is available, plot CDF
    if "lipschitz_values" in lipschitz_data:
        values = np.sort(lipschitz_data["lipschitz_values"])
        cdf = np.arange(1, len(values) + 1) / len(values)
        
        ax2.plot(values, cdf, color="#2563eb", linewidth=2)
        
        # Mark percentiles on CDF
        for key, (color, label) in percentile_colors.items():
            val = stats[key]
            percentile_val = float(key.split("_")[1]) / 100
            ax2.axhline(percentile_val, color=color, linestyle=":", alpha=0.5)
            ax2.axvline(val, color=color, linestyle="--", linewidth=1.5)
            ax2.scatter([val], [percentile_val], color=color, s=50, zorder=5)
        
        ax2.set_xlabel("Lipschitz Value", fontsize=11)
        ax2.set_ylabel("Cumulative Probability", fontsize=11)
        ax2.set_title("Cumulative Distribution Function", fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
    else:
        # Show statistics as text if no raw values
        stats_text = (
            f"Max: {stats['max']:.2f}\n"
            f"99th percentile: {stats['percentile_99']:.2f}\n"
            f"95th percentile: {stats['percentile_95']:.2f}\n"
            f"90th percentile: {stats['percentile_90']:.2f}\n"
            f"Mean: {stats['mean']:.2f}\n"
            f"Median: {stats['median']:.2f}\n"
            f"Std: {stats['std']:.2f}\n"
            f"Min: {stats['min']:.2f}"
        )
        ax2.text(
            0.5, 0.5, stats_text,
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment='center',
            horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        ax2.set_title("Statistics Summary", fontsize=12)
        ax2.axis('off')

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, output_path, close=False)

    return fig


def plot_persistence_diagram(
    dgms: list[np.ndarray],
    output_path: Path | str | None = None,
    title: str = "Persistence Diagram",
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot persistence diagrams.
    
    Creates a two-panel plot:
        - Left: Persistence diagram (birth vs death scatter plot)
        - Right: Barcode diagram
    
    Args:
        dgms: List of persistence diagrams (numpy arrays with birth/death pairs).
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Plot title.
        figsize: Figure size as (width, height).
    
    Returns:
        matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    
    # Left: Persistence diagram (birth vs death)
    max_val = 0
    for dim, dgm in enumerate(dgms):
        if len(dgm) > 0:
            finite_mask = ~np.isinf(dgm[:, 1])
            finite_dgm = dgm[finite_mask]
            infinite_dgm = dgm[~finite_mask]
            
            color = colors[dim % len(colors)]
            
            if len(finite_dgm) > 0:
                ax1.scatter(
                    finite_dgm[:, 0], finite_dgm[:, 1],
                    c=color, alpha=0.6, s=20, label=f"H{dim}"
                )
                max_val = max(max_val, np.max(finite_dgm))
            
            # Plot infinite features at the top
            if len(infinite_dgm) > 0:
                ax1.scatter(
                    infinite_dgm[:, 0], 
                    np.full(len(infinite_dgm), max_val * 1.1 if max_val > 0 else 1),
                    c=color, alpha=0.6, s=20, marker="^"
                )
    
    # Add diagonal line
    if max_val > 0:
        ax1.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3)
        ax1.set_xlim(-0.05 * max_val, max_val * 1.15)
        ax1.set_ylim(-0.05 * max_val, max_val * 1.15)
    
    ax1.set_xlabel("Birth", fontsize=11)
    ax1.set_ylabel("Death", fontsize=11)
    ax1.set_title("Persistence Diagram", fontsize=12)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    # Right: Barcode diagram
    y_offset = 0
    for dim, dgm in enumerate(dgms):
        if len(dgm) > 0:
            color = colors[dim % len(colors)]
            for birth, death in dgm:
                if np.isinf(death):
                    death = max_val * 1.1 if max_val > 0 else 1
                ax2.plot([birth, death], [y_offset, y_offset], c=color, linewidth=1.5)
                y_offset += 1
            y_offset += 2  # Gap between dimensions
    
    ax2.set_xlabel("Filtration Value", fontsize=11)
    ax2.set_ylabel("Feature Index", fontsize=11)
    ax2.set_title("Barcode Diagram", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_persistence_landscape(
    dgms: list[np.ndarray],
    output_path: Path | str | None = None,
    *,
    start: float = 0.0,
    stop: float = 1.0,
    num_steps: int = 500,
    max_layers: int = 5,
    title: str = "Persistence landscapes",
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """
    Plot Bubenik persistence landscapes (piecewise-linear / tent functions via persim).

    One panel per homology dimension that has at least one interval; overlays the
    first ``max_layers`` landscape functions λ₁, λ₂, … (``PersLandscapeApprox``).

    ``dgms`` should use finite death values (replace ``inf`` with the grid endpoint).
    """
    try:
        from persim import PersLandscapeApprox
    except ImportError:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "persim is not installed; cannot plot persistence landscapes.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.suptitle(title, fontsize=14, fontweight="bold")
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    dims_with_points = [i for i, d in enumerate(dgms) if len(d) > 0]
    if not dims_with_points:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No finite persistence intervals to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.suptitle(title, fontsize=14, fontweight="bold")
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    n_panels = len(dims_with_points)
    if figsize is None:
        figsize = (max(12, 5 * n_panels), 5)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()
    t = np.linspace(start, stop, num_steps, dtype=float)
    layer_colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ca8a04"]

    for ax_idx, dim in enumerate(dims_with_points):
        ax = axes_flat[ax_idx]
        pla = PersLandscapeApprox(
            dgms=dgms,
            hom_deg=dim,
            start=start,
            stop=stop,
            num_steps=num_steps,
        )
        # Number of landscape functions can be smaller than |diagram| (persim grid).
        n_land = int(pla.values.shape[0])
        n_layers_plot = min(max_layers, max(1, n_land))
        for k in range(n_layers_plot):
            y = np.asarray(pla[k], dtype=float)
            c = layer_colors[k % len(layer_colors)]
            ax.plot(t, y, color=c, linewidth=1.5, label=f"λ_{k + 1}")
        ax.set_xlabel("Filtration parameter t", fontsize=11)
        ax.set_ylabel(f"λ_k(t)  (H{dim})", fontsize=11)
        ax.set_title(f"H{dim} persistence landscape", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_threshold_comparison(
    results: dict,
    thresholds: list[float],
    output_path: Path | str | None = None,
    title: str = "Persistent Homology vs Similarity Threshold",
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Create a comparison plot showing how persistent homology changes across similarity thresholds.

    Creates a 2x2 grid showing:
        - Number of features vs threshold
        - Max persistence vs threshold
        - Mean persistence vs threshold
        - Total persistence vs threshold

    Args:
        results: Dictionary mapping threshold to PH results with 'statistics' key.
        thresholds: List of similarity thresholds.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Overall title of the plot.
        figsize: Figure size as (width, height).

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Collect statistics across thresholds
    dims_to_plot = [0, 1]
    metrics = ["n_features", "max_persistence", "mean_persistence", "total_persistence"]
    metric_labels = ["Number of Features", "Max Persistence", "Mean Persistence", "Total Persistence"]
    
    for ax, metric, label in zip(axes.flatten(), metrics, metric_labels):
        for dim in dims_to_plot:
            values = []
            for threshold in thresholds:
                stats = results[threshold]["statistics"]
                dim_key = f"dim_{dim}"
                if dim_key in stats and metric in stats[dim_key]:
                    values.append(stats[dim_key][metric])
                else:
                    values.append(0)
            
            ax.plot(thresholds, values, marker="o", linewidth=2, markersize=6, label=f"H{dim}")
        
        ax.set_xlabel("Similarity Threshold", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(loc="best")
        _format_axes(ax)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_accuracy_vs_threshold(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    title: str = "Model Performance vs Similarity Threshold",
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot model performance metrics as a function of similarity threshold.

    Creates a two-panel plot:
        - Left: Accuracy, Precision, Recall, F1 vs threshold
        - Right: Class distribution (positive/negative samples) vs threshold

    Args:
        df: DataFrame with columns: similarity_threshold, accuracy, precision,
            recall, f1, n_positive, n_negative.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Overall title of the plot.
        figsize: Figure size as (width, height).

    Returns:
        matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    thresholds = df["similarity_threshold"]

    # Left panel: Performance metrics
    metrics = [
        ("accuracy", "#2563eb", "Accuracy"),
        ("precision", "#dc2626", "Precision"),
        ("recall", "#16a34a", "Recall"),
        ("f1", "#9333ea", "F1 Score"),
    ]

    for col, color, label in metrics:
        ax1.plot(
            thresholds, df[col],
            marker="o", linewidth=2, markersize=6,
            color=color, label=label
        )

    ax1.set_xlabel("Similarity Threshold", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Performance Metrics", fontsize=12)
    ax1.set_xlim(thresholds.min() - 0.05, thresholds.max() + 0.05)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right panel: Class distribution
    width = 0.035
    x = np.array(thresholds)
    
    ax2.bar(x - width/2, df["n_positive"], width, label="Positive (Cliffs)", color="#dc2626", alpha=0.7)
    ax2.bar(x + width/2, df["n_negative"], width, label="Negative (Non-cliffs)", color="#2563eb", alpha=0.7)

    ax2.set_xlabel("Similarity Threshold", fontsize=11)
    ax2.set_ylabel("Number of Samples", fontsize=11)
    ax2.set_title("Test Set Class Distribution", fontsize=12)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, output_path, close=False)

    return fig
def plot_vector_length_statistics_multi(
    embedding_stats: dict[str, dict],
    output_path: Path | str | None = None,
    title: str = "Vector Length Statistics by Embedding",
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """
    Plot vector length statistics for multiple embeddings as a multi-panel figure.
    
    Args:
        embedding_stats: Dict mapping embedding name to statistics dict with 'normal_fit'.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Plot title.
        figsize: Figure size as (width, height). Auto-calculated if None.
    
    Returns:
        matplotlib Figure object.
    """
    fig, axes = _setup_multi_panel_figure(len(embedding_stats), n_cols=3, figsize=figsize)
    
    for idx, (emb_name, stats) in enumerate(embedding_stats.items()):
        ax = axes[idx]
        
        # Get vector lengths from stats
        vector_lengths = stats.get("_vector_lengths", None)
        normal_fit = stats.get("normal_fit", {})
        
        if vector_lengths is not None and len(vector_lengths) > 0:
            mu = normal_fit.get("mu", np.mean(vector_lengths))
            sigma = normal_fit.get("sigma", np.std(vector_lengths))
            
            counts, bin_edges, _ = ax.hist(
                vector_lengths, bins=50, color="#2563eb", alpha=0.7,
                edgecolor="white", density=True, label="Data"
            )
            
            # Overlay Gaussian fit
            x_fit = np.linspace(bin_edges[0], bin_edges[-1], 200)
            y_fit = scipy_stats.norm.pdf(x_fit, mu, sigma)
            ax.plot(x_fit, y_fit, color="#dc2626", linewidth=2,
                    label=f"μ={mu:.2f}, σ={sigma:.2f}")
            
            ax.set_ylabel("Density", fontsize=9)
        
        ax.set_xlabel("Vector Length (L2 Norm)", fontsize=9)
        ax.set_title(emb_name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    _hide_unused_axes(axes, len(embedding_stats))
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, output_path)
    
    return fig


def plot_umap_projections_multi(
    umap_projections: dict[str, np.ndarray],
    activities: np.ndarray,
    output_path: Path | str | None = None,
    title: str = "UMAP Projections by Embedding",
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """
    Plot UMAP projections for multiple embeddings as a multi-panel figure.
    
    Args:
        umap_projections: Dict mapping embedding name to projection array (n_samples, 2).
        activities: Array of activity values for coloring points.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Plot title.
        figsize: Figure size as (width, height). Auto-calculated if None.
    
    Returns:
        matplotlib Figure object.
    """
    fig, axes = _setup_multi_panel_figure(len(umap_projections), n_cols=3, figsize=figsize)
    
    for idx, (emb_name, projection) in enumerate(umap_projections.items()):
        ax = axes[idx]
        
        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c=activities,
            cmap="viridis",
            alpha=0.7,
            s=10,
            edgecolors="none",
        )
        
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.set_title(emb_name, fontsize=10, fontweight="bold")
        _format_axes(ax)
    
    _hide_unused_axes(axes, len(umap_projections))
    
    # Add colorbar attached to the figure, not to a specific panel
    if len(umap_projections) > 0:
        # Create a dedicated axes for the colorbar on the right side of the figure
        # Position: [left, bottom, width, height] in figure coordinates (0-1)
        # Leave some margin at top and bottom, and position on the right
        cbar_left = 0.92  # Position on the right side
        cbar_bottom = 0.1  # Leave some margin at bottom
        cbar_width = 0.02  # Width of colorbar
        cbar_height = 0.8  # Height (most of the figure)
        
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        
        # Create colorbar using the scatter plot's colormap and values
        cbar = fig.colorbar(scatter, cax=cbar_ax, label="Log Activity")
        cbar.ax.yaxis.label.set_fontsize(9)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    # Adjust layout to leave space on the right for colorbar
    plt.tight_layout(rect=[0, 0, 0.90, 1])
    
    _save_figure(fig, output_path)
    return fig


def plot_distance_from_center_multi(
    center_stats: dict[str, dict],
    output_path: Path | str | None = None,
    title: str = "Distance from Center by Embedding",
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """
    Plot distance from center statistics for multiple embeddings as a multi-panel figure.
    
    Args:
        center_stats: Dict mapping embedding name to stats dict with '_distances'.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Plot title.
        figsize: Figure size as (width, height). Auto-calculated if None.
    
    Returns:
        matplotlib Figure object.
    """
    n_embeddings = len(center_stats)
    n_cols = min(3, n_embeddings)
    n_rows = math.ceil(n_embeddings / n_cols)
    
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_embeddings == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (emb_name, stats) in enumerate(center_stats.items()):
        ax = axes[idx]
        
        distances = stats.get("_distances", None)
        
        if distances is not None and len(distances) > 0:
            ax.hist(distances, bins=50, color="#0891b2", alpha=0.7, edgecolor="white")
            ax.axvline(np.mean(distances), color="#dc2626", linestyle="--",
                       label=f"Mean: {np.mean(distances):.3f}")
            ax.axvline(np.median(distances), color="#16a34a", linestyle="--",
                       label=f"Median: {np.median(distances):.3f}")
            ax.legend(fontsize=7)
        
        ax.set_xlabel("Distance from Center", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(emb_name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Hide unused axes
    for idx in range(n_embeddings, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    _save_figure(fig, output_path)
    return fig


def plot_activity_statistics(
    all_activities: np.ndarray,
    uncensored_activities: np.ndarray,
    mixture_fit: dict,
    output_path: Path | str | None = None,
    title: str = "Activity Statistics",
    figsize: tuple[int, int] = (16, 5),
) -> plt.Figure:
    """
    Plot activity statistics as a 1x3 panel.
    
    Creates a 1x3 panel plot:
        - Left: Histogram of all compounds activities
        - Center: Histogram of uncensored activities with Normal-Uniform mixture fit
        - Right: Q-Q plot for uncensored activities vs Normal-Uniform mixture
    
    Args:
        all_activities: Array of all log activity values.
        uncensored_activities: Array of uncensored log activity values.
        mixture_fit: Dictionary with mixture distribution fit parameters.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Plot title.
        figsize: Figure size as (width, height).
    
    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Left: Histogram of all activities
    axes[0].hist(all_activities, bins=50, color="#9333ea", alpha=0.7, edgecolor="white")
    axes[0].set_xlabel("Log Activity (-log₁₀)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title(f"All Compounds (n={len(all_activities):,})", fontsize=12)
    axes[0].axvline(np.mean(all_activities), color="#dc2626", linestyle="--",
                    label=f"Mean: {np.mean(all_activities):.3f}")
    axes[0].axvline(np.median(all_activities), color="#16a34a", linestyle="--",
                    label=f"Median: {np.median(all_activities):.3f}")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    
    # Center: Histogram of uncensored activities with mixture fit
    p = mixture_fit["p"]
    mu = mixture_fit["mu"]
    sigma = mixture_fit["sigma"]
    a = mixture_fit["a"]
    b = mixture_fit["b"]
    
    counts, bin_edges, _ = axes[1].hist(
        uncensored_activities, bins=50, color="#2563eb", alpha=0.7, 
        edgecolor="white", density=True, label="Data"
    )
    
    # Overlay mixture distribution
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 200)
    normal_part = p * scipy_stats.norm.pdf(x_fit, mu, sigma)
    uniform_part = (1 - p) * scipy_stats.uniform.pdf(x_fit, loc=a, scale=b - a)
    y_fit = normal_part + uniform_part
    
    axes[1].plot(x_fit, y_fit, color="#dc2626", linewidth=2.5,
                 label=f"Mixture (p={p:.2f})")
    axes[1].plot(x_fit, normal_part, color="#f59e0b", linewidth=1.5, linestyle="--",
                 alpha=0.7, label=f"Normal (μ={mu:.2f}, σ={sigma:.2f})")
    axes[1].plot(x_fit, uniform_part, color="#10b981", linewidth=1.5, linestyle="--",
                 alpha=0.7, label=f"Uniform (a={a:.2f}, b={b:.2f})")
    
    axes[1].set_xlabel("Log Activity (-log₁₀)", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title(f"Uncensored Compounds (n={len(uncensored_activities):,})", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    
    # Right: Q-Q plot for uncensored activities vs mixture
    # For Q-Q plot, we need to compute theoretical quantiles from the mixture
    sorted_data = np.sort(uncensored_activities)
    n = len(sorted_data)
    theoretical_quantiles = np.linspace(0.01, 0.99, n)  # Avoid 0 and 1 for numerical stability
    
    # Compute quantiles from mixture CDF (inverse CDF)
    def mixture_cdf(x, p, mu, sigma, a, b):
        """Mixture CDF"""
        normal_cdf = p * scipy_stats.norm.cdf(x, mu, sigma)
        uniform_cdf = (1 - p) * scipy_stats.uniform.cdf(x, loc=a, scale=b - a)
        return normal_cdf + uniform_cdf
    
    # Find quantiles by solving mixture_cdf(x) = q for each q
    mixture_quantiles = []
    for q in theoretical_quantiles:
        # Use binary search to find quantile
        x_low = np.min(uncensored_activities) - 5 * sigma
        x_high = np.max(uncensored_activities) + 5 * sigma
        for _ in range(50):  # Binary search iterations
            x_mid = (x_low + x_high) / 2
            cdf_val = mixture_cdf(x_mid, p, mu, sigma, a, b)
            if cdf_val < q:
                x_low = x_mid
            else:
                x_high = x_mid
        mixture_quantiles.append((x_low + x_high) / 2)
    
    mixture_quantiles = np.array(mixture_quantiles)
    
    axes[2].scatter(mixture_quantiles, sorted_data, alpha=0.6, s=20, color="#2563eb")
    
    # Add diagonal reference line
    min_val = min(np.min(mixture_quantiles), np.min(sorted_data))
    max_val = max(np.max(mixture_quantiles), np.max(sorted_data))
    axes[2].plot([min_val, max_val], [min_val, max_val], 
                 color="#dc2626", linestyle="--", linewidth=1.5, label="y=x")
    
    axes[2].set_xlabel("Theoretical Quantiles (Mixture)", fontsize=11)
    axes[2].set_ylabel("Sample Quantiles", fontsize=11)
    axes[2].set_title("Q-Q Plot (vs Mixture)", fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    _save_figure(fig, output_path)
    return fig


def plot_log_ratio_statistics(
    log_ratios: np.ndarray,
    log_ratio_stats: dict,
    output_path: Path | str | None = None,
    title: str = "Pairwise Activity Differences (log_ratio)",
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot histogram of log_ratio values with Kohlrausch-Williams-Watts decay fit.
    
    Args:
        log_ratios: Array of log_ratio values.
        log_ratio_stats: Dictionary from calculate_log_ratio_statistics.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Plot title.
        figsize: Figure size as (width, height).
    
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    stats = log_ratio_stats["statistics"]
    kww_fit = log_ratio_stats.get("kww_fit", {})
    
    # Create histogram
    counts, bin_edges = np.histogram(log_ratios, bins=50, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Plot histogram
    ax.bar(
        bin_centers,
        counts,
        width=bin_widths,
        color="#2563eb",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        label="Data",
    )
    
    # Overlay Kohlrausch-Williams-Watts decay fit if available
    a = kww_fit.get("a")
    k = kww_fit.get("k")
    b = kww_fit.get("b")
    if a is not None and k is not None and b is not None and a > 0:
        # Scale fit to histogram counts
        x_fit = np.linspace(max(bin_edges[0], 0.01), bin_edges[-1], 200)
        bin_width = bin_widths[0]
        n_filtered = kww_fit.get("n_values_used", len(log_ratios))
        y_fit = kww_decay(x_fit, a, k, b) * n_filtered * bin_width
        r_squared = kww_fit.get("r_squared", 0)
        ax.plot(
            x_fit, y_fit, 
            color="#dc2626", 
            linewidth=2.5, 
            linestyle="-",
            label=f"KWW: a={a:.2f}, k={k:.2f}, b={b:.2f}, R²={r_squared:.3f}",
        )
    
    # Set y-axis limit to maximum of histogram counts
    ax.set_ylim(0, counts.max() * 1.1)
    ax.set_xlabel("Log Ratio (|log_activity₁ - log_activity₂|)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{title} (n={stats['n_pairs']:,} pairs)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    _format_axes(ax, grid_axis="y")
    
    plt.tight_layout()
    
    _save_figure(fig, output_path)
    return fig


def plot_embedding_center_statistics_multi(
    center_stats: dict[str, dict],
    output_path: Path | str | None = None,
    title: str = "Mean per Dimension histograms by embedding",
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """
    Plot histogram of mean per dimension for multiple embeddings with Gaussian fit.
    
    Args:
        center_stats: Dict mapping embedding name to stats dict with '_mean_per_dim_array'.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Plot title.
        figsize: Figure size as (width, height). Auto-calculated if None.
    
    Returns:
        matplotlib Figure object.
    """
    n_embeddings = len(center_stats)
    n_cols = min(3, n_embeddings)
    n_rows = math.ceil(n_embeddings / n_cols)
    
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_embeddings == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (emb_name, stats) in enumerate(center_stats.items()):
        ax = axes[idx]
        
        # Get mean per dimension from stats
        mean_per_dim = stats.get("_mean_per_dim_array", None)
        
        if mean_per_dim is not None and len(mean_per_dim) > 0:
            # Determine embedding type and fit accordingly
            # Use explicit list check to ensure matching
            is_continuous = emb_name in CONTINUOUS_EMBEDDING_METHODS
            is_bit = emb_name in BIT_EMBEDDING_METHODS
            
            # Plot histogram with density=True
            counts, bin_edges, _ = ax.hist(
                mean_per_dim, bins=50, color="#2563eb", alpha=0.7,
                edgecolor="white", density=True, label="Data"
            )
            
            x_fit = np.linspace(bin_edges[0], bin_edges[-1], 200)
            
            if is_continuous:
                # Fit generalized Gaussian for continuous embeddings
                gen_gauss_fit, bin_centers_fit, fitted_values_fit = fit_generalized_gaussian(
                    mean_per_dim, n_bins=50, percentile_cutoff=99.0
                )
                
                if gen_gauss_fit.get("a", 0) > 0 and len(bin_centers_fit) > 0:
                    a = gen_gauss_fit["a"]
                    u = gen_gauss_fit["u"]
                    s = gen_gauss_fit["s"]
                    b = gen_gauss_fit["b"]
                    r_squared = gen_gauss_fit.get("r_squared", 0.0)
                    
                    # Overlay generalized Gaussian fit
                    y_fit = generalized_gaussian(x_fit, a, u, s, b)
                    ax.plot(x_fit, y_fit, color="#dc2626", linewidth=2,
                            label=f"Gen. Gauss (u={u:.3f}, s={s:.3f}, b={b:.3f}, R²={r_squared:.3f})")
                else:
                    # Fallback to normal Gaussian if fit failed
                    normal_fit = fit_normal(mean_per_dim)
                    mu = normal_fit.get("mu", np.mean(mean_per_dim))
                    sigma = normal_fit.get("sigma", np.std(mean_per_dim))
                    y_fit = scipy_stats.norm.pdf(x_fit, mu, sigma)
                    ax.plot(x_fit, y_fit, color="#dc2626", linewidth=2,
                            label=f"Gaussian (μ={mu:.3f}, σ={sigma:.3f})")
            
            elif is_bit:
                # Fit KWW decay for bit embeddings
                kww_fit, bin_centers_fit, fitted_values_fit = fit_kww_decay(
                    mean_per_dim, n_bins=50, percentile_cutoff=99.0
                )
                
                # Check if fit succeeded - be more lenient with the check
                a_val = kww_fit.get("a", 0.0)
                k_val = kww_fit.get("k", 0.0)
                b_val = kww_fit.get("b", 1.0)
                r_squared = kww_fit.get("r_squared", 0.0)
                
                # If we have valid bin centers and fitted values, use KWW even if a is small
                if len(bin_centers_fit) > 0 and len(fitted_values_fit) > 0 and np.any(fitted_values_fit > 0):
                    # Plot KWW fit at bin centers (fitted_values_fit is already scaled to density)
                    ax.plot(bin_centers_fit, fitted_values_fit, color="#dc2626", linewidth=2,
                            label=f"KWW (a={a_val:.3f}, k={k_val:.3f}, b={b_val:.3f}, R²={r_squared:.3f})")
                else:
                    # If KWW fit completely failed, try a simpler approach: fit KWW directly to histogram
                    # Create histogram for direct fitting
                    counts, bin_edges = np.histogram(mean_per_dim, bins=50, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    bin_widths = bin_edges[1:] - bin_edges[:-1]
                    
                    # Try to fit KWW to the histogram density
                    try:
                        # Initial guess for KWW parameters
                        a0 = np.max(counts)
                        k0 = 1.0 / (np.mean(bin_centers) + 1e-10)
                        b0 = 0.5
                        
                        # Filter out zero counts
                        mask = counts > 0
                        if np.sum(mask) >= 5:
                            popt, _ = curve_fit(
                                kww_decay,
                                bin_centers[mask],
                                counts[mask],
                                p0=[a0, k0, b0],
                                bounds=([0, 0, 0.01], [np.inf, np.inf, 2.0]),
                                maxfev=10000,
                            )
                            a_fit, k_fit, b_fit = popt
                            # Evaluate fit
                            y_fit_kww = kww_decay(x_fit, a_fit, k_fit, b_fit)
                            ax.plot(x_fit, y_fit_kww, color="#dc2626", linewidth=2,
                                    label=f"KWW (a={a_fit:.3f}, k={k_fit:.3f}, b={b_fit:.3f})")
                        else:
                            raise ValueError("Not enough data points")
                    except Exception:
                        # Final fallback to normal Gaussian if all KWW attempts fail
                        normal_fit = fit_normal(mean_per_dim)
                        mu = normal_fit.get("mu", np.mean(mean_per_dim))
                        sigma = normal_fit.get("sigma", np.std(mean_per_dim))
                        y_fit = scipy_stats.norm.pdf(x_fit, mu, sigma)
                        ax.plot(x_fit, y_fit, color="#dc2626", linewidth=2,
                                label=f"Gaussian (μ={mu:.3f}, σ={sigma:.3f})")
            else:
                # Unknown type, use normal Gaussian
                normal_fit = fit_normal(mean_per_dim)
                mu = normal_fit.get("mu", np.mean(mean_per_dim))
                sigma = normal_fit.get("sigma", np.std(mean_per_dim))
                y_fit = scipy_stats.norm.pdf(x_fit, mu, sigma)
                ax.plot(x_fit, y_fit, color="#dc2626", linewidth=2,
                        label=f"Gaussian (μ={mu:.3f}, σ={sigma:.3f})")
            
            ax.set_ylabel("Density", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel("Count", fontsize=9)
        
        ax.set_xlabel("Mean per Dimension", fontsize=9)
        ax.set_title(emb_name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Hide unused axes
    for idx in range(n_embeddings, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    _save_figure(fig, output_path)
    return fig



def plot_distance_distributions(
    raw_distances: np.ndarray,
    scaled_distances: np.ndarray | None,
    raw_gaussian_fit: dict,
    scaled_gaussian_fit: dict,
    *,
    scaled_bic_comparison: dict | None = None,
    output_path: Path | str | None = None,
    title: str = "Distance Distributions",
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot histograms of raw and scaled distance distributions with Gaussian fits.
    
    Creates a two-panel plot:
        - Left panel: Raw distance distribution with Gaussian fit
        - Right panel: Scaled distance distribution with best BIC distribution (if
          ``scaled_bic_comparison`` is provided) or Gaussian fit from ``scaled_gaussian_fit``
    
    Args:
        raw_distances: Array of raw distance values.
        scaled_distances: Array of scaled distance values. If None, the right panel
            will show a message that no scaling was applied.
        raw_gaussian_fit: Dictionary with Gaussian fit parameters for raw distances.
            Should contain 'mu' and 'sigma' keys.
        scaled_gaussian_fit: Dictionary with Gaussian fit parameters for scaled distances.
            Should contain 'mu' and 'sigma' keys.
        scaled_bic_comparison: Optional result of ``fit_scaled_distances_bic_comparison``;
            the lowest-BIC distribution is overlaid on the scaled histogram.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Overall title of the plot.
        figsize: Figure size as (width, height).
    
    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel 1: Raw distances
    ax1 = axes[0]
    if len(raw_distances) > 0:
        counts, bins, patches = ax1.hist(
            raw_distances, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black'
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Plot Gaussian fit
        if raw_gaussian_fit and "mu" in raw_gaussian_fit and "sigma" in raw_gaussian_fit:
            mu = raw_gaussian_fit["mu"]
            sigma = raw_gaussian_fit["sigma"]
            x_fit = np.linspace(raw_distances.min(), raw_distances.max(), 200)
            y_fit = scipy_stats.norm.pdf(x_fit, mu, sigma)
            ax1.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Gaussian fit\nμ={mu:.4f}, σ={sigma:.4f}')
            ax1.legend()
        
        ax1.set_xlabel('Raw Distance', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Raw Distance Distribution', fontsize=14, fontweight='bold')
        _format_axes(ax1)
    
    # Panel 2: Scaled distances
    ax2 = axes[1]
    if scaled_distances is not None and len(scaled_distances) > 0:
        counts, bins, patches = ax2.hist(
            scaled_distances, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black'
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Best BIC fit (preferred) or Gaussian overlay
        x_fit = np.linspace(float(scaled_distances.min()), float(scaled_distances.max()), 200)
        plotted = False
        if scaled_bic_comparison and scaled_bic_comparison.get("best_by_bic"):
            best_name = scaled_bic_comparison["best_by_bic"]
            best_fd = (scaled_bic_comparison.get("fits") or {}).get(best_name) or {}
            if best_fd.get("success") and best_fd.get("parameters"):
                try:
                    y_fit = evaluate_scaled_distance_fit_pdf(
                        best_name, best_fd["parameters"], x_fit
                    )
                    bic = best_fd.get("bic")
                    bic_s = f"{bic:.4g}" if bic is not None and np.isfinite(bic) else "N/A"
                    ax2.plot(
                        x_fit,
                        y_fit,
                        "r-",
                        linewidth=2,
                        label=f"Best BIC: {best_name}\nBIC={bic_s}",
                    )
                    plotted = True
                except Exception:
                    plotted = False
        if not plotted and scaled_gaussian_fit and "mu" in scaled_gaussian_fit and "sigma" in scaled_gaussian_fit:
            mu = scaled_gaussian_fit["mu"]
            sigma = scaled_gaussian_fit["sigma"]
            y_fit = scipy_stats.norm.pdf(x_fit, mu, sigma)
            ax2.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Gaussian fit\nμ={mu:.4f}, σ={sigma:.4f}')
            plotted = True
        if plotted:
            ax2.legend()
        
        ax2.set_xlabel('Scaled Distance', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Scaled Distance Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No scaling applied', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Scaled Distance Distribution', fontsize=14, fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    _save_figure(fig, output_path)
    return fig


def plot_distance_statistics(
    pairwise_results: dict[str, dict[str, float]],
    distances_from_center: np.ndarray,
    output_path: Path | str | None = None,
    title: str = "Distance Statistics",
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Plot histograms of distance statistics in a 2x2 grid.
    
    Excludes pairs with status='excluded' from the plots.
    
    Creates a 2x2 panel plot showing distributions of:
    - Top-left: Pairwise structural distance
    - Top-right: Distance from center
    - Bottom-left: Activity difference
    - Bottom-right: Log ratio
    
    Args:
        pairwise_results: Dictionary of pairwise distance results.
        distances_from_center: Array of distances from the center of embedding space.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Plot title.
        figsize: Figure size as (width, height).
    
    Returns:
        matplotlib Figure object.
    """
    # Filter out excluded pairs
    valid_pairs = [
        v for v in pairwise_results.values() 
        if v.get("status", "correct") != "excluded"
    ]
    
    # Support both 'distance' and 'distance_raw' keys for backward compatibility
    distances = np.array([v.get("distance_raw", v.get("distance", np.nan)) for v in valid_pairs])
    activity_diffs = np.array([v["activity_difference"] for v in valid_pairs])
    log_ratios = np.array([v["log_ratio"] for v in valid_pairs])
    
    # Filter out any remaining NaN values
    distances = distances[~np.isnan(distances)]
    activity_diffs = activity_diffs[~np.isnan(activity_diffs)]
    log_ratios = log_ratios[~np.isnan(log_ratios)]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Top-left: Pairwise distance distribution
    axes[0, 0].hist(distances, bins=50, color="#2563eb", alpha=0.7, edgecolor="white")
    axes[0, 0].set_xlabel("Pairwise Structural Distance", fontsize=11)
    axes[0, 0].set_ylabel("Count", fontsize=11)
    axes[0, 0].set_title("Pairwise Distance Distribution", fontsize=12)
    axes[0, 0].axvline(np.mean(distances), color="#dc2626", linestyle="--", 
                       label=f"Mean: {np.mean(distances):.3f}")
    axes[0, 0].axvline(np.median(distances), color="#16a34a", linestyle="--",
                       label=f"Median: {np.median(distances):.3f}")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)
    
    # Top-right: Distance from center distribution
    axes[0, 1].hist(distances_from_center, bins=50, color="#0891b2", alpha=0.7, edgecolor="white")
    axes[0, 1].set_xlabel("Distance from Center", fontsize=11)
    axes[0, 1].set_ylabel("Count", fontsize=11)
    axes[0, 1].set_title("Distance from Center Distribution", fontsize=12)
    axes[0, 1].axvline(np.mean(distances_from_center), color="#dc2626", linestyle="--",
                       label=f"Mean: {np.mean(distances_from_center):.3f}")
    axes[0, 1].axvline(np.median(distances_from_center), color="#16a34a", linestyle="--",
                       label=f"Median: {np.median(distances_from_center):.3f}")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)
    
    # Bottom-left: Activity difference distribution
    axes[1, 0].hist(activity_diffs, bins=50, color="#16a34a", alpha=0.7, edgecolor="white")
    axes[1, 0].set_xlabel("Activity Difference", fontsize=11)
    axes[1, 0].set_ylabel("Count", fontsize=11)
    axes[1, 0].set_title("Activity Difference Distribution", fontsize=12)
    axes[1, 0].axvline(np.mean(activity_diffs), color="#dc2626", linestyle="--",
                       label=f"Mean: {np.mean(activity_diffs):.3f}")
    axes[1, 0].axvline(np.median(activity_diffs), color="#2563eb", linestyle="--",
                       label=f"Median: {np.median(activity_diffs):.3f}")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)
    
    # Bottom-right: Log ratio distribution
    axes[1, 1].hist(log_ratios, bins=50, color="#9333ea", alpha=0.7, edgecolor="white")
    axes[1, 1].set_xlabel("Log Ratio", fontsize=11)
    axes[1, 1].set_ylabel("Count", fontsize=11)
    axes[1, 1].set_title("Log Ratio Distribution", fontsize=12)
    axes[1, 1].axvline(np.mean(log_ratios), color="#dc2626", linestyle="--",
                       label=f"Mean: {np.mean(log_ratios):.3f}")
    axes[1, 1].axvline(np.median(log_ratios), color="#2563eb", linestyle="--",
                       label=f"Median: {np.median(log_ratios):.3f}")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    _save_figure(fig, output_path)
    return fig


def plot_distance_and_similarity_matrices(
    distance_matrix: np.ndarray,
    similarity_matrix: np.ndarray,
    activity_distance_matrix: np.ndarray,
    activity_similarity_matrix: np.ndarray,
    output_path: Path | str | None = None,
    title: str = "Distance and Similarity Matrices",
    figsize: tuple[int, int] = (14, 12),
) -> plt.Figure:
    """
    Plot structural and activity distance/similarity matrices in a 2x2 grid.
    
    Layout:
    - Top-left: Structural distance matrix
    - Top-right: Structural similarity matrix
    - Bottom-left: Activity distance matrix (log_ratio)
    - Bottom-right: Activity similarity matrix
    
    Colors are matched such that:
    - High distance (dark in distance plot) = Low similarity (dark in similarity plot)
    - Low distance (light in distance plot) = High similarity (light in similarity plot)
    - NaN values (excluded pairs) are shown in white
    
    Args:
        distance_matrix: Matrix of structural distance values.
        similarity_matrix: Matrix of structural similarity values.
        activity_distance_matrix: Matrix of activity distance values (log_ratio).
        activity_similarity_matrix: Matrix of activity similarity values.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Plot title.
        figsize: Figure size as (width, height).
    
    Returns:
        matplotlib Figure object.
    """
    import matplotlib.colors as mcolors
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Create colormaps with white for NaN (bad) values
    cmap_viridis_r = plt.cm.viridis_r.copy()
    cmap_viridis_r.set_bad(color='white')
    
    cmap_viridis = plt.cm.viridis.copy()
    cmap_viridis.set_bad(color='white')
    
    cmap_magma_r = plt.cm.magma_r.copy()
    cmap_magma_r.set_bad(color='white')
    
    cmap_magma = plt.cm.magma.copy()
    cmap_magma.set_bad(color='white')
    
    # Create masked arrays to properly handle NaN
    distance_masked = np.ma.masked_invalid(distance_matrix)
    similarity_masked = np.ma.masked_invalid(similarity_matrix)
    activity_distance_masked = np.ma.masked_invalid(activity_distance_matrix)
    activity_similarity_masked = np.ma.masked_invalid(activity_similarity_matrix)
    
    # Top-left: Structural distance matrix (reversed colormap: high distance = dark)
    im1 = axes[0, 0].imshow(distance_masked, cmap=cmap_viridis_r, aspect="auto")
    axes[0, 0].set_title("Structural Distance Matrix", fontsize=12)
    axes[0, 0].set_xlabel("Compound Index", fontsize=11)
    axes[0, 0].set_ylabel("Compound Index", fontsize=11)
    cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar1.set_label("Distance", fontsize=10)
    
    # Top-right: Structural similarity matrix (normal colormap: high similarity = light)
    im2 = axes[0, 1].imshow(similarity_masked, cmap=cmap_viridis, aspect="auto")
    axes[0, 1].set_title("Structural Similarity Matrix", fontsize=12)
    axes[0, 1].set_xlabel("Compound Index", fontsize=11)
    axes[0, 1].set_ylabel("Compound Index", fontsize=11)
    cbar2 = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar2.set_label("Similarity", fontsize=10)
    
    # Bottom-left: Activity distance matrix (reversed colormap: high distance = dark)
    im3 = axes[1, 0].imshow(activity_distance_masked, cmap=cmap_magma_r, aspect="auto")
    axes[1, 0].set_title("Activity Distance Matrix (Log Ratio)", fontsize=12)
    axes[1, 0].set_xlabel("Compound Index", fontsize=11)
    axes[1, 0].set_ylabel("Compound Index", fontsize=11)
    cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar3.set_label("Log Ratio", fontsize=10)
    
    # Bottom-right: Activity similarity matrix (normal colormap: high similarity = light)
    im4 = axes[1, 1].imshow(activity_similarity_masked, cmap=cmap_magma, aspect="auto")
    axes[1, 1].set_title("Activity Similarity Matrix", fontsize=12)
    axes[1, 1].set_xlabel("Compound Index", fontsize=11)
    axes[1, 1].set_ylabel("Compound Index", fontsize=11)
    cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar4.set_label("Similarity", fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    _save_figure(fig, output_path)
    return fig


def plot_distance_matrices(
    structural_distance_matrix: np.ndarray,
    activity_distance_matrix: np.ndarray,
    output_path: Path | str | None = None,
    title: str = "Distance Matrices",
    figsize: tuple[int, int] = (16, 7),
) -> plt.Figure:
    """
    Plot structural and activity distance matrices as heatmaps in a two-panel plot.
    
    Creates a 1x2 panel plot showing:
    - Left panel: Structural distance matrix
    - Right panel: Activity distance matrix (log_ratio)
    
    Args:
        structural_distance_matrix: Matrix of structural distance values.
        activity_distance_matrix: Matrix of activity distance values (log_ratio).
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Overall title of the plot.
        figsize: Figure size as (width, height).
    
    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel 1: Structural distance matrix
    ax1 = axes[0]
    im1 = ax1.imshow(structural_distance_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    ax1.set_title('Structural Distance Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Compound Index', fontsize=12)
    ax1.set_ylabel('Compound Index', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Distance')
    
    # Panel 2: Activity distance matrix
    ax2 = axes[1]
    im2 = ax2.imshow(activity_distance_matrix, cmap='plasma', aspect='auto', interpolation='nearest')
    ax2.set_title('Activity Distance Matrix (log_ratio)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Compound Index', fontsize=12)
    ax2.set_ylabel('Compound Index', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Distance')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    _save_figure(fig, output_path)
    return fig


# ============================================================================
# Correlation Analysis Plotting Functions
# ============================================================================

def plot_lipschitz_distribution(
    lipschitz_data: dict,
    output_path: Path | str | None = None,
    title: str = "Lipschitz Statistics",
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot Lipschitz statistics as a 1x2 panel.
    
    Creates a 1x2 panel plot:
        - Left: Histogram of raw Lipschitz values with KWW decay fit
        - Right: Cumulative distribution of raw values with percentile markers
    
    Args:
        lipschitz_data: Dictionary from calculate_lipschitz_statistics with:
            - 'lipschitz_values': array of values
            - 'statistics': dict with max, percentile_90, percentile_95, etc.
            - 'histogram': dict with bin_edges, counts, percentile_cutoff
            - 'kww_fit': dict with a, k, b, r_squared, etc.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Overall title of the plot.
        figsize: Figure size as (width, height).
    
    Returns:
        matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    stats = lipschitz_data["statistics"]
    histogram = lipschitz_data["histogram"]
    kww_fit = lipschitz_data.get("kww_fit") or {}
    lipschitz_values = lipschitz_data.get("lipschitz_values", np.array([]))
    
    bin_edges = np.array(histogram["bin_edges"])
    counts = np.array(histogram["counts"])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # -------------------------------------------------------------------------
    # Left: Histogram with KWW decay fit
    # -------------------------------------------------------------------------
    ax1.bar(
        bin_centers,
        counts,
        width=bin_widths,
        color="#2563eb",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        label="Data",
    )

    bin_width = float(bin_widths[0]) if len(bin_widths) > 0 else 1.0
    n_filtered = int(kww_fit.get("n_values_used", len(lipschitz_values)))
    ka = kww_fit.get("a")
    k = kww_fit.get("k")
    kb = kww_fit.get("b")
    if ka is not None and k is not None and kb is not None:
        x_fit = np.linspace(max(bin_edges[0], 0.01), bin_edges[-1], 200)
        y_fit = kww_decay(x_fit, ka, k, kb) * n_filtered * bin_width
        r_squared = kww_fit.get("r_squared", 0)
        ax1.plot(
            x_fit,
            y_fit,
            color="#dc2626",
            linewidth=2.5,
            linestyle="-",
            label=f"KWW: a={ka:.2f}, k={k:.2f}, b={kb:.2f}, R²={r_squared:.3f}",
        )

    # Set y-axis limit to maximum of histogram counts
    ax1.set_ylim(0, counts.max() * 1.1)
    ax1.set_xlabel("Lipschitz Value", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution (up to 99th percentile)", fontsize=12)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Right: CDF with percentile markers
    # -------------------------------------------------------------------------
    if len(lipschitz_values) > 0:
        # Filter to 99th percentile for display
        cutoff = stats["percentile_99"]
        filtered = lipschitz_values[lipschitz_values <= cutoff]
        sorted_vals = np.sort(filtered)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(lipschitz_values)
        ax2.plot(sorted_vals, cdf, color="#2563eb", linewidth=2)

    # Add percentile markers
    percentile_colors = {
        "percentile_90": ("#f59e0b", "90th"),
        "percentile_95": ("#ef4444", "95th"),
        "percentile_99": ("#7c3aed", "99th"),
    }
    
    for key, (color, label) in percentile_colors.items():
        val = stats[key]
        ax2.axvline(val, color=color, linestyle="--", linewidth=2, label=f"{label}: {val:.2f}")

    ax2.set_xlabel("Lipschitz Value", fontsize=11)
    ax2.set_ylabel("Cumulative Probability", fontsize=11)
    ax2.set_title("Cumulative Distribution", fontsize=12)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlim(0, stats["percentile_99"] * 1.05)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_distance_distribution_beta(
    distance_values: np.ndarray,
    beta_fit: dict,
    output_path: Path | str | None = None,
    title: str = "Scaled structural distance (distance_scaled) — Beta fit",
    *,
    n_bins: int = 100,
    percentile_cutoff: float = 99.0,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Histogram of scaled structural distances (``distance_scaled``) with Beta(a,b) PDF overlay.

    Uses the same truncation and scaling as :func:`src.fitting.fit_beta_distance`.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    vals = np.asarray(distance_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        ax.text(0.5, 0.5, "No finite distances", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.suptitle(title, fontsize=14, fontweight="bold")
        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig

    cutoff = float(np.percentile(vals, percentile_cutoff))
    filtered = vals[vals <= cutoff]
    if len(filtered) < 3:
        filtered = vals

    counts, bin_edges = np.histogram(filtered, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    ax.bar(
        bin_centers,
        counts,
        width=bin_widths,
        color="#0d9488",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        label="Data",
    )

    n_used = int(len(filtered))
    bin_w = float(bin_widths[0]) if len(bin_widths) > 0 else 1.0
    if beta_fit.get("a") is not None and beta_fit.get("b") is not None:
        ba, bb = float(beta_fit["a"]), float(beta_fit["b"])
        lo = beta_fit.get("scale_lo")
        hi = beta_fit.get("scale_hi")
        if lo is not None and hi is not None and hi > lo:
            x_fit = np.linspace(float(bin_edges[0]), float(bin_edges[-1]), 200)
            span = hi - lo
            u = (x_fit - lo) / span
            u = np.clip(u, 1e-9, 1.0 - 1e-9)
            pdf_u = scipy_stats.beta.pdf(u, ba, bb)
            pdf_x = pdf_u / span
            y_fit = pdf_x * n_used * bin_w
            r2 = beta_fit.get("r_squared", 0)
            ax.plot(
                x_fit,
                y_fit,
                color="#c026d3",
                linewidth=2.5,
                label=f"Beta: a={ba:.2f}, b={bb:.2f}, R²={r2:.3f}",
            )

    ax.set_ylim(0, max(counts.max() * 1.1, 1))
    ax.set_xlabel("Scaled structural distance (distance_scaled)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Distribution (≤ {percentile_cutoff:g}th percentile)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_pair_subset_three_panels(
    similarity_values: np.ndarray,
    log_ratio_values: np.ndarray,
    lipschitz_values: np.ndarray,
    statistics: dict[str, dict[str, float] | None],
    output_path: Path | str | None = None,
    title: str = "Pair subset statistics",
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot three histograms: similarity, log_ratio (activity difference), and regularized Lipschitz.

    Args:
        similarity_values: Array of similarity values for the pair subset.
        log_ratio_values: Array of log_ratio values (|Δlog_activity|).
        lipschitz_values: Array of regularized Lipschitz values.
        statistics: Dict with keys 'similarity', 'log_ratio', 'regularized_lipschitz', each a stats dict or None.
        output_path: Path to save the figure.
        title: Overall figure title.
        figsize: (width, height).

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    colors = ["#2563eb", "#16a34a", "#dc2626"]
    labels = ["Similarity", "Log ratio (|Δlog_activity|)", "Regularized Lipschitz"]
    data_arrays = [similarity_values, log_ratio_values, lipschitz_values]
    stat_keys = ["similarity", "log_ratio", "regularized_lipschitz"]

    for ax, arr, color, label, key in zip(axes, data_arrays, colors, labels, stat_keys):
        if len(arr) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            continue
        n_bins = min(50, max(10, len(arr) // 5))
        try:
            counts, bin_edges = np.histogram(arr, bins=n_bins)
        except ValueError:
            # Values differ only by floating-point noise (e.g. stereoisomers with identical
            # embeddings, similarity 1 ± 1e-16): too narrow a range for n_bins bins, so plot
            # them as the single value they effectively are
            counts, bin_edges = np.histogram(np.full(len(arr), np.mean(arr)), bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        ax.bar(
            bin_centers,
            counts,
            width=bin_widths,
            color=color,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        stats_dict = (statistics or {}).get(key) if statistics else None
        if stats_dict:
            for pct, pct_name in [(90, "90th"), (95, "95th"), (99, "99th")]:
                pkey = f"percentile_{pct}"
                if pkey in stats_dict:
                    ax.axvline(stats_dict[pkey], color="black", linestyle="--", linewidth=1, alpha=0.7)
                    ax.text(
                        stats_dict[pkey],
                        ax.get_ylim()[1] * 0.95,
                        f"p{pct}",
                        fontsize=8,
                        ha="left",
                        rotation=90,
                    )
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{label} (n={len(arr):,})", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_binned_log_lipschitz(
    pairwise_data: dict[str, dict[str, float]],
    output_path: Path | str | None = None,
    title: str = "Log-Lipschitz Distribution by Similarity Bin",
    figsize: tuple[int, int] = (14, 20),
) -> plt.Figure:
    """
    Plot log(Lipschitz) distribution and Q-Q plot for each similarity bin.
    
    Creates a 5x2 panel plot (one row per similarity bin):
        - Left column: Histogram of log(Lipschitz values)
        - Right column: Q-Q plot vs Normal distribution
    
    Similarity bins: <0.5, 0.5-0.7, 0.7-0.8, 0.8-0.9, >0.9
    
    Args:
        pairwise_data: Dictionary containing 'regularized_lipschitz' and 'similarity' for each pair.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Overall title of the plot.
        figsize: Figure size as (width, height).
    
    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(5, 2, figsize=figsize)
    
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed"]
    
    for row_idx, (lower, upper, label) in enumerate(SIMILARITY_BINS):
        ax_hist = axes[row_idx, 0]
        ax_qq = axes[row_idx, 1]
        color = colors[row_idx]
        
        # Filter pairs in this similarity bin (exclude status='excluded' and NaN values)
        if upper == 1.0:
            # Include upper bound for the last bin
            pairs_in_bin = [
                data for data in pairwise_data.values()
                if data.get("status", "correct") != "excluded"
                and not np.isnan(data.get("similarity", np.nan))
                and lower <= data["similarity"] <= upper
            ]
        else:
            pairs_in_bin = [
                data for data in pairwise_data.values()
                if data.get("status", "correct") != "excluded"
                and not np.isnan(data.get("similarity", np.nan))
                and lower <= data["similarity"] < upper
            ]
        
        n_pairs = len(pairs_in_bin)
        
        if n_pairs > 0:
            # Get Lipschitz values and filter out zeros
            lipschitz_values = np.array([d["regularized_lipschitz"] for d in pairs_in_bin])
            positive_values = lipschitz_values[lipschitz_values > 0]
            
            if len(positive_values) > 0:
                log_values = np.log(positive_values)
                
                # Left panel: Histogram of log(Lipschitz)
                ax_hist.hist(
                    log_values,
                    bins=30,
                    color=color,
                    alpha=0.7,
                    edgecolor="white",
                    linewidth=0.5,
                )
                
                # Add mean and median lines
                log_mean = np.mean(log_values)
                log_median = np.median(log_values)
                log_std = np.std(log_values)
                ax_hist.axvline(log_mean, color="#dc2626", linestyle="--", linewidth=2, 
                                label=f"Mean: {log_mean:.2f}")
                ax_hist.axvline(log_median, color="#374151", linestyle="--", linewidth=2,
                                label=f"Median: {log_median:.2f}")
                
                # Right panel: Q-Q plot
                sorted_log = np.sort(log_values)
                n = len(sorted_log)
                theoretical_quantiles = scipy_stats.norm.ppf(np.linspace(0.001, 0.999, n))
                
                ax_qq.scatter(theoretical_quantiles, sorted_log, alpha=0.5, s=10, color=color)
                
                # Add reference line (perfect normal)
                z_min, z_max = theoretical_quantiles.min(), theoretical_quantiles.max()
                ref_line = log_mean + log_std * np.array([z_min, z_max])
                ax_qq.plot([z_min, z_max], ref_line, color="#dc2626", linestyle="--", linewidth=2,
                           label=f"Normal (μ={log_mean:.2f}, σ={log_std:.2f})")
            else:
                ax_hist.text(0.5, 0.5, "No positive values", ha="center", va="center", 
                             transform=ax_hist.transAxes, fontsize=12)
                ax_qq.text(0.5, 0.5, "No positive values", ha="center", va="center",
                           transform=ax_qq.transAxes, fontsize=12)
        else:
            ax_hist.text(0.5, 0.5, "No pairs in bin", ha="center", va="center",
                         transform=ax_hist.transAxes, fontsize=12)
            ax_qq.text(0.5, 0.5, "No pairs in bin", ha="center", va="center",
                       transform=ax_qq.transAxes, fontsize=12)
        
        # Formatting for histogram
        ax_hist.set_xlabel("log(Lipschitz Value)", fontsize=10)
        ax_hist.set_ylabel("Count", fontsize=10)
        ax_hist.set_title(f"Similarity {label} (n={n_pairs:,})", fontsize=11)
        ax_hist.legend(fontsize=8, loc="upper right")
        ax_hist.grid(True, alpha=0.3, axis="y")
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        
        # Formatting for Q-Q plot
        ax_qq.set_xlabel("Theoretical Quantiles (Normal)", fontsize=10)
        ax_qq.set_ylabel("Sample Quantiles", fontsize=10)
        ax_qq.set_title(f"Q-Q Plot - Similarity {label}", fontsize=11)
        ax_qq.legend(fontsize=8, loc="upper left")
        ax_qq.grid(True, alpha=0.3)
        ax_qq.spines["top"].set_visible(False)
        ax_qq.spines["right"].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def plot_lipschitz_heatmap(
    lipschitz_matrix: np.ndarray,
    output_path: Path | str | None = None,
    title: str = "Regularized Lipschitz Heatmap",
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot heatmap of regularized Lipschitz values.
    
    Colorbar spans from 0 to 90th percentile. Values above 90th percentile
    are clipped to the same color. NaN values (excluded pairs) are shown in white.
    
    Args:
        lipschitz_matrix: Matrix of regularized Lipschitz values.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Plot title.
        figsize: Figure size as (width, height).
    
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate 90th percentile for colorbar range (exclude diagonal zeros and NaN)
    non_diagonal = lipschitz_matrix[~np.eye(lipschitz_matrix.shape[0], dtype=bool)]
    valid_values = non_diagonal[~np.isnan(non_diagonal)]
    percentile_90 = np.percentile(valid_values, 90) if len(valid_values) > 0 else 1.0
    
    # Create colormap with white for NaN values
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='white')
    
    # Create masked array to handle NaN
    masked_matrix = np.ma.masked_invalid(lipschitz_matrix)
    
    # Plot with colorbar from 0 to 90th percentile
    im = ax.imshow(
        masked_matrix, 
        cmap=cmap, 
        aspect="auto",
        vmin=0,
        vmax=percentile_90,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Compound Index", fontsize=12)
    ax.set_ylabel("Compound Index", fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Regularized Lipschitz Value (0-90th percentile: {percentile_90:.2f})", fontsize=11)
    
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def plot_binned_cliff_probability(
    df: pd.DataFrame,
    activity_threshold: float,
    output_path: Path | str | None = None,
    title: str = "Activity Cliff Probability vs Similarity Bin",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot activity cliff probability as a function of similarity bin with number of pairs as bars.
    
    Creates a plot with:
        - Bar chart showing number of pairs in each bin (secondary y-axis)
        - Line plot showing cliff probability vs bin center (primary y-axis)
    
    Args:
        df: DataFrame from calculate_binned_cliff_probability with columns:
            'bin_center', 'n_pairs', 'cliff_probability', etc.
        activity_threshold: Threshold used for defining cliffs (for plot annotation).
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Plot title.
        figsize: Figure size as (width, height).
    
    Returns:
        matplotlib Figure object.
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    bin_centers = df["bin_center"].values
    n_pairs = df["n_pairs"].values
    cliff_prob = df["cliff_probability"].values
    bin_width = df["bin_upper"].values[0] - df["bin_lower"].values[0]
    
    # Bar chart for number of pairs (secondary y-axis)
    ax2 = ax1.twinx()
    ax2.bar(
        bin_centers,
        n_pairs,
        width=bin_width * 0.8,
        color="#94a3b8",
        alpha=0.5,
        edgecolor="white",
        linewidth=0.5,
        label="Number of Pairs",
        zorder=1,
    )
    ax2.set_ylabel("Number of Pairs", fontsize=11, color="#64748b")
    ax2.tick_params(axis="y", labelcolor="#64748b")
    ax2.spines["right"].set_color("#64748b")
    
    # Line plot for cliff probability (primary y-axis)
    ax1.plot(
        bin_centers,
        cliff_prob,
        color="#16a34a",
        linewidth=2.5,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=2,
        label=f"P(log_ratio > {activity_threshold})",
        zorder=3,
    )
    
    ax1.set_xlabel("Similarity Bin Center", fontsize=11)
    ax1.set_ylabel("Cliff Probability", fontsize=11, color="#16a34a")
    ax1.tick_params(axis="y", labelcolor="#16a34a")
    ax1.spines["left"].set_color("#16a34a")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Grid
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.spines["top"].set_visible(False)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def plot_binned_correlation(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    title: str = "Pearson Correlation vs Similarity Bin",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot Pearson correlation as a function of similarity bin with number of pairs as bars.
    
    Creates a plot with:
        - Bar chart showing number of pairs in each bin (secondary y-axis)
        - Line plot showing Pearson correlation vs bin center (primary y-axis)
    
    Args:
        df: DataFrame from calculate_binned_pearson_correlation with columns:
            'bin_center', 'n_pairs', 'pearson_r', etc.
        output_path: Path to save the plot. If None, the plot is not saved.
        title: Plot title.
        figsize: Figure size as (width, height).
    
    Returns:
        matplotlib Figure object.
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    bin_centers = df["bin_center"].values
    n_pairs = df["n_pairs"].values
    pearson_r = df["pearson_r"].values
    bin_width = df["bin_upper"].values[0] - df["bin_lower"].values[0]
    
    # Bar chart for number of pairs (secondary y-axis)
    ax2 = ax1.twinx()
    bars = ax2.bar(
        bin_centers,
        n_pairs,
        width=bin_width * 0.8,
        color="#94a3b8",
        alpha=0.5,
        edgecolor="white",
        linewidth=0.5,
        label="Number of Pairs",
        zorder=1,
    )
    ax2.set_ylabel("Number of Pairs", fontsize=11, color="#64748b")
    ax2.tick_params(axis="y", labelcolor="#64748b")
    ax2.spines["right"].set_color("#64748b")
    
    # Line plot for Pearson correlation (primary y-axis)
    # Filter out NaN values for plotting
    valid_mask = ~np.isnan(pearson_r)
    ax1.plot(
        bin_centers[valid_mask],
        pearson_r[valid_mask],
        color="#dc2626",
        linewidth=2.5,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Pearson r",
        zorder=3,
    )
    
    # Add zero line
    ax1.axhline(0, color="#374151", linestyle="--", linewidth=1, alpha=0.5)
    
    ax1.set_xlabel("Similarity Bin Center", fontsize=11)
    ax1.set_ylabel("Pearson Correlation (r)", fontsize=11, color="#dc2626")
    ax1.tick_params(axis="y", labelcolor="#dc2626")
    ax1.spines["left"].set_color("#dc2626")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-1, 1)
    
    # Grid
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.spines["top"].set_visible(False)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def plot_cliff_fitting_constants_comparison(fit_params_list: list[dict], output_dir: Path) -> None:
    """Plot comparison of cliff identification fitting constants."""
    config_names = []
    embedding_methods = []
    params = {
        "activity_cliff_fraction": {"c": [], "a": [], "k": [], "x0": []},
        "cliff_fraction_vs_pair_rank": {"c": [], "a": [], "k": [], "x0": []},
    }

    for fit_params in fit_params_list:
        config_names.append(fit_params["config_name"])
        embedding_methods.append(fit_params["embedding_method"])
        acf = fit_params.get("activity_cliff_fraction", {})
        params["activity_cliff_fraction"]["c"].append(acf.get("c", np.nan))
        params["activity_cliff_fraction"]["a"].append(acf.get("a", np.nan))
        params["activity_cliff_fraction"]["k"].append(acf.get("k", np.nan))
        params["activity_cliff_fraction"]["x0"].append(acf.get("x0", np.nan))
        cfr = fit_params.get("cliff_fraction_vs_pair_rank", {})
        params["cliff_fraction_vs_pair_rank"]["c"].append(cfr.get("c", np.nan))
        params["cliff_fraction_vs_pair_rank"]["a"].append(cfr.get("a", np.nan))
        params["cliff_fraction_vs_pair_rank"]["k"].append(cfr.get("k", np.nan))
        params["cliff_fraction_vs_pair_rank"]["x0"].append(cfr.get("x0", np.nan))

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle("Cliff Identification Fitting Constants Comparison", fontsize=16, fontweight="bold")
    unique_embeddings = sorted(list(set(embedding_methods)))
    tab10_colors = plt.cm.tab10.colors
    embedding_to_color = {emb: tab10_colors[i % len(tab10_colors)] for i, emb in enumerate(unique_embeddings)}
    n_configs = len(config_names)
    n_embeddings = len(unique_embeddings)
    x = np.arange(n_configs)
    width = 0.8 / n_embeddings

    for idx, param_name in enumerate(["c", "a", "k", "x0"]):
        ax = axes[0, idx]
        values = params["activity_cliff_fraction"][param_name]
        for emb_idx, emb in enumerate(unique_embeddings):
            emb_positions = [i for i, emb_m in enumerate(embedding_methods) if emb_m == emb]
            bar_positions = [x[pos] + (emb_idx - n_embeddings / 2 + 0.5) * width for pos in emb_positions]
            bar_values = [values[pos] for pos in emb_positions]
            ax.bar(bar_positions, bar_values, width, label=emb, color=embedding_to_color[emb], alpha=0.7)
        ax.set_xlabel("Configuration")
        ax.set_ylabel(f"Parameter {param_name}")
        ax.set_title(f"Activity Cliff Fraction: {param_name}")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels([name.split("/")[-1] for name in config_names], rotation=45, ha="right")

    for idx, param_name in enumerate(["c", "a", "k", "x0"]):
        ax = axes[1, idx]
        values = params["cliff_fraction_vs_pair_rank"][param_name]
        for emb_idx, emb in enumerate(unique_embeddings):
            emb_positions = [i for i, emb_m in enumerate(embedding_methods) if emb_m == emb]
            bar_positions = [x[pos] + (emb_idx - n_embeddings / 2 + 0.5) * width for pos in emb_positions]
            bar_values = [values[pos] for pos in emb_positions]
            ax.bar(bar_positions, bar_values, width, label=emb, color=embedding_to_color[emb], alpha=0.7)
        ax.set_xlabel("Configuration")
        ax.set_ylabel(f"Parameter {param_name}")
        ax.set_title(f"Cliff Fraction vs Pair Rank: {param_name}")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels([name.split("/")[-1] for name in config_names], rotation=45, ha="right")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=embedding_to_color[emb], alpha=0.7, label=emb)
               for emb in unique_embeddings]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    plt.tight_layout()
    plt.savefig(output_dir / "cliff_fitting_constants_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cliff_sets_dice_similarity_heatmap(
    config_labels: list[str],
    dice_matrix: np.ndarray,
    output_dir: Path,
    *,
    figsize_per_cell: float = 0.7,
    fontsize_value: float = 6,
) -> None:
    """
    Plot a heatmap of Dice similarity between cliff sets (95% threshold) across configs.

    Configs should be ordered so that same embedding method are colocated (e.g. sorted by embedding).
    Values are shown in each cell, rounded to 2 decimal places.
    """
    n = len(config_labels)
    if n == 0 or dice_matrix.shape != (n, n):
        return
    # Mask NaN so they are drawn as white (both sets empty)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    masked = np.ma.masked_invalid(dice_matrix)
    fig, ax = plt.subplots(figsize=(figsize_per_cell * n, figsize_per_cell * n))
    im = ax.imshow(masked, cmap=cmap, aspect="equal", vmin=0, vmax=1)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(config_labels, fontsize=min(8, max(5, 10 - n // 4)))
    ax.set_yticklabels(config_labels, fontsize=min(8, max(5, 10 - n // 4)))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(n):
        for j in range(n):
            val = dice_matrix[i, j]
            if np.isnan(val):
                text_str = "NaN"
            else:
                text_str = f"{float(val):.2f}"
            text = ax.text(
                j, i, text_str,
                ha="center", va="center", color="white" if not np.isnan(val) and val < 0.5 else "black",
                fontsize=fontsize_value, fontweight="bold",
            )
    ax.set_title("Dice similarity of cliff sets (95% structural similarity)")
    fig.colorbar(im, ax=ax, label="Dice similarity")
    plt.tight_layout()
    plt.savefig(output_dir / "cliff_sets_dice_similarity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_metrics_comparison_grouped(
    metrics_list: list[dict],
    model_type: str,
    output_dir: Path,
    *,
    split_name: str | None = None,
    short_labels: list[str] | None = None,
) -> None:
    """
    Plot model metrics as bar charts: one bar per config, colored by embedding, configs ordered
    so same embedding are grouped. Uses short_labels on x-axis for readability.
    """
    if not metrics_list:
        return
    if short_labels is None:
        short_labels = [f"{m['embedding_method']} / {m['distance_metric']}" for m in metrics_list]
    assert len(short_labels) == len(metrics_list)

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "binary_cross_entropy"]
    unique_embeddings = sorted(set(m["embedding_method"] for m in metrics_list))
    tab10_colors = plt.cm.tab10.colors
    embedding_to_color = {emb: tab10_colors[i % len(tab10_colors)] for i, emb in enumerate(unique_embeddings)}

    n_configs = len(metrics_list)
    x = np.arange(n_configs)
    width = 0.75

    fig, axes = plt.subplots(2, 3, figsize=(max(14, n_configs * 1.2), 10))
    title = f"{model_type.replace('_', ' ').title()} Model Metrics"
    if split_name:
        title = f"{split_name.capitalize()} split — {title}"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        values = [m.get(metric_name, np.nan) for m in metrics_list]
        colors = [embedding_to_color[m["embedding_method"]] for m in metrics_list]
        bars = ax.bar(x, values, width, color=colors, alpha=0.85, edgecolor="gray", linewidth=0.5)
        ax.set_xlabel("Configuration", fontsize=11)
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=11)
        ax.set_title(metric_name.replace("_", " ").title(), fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
        ax.tick_params(axis="both", labelsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=embedding_to_color[emb], alpha=0.85, label=emb)
        for emb in unique_embeddings
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=11)
    plt.tight_layout()
    fname = f"{split_name}_{model_type}_model_metrics_comparison.png" if split_name else f"{model_type}_model_metrics_comparison.png"
    plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_persistence_landscape_l2_distance_heatmap(
    config_labels: list[str],
    distance_matrix: np.ndarray,
    output_path: Path,
    *,
    figsize_per_cell: float = 0.7,
    fontsize_value: float = 6,
) -> None:
    """
    Heatmap of L2 distance between each pair of persistence landscapes.
    Configs ordered so same embedding are colocated. Values in cells rounded to 2 decimals.
    """
    n = len(config_labels)
    if n == 0 or distance_matrix.shape != (n, n):
        return
    fig, ax = plt.subplots(figsize=(figsize_per_cell * n, figsize_per_cell * n))
    im = ax.imshow(distance_matrix, cmap="viridis", aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(config_labels, fontsize=min(8, max(5, 10 - n // 4)))
    ax.set_yticklabels(config_labels, fontsize=min(8, max(5, 10 - n // 4)))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    vmin, vmax = np.nanmin(distance_matrix), np.nanmax(distance_matrix)
    for i in range(n):
        for j in range(n):
            val = distance_matrix[i, j]
            if np.isnan(val):
                text_str = "NaN"
            else:
                text_str = f"{float(val):.2f}"
            # Light text on dark cells, dark text on light cells
            cell_val = val if not np.isnan(val) else 0
            thresh = (vmin + vmax) / 2 if vmax > vmin else vmax
            color = "white" if cell_val > thresh else "black"
            ax.text(
                j, i, text_str,
                ha="center", va="center", color=color,
                fontsize=fontsize_value, fontweight="bold",
            )
    ax.set_title("L2 distance between persistence landscapes")
    fig.colorbar(im, ax=ax, label="L2 distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_persistence_landscape_mean_and_variance(
    t_grid: np.ndarray,
    mean_lambda1: np.ndarray,
    var_lambda1: np.ndarray,
    output_path: Path,
) -> None:
    """
    Two-panel plot: pointwise mean λ₁(t) and variance Var(λ₁(t)) across configs.
    """
    fig, (ax_mean, ax_var) = plt.subplots(1, 2, figsize=(12, 5))
    ax_mean.plot(t_grid, mean_lambda1, color="steelblue", linewidth=1.5)
    ax_mean.set_xlabel("Parameter t", fontsize=12)
    ax_mean.set_ylabel(r"Mean λ₁(t)", fontsize=12)
    ax_mean.set_title("Pointwise mean of first persistence landscape λ₁", fontsize=12, fontweight="bold")
    ax_mean.grid(True, alpha=0.3)
    ax_var.plot(t_grid, var_lambda1, color="coral", linewidth=1.5)
    ax_var.set_xlabel("Parameter t", fontsize=12)
    ax_var.set_ylabel(r"Var(λ₁(t))", fontsize=12)
    ax_var.set_title("Variance of first persistence landscape λ₁", fontsize=12, fontweight="bold")
    ax_var.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_persistence_landscape_l2_norm(
    short_labels: list[str],
    l2_norms: list[float],
    embedding_methods: list[str],
    output_path: Path,
) -> None:
    """
    Bar plot of L2 norm of persistence landscape per config; same embedding grouped and colored.
    """
    if not short_labels or not l2_norms:
        return
    n = len(short_labels)
    assert n == len(l2_norms) == len(embedding_methods)
    unique_embeddings = sorted(set(embedding_methods))
    tab10_colors = plt.cm.tab10.colors
    embedding_to_color = {emb: tab10_colors[i % len(tab10_colors)] for i, emb in enumerate(unique_embeddings)}
    colors = [embedding_to_color[emb] for emb in embedding_methods]
    x = np.arange(n)
    width = 0.75
    fig, ax = plt.subplots(figsize=(max(8, n * 1.0), 5))
    ax.bar(x, l2_norms, width, color=colors, alpha=0.85, edgecolor="gray", linewidth=0.5)
    ax.set_xlabel("Configuration", fontsize=11)
    ax.set_ylabel("L2 norm of persistence landscape", fontsize=11)
    ax.set_title("L2 norm of persistence landscape by config", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=embedding_to_color[emb], alpha=0.85, label=emb)
        for emb in unique_embeddings
    ]
    ax.legend(handles=handles, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _embedding_order_key(emb: str) -> str:
    """Sort key for embedding names."""
    return emb


def plot_comparison_cliffs_5pct(
    labels: list[str],
    n_cliffs_5: list[float],
    embedding_per_label: list[str],
    output_path: Path,
) -> None:
    """Barplot: cliffs in 5% most similar pairs per config; grouped by embedding."""
    n = len(labels)
    if n == 0:
        return
    unique_embeddings = sorted(set(embedding_per_label), key=_embedding_order_key)
    tab10 = plt.cm.tab10.colors
    emb_color = {emb: tab10[i % len(tab10)] for i, emb in enumerate(unique_embeddings)}
    colors = [emb_color[e] for e in embedding_per_label]
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), 5))
    ax.bar(x, n_cliffs_5, width=0.75, color=colors, alpha=0.9, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of cliffs")
    ax.set_title("Cliffs in 5% most similar pairs")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=emb_color[emb], alpha=0.85, label=emb) for emb in unique_embeddings]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_lipschitz_90(
    labels: list[str],
    values: list[float],
    embedding_per_label: list[str],
    output_path: Path,
) -> None:
    """Barplot: 90th percentile of Lipschitz constant per config, grouped by embedding."""
    n = len(labels)
    if n == 0:
        return
    unique_embeddings = sorted(set(embedding_per_label), key=_embedding_order_key)
    tab10 = plt.cm.tab10.colors
    emb_color = {emb: tab10[i % len(tab10)] for i, emb in enumerate(unique_embeddings)}
    colors = [emb_color[e] for e in embedding_per_label]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), 5))
    ax.bar(np.arange(n), values, width=0.75, color=colors, alpha=0.85, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("90th percentile of Lipschitz constant")
    ax.set_title("Lipschitz 90th percentile by config")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=emb_color[emb], alpha=0.85, label=emb) for emb in unique_embeddings]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_persistence_mean_max(
    labels: list[str],
    mean_persistence: list[float],
    max_persistence: list[float],
    embedding_per_label: list[str],
    output_path: Path,
) -> None:
    """Two-panel: mean persistence and longest (max) persistence per config."""
    n = len(labels)
    if n == 0:
        return
    unique_embeddings = sorted(set(embedding_per_label), key=_embedding_order_key)
    tab10 = plt.cm.tab10.colors
    emb_color = {emb: tab10[i % len(tab10)] for i, emb in enumerate(unique_embeddings)}
    colors = [emb_color[e] for e in embedding_per_label]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n * 0.6), 5))
    x = np.arange(n)
    w = 0.75
    ax1.bar(x, mean_persistence, w, color=colors, alpha=0.85, edgecolor="gray", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Mean persistence")
    ax1.set_title("Mean persistence by config")
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.bar(x, max_persistence, w, color=colors, alpha=0.85, edgecolor="gray", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Longest (max) persistence")
    ax2.set_title("Longest persistence by config")
    ax2.grid(True, alpha=0.3, axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=emb_color[emb], alpha=0.85, label=emb) for emb in unique_embeddings]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_chiral_mmp_2x2(
    chiral_histogram: dict | None,
    chiral_mean_std: tuple[float, float] | None,
    chiral_std_per_config: list[tuple[str, float]],
    mmp_histogram: dict | None,
    mmp_mean_std: tuple[float, float] | None,
    mmp_std_per_config: list[tuple[str, float]],
    embedding_per_label: list[str],
    output_path: Path,
) -> None:
    """2x2: top=chiral (left=Gaussian fit for example config, right=bar of std per config);
    bottom=MMP same.
    """
    from scipy.stats import norm as scipy_norm
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Top-left: Chiral histogram + Gaussian fit
    ax = axes[0, 0]
    if chiral_histogram and chiral_mean_std:
        be = np.array(chiral_histogram["bin_edges"])
        counts = np.array(chiral_histogram["counts"])
        bc = (be[:-1] + be[1:]) / 2
        bw = be[1:] - be[:-1]
        ax.bar(bc, counts, width=bw * 0.9, color="steelblue", alpha=0.7, label="Chiral")
        mu, sig = chiral_mean_std
        if sig > 0:
            x = np.linspace(be[0], be[-1], 200)
            scale = counts.sum() * (be[1] - be[0]) if len(be) > 1 else 1
            ax.plot(x, scale * scipy_norm.pdf(x, mu, sig), "r-", lw=2, label="Gaussian fit")
        ax.set_title("Chiral — distances (example config)")
        ax.legend()
    else:
        ax.set_title("Chiral — no data")
    ax.grid(True, alpha=0.3)
    # Top-right: Chiral std per config
    ax = axes[0, 1]
    if chiral_std_per_config:
        labels = [t[0] for t in chiral_std_per_config]
        stds = [t[1] for t in chiral_std_per_config]
        uniq = sorted(set(embedding_per_label), key=_embedding_order_key)
        tab10 = plt.cm.tab10.colors
        emb_color = {e: tab10[i % len(tab10)] for i, e in enumerate(uniq)}
        colors = [emb_color.get(embedding_per_label[i], "gray") for i in range(min(len(labels), len(embedding_per_label)))]
        if len(colors) < len(labels):
            colors.extend(["gray"] * (len(labels) - len(colors)))
        ax.bar(np.arange(len(labels)), stds, width=0.75, color=colors, alpha=0.85)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Std")
        ax.set_title("Chiral — std by config")
    ax.grid(True, alpha=0.3, axis="y")
    # Bottom-left: MMP histogram + Gaussian
    ax = axes[1, 0]
    if mmp_histogram and mmp_mean_std:
        be = np.array(mmp_histogram["bin_edges"])
        counts = np.array(mmp_histogram["counts"])
        bc = (be[:-1] + be[1:]) / 2
        bw = be[1:] - be[:-1]
        ax.bar(bc, counts, width=bw * 0.9, color="coral", alpha=0.7, label="MMP")
        mu, sig = mmp_mean_std
        if sig > 0:
            x = np.linspace(be[0], be[-1], 200)
            scale = counts.sum() * (be[1] - be[0]) if len(be) > 1 else 1
            ax.plot(x, scale * scipy_norm.pdf(x, mu, sig), "r-", lw=2, label="Gaussian fit")
        ax.set_title("MMP — distances (example config)")
        ax.legend()
    else:
        ax.set_title("MMP — no data")
    ax.grid(True, alpha=0.3)
    # Bottom-right: MMP std per config
    ax = axes[1, 1]
    if mmp_std_per_config:
        labels = [t[0] for t in mmp_std_per_config]
        stds = [t[1] for t in mmp_std_per_config]
        uniq = sorted(set(embedding_per_label), key=_embedding_order_key)
        tab10 = plt.cm.tab10.colors
        emb_color = {e: tab10[i % len(tab10)] for i, e in enumerate(uniq)}
        colors = [emb_color.get(embedding_per_label[i], "gray") for i in range(min(len(labels), len(embedding_per_label)))]
        if len(colors) < len(labels):
            colors.extend(["gray"] * (len(labels) - len(colors)))
        ax.bar(np.arange(len(labels)), stds, width=0.75, color=colors, alpha=0.85)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Std")
        ax.set_title("MMP — std by config")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

