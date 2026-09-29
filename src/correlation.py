"""
Correlation analysis module.

This module provides functions for analyzing the correlation between structural
similarity and activity differences, including Lipschitz statistics, binned
correlation analysis, and cliff probability calculations.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from src.chemistry import get_inchi_key
from src.fitting import fit_beta_distance, fit_kww_decay, kww_decay, kww_theoretical_statistics
from src.plotting import (
    SIMILARITY_BINS,
    plot_binned_cliff_probability,
    plot_binned_correlation,
    plot_binned_log_lipschitz,
    plot_distance_distribution_beta,
    plot_lipschitz_distribution,
    plot_lipschitz_heatmap,
    plot_pair_subset_three_panels,
)

# ============================================================================
# Constants
# ============================================================================

REGULARIZATION_FACTOR = 0.01
TAIL_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# ============================================================================
# Helper Functions
# ============================================================================

def _filter_valid_pairs(
    pairwise_data: dict[str, dict[str, float]],
    require_similarity: bool = True,
    require_log_ratio: bool = True,
) -> list[dict[str, float]]:
    """
    Filter out excluded pairs and optionally filter by NaN values.

    Args:
        pairwise_data: Dictionary of pairwise data.
        require_similarity: If True, filter out pairs with NaN similarity.
        require_log_ratio: If True, filter out pairs with NaN log_ratio.

    Returns:
        List of valid pair data dictionaries.
    """
    valid_pairs = []
    for data in pairwise_data.values():
        if data.get("status", "correct") == "excluded":
            continue
        if require_similarity and np.isnan(data.get("similarity", np.nan)):
            continue
        if require_log_ratio and np.isnan(data.get("log_ratio", np.nan)):
            continue
        valid_pairs.append(data)
    return valid_pairs


def _filter_pairs_in_interval(
    pairwise_data: dict[str, dict[str, float]],
    lower: float,
    upper: float,
    require_similarity: bool = True,
    require_log_ratio: bool = True,
) -> list[dict[str, float]]:
    """
    Filter pairs within a similarity interval.

    Args:
        pairwise_data: Dictionary of pairwise data.
        lower: Lower bound of similarity interval.
        upper: Upper bound of similarity interval.
        require_similarity: If True, filter out pairs with NaN similarity.
        require_log_ratio: If True, filter out pairs with NaN log_ratio.

    Returns:
        List of pair data dictionaries within the interval.
    """
    valid_pairs = _filter_valid_pairs(
        pairwise_data,
        require_similarity=require_similarity,
        require_log_ratio=require_log_ratio,
    )

    pairs_in_interval = []
    for data in valid_pairs:
        similarity = data.get("similarity", np.nan)
        if upper == 1.0:
            if lower <= similarity <= upper:
                pairs_in_interval.append(data)
        else:
            if lower <= similarity < upper:
                pairs_in_interval.append(data)

    return pairs_in_interval


def _create_similarity_bins(bin_width: float) -> list[tuple[float, float]]:
    """
    Create similarity bins from 0 to 1.

    Args:
        bin_width: Width of each bin.

    Returns:
        List of (lower, upper) tuples for each bin.
    """
    n_bins = int(1.0 / bin_width)
    return [(i * bin_width, (i + 1) * bin_width) for i in range(n_bins)]


def _pair_subset_statistics(
    pairwise_subset: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]] | None:
    """
    Compute statistics for similarity, log_ratio, and regularized_lipschitz across a subset of pairs.

    Args:
        pairwise_subset: Dictionary of pair_key -> data with 'similarity', 'log_ratio', 'regularized_lipschitz'.

    Returns:
        Dict with keys 'similarity', 'log_ratio', 'regularized_lipschitz', each mapping to a stats dict
        (min, max, mean, median, std, percentile_90, percentile_95, percentile_99), or None if subset is empty.
    """
    if not pairwise_subset:
        return None
    sim_vals = []
    log_vals = []
    lip_vals = []
    for data in pairwise_subset.values():
        s = data.get("similarity", np.nan)
        lr = data.get("log_ratio", np.nan)
        lip = data.get("regularized_lipschitz", np.nan)
        if not np.isnan(s):
            sim_vals.append(s)
        if not np.isnan(lr):
            log_vals.append(lr)
        if not np.isnan(lip):
            lip_vals.append(lip)
    if not sim_vals and not log_vals and not lip_vals:
        return None

    def _stats(arr: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "percentile_90": float(np.percentile(arr, 90)),
            "percentile_95": float(np.percentile(arr, 95)),
            "percentile_99": float(np.percentile(arr, 99)),
        }

    result = {}
    if sim_vals:
        result["similarity"] = _stats(np.array(sim_vals))
    else:
        result["similarity"] = None
    if log_vals:
        result["log_ratio"] = _stats(np.array(log_vals))
    else:
        result["log_ratio"] = None
    if lip_vals:
        result["regularized_lipschitz"] = _stats(np.array(lip_vals))
    else:
        result["regularized_lipschitz"] = None
    return result


def _scaled_distance_zero_count_and_fraction(
    pairwise_subset: dict[str, dict[str, float]],
) -> tuple[int, float]:
    """Count pairs with structural ``distance_scaled`` exactly zero; fraction w.r.t. subset size."""
    if not pairwise_subset:
        return 0, 0.0
    n_zero = 0
    for data in pairwise_subset.values():
        ds = data.get("distance_scaled")
        if ds is None:
            continue
        try:
            v = float(ds)
        except (TypeError, ValueError):
            continue
        if np.isnan(v):
            continue
        if v == 0.0:
            n_zero += 1
    n = len(pairwise_subset)
    return n_zero, (n_zero / n) if n else 0.0


def _pairwise_scaled_distances(pairwise_data: dict[str, dict]) -> np.ndarray:
    """
    Finite scaled structural distances for each pair.

    Uses ``distance_scaled`` (same as in Lipschitz / similarity: ``similarity = 1 - distance_scaled``).
    """
    vals: list[float] = []
    for d in pairwise_data.values():
        dist = d.get("distance_scaled", np.nan)
        if np.isfinite(dist):
            vals.append(float(dist))
    return np.asarray(vals, dtype=float)


# ============================================================================
# Main Functions
# ============================================================================

def calculate_regularized_lipschitz_values(
    pairwise_data: dict[str, dict[str, float]],
    regularization_factor: float = REGULARIZATION_FACTOR,
) -> dict[str, dict[str, float]]:
    """
    Calculate regularized Lipschitz value for each pair and add it to pairwise_data.

    The regularized Lipschitz value is: log_ratio / max(1 - similarity, regularization_factor)
    Excluded pairs (status='excluded') get NaN values.

    Args:
        pairwise_data: Dictionary mapping "inchi_key_1,inchi_key_2" pairs to a dict
            containing 'similarity', 'log_ratio', 'status', etc.
        regularization_factor: Minimum denominator value to avoid division by near-zero.

    Returns:
        Updated pairwise_data with 'regularized_lipschitz' added to each entry.
    """
    for pair_key, data in pairwise_data.items():
        # Skip excluded pairs
        if data.get("status") == "excluded":
            data["regularized_lipschitz"] = np.nan
            continue

        # Handle NaN values in similarity or log_ratio
        if np.isnan(data.get("similarity", 0)) or np.isnan(data.get("log_ratio", 0)):
            data["regularized_lipschitz"] = np.nan
            continue

        denominator = max(1 - data["similarity"], regularization_factor)
        data["regularized_lipschitz"] = data["log_ratio"] / denominator

    return pairwise_data


def build_lipschitz_matrix(
    pairwise_data: dict[str, dict[str, float]],
    inchi_keys: list[str],
) -> np.ndarray:
    """
    Build a symmetric matrix of regularized Lipschitz values.

    Excluded pairs (with NaN values) are preserved as NaN in the matrix.

    Args:
        pairwise_data: Dictionary containing 'regularized_lipschitz' for each pair.
        inchi_keys: List of InChI keys defining the order of molecules.

    Returns:
        Square symmetric matrix of regularized Lipschitz values.
    """
    n = len(inchi_keys)
    key_to_idx = {k: i for i, k in enumerate(inchi_keys)}

    # Initialize matrix with NaN, set diagonal to 0
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 0.0)

    for pair_key, data in pairwise_data.items():
        parts = pair_key.split(",")
        i = key_to_idx.get(parts[0])
        j = key_to_idx.get(parts[1])

        if i is not None and j is not None:
            value = data.get("regularized_lipschitz", np.nan)
            matrix[i, j] = value
            matrix[j, i] = value

    return matrix


def calculate_binned_cliff_probability(
    pairwise_data: dict[str, dict[str, float]],
    activity_threshold: float,
    bin_width: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate probability of log_ratio > activity_threshold for each similarity bin.

    Pairs with status='excluded' are filtered out from the analysis.

    Args:
        pairwise_data: Dictionary mapping "inchi_key_1,inchi_key_2" pairs to a dict
            containing 'similarity', 'log_ratio', 'status', etc.
        activity_threshold: Threshold for log_ratio to consider a pair an activity cliff.
        bin_width: Width of similarity bins (default 0.05).

    Returns:
        DataFrame with columns:
            - 'bin_center': center of the similarity bin
            - 'bin_lower': lower bound of the bin
            - 'bin_upper': upper bound of the bin
            - 'n_pairs': number of pairs in the bin
            - 'n_cliffs': number of pairs with log_ratio > activity_threshold
            - 'cliff_probability': probability of being an activity cliff
    """
    bins = _create_similarity_bins(bin_width)

    rows = []
    for lower, upper in bins:
        pairs_in_bin = _filter_pairs_in_interval(
            pairwise_data, lower, upper, require_similarity=True, require_log_ratio=True
        )

        n_pairs = len(pairs_in_bin)
        bin_center = (lower + upper) / 2

        # Count activity cliffs
        n_cliffs = sum(1 for d in pairs_in_bin if d["log_ratio"] > activity_threshold)
        cliff_probability = n_cliffs / n_pairs if n_pairs > 0 else 0.0

        rows.append({
            "bin_center": bin_center,
            "bin_lower": lower,
            "bin_upper": upper,
            "n_pairs": n_pairs,
            "n_cliffs": n_cliffs,
            "cliff_probability": cliff_probability,
        })

    return pd.DataFrame(rows)


def calculate_binned_pearson_correlation(
    pairwise_data: dict[str, dict[str, float]],
    bin_width: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate Pearson correlation between similarity and log_ratio for each similarity bin.

    Pairs with status='excluded' are filtered out from the analysis.

    Args:
        pairwise_data: Dictionary mapping "inchi_key_1,inchi_key_2" pairs to a dict
            containing 'similarity', 'log_ratio', 'status', etc.
        bin_width: Width of similarity bins (default 0.05).

    Returns:
        DataFrame with columns:
            - 'bin_center': center of the similarity bin
            - 'bin_lower': lower bound of the bin
            - 'bin_upper': upper bound of the bin
            - 'n_pairs': number of pairs in the bin
            - 'pearson_r': Pearson correlation coefficient
            - 'p_value': p-value of the correlation
            - 'mean_similarity': mean similarity in the bin
            - 'mean_log_ratio': mean log_ratio in the bin
    """
    bins = _create_similarity_bins(bin_width)

    rows = []
    for lower, upper in bins:
        pairs_in_bin = _filter_pairs_in_interval(
            pairwise_data, lower, upper, require_similarity=True, require_log_ratio=True
        )

        n_pairs = len(pairs_in_bin)
        bin_center = (lower + upper) / 2

        if n_pairs >= 3:  # Need at least 3 points for correlation
            similarities = np.array([d["similarity"] for d in pairs_in_bin])
            log_ratios = np.array([d["log_ratio"] for d in pairs_in_bin])

            # Check for constant values (would cause correlation to be undefined)
            if np.std(similarities) > 1e-10 and np.std(log_ratios) > 1e-10:
                r, p_value = pearsonr(similarities, log_ratios)
            else:
                r, p_value = np.nan, np.nan

            mean_sim = float(np.mean(similarities))
            mean_log_ratio = float(np.mean(log_ratios))
        else:
            r, p_value = np.nan, np.nan
            mean_sim = np.nan
            mean_log_ratio = np.nan

        rows.append({
            "bin_center": bin_center,
            "bin_lower": lower,
            "bin_upper": upper,
            "n_pairs": n_pairs,
            "pearson_r": r,
            "p_value": p_value,
            "mean_similarity": mean_sim,
            "mean_log_ratio": mean_log_ratio,
        })

    return pd.DataFrame(rows)


def calculate_conditional_probability(
    pairwise_data: dict[str, dict[str, float]],
    activity_threshold: float,
    similarity_intervals: list[tuple[float, float]] | None = None,
    tail_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Calculate P(log_ratio >= activity_threshold | similarity in interval) for different intervals.

    Also calculates statistics (mean, median, variance, std) and tail probabilities of log_ratio.
    Pairs with status='excluded' are filtered out from the analysis.

    Args:
        pairwise_data: Dictionary mapping "inchi_key_1,inchi_key_2" pairs to a dict
            containing 'similarity', 'log_ratio', 'distance', 'status', etc.
        activity_threshold: Threshold for log_ratio to consider a pair an activity cliff.
        similarity_intervals: List of (lower, upper) tuples defining similarity intervals.
            If None, uses intervals of width 0.1 from 0.0 to 1.0.
        tail_thresholds: List of thresholds for tail probability calculation.
            If None, uses [0.5, 1.0, 1.5, 2.0, 2.5, 3.0].

    Returns:
        DataFrame with columns:
            - 'interval_lower': lower bound of similarity interval
            - 'interval_upper': upper bound of similarity interval
            - 'n_pairs': number of pairs in the interval
            - 'n_cliffs': number of activity cliffs in the interval
            - 'conditional_probability': P(log_ratio >= threshold | similarity in interval)
            - 'mean_log_ratio': mean of log_ratio in interval
            - 'median_log_ratio': median of log_ratio in interval
            - 'variance_log_ratio': variance of log_ratio in interval
            - 'std_log_ratio': std of log_ratio in interval
            - 'tail_prob_X.X': P(log_ratio >= X.X | similarity in interval) for each tail threshold
    """
    if similarity_intervals is None:
        similarity_intervals = [(i * 0.1, (i + 1) * 0.1) for i in range(10)]

    if tail_thresholds is None:
        tail_thresholds = TAIL_THRESHOLDS

    rows = []
    for lower, upper in similarity_intervals:
        pairs_in_interval = _filter_pairs_in_interval(
            pairwise_data, lower, upper, require_similarity=True, require_log_ratio=True
        )

        n_pairs = len(pairs_in_interval)

        # Count activity cliffs in interval
        n_cliffs = sum(1 for data in pairs_in_interval if data["log_ratio"] >= activity_threshold)

        # Calculate statistics and tail probabilities
        if n_pairs > 0:
            prob = n_cliffs / n_pairs
            log_ratios = np.array([data["log_ratio"] for data in pairs_in_interval])

            # Basic statistics
            mean_log_ratio = float(np.mean(log_ratios))
            median_log_ratio = float(np.median(log_ratios))
            variance_log_ratio = float(np.var(log_ratios))
            std_log_ratio = float(np.std(log_ratios))

            # Tail probabilities
            tail_probs = {f"tail_prob_{t:.1f}": float(np.mean(log_ratios >= t)) for t in tail_thresholds}
        else:
            prob = 0.0
            mean_log_ratio = np.nan
            median_log_ratio = np.nan
            variance_log_ratio = np.nan
            std_log_ratio = np.nan
            tail_probs = {f"tail_prob_{t:.1f}": np.nan for t in tail_thresholds}

        row = {
            "interval_lower": lower,
            "interval_upper": upper,
            "n_pairs": n_pairs,
            "n_cliffs": n_cliffs,
            "conditional_probability": prob,
            "mean_log_ratio": mean_log_ratio,
            "median_log_ratio": median_log_ratio,
            "variance_log_ratio": variance_log_ratio,
            "std_log_ratio": std_log_ratio,
            **tail_probs,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_conditional_probability_varying_width(
    pairwise_data: dict[str, dict[str, float]],
    activity_threshold: float,
    interval_widths: list[float] | None = None,
) -> dict[float, pd.DataFrame]:
    """
    Calculate conditional probabilities for different interval widths.

    Args:
        pairwise_data: Dictionary mapping "inchi_key_1,inchi_key_2" pairs to a dict
            containing 'similarity', 'log_ratio', 'distance', etc.
        activity_threshold: Threshold for log_ratio to consider a pair an activity cliff.
        interval_widths: List of interval widths to use.
            If None, uses [0.05, 0.1, 0.2].

    Returns:
        Dictionary mapping interval width to DataFrame with conditional probabilities.
    """
    if interval_widths is None:
        interval_widths = [0.05, 0.1, 0.2]

    results = {}
    for width in interval_widths:
        # Generate intervals of the given width
        intervals = _create_similarity_bins(width)

        df = calculate_conditional_probability(
            pairwise_data=pairwise_data,
            activity_threshold=activity_threshold,
            similarity_intervals=intervals,
        )
        results[width] = df

    return results


def calculate_lipschitz_statistics(
    pairwise_data: dict[str, dict[str, float]],
    n_bins: int = 100,
    percentile_cutoff: float = 99.0,
) -> dict:
    """
    Calculate Lipschitz-like statistics from pre-calculated regularized_lipschitz values.

    Assumes pairwise_data already contains 'regularized_lipschitz' for each pair
    (calculated by calculate_regularized_lipschitz_values).
    Pairs with status='excluded' are filtered out from the analysis.

    Args:
        pairwise_data: Dictionary mapping "inchi_key_1,inchi_key_2" pairs to a dict
            containing 'regularized_lipschitz', 'similarity', 'log_ratio', 'status', etc.
        n_bins: Number of bins for histogram.
        percentile_cutoff: Percentile cutoff for KWW decay fitting.

    Returns:
        Dictionary with:
            - 'lipschitz_values': array of all Lipschitz-like values (valid only)
            - 'n_excluded': number of excluded pairs
            - 'statistics': dict with max, percentile_90, percentile_95, percentile_99, mean, median, std
            - 'histogram': dict with bin_edges and counts (up to 99th percentile)
            - 'kww_fit': dict with fit parameters (a, k, b, r_squared, etc.)
            - 'kww_theoretical': dict with theoretical r95 and mean_Er from KWW/Weibull
    """
    # Extract pre-calculated Lipschitz values, excluding status='excluded' and NaN
    valid_pairs = _filter_valid_pairs(pairwise_data, require_similarity=False, require_log_ratio=False)
    all_values = [data.get("regularized_lipschitz", np.nan) for data in valid_pairs]
    lipschitz_values = np.array([v for v in all_values if not np.isnan(v)])
    n_excluded = sum(1 for data in pairwise_data.values() if data.get("status") == "excluded")

    # Calculate statistics
    statistics = {
        "max": float(np.max(lipschitz_values)),
        "percentile_90": float(np.percentile(lipschitz_values, 90)),
        "percentile_95": float(np.percentile(lipschitz_values, 95)),
        "percentile_99": float(np.percentile(lipschitz_values, 99)),
        "mean": float(np.mean(lipschitz_values)),
        "median": float(np.median(lipschitz_values)),
        "std": float(np.std(lipschitz_values)),
        "min": float(np.min(lipschitz_values)),
    }

    # Build histogram up to 99th percentile
    cutoff = statistics["percentile_99"]
    filtered_values = lipschitz_values[lipschitz_values <= cutoff]
    counts, bin_edges = np.histogram(filtered_values, bins=n_bins)
    histogram = {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "percentile_cutoff": percentile_cutoff,
    }

    # Fit KWW decay using function from fitting.py
    kww_fit, _, _ = fit_kww_decay(
        lipschitz_values,
        n_bins=n_bins,
        percentile_cutoff=percentile_cutoff,
    )

    kww_theoretical = kww_theoretical_statistics(
        kww_fit.get("a"),
        kww_fit.get("k"),
        kww_fit.get("b"),
    )

    return {
        "lipschitz_values": lipschitz_values,
        "n_excluded": n_excluded,
        "n_valid": len(lipschitz_values),
        "statistics": statistics,
        "histogram": histogram,
        "kww_fit": kww_fit,
        "kww_theoretical": kww_theoretical,
    }


def correlate_similarity_vs_activity(
    pairwise_data: dict[str, dict[str, float]],
    raw_data: pd.DataFrame,
    results_dir: Path,
    activity_threshold: float = 1.0,
    regularization_factor: float = REGULARIZATION_FACTOR,
    mmp_file: str | Path | None = None,
) -> dict:
    """
    Correlate similarity vs activity by calculating Lipschitz statistics and correlations.

    Steps:
        3a: Calculate regularized Lipschitz values for each pair
        3b: Build and plot Lipschitz heatmap
        3c: Calculate Lipschitz statistics, fit KWW decay, and plot
        3d: Analyze log-Lipschitz distribution by similarity bin (<0.5, 0.5-0.7, 0.7-0.8, 0.8-0.9, >0.9)
        3e: Calculate Pearson correlation vs similarity bin (0.05 width)
        3f: Calculate cliff probability vs similarity bin (0.05 width)
        3g: Chiral pairs — KWW on Lipschitz; Beta(a,b) on structural distance only
        3h: Matched molecular pairs — same as 3g
        3i: Save all results

    Args:
        pairwise_data: Dictionary mapping "inchi_key_1,inchi_key_2" pairs to a dict
            containing 'similarity', 'log_ratio', 'distance', etc.
            This is the output from create_pairwise_data().
        raw_data: DataFrame with 'inchi_key' column defining molecule order.
        results_dir: Base directory for saving results.
        activity_threshold: Threshold for log_ratio to consider a pair an activity cliff.
        regularization_factor: Minimum denominator value for Lipschitz calculation.
        mmp_file: Optional path to CSV with smiles1, smiles2 (or smiles_1, smiles_2) columns for matched molecular pairs analysis.

    Returns:
        Dictionary with 'cliff_probability', 'lipschitz_statistics',
        'binned_correlation', and updated 'pairwise_data'.
    """
    console = Console()

    console.print("[bold cyan]Step 3 - Correlating similarity vs activity...[/bold cyan]")

    correlation_dir = results_dir / "correlation"
    correlation_dir.mkdir(parents=True, exist_ok=True)

    # Get inchi_keys for matrix ordering
    inchi_keys = raw_data["inchi_key"].tolist()

    # Count excluded pairs
    n_excluded = sum(1 for data in pairwise_data.values() if data.get("status") == "excluded")
    n_valid = len(pairwise_data) - n_excluded
    if n_excluded > 0:
        console.print(f"  Valid pairs: {n_valid:,} (excluded: {n_excluded:,})")

    # -------------------------------------------------------------------------
    # Step 3a: Calculate regularized Lipschitz values
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 3a: Calculating regularized Lipschitz values...[/cyan]")

    pairwise_data = calculate_regularized_lipschitz_values(
        pairwise_data=pairwise_data,
        regularization_factor=regularization_factor,
    )
    console.print(f"[green]  ✓ Regularized Lipschitz values calculated (regularization={regularization_factor})[/green]")

    # -------------------------------------------------------------------------
    # Step 3b: Build and plot Lipschitz heatmap
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 3b: Building and plotting Lipschitz heatmap...[/cyan]")

    lipschitz_matrix = build_lipschitz_matrix(pairwise_data, inchi_keys)
    np.save(correlation_dir / "lipschitz_matrix.npy", lipschitz_matrix)

    plot_lipschitz_heatmap(
        lipschitz_matrix=lipschitz_matrix,
        output_path=correlation_dir / "lipschitz_heatmap.png",
        title="Regularized Lipschitz Heatmap",
    )
    console.print("[green]  ✓ Lipschitz heatmap plotted[/green]")

    # -------------------------------------------------------------------------
    # Step 3c: Calculate Lipschitz statistics, fit KWW decay, and plot
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 3c: Calculating Lipschitz statistics and fitting KWW decay...[/cyan]")

    lipschitz_results = calculate_lipschitz_statistics(pairwise_data=pairwise_data)

    # Save Lipschitz statistics (without the large array)
    lipschitz_to_save = {
        "statistics": lipschitz_results["statistics"],
        "histogram": lipschitz_results["histogram"],
        "kww_fit": lipschitz_results["kww_fit"],
        "kww_theoretical": lipschitz_results["kww_theoretical"],
        "regularization_factor": regularization_factor,
    }
    with open(correlation_dir / "lipschitz_statistics.json", "w") as f:
        json.dump(lipschitz_to_save, f, indent=2)

    # Plot Lipschitz distribution with KWW decay fit
    plot_lipschitz_distribution(
        lipschitz_data=lipschitz_results,
        output_path=correlation_dir / "lipschitz_statistics.png",
        title="Lipschitz Statistics",
    )

    kww_fit = lipschitz_results["kww_fit"]
    if kww_fit.get("a") is not None:
        console.print(
            f"[green]  ✓ Lipschitz statistics calculated, KWW decay fit: "
            f"a={kww_fit['a']:.3f}, k={kww_fit['k']:.3f}, b={kww_fit['b']:.3f}, R²={kww_fit['r_squared']:.3f}[/green]"
        )
    else:
        console.print("[yellow]  ⚠ Lipschitz statistics calculated, KWW decay fit failed[/yellow]")

    # -------------------------------------------------------------------------
    # Step 3d: Binned log-Lipschitz analysis (by similarity bin)
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 3d: Analyzing log-Lipschitz distribution by similarity bin...[/cyan]")

    plot_binned_log_lipschitz(
        pairwise_data=pairwise_data,
        output_path=correlation_dir / "binned_log_lipschitz.png",
        title="Log-Lipschitz Distribution by Similarity Bin",
    )
    console.print("[green]  ✓ Binned log-Lipschitz analysis complete[/green]")

    # -------------------------------------------------------------------------
    # Step 3e: Calculate Pearson correlation vs similarity bin
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 3e: Calculating Pearson correlation vs similarity bin...[/cyan]")

    binned_correlation_df = calculate_binned_pearson_correlation(
        pairwise_data=pairwise_data,
        bin_width=0.05,
    )

    # Save binned correlation results
    binned_correlation_df.to_csv(correlation_dir / "binned_pearson_correlation.csv", index=False)

    # Plot binned correlation
    plot_binned_correlation(
        df=binned_correlation_df,
        output_path=correlation_dir / "binned_pearson_correlation.png",
        title="Pearson Correlation vs Similarity Bin",
    )

    # Calculate overall statistics
    valid_correlations = binned_correlation_df["pearson_r"].dropna()
    if len(valid_correlations) > 0:
        mean_r = valid_correlations.mean()
        console.print(f"[green]  ✓ Binned Pearson correlation calculated (mean r={mean_r:.3f})[/green]")
    else:
        console.print("[green]  ✓ Binned Pearson correlation calculated[/green]")

    # -------------------------------------------------------------------------
    # Step 3f: Calculate cliff probability vs similarity bin
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 3f: Calculating cliff probability vs similarity bin...[/cyan]")

    cliff_probability_df = calculate_binned_cliff_probability(
        pairwise_data=pairwise_data,
        activity_threshold=activity_threshold,
        bin_width=0.05,
    )

    # Save cliff probability results
    cliff_probability_df.to_csv(correlation_dir / "binned_cliff_probability.csv", index=False)

    # Plot cliff probability
    plot_binned_cliff_probability(
        df=cliff_probability_df,
        activity_threshold=activity_threshold,
        output_path=correlation_dir / "binned_cliff_probability.png",
        title=f"Activity Cliff Probability vs Similarity (threshold={activity_threshold})",
    )

    # Calculate summary statistics
    total_cliffs = cliff_probability_df["n_cliffs"].sum()
    total_pairs = cliff_probability_df["n_pairs"].sum()
    overall_cliff_prob = total_cliffs / total_pairs if total_pairs > 0 else 0.0
    console.print(f"[green]  ✓ Cliff probability calculated (overall: {overall_cliff_prob:.3f})[/green]")

    # -------------------------------------------------------------------------
    # Step 3g: Chiral compound pairs (same first InChI block, |Δlog_activity| > threshold)
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 3g: Analyzing chiral compound pairs...[/cyan]")

    # First part of inchi_key (connectivity); same first part => potential stereoisomers
    inchi_prefix = raw_data["inchi_key"].str.split("-").str[0]
    all_chiral_pair_keys = []  # all valid chiral pairs (in pairwise_data, not excluded)
    chiral_pair_keys = []      # subset that are cliffs (log_ratio > threshold)
    for _prefix, idx in inchi_prefix.groupby(inchi_prefix, sort=False).groups.items():
        group = raw_data.loc[idx]
        if len(group) < 2:
            continue
        inchi_keys = group["inchi_key"].tolist()
        for i in range(len(inchi_keys)):
            for j in range(i + 1, len(inchi_keys)):
                k1, k2 = inchi_keys[i], inchi_keys[j]
                for pair_key in (f"{k1},{k2}", f"{k2},{k1}"):
                    if pair_key not in pairwise_data:
                        continue
                    data = pairwise_data[pair_key]
                    if data.get("status") == "excluded":
                        continue
                    if np.isnan(data.get("log_ratio", np.nan)):
                        continue
                    all_chiral_pair_keys.append(pair_key)
                    if data["log_ratio"] > activity_threshold:
                        chiral_pair_keys.append(pair_key)
                    break

    n_total_chiral_pairs = len(all_chiral_pair_keys)
    chiral_pairwise_data = {k: pairwise_data[k] for k in chiral_pair_keys}
    n_chiral_pairs = len(chiral_pairwise_data)
    fraction_chiral_cliffs = n_chiral_pairs / n_total_chiral_pairs if n_total_chiral_pairs > 0 else 0.0

    if n_chiral_pairs > 0:
        lipschitz_results_chiral = calculate_lipschitz_statistics(pairwise_data=chiral_pairwise_data)
        pair_subset_stats_chiral = _pair_subset_statistics(chiral_pairwise_data)
        sim_arr = np.array([d["similarity"] for d in chiral_pairwise_data.values() if not np.isnan(d.get("similarity", np.nan))])
        log_arr = np.array([d["log_ratio"] for d in chiral_pairwise_data.values() if not np.isnan(d.get("log_ratio", np.nan))])
        lip_arr = lipschitz_results_chiral["lipschitz_values"]
        dist_arr_chiral = _pairwise_scaled_distances(chiral_pairwise_data)
        distance_beta_chiral = fit_beta_distance(dist_arr_chiral)
        n_chiral_scaled_zero, frac_chiral_scaled_zero = _scaled_distance_zero_count_and_fraction(
            chiral_pairwise_data
        )

        chiral_to_save = {
            "n_total_pairs": n_total_chiral_pairs,
            "n_pairs": n_chiral_pairs,
            "fraction_cliffs": fraction_chiral_cliffs,
            "activity_threshold": activity_threshold,
            "statistics": lipschitz_results_chiral["statistics"],
            "statistics_similarity": pair_subset_stats_chiral.get("similarity") if pair_subset_stats_chiral else None,
            "statistics_log_ratio": pair_subset_stats_chiral.get("log_ratio") if pair_subset_stats_chiral else None,
            "statistics_regularized_lipschitz": pair_subset_stats_chiral.get("regularized_lipschitz") if pair_subset_stats_chiral else None,
            "histogram": lipschitz_results_chiral["histogram"],
            "kww_fit": lipschitz_results_chiral["kww_fit"],
            "kww_theoretical": lipschitz_results_chiral["kww_theoretical"],
            "distance_beta_fit": distance_beta_chiral,
            "regularization_factor": regularization_factor,
            "n_pairs_scaled_distance_zero": n_chiral_scaled_zero,
            "fraction_scaled_distance_zero": frac_chiral_scaled_zero,
        }
        with open(correlation_dir / "chiral_cliff_lipschitz_statistics.json", "w") as f:
            json.dump(chiral_to_save, f, indent=2)

        plot_lipschitz_distribution(
            lipschitz_data=lipschitz_results_chiral,
            output_path=correlation_dir / "chiral_cliff_lipschitz_statistics.png",
            title="Chiral Compound Pairs (Activity Cliff) — Lipschitz Statistics",
        )
        plot_distance_distribution_beta(
            distance_values=dist_arr_chiral,
            beta_fit=distance_beta_chiral,
            output_path=correlation_dir / "chiral_distance_beta.png",
            title="Chiral Compound Pairs (Activity Cliff) — Scaled distance (Beta fit)",
        )
        plot_pair_subset_three_panels(
            similarity_values=sim_arr,
            log_ratio_values=log_arr,
            lipschitz_values=lip_arr,
            statistics=pair_subset_stats_chiral,
            output_path=correlation_dir / "chiral_cliff_three_panels.png",
            title="Chiral Compound Pairs (Activity Cliff) — Similarity, Log Ratio, Lipschitz",
        )

        kww_chiral = lipschitz_results_chiral["kww_fit"]
        if kww_chiral.get("a") is not None:
            console.print(
                f"[green]  ✓ Chiral: {n_chiral_pairs} cliff pairs (of {n_total_chiral_pairs} total), fraction cliffs={fraction_chiral_cliffs:.3f}, "
                f"Lipschitz max={lipschitz_results_chiral['statistics']['max']:.2f}, KWW R²={kww_chiral['r_squared']:.3f}[/green]"
            )
        else:
            console.print(f"[green]  ✓ Chiral: {n_chiral_pairs} cliff pairs (of {n_total_chiral_pairs} total), fraction cliffs={fraction_chiral_cliffs:.3f}, Lipschitz max={lipschitz_results_chiral['statistics']['max']:.2f}[/green]")
        db = distance_beta_chiral
        if db.get("a") is not None:
            console.print(
                f"[green]    Distance Beta(a,b)=({db['a']:.3f},{db['b']:.3f}), m={db['m']:.3f}, f={db['f']:.3f}, R²={db['r_squared']:.3f}[/green]"
            )
    else:
        chiral_to_save = {
            "n_total_pairs": n_total_chiral_pairs,
            "n_pairs": 0,
            "fraction_cliffs": fraction_chiral_cliffs,
            "activity_threshold": activity_threshold,
            "statistics": None,
            "statistics_similarity": None,
            "statistics_log_ratio": None,
            "statistics_regularized_lipschitz": None,
            "histogram": None,
            "kww_fit": None,
            "kww_theoretical": None,
            "distance_beta_fit": None,
            "regularization_factor": regularization_factor,
            "n_pairs_scaled_distance_zero": 0,
            "fraction_scaled_distance_zero": 0.0,
        }
        with open(correlation_dir / "chiral_cliff_lipschitz_statistics.json", "w") as f:
            json.dump(chiral_to_save, f, indent=2)
        lipschitz_results_chiral = None
        pair_subset_stats_chiral = None
        console.print("[yellow]  ⚠ No chiral compound pairs with activity difference > threshold[/yellow]")

    # -------------------------------------------------------------------------
    # Step 3h: Matched molecular pairs (from mmp_file CSV)
    # -------------------------------------------------------------------------
    lipschitz_results_mmp = None
    n_mmp_pairs = 0
    mmp_summary = {
        "n_total_pairs": 0,
        "n_pairs": 0,
        "fraction_cliffs": 0.0,
        "activity_threshold": activity_threshold,
        "statistics": None,
        "kww_fit": None,
        "kww_theoretical": None,
        "distance_beta_fit": None,
    }
    if mmp_file is not None:
        mmp_path = Path(mmp_file)
        console.print("[cyan]Step 3h: Analyzing matched molecular pairs...[/cyan]")
        if not mmp_path.exists():
            console.print(f"[yellow]  ⚠ mmp_file not found: {mmp_path}[/yellow]")
            mmp_summary = {
                "n_pairs": 0,
                "activity_threshold": activity_threshold,
                "statistics": None,
                "kww_fit": None,
                "kww_theoretical": None,
                "distance_beta_fit": None,
                "error": "file_not_found",
                "mmp_file": str(mmp_path),
            }
            with open(correlation_dir / "mmp_cliff_lipschitz_statistics.json", "w") as f:
                json.dump({**mmp_summary, "regularization_factor": regularization_factor}, f, indent=2)
        else:
            mmp_df = pd.read_csv(mmp_path)
            # Accept smiles_1/smiles_2 (as in the standard dataset sources) as well as smiles1/smiles2
            mmp_df = mmp_df.rename(columns={"smiles_1": "smiles1", "smiles_2": "smiles2"})
            if "smiles1" not in mmp_df.columns or "smiles2" not in mmp_df.columns:
                console.print("[yellow]  ⚠ mmp_file must have 'smiles1' and 'smiles2' columns; skipping MMP analysis[/yellow]")
                mmp_summary = {
                    "n_pairs": 0,
                    "activity_threshold": activity_threshold,
                    "statistics": None,
                    "kww_fit": None,
                    "kww_theoretical": None,
                    "distance_beta_fit": None,
                    "error": "missing_columns",
                    "mmp_file": str(mmp_path),
                }
                with open(correlation_dir / "mmp_cliff_lipschitz_statistics.json", "w") as f:
                    json.dump({**mmp_summary, "regularization_factor": regularization_factor}, f, indent=2)
            else:
                all_mmp_matched = set()
                mmp_pair_keys = []
                for _, row in mmp_df.iterrows():
                    s1, s2 = str(row["smiles1"]).strip(), str(row["smiles2"]).strip()
                    try:
                        k1, k2 = get_inchi_key(s1), get_inchi_key(s2)
                    except (ValueError, TypeError):
                        continue
                    for pair_key in (f"{k1},{k2}", f"{k2},{k1}"):
                        if pair_key not in pairwise_data:
                            continue
                        data = pairwise_data[pair_key]
                        if data.get("status") == "excluded":
                            continue
                        if np.isnan(data.get("log_ratio", np.nan)):
                            continue
                        all_mmp_matched.add(pair_key)
                        if data["log_ratio"] > activity_threshold:
                            mmp_pair_keys.append(pair_key)
                        break
                n_total_mmp_pairs = len(all_mmp_matched)
                mmp_pairwise_data = {k: pairwise_data[k] for k in mmp_pair_keys}
                n_mmp_pairs = len(mmp_pairwise_data)
                fraction_mmp_cliffs = n_mmp_pairs / n_total_mmp_pairs if n_total_mmp_pairs > 0 else 0.0
                if n_mmp_pairs > 0:
                    lipschitz_results_mmp = calculate_lipschitz_statistics(pairwise_data=mmp_pairwise_data)
                    pair_subset_stats_mmp = _pair_subset_statistics(mmp_pairwise_data)
                    sim_arr_mmp = np.array([d["similarity"] for d in mmp_pairwise_data.values() if not np.isnan(d.get("similarity", np.nan))])
                    log_arr_mmp = np.array([d["log_ratio"] for d in mmp_pairwise_data.values() if not np.isnan(d.get("log_ratio", np.nan))])
                    lip_arr_mmp = lipschitz_results_mmp["lipschitz_values"]
                    dist_arr_mmp = _pairwise_scaled_distances(mmp_pairwise_data)
                    distance_beta_mmp = fit_beta_distance(dist_arr_mmp)
                    mmp_to_save = {
                        "n_total_pairs": n_total_mmp_pairs,
                        "n_pairs": n_mmp_pairs,
                        "fraction_cliffs": fraction_mmp_cliffs,
                        "activity_threshold": activity_threshold,
                        "mmp_file": str(mmp_path),
                        "statistics": lipschitz_results_mmp["statistics"],
                        "statistics_similarity": pair_subset_stats_mmp.get("similarity") if pair_subset_stats_mmp else None,
                        "statistics_log_ratio": pair_subset_stats_mmp.get("log_ratio") if pair_subset_stats_mmp else None,
                        "statistics_regularized_lipschitz": pair_subset_stats_mmp.get("regularized_lipschitz") if pair_subset_stats_mmp else None,
                        "histogram": lipschitz_results_mmp["histogram"],
                        "kww_fit": lipschitz_results_mmp["kww_fit"],
                        "kww_theoretical": lipschitz_results_mmp["kww_theoretical"],
                        "distance_beta_fit": distance_beta_mmp,
                        "regularization_factor": regularization_factor,
                    }
                    with open(correlation_dir / "mmp_cliff_lipschitz_statistics.json", "w") as f:
                        json.dump(mmp_to_save, f, indent=2)
                    plot_lipschitz_distribution(
                        lipschitz_data=lipschitz_results_mmp,
                        output_path=correlation_dir / "mmp_cliff_lipschitz_statistics.png",
                        title="Matched Molecular Pairs (Activity Cliff) — Lipschitz Statistics",
                    )
                    plot_distance_distribution_beta(
                        distance_values=dist_arr_mmp,
                        beta_fit=distance_beta_mmp,
                        output_path=correlation_dir / "mmp_distance_beta.png",
                        title="Matched Molecular Pairs (Activity Cliff) — Scaled distance (Beta fit)",
                    )
                    plot_pair_subset_three_panels(
                        similarity_values=sim_arr_mmp,
                        log_ratio_values=log_arr_mmp,
                        lipschitz_values=lip_arr_mmp,
                        statistics=pair_subset_stats_mmp,
                        output_path=correlation_dir / "mmp_cliff_three_panels.png",
                        title="Matched Molecular Pairs (Activity Cliff) — Similarity, Log Ratio, Lipschitz",
                    )
                    kww_mmp = lipschitz_results_mmp["kww_fit"]
                    if kww_mmp.get("a") is not None:
                        console.print(
                            f"[green]  ✓ MMP: {n_mmp_pairs} cliff pairs (of {n_total_mmp_pairs} total), fraction cliffs={fraction_mmp_cliffs:.3f}, "
                            f"Lipschitz max={lipschitz_results_mmp['statistics']['max']:.2f}, KWW R²={kww_mmp['r_squared']:.3f}[/green]"
                        )
                    else:
                        console.print(f"[green]  ✓ MMP: {n_mmp_pairs} cliff pairs (of {n_total_mmp_pairs} total), fraction cliffs={fraction_mmp_cliffs:.3f}, Lipschitz max={lipschitz_results_mmp['statistics']['max']:.2f}[/green]")
                    dbm = distance_beta_mmp
                    if dbm.get("a") is not None:
                        console.print(
                            f"[green]    Distance Beta(a,b)=({dbm['a']:.3f},{dbm['b']:.3f}), m={dbm['m']:.3f}, f={dbm['f']:.3f}, R²={dbm['r_squared']:.3f}[/green]"
                        )
                    mmp_summary = {
                        "n_total_pairs": n_total_mmp_pairs,
                        "n_pairs": n_mmp_pairs,
                        "fraction_cliffs": fraction_mmp_cliffs,
                        "activity_threshold": activity_threshold,
                        "statistics": lipschitz_results_mmp["statistics"],
                        "kww_fit": lipschitz_results_mmp["kww_fit"],
                        "kww_theoretical": lipschitz_results_mmp["kww_theoretical"],
                        "distance_beta_fit": distance_beta_mmp,
                        "mmp_file": str(mmp_path),
                        "statistics_similarity": mmp_to_save.get("statistics_similarity"),
                        "statistics_log_ratio": mmp_to_save.get("statistics_log_ratio"),
                        "statistics_regularized_lipschitz": mmp_to_save.get("statistics_regularized_lipschitz"),
                    }
                else:
                    mmp_to_save = {
                        "n_total_pairs": n_total_mmp_pairs,
                        "n_pairs": 0,
                        "fraction_cliffs": fraction_mmp_cliffs,
                        "activity_threshold": activity_threshold,
                        "mmp_file": str(mmp_path),
                        "statistics": None,
                        "statistics_similarity": None,
                        "statistics_log_ratio": None,
                        "statistics_regularized_lipschitz": None,
                        "histogram": None,
                        "kww_fit": None,
                        "kww_theoretical": None,
                        "distance_beta_fit": None,
                        "regularization_factor": regularization_factor,
                    }
                    with open(correlation_dir / "mmp_cliff_lipschitz_statistics.json", "w") as f:
                        json.dump(mmp_to_save, f, indent=2)
                    mmp_summary = {
                        "n_total_pairs": n_total_mmp_pairs,
                        "n_pairs": 0,
                        "fraction_cliffs": fraction_mmp_cliffs,
                        "activity_threshold": activity_threshold,
                        "statistics": None,
                        "kww_fit": None,
                        "kww_theoretical": None,
                        "distance_beta_fit": None,
                        "mmp_file": str(mmp_path),
                        "statistics_similarity": None,
                        "statistics_log_ratio": None,
                        "statistics_regularized_lipschitz": None,
                    }
                    console.print("[yellow]  ⚠ No MMP pairs with activity difference > threshold found in pairwise data[/yellow]")
    else:
        mmp_summary = {
            "n_pairs": 0,
            "activity_threshold": activity_threshold,
            "statistics": None,
            "kww_fit": None,
            "kww_theoretical": None,
            "distance_beta_fit": None,
        }

    # -------------------------------------------------------------------------
    # Step 3i: Save results
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 3i: Saving correlation results...[/cyan]")

    # Calculate binned correlation summary statistics
    valid_r = binned_correlation_df["pearson_r"].dropna()
    binned_corr_summary = {
        "bin_width": 0.05,
        "n_bins": len(binned_correlation_df),
        "n_valid_bins": len(valid_r),
        "mean_pearson_r": float(valid_r.mean()) if len(valid_r) > 0 else None,
        "std_pearson_r": float(valid_r.std()) if len(valid_r) > 0 else None,
        "min_pearson_r": float(valid_r.min()) if len(valid_r) > 0 else None,
        "max_pearson_r": float(valid_r.max()) if len(valid_r) > 0 else None,
    }

    # Calculate cliff probability summary
    cliff_prob_summary = {
        "bin_width": 0.05,
        "activity_threshold": activity_threshold,
        "n_bins": len(cliff_probability_df),
        "total_pairs": int(total_pairs),
        "total_cliffs": int(total_cliffs),
        "overall_cliff_probability": float(overall_cliff_prob),
    }

    # Chiral cliff summary for report
    chiral_summary = {
        "n_total_pairs": n_total_chiral_pairs,
        "n_pairs": n_chiral_pairs,
        "fraction_cliffs": fraction_chiral_cliffs,
        "activity_threshold": activity_threshold,
    }
    if lipschitz_results_chiral is not None:
        chiral_summary["statistics"] = lipschitz_results_chiral["statistics"]
        chiral_summary["kww_fit"] = lipschitz_results_chiral["kww_fit"]
        chiral_summary["kww_theoretical"] = lipschitz_results_chiral["kww_theoretical"]
        chiral_summary["distance_beta_fit"] = chiral_to_save.get("distance_beta_fit")
        chiral_summary["statistics_similarity"] = chiral_to_save.get("statistics_similarity")
        chiral_summary["statistics_log_ratio"] = chiral_to_save.get("statistics_log_ratio")
        chiral_summary["statistics_regularized_lipschitz"] = chiral_to_save.get("statistics_regularized_lipschitz")
    else:
        chiral_summary["statistics"] = None
        chiral_summary["kww_fit"] = None
        chiral_summary["kww_theoretical"] = None
        chiral_summary["distance_beta_fit"] = None
        chiral_summary["statistics_similarity"] = None
        chiral_summary["statistics_log_ratio"] = None
        chiral_summary["statistics_regularized_lipschitz"] = None

    # Save summary
    summary = {
        "activity_threshold": activity_threshold,
        "total_pairs": len(pairwise_data),
        "excluded_pairs": n_excluded,
        "valid_pairs": n_valid,
        "regularization_factor": regularization_factor,
        "lipschitz_max": lipschitz_results["statistics"]["max"],
        "lipschitz_percentile_90": lipschitz_results["statistics"]["percentile_90"],
        "lipschitz_percentile_99": lipschitz_results["statistics"]["percentile_99"],
        "kww_fit": lipschitz_results["kww_fit"],
        "kww_theoretical": lipschitz_results["kww_theoretical"],
        "binned_correlation": binned_corr_summary,
        "cliff_probability": cliff_prob_summary,
        "chiral_cliff_lipschitz": chiral_summary,
        "mmp_cliff_lipschitz": mmp_summary,
    }
    with open(correlation_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    console.print("[green]  ✓ Results saved[/green]")
    console.print("[green]  ✓ Correlation analysis complete[/green]")
    console.print(f"  Results saved to: {correlation_dir}")
    console.print(
        f"  Lipschitz stats: max={lipschitz_results['statistics']['max']:.2f}, "
        f"99th percentile={lipschitz_results['statistics']['percentile_99']:.2f}"
    )

    return {
        "cliff_probability": cliff_probability_df,
        "lipschitz_statistics": lipschitz_results,
        "binned_correlation": binned_correlation_df,
        "pairwise_data": pairwise_data,
        "chiral_cliff_lipschitz": lipschitz_results_chiral,
        "chiral_cliff_n_pairs": n_chiral_pairs,
        "mmp_cliff_lipschitz": lipschitz_results_mmp,
        "mmp_cliff_n_pairs": n_mmp_pairs,
    }
