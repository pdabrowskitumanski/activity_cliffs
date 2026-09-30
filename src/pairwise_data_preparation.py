"""
Pairwise data preparation module for activity cliffs analysis.

This module provides functions for:
- Calculating pairwise distances between molecular embeddings
- Scaling distances to similarity values
- Computing distance statistics
- Building and visualizing distance matrices
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.compound_similarity import calculate_vector_distance, get_scaling_function
from src.fitting import fit_normal, fit_scaled_distances_bic_comparison
from src.plotting import plot_distance_distributions, plot_distance_matrices
from src.utils import convert_to_native_types


# ============================================================================
# Pairwise Distance Calculations
# ============================================================================

def _make_json_safe(val):
    """Convert NaN to None for JSON serialization."""
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


def _determine_pair_status(censored_i: bool, censored_j: bool) -> str:
    """
    Determine the status of a pair based on censored flags.

    Args:
        censored_i: Whether the first compound is censored.
        censored_j: Whether the second compound is censored.

    Returns:
        Status string: 'excluded', 'censored', or 'correct'.
    """
    if censored_i and censored_j:
        return "excluded"
    elif censored_i or censored_j:
        return "censored"
    return "correct"


def calculate_pairwise_distances(
    df: pd.DataFrame,
    results_dir: Path,
    distance_metric: str = "tanimoto",
) -> dict[str, dict[str, float]]:
    """
    Calculate pairwise distances between all molecules in the dataset.

    For each pair of molecules, calculates:
    - Structural distance using the specified metric
    - Activity difference (absolute)
    - Log ratio (absolute difference of -log10 activities)
    - Status based on censored status of compounds

    Args:
        df: DataFrame with columns 'smiles', 'inchi_key', 'embedding',
            'activity', 'log_activity', and optionally 'censored'.
        results_dir: Directory to save results.
        distance_metric: Distance metric to use (default: 'tanimoto').

    Returns:
        Dictionary mapping "inchi_key_1,inchi_key_2" pairs to a dict containing:
        - 'distance_raw': Structural distance (0 = identical, NaN if excluded)
        - 'activity_difference': Absolute activity difference (NaN if excluded)
        - 'log_ratio': Absolute log ratio difference
        - 'status': 'correct', 'censored', or 'excluded'
        - 'embedding1': First molecule's embedding
        - 'embedding2': Second molecule's embedding
    """
    n_molecules = len(df)
    n_pairs = n_molecules * (n_molecules - 1) // 2

    # Check for censored column
    has_censored = "censored" in df.columns

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task_pairs = progress.add_task("[cyan]Calculating pairwise metrics...", total=n_pairs)

        results: dict[str, dict[str, float]] = {}
        structurally_similar_pairs: list[dict] = []  # distance == 0
        activity_similar_pairs: list[dict] = []  # log_ratio == 0.0

        # Status counters
        n_correct = 0
        n_censored = 0
        n_excluded = 0

        # Create index-based lookup for faster access
        smiles_list = df["smiles"].tolist()
        inchi_keys_list = df["inchi_key"].tolist()
        embeddings_list = df["embedding"].tolist()
        activities_list = df["activity"].tolist()
        log_activities_list = df["log_activity"].tolist()

        # Get censored status list
        if has_censored:
            censored_list = df["censored"].tolist()
        else:
            censored_list = [False] * n_molecules

        for i, j in combinations(range(n_molecules), 2):
            inchi_key_1 = inchi_keys_list[i]
            inchi_key_2 = inchi_keys_list[j]

            # Determine status based on censored values
            censored_i = censored_list[i] == True
            censored_j = censored_list[j] == True
            status = _determine_pair_status(censored_i, censored_j)

            if status == "excluded":
                n_excluded += 1
            elif status == "censored":
                n_censored += 1
            else:
                n_correct += 1

            # Calculate distance, activity_diff, and log_ratio (NaN for excluded pairs)
            if status == "excluded":
                distance = np.nan
                activity_diff = np.nan
                log_ratio = np.nan
            else:
                distance = calculate_vector_distance(
                    embeddings_list[i], embeddings_list[j], metric=distance_metric
                )
                activity_diff = abs(activities_list[i] - activities_list[j])
                log_ratio = abs(log_activities_list[i] - log_activities_list[j])

            # Use string key for JSON serialization
            pair_key = f"{inchi_key_1},{inchi_key_2}"
            results[pair_key] = {
                "distance_raw": distance,
                "activity_difference": activity_diff,
                "log_ratio": log_ratio,
                "status": status,
                "embedding1": embeddings_list[i],
                "embedding2": embeddings_list[j],
            }

            # Track pairs with distance_raw == 0 (structurally identical) - only for non-excluded
            if status != "excluded" and distance == 0.0:
                structurally_similar_pairs.append({
                    "smiles1": smiles_list[i],
                    "inchi_key1": inchi_key_1,
                    "activity1": activities_list[i],
                    "smiles2": smiles_list[j],
                    "inchi_key2": inchi_key_2,
                    "activity2": activities_list[j],
                    "distance": distance,
                    "log_ratio": log_ratio,
                    "status": status,
                })

            # Track pairs with log_ratio == 0.0 (similar activity) - only for non-excluded
            if status != "excluded" and log_ratio == 0.0:
                activity_similar_pairs.append({
                    "smiles1": smiles_list[i],
                    "inchi_key1": inchi_key_1,
                    "activity1": activities_list[i],
                    "smiles2": smiles_list[j],
                    "inchi_key2": inchi_key_2,
                    "activity2": activities_list[j],
                    "distance": distance,
                    "log_ratio": log_ratio,
                    "status": status,
                })

            progress.update(task_pairs, advance=1)

        # Save raw distances to JSON (excluding embeddings for JSON serialization)
        task_save = progress.add_task("[cyan]Saving results...", total=3)

        results_for_json = {
            k: {key: _make_json_safe(val) for key, val in v.items()
                if key not in ("embedding1", "embedding2")}
            for k, v in results.items()
        }
        with open(results_dir / "raw_distances.json", "w") as f:
            json.dump(results_for_json, f, indent=2)
        progress.update(task_save, advance=1)

        # Save structurally similar pairs (distance == 0)
        structurally_similar_df = pd.DataFrame(structurally_similar_pairs)
        structurally_similar_df.to_csv(results_dir / "structurally_similar_pairs.csv", index=False)
        if len(structurally_similar_pairs) > 0:
            progress.console.print(
                f"[yellow]Found {len(structurally_similar_pairs)} pairs with distance = 0.0[/yellow]"
            )
        progress.update(task_save, advance=1)

        # Save activity similar pairs (log_ratio == 0.0)
        activity_similar_df = pd.DataFrame(activity_similar_pairs)
        activity_similar_df.to_csv(results_dir / "activity_similar_pairs.csv", index=False)
        if len(activity_similar_pairs) > 0:
            progress.console.print(
                f"[yellow]Found {len(activity_similar_pairs)} pairs with log_ratio == 0.0[/yellow]"
            )
        progress.update(task_save, advance=1)

        # Print status summary
        if has_censored:
            progress.console.print(
                f"[cyan]Pair status: {n_correct:,} correct, {n_censored:,} censored, "
                f"{n_excluded:,} excluded[/cyan]"
            )

    return results


# ============================================================================
# Distance Statistics
# ============================================================================

def _compute_stats(arr: np.ndarray, name: str) -> dict:
    """
    Compute statistical measures for an array.

    Args:
        arr: Array of values.
        name: Base name for the statistics keys.

    Returns:
        Dictionary with statistics: min, max, mean, median, std, q25, q75, q90, q99.
    """
    if len(arr) == 0:
        return {
            f"{name}_min": None,
            f"{name}_max": None,
            f"{name}_mean": None,
            f"{name}_median": None,
            f"{name}_std": None,
            f"{name}_q25": None,
            f"{name}_q75": None,
            f"{name}_q90": None,
            f"{name}_q99": None,
        }
    return {
        f"{name}_min": float(np.min(arr)),
        f"{name}_max": float(np.max(arr)),
        f"{name}_mean": float(np.mean(arr)),
        f"{name}_median": float(np.median(arr)),
        f"{name}_std": float(np.std(arr)),
        f"{name}_q25": float(np.percentile(arr, 25)),
        f"{name}_q75": float(np.percentile(arr, 75)),
        f"{name}_q90": float(np.percentile(arr, 90)),
        f"{name}_q99": float(np.percentile(arr, 99)),
    }


def calculate_distance_statistics(
    pairwise_results: dict[str, dict[str, float]],
) -> dict:
    """
    Calculate statistics of pairwise distances.

    Excludes pairs with status='excluded' from statistics calculation.

    Args:
        pairwise_results: Dictionary of pairwise distance results.

    Returns:
        Dictionary with statistics for distance, activity_difference, and log_ratio,
        including pair counts by status.
    """
    # Filter out excluded pairs and NaN values
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

    # Count pairs by status
    n_total = len(pairwise_results)
    n_excluded = sum(1 for v in pairwise_results.values() if v.get("status") == "excluded")
    n_censored = sum(1 for v in pairwise_results.values() if v.get("status") == "censored")
    n_correct = sum(1 for v in pairwise_results.values() if v.get("status", "correct") == "correct")

    stats = {
        "n_pairs": n_total,
        "n_pairs_correct": n_correct,
        "n_pairs_censored": n_censored,
        "n_pairs_excluded": n_excluded,
        "n_pairs_valid": len(valid_pairs),
        **_compute_stats(distances, "distance"),
        **_compute_stats(activity_diffs, "activity_difference"),
        **_compute_stats(log_ratios, "log_ratio"),
    }

    return stats


# ============================================================================
# Distance and Similarity Matrices
# ============================================================================

def build_distance_matrix(
    pairwise_results: dict[str, dict[str, float]],
    inchi_keys: list[str],
    metric: str = "distance",
    warn_missing: bool = True,
    preserve_nan: bool = True,
) -> tuple[np.ndarray, list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Build a symmetric distance matrix from pairwise results.

    Args:
        pairwise_results: Dictionary of pairwise results.
        inchi_keys: List of InChI keys defining matrix order.
        metric: Which metric to use ('distance', 'activity_difference', 'log_ratio',
            'distance_scaled').
        warn_missing: If True, warn about missing pairs (off-diagonal zeros).
        preserve_nan: If True, keep NaN values for excluded pairs (for white display).

    Returns:
        Tuple of:
        - Symmetric matrix of shape (n, n)
        - List of missing pairs as (inchi_key1, inchi_key2) tuples
        - List of excluded pairs as (inchi_key1, inchi_key2) tuples
    """
    n = len(inchi_keys)
    key_to_idx = {k: i for i, k in enumerate(inchi_keys)}

    # Initialize matrix with NaN to detect missing pairs
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 0.0)  # Self-distance is 0

    excluded_pairs = []

    for pair_key, data in pairwise_results.items():
        parts = pair_key.split(",")
        i = key_to_idx.get(parts[0])
        j = key_to_idx.get(parts[1])

        if i is not None and j is not None:
            value = data[metric]
            status = data.get("status", "correct")

            # Track excluded pairs
            if status == "excluded":
                excluded_pairs.append((parts[0], parts[1]))

            # Set value in matrix (NaN will remain for excluded pairs)
            matrix[i, j] = value
            matrix[j, i] = value

    # Check for truly missing pairs (not from excluded status)
    missing_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(matrix[i, j]):
                pair_key = f"{inchi_keys[i]},{inchi_keys[j]}"
                alt_pair_key = f"{inchi_keys[j]},{inchi_keys[i]}"

                # Check if this pair exists in results
                if pair_key not in pairwise_results and alt_pair_key not in pairwise_results:
                    missing_pairs.append((inchi_keys[i], inchi_keys[j]))

    if missing_pairs and warn_missing:
        console = Console()
        console.print(
            f"[bold yellow]⚠ Warning: Found {len(missing_pairs)} truly missing pairs![/bold yellow]"
        )
        if len(missing_pairs) <= 10:
            for pair in missing_pairs:
                console.print(f"[dim]    - {pair[0]}, {pair[1]}[/dim]")
        else:
            for pair in missing_pairs[:5]:
                console.print(f"[dim]    - {pair[0]}, {pair[1]}[/dim]")
            console.print(f"[dim]    ... and {len(missing_pairs) - 5} more[/dim]")

    # Only replace truly missing pairs with 0, keep excluded as NaN
    if not preserve_nan:
        matrix = np.nan_to_num(matrix, nan=0.0)

    return matrix, missing_pairs, excluded_pairs


# ============================================================================
# Main Analysis Function
# ============================================================================

def _embeddings_are_stackable(embedding_list: list) -> bool:
    """
    Return True if all embeddings have the same shape so they can be stacked
    into a single (n_samples, n_features) array. Variable-shape embeddings
    (e.g. per-atom (n_atoms, dim)) return False.
    """
    if not embedding_list:
        return True
    arr0 = np.asarray(embedding_list[0])
    if arr0.ndim != 1:
        return False
    shape0 = arr0.shape
    for i in range(1, len(embedding_list)):
        arr = np.asarray(embedding_list[i])
        if arr.ndim != 1 or arr.shape != shape0:
            return False
    return True


def _preprocess_embeddings(
    embeddings: np.ndarray,
    center: bool = False,
    large_coords_to_trim: int | None = None,
    console: Console | None = None,
) -> np.ndarray:
    """
    Preprocess embeddings by trimming large coordinates and/or centering.

    Args:
        embeddings: Array of embeddings of shape (n_samples, n_features).
        center: Whether to center embeddings by subtracting the mean.
        large_coords_to_trim: If specified, set n coordinates with largest mean to 0.
        console: Console for printing progress messages.

    Returns:
        Preprocessed embeddings array.
    """
    # Calculate mean on each coordinate
    mean_per_coordinate = np.mean(embeddings, axis=0)

    # If large_coords_to_trim is specified, set n coordinates with largest mean to 0
    if large_coords_to_trim is not None and large_coords_to_trim > 0:
        n_coords = embeddings.shape[1]
        if large_coords_to_trim < n_coords:
            # Get indices sorted by mean (descending)
            sorted_indices = np.argsort(mean_per_coordinate)[::-1]
            # Select the top n coordinates to set to 0
            coords_to_zero = sorted_indices[:large_coords_to_trim]
            # Set those coordinates to 0 for all embeddings
            embeddings[:, coords_to_zero] = 0
            if console:
                console.print(f"  [dim]  Set {large_coords_to_trim} coordinates with largest mean to 0[/dim]")
            # Recalculate mean after setting coordinates to 0
            mean_per_coordinate = np.mean(embeddings, axis=0)

    # Center embeddings if requested (subtract mean from all vectors)
    if center:
        embeddings = embeddings - mean_per_coordinate
        if console:
            console.print("  [dim]  Centered embeddings (subtracted mean)[/dim]")

    return embeddings


def _scale_distances(
    raw_distances: np.ndarray,
    pairwise_results: dict[str, dict[str, float]],
    scaling_function: str,
    scaling_cutoff_percentile: int,
    console: Console | None = None,
) -> tuple[np.ndarray | None, dict, dict, float | None, dict]:
    """
    Scale raw distances and update pairwise results.

    Args:
        raw_distances: Array of raw distance values.
        pairwise_results: Dictionary of pairwise results to update.
        scaling_function: Name of the scaling function to use.
        scaling_cutoff_percentile: Percentile to use for clipping distances.
        console: Console for printing progress messages.

    Returns:
        Tuple of:
        - Scaled distances array (or None if no scaling)
        - Scaled statistics dictionary
        - Scaled Gaussian fit dictionary (MLE Gaussian; for backward compatibility)
        - Cutoff value (or None)
        - BIC comparison of Gaussian, Beta, Gamma, log-normal, Uniform (see ``fit_scaled_distances_bic_comparison``)
    """
    if scaling_cutoff_percentile is None or len(raw_distances) == 0:
        # No scaling - set distance_scaled to same as distance_raw and calculate similarity
        for pair_data in pairwise_results.values():
            pair_data["distance_scaled"] = pair_data["distance_raw"]
            if not np.isnan(pair_data["distance_raw"]):
                pair_data["similarity"] = float(1.0 - pair_data["distance_raw"])
            else:
                pair_data["similarity"] = np.nan
        if console:
            console.print("[dim]  No scaling applied (scaling_cutoff_percentile not given)[/dim]")
        return None, {}, {}, None, {}

    # Calculate cutoff value from raw distances
    cutoff_value = float(np.percentile(raw_distances, scaling_cutoff_percentile))
    if console:
        console.print(f"  [dim]  Scaling cutoff (percentile {scaling_cutoff_percentile}): {cutoff_value:.6f}[/dim]")

    # Get scaling function
    scaler = get_scaling_function(scaling_function)

    # Scale distances: clip to cutoff, then scale, then set everything above cutoff to 1
    clipped_distances = np.clip(raw_distances, None, cutoff_value)
    scaled_clipped = scaler(clipped_distances)

    # Create scaled distances array (same length as raw_distances)
    scaled_distances = scaled_clipped.copy()
    # Everything that was above cutoff gets value 1
    above_cutoff_mask = raw_distances > cutoff_value
    scaled_distances[above_cutoff_mask] = 1.0

    # Update pairwise_results with scaled distances and similarity
    valid_idx = 0
    for pair_data in pairwise_results.values():
        if pair_data.get("status", "correct") != "excluded":
            if not np.isnan(pair_data["distance_raw"]):
                pair_data["distance_scaled"] = float(scaled_distances[valid_idx])
                pair_data["similarity"] = float(1.0 - scaled_distances[valid_idx])
                valid_idx += 1
            else:
                pair_data["distance_scaled"] = np.nan
                pair_data["similarity"] = np.nan
        else:
            pair_data["distance_scaled"] = np.nan
            pair_data["similarity"] = np.nan

    # Calculate statistics of scaled distances
    scaled_stats = {
        "distance_scaled_min": float(np.min(scaled_distances)),
        "distance_scaled_max": float(np.max(scaled_distances)),
        "distance_scaled_mean": float(np.mean(scaled_distances)),
        "distance_scaled_median": float(np.median(scaled_distances)),
        "distance_scaled_std": float(np.std(scaled_distances)),
        "distance_scaled_q25": float(np.percentile(scaled_distances, 25)),
        "distance_scaled_q75": float(np.percentile(scaled_distances, 75)),
        "distance_scaled_q90": float(np.percentile(scaled_distances, 90)),
        "distance_scaled_q99": float(np.percentile(scaled_distances, 99)),
    }
    scaled_bic_comparison: dict = {}
    if len(scaled_distances) > 0:
        scaled_bic_comparison = fit_scaled_distances_bic_comparison(scaled_distances)
        gfit = scaled_bic_comparison.get("fits", {}).get("gaussian") or {}
        if gfit.get("success") and gfit.get("parameters"):
            p = gfit["parameters"]
            scaled_gaussian_fit = {
                "mu": p["mu"],
                "sigma": p["sigma"],
                "n_values": gfit.get("n_samples", len(scaled_distances)),
                "log_likelihood": gfit["log_likelihood"],
                "aic": gfit["aic"],
                "bic": gfit["bic"],
            }
        else:
            scaled_gaussian_fit = fit_normal(scaled_distances)
        if console and scaled_bic_comparison.get("best_by_bic"):
            console.print(
                f"  [dim]  Best scaled-distance fit (lowest BIC): "
                f"{scaled_bic_comparison['best_by_bic']} (BIC={scaled_bic_comparison.get('best_bic'):.4g})[/dim]"
            )
    else:
        scaled_gaussian_fit = {}
    if console:
        console.print("[green]  ✓ Distances scaled[/green]")

    return scaled_distances, scaled_stats, scaled_gaussian_fit, cutoff_value, scaled_bic_comparison


def create_pairwise_data(
    df: pd.DataFrame,
    results_dir: Path,
    distance_metric: str = "tanimoto",
    scaling_function: str = "linear",
    scaling_cutoff_percentile: int = 99,
    center: bool = False,
    large_coords_to_trim: int | None = None,
) -> dict[str, dict[str, float]]:
    """
    Create pairwise data dictionary and perform basic statistics.

    This function performs the following steps:
    Step 1 - Pairwise Data Creation:
        1a. Embeddings normalizing and shifting
            - Calculate mean on each coordinate
            - If large_coords_to_trim is specified, set n coordinates with largest mean to 0
            - If center is True, subtract the mean from all vectors
        1b. Calculate pairwise distances and statistics
            - Calculate pairwise distances (saved as 'distance_raw')
            - Calculate statistics of raw distances
            - If scaling_cutoff_percentile is given, scale distances with the given scaler
              (everything above the percentile threshold gets value 1)
            - Save scaled distances as 'distance_scaled'
            - Calculate similarity as 1 - distance_scaled
            - Calculate statistics of scaled distances
            - Fit Gaussian, Beta, Gamma, log-normal, and Uniform to scaled distances; compare by BIC (lowest wins); save ``scaled_distance_distribution_fits.json``
            - Plot histograms of raw and scaled distances (scaled panel: best BIC fit)
        1c. Build and plot distance matrices
            - Build structural distance matrix from 'distance_scaled'
            - Build activity distance matrix from 'log_ratio'
            - Plot both matrices as heatmaps in a two-panel plot

    All results are saved to a subdirectory within results_dir.

    Args:
        df: DataFrame with columns:
            - 'smiles': SMILES strings
            - 'inchi_key': InChI keys
            - 'embedding': Molecular embeddings (list/array)
            - 'activity': Activity values
            - 'log_activity': -log10(activity) values
            - 'censored': Boolean indicating if the activity is censored (optional)
        results_dir: Base directory for saving results.
        distance_metric: Metric for calculating structural distance.
            Options: 'tanimoto', 'dice', 'cosine', 'euclidean', etc.
        scaling_function: Function for scaling distances to [0, 1].
            Options: 'linear', 'sigmoid', 'exponential'.
        scaling_cutoff_percentile: Percentile to use for clipping distances before scaling.
            Default is 99.
        center: Whether to center embeddings by subtracting the mean. Default is False.
        large_coords_to_trim: If specified, set n coordinates with largest mean to 0.
            Default is None.

    Returns:
        Dictionary mapping "inchi_key_1,inchi_key_2" pairs to a dict containing:
        - 'distance_raw': Structural distance (0 = identical)
        - 'distance_scaled': Scaled structural distance
        - 'activity_difference': Absolute activity difference
        - 'log_ratio': Absolute log ratio difference
        - 'similarity': Structural similarity (scaled and inverted distance)
        - 'status': 'correct', 'censored', or 'excluded'
        - 'embedding1': First molecule's embedding
        - 'embedding2': Second molecule's embedding
    """
    console = Console()

    # Create analysis subdirectory
    analysis_dir = results_dir / "pairwise_data_creation"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Step 1 - Starting pairwise data creation...[/bold cyan]")

    # -------------------------------------------------------------------------
    # Step 1a: Embeddings normalizing and shifting
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 1a: Embeddings normalizing and shifting...[/cyan]")

    df = df.copy()  # Work on a copy to avoid modifying original

    embedding_list = df["embedding"].tolist()
    if _embeddings_are_stackable(embedding_list):
        # Fixed-shape vectors: stack and preprocess (center, trim)
        embeddings = np.array(embedding_list)
        embeddings = _preprocess_embeddings(
            embeddings, center=center, large_coords_to_trim=large_coords_to_trim, console=console
        )
        df["embedding"] = embeddings.tolist()
        console.print("[green]  ✓ Embeddings preprocessed[/green]")
    else:
        # Variable-shape (e.g. per-atom (n_atoms, dim)): skip stacking/preprocessing
        if center or large_coords_to_trim:
            console.print(
                "[yellow]  ⚠ Variable-shape embeddings detected; "
                "skipping center/trim (use metric='tensor' for distances).[/yellow]"
            )
        else:
            console.print(
                "[dim]  Variable-shape embeddings: skipping preprocessing "
                "(distances via metric='tensor').[/dim]"
            )

    # Check for censored column and create mask for uncensored compounds
    has_censored = "censored" in df.columns
    if has_censored:
        uncensored_mask = (df["censored"] != True).values
        n_censored = (~uncensored_mask).sum()
        n_uncensored = uncensored_mask.sum()
    else:
        uncensored_mask = np.ones(len(df), dtype=bool)
        n_censored = 0
        n_uncensored = len(df)

    # -------------------------------------------------------------------------
    # Step 1b: Calculate pairwise distances and statistics
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 1b: Calculating pairwise distances and statistics...[/cyan]")

    # Calculate pairwise distances
    pairwise_results = calculate_pairwise_distances(
        df, analysis_dir, distance_metric=distance_metric
    )
    console.print("[green]  ✓ Pairwise distances calculated[/green]")

    # Filter out excluded pairs for statistics
    valid_pairs = [
        v for v in pairwise_results.values()
        if v.get("status", "correct") != "excluded"
    ]
    raw_distances = np.array([v["distance_raw"] for v in valid_pairs])
    raw_distances = raw_distances[~np.isnan(raw_distances)]

    # Calculate statistics of raw distances
    raw_stats = calculate_distance_statistics(pairwise_results)
    raw_gaussian_fit = fit_normal(raw_distances) if len(raw_distances) > 0 else {}

    # Scale distances
    scaled_distances, scaled_stats, scaled_gaussian_fit, cutoff_value, scaled_bic_comparison = _scale_distances(
        raw_distances,
        pairwise_results,
        scaling_function,
        scaling_cutoff_percentile,
        console=console,
    )

    if scaled_bic_comparison:
        bic_path = analysis_dir / "scaled_distance_distribution_fits.json"
        with open(bic_path, "w") as f:
            json.dump(convert_to_native_types(scaled_bic_comparison), f, indent=2)
        console.print(f"[green]  ✓ Scaled distance BIC fits saved to {bic_path.name}[/green]")

    # Create plot with two panels: raw and scaled distances
    plot_distance_distributions(
        raw_distances=raw_distances,
        scaled_distances=scaled_distances,
        raw_gaussian_fit=raw_gaussian_fit,
        scaled_gaussian_fit=scaled_gaussian_fit,
        scaled_bic_comparison=scaled_bic_comparison,
        output_path=analysis_dir / "distance_statistics.png",
        title="Distance Distributions",
    )
    console.print("[green]  ✓ Distance statistics plot saved[/green]")

    # Save statistics to compound_statistics.json
    compound_distance_statistics = {
        "raw_distance": {
            **raw_stats,
            "gaussian_fit": raw_gaussian_fit,
        },
        "scaled_distance": {
            **scaled_stats,
            "gaussian_fit": scaled_gaussian_fit,
            "distribution_fits_bic": convert_to_native_types(scaled_bic_comparison)
            if scaled_bic_comparison
            else {},
        },
        "scaling": {
            "scaling_function": scaling_function,
            "scaling_cutoff_percentile": scaling_cutoff_percentile,
            "cutoff_value": float(cutoff_value) if cutoff_value is not None else None,
        },
    }

    with open(analysis_dir / "compound_distance_statistics.json", "w") as f:
        json.dump(compound_distance_statistics, f, indent=2)

    console.print("[green]  ✓ Distance statistics saved[/green]")

    # -------------------------------------------------------------------------
    # Step 1c: Build and plot distance matrices
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 1c: Building and plotting distance matrices...[/cyan]")

    inchi_keys = df["inchi_key"].tolist()

    # Build structural distance matrix from distance_scaled
    structural_distance_matrix, missing_pairs, excluded_pairs = build_distance_matrix(
        pairwise_results, inchi_keys, metric="distance_scaled", warn_missing=True, preserve_nan=True
    )

    # Save missing pairs if any
    if missing_pairs:
        missing_pairs_df = pd.DataFrame(missing_pairs, columns=["inchi_key1", "inchi_key2"])
        missing_pairs_df.to_csv(analysis_dir / "missing_pairs.csv", index=False)

    # Log excluded pairs count
    if excluded_pairs:
        console.print(f"  [yellow]Excluded {len(excluded_pairs)} pairs (both compounds censored)[/yellow]")

    # Build activity distance matrix from log_ratio values
    activity_distance_matrix, _, _ = build_distance_matrix(
        pairwise_results, inchi_keys, metric="log_ratio", warn_missing=False, preserve_nan=True
    )

    # Save distance matrices only
    np.save(analysis_dir / "structural_distance_matrix.npy", structural_distance_matrix)
    np.save(analysis_dir / "activity_distance_matrix.npy", activity_distance_matrix)
    # Row/column order of both matrices
    with open(analysis_dir / "matrix_inchi_keys.json", "w") as f:
        json.dump(inchi_keys, f)

    console.print("[green]  ✓ Distance matrices built[/green]")

    # Plot both distance matrices as heatmaps
    plot_distance_matrices(
        structural_distance_matrix=structural_distance_matrix,
        activity_distance_matrix=activity_distance_matrix,
        output_path=analysis_dir / "distance_matrices.png",
        title="Distance Matrices",
    )
    console.print("[green]  ✓ Distance matrices plotted[/green]")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    summary = {
        "n_compounds": int(len(df)),
        "n_compounds_uncensored": int(n_uncensored),
        "n_compounds_censored": int(n_censored),
        "n_pairs": int(raw_stats["n_pairs"]),
        "n_pairs_correct": int(raw_stats.get("n_pairs_correct", raw_stats["n_pairs"])),
        "n_pairs_censored": int(raw_stats.get("n_pairs_censored", 0)),
        "n_pairs_excluded": int(raw_stats.get("n_pairs_excluded", 0)),
        "n_missing_pairs": int(len(missing_pairs)),
        "n_excluded_pairs": int(len(excluded_pairs)),
        "distance_metric": distance_metric,
        "scaling_function": scaling_function,
        "results_dir": str(analysis_dir),
        "distance_statistics": convert_to_native_types(raw_stats),
        "scaled_distance_best_bic_distribution": scaled_bic_comparison.get("best_by_bic")
        if scaled_bic_comparison
        else None,
        "scaled_distance_distribution_fits_file": "scaled_distance_distribution_fits.json"
        if scaled_bic_comparison
        else None,
    }
    with open(analysis_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[bold green]✓ Pairwise data creation complete![/bold green]")
    console.print(f"  Results saved to: {analysis_dir}")
    console.print(f"  Compounds: {len(df)}")
    console.print(f"  Total pairs: {raw_stats['n_pairs']}")
    console.print(f"    - Correct: {raw_stats.get('n_pairs_correct', raw_stats['n_pairs']):,}")
    if raw_stats.get('n_pairs_censored', 0) > 0:
        console.print(f"    - Censored: {raw_stats['n_pairs_censored']:,}")
    if raw_stats.get('n_pairs_excluded', 0) > 0:
        console.print(f"    - [yellow]Excluded: {raw_stats['n_pairs_excluded']:,}[/yellow]")
    if missing_pairs:
        console.print(f"  [yellow]Missing pairs: {len(missing_pairs)}[/yellow]")
    console.print(f"  Distance metric: {distance_metric}")
    if scaling_cutoff_percentile is not None:
        console.print(f"  Scaling function: {scaling_function} (cutoff: {scaling_cutoff_percentile}th percentile)")
    else:
        console.print(f"  Scaling function: None (no scaling applied)")

    return pairwise_results
