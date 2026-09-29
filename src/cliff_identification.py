"""
Activity cliff identification module.

This module provides functions for identifying activity cliffs in molecular datasets,
including calculating cliff fractions, fitting sigmoid curves, and analyzing cliff
distributions across similarity rankings.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from scipy.stats import skew as scipy_skew

from src.fitting import (
    fit_power_law_cumulative_cliff_fraction,
    fit_sigmoid,
    gini_from_power_law_exponent,
)
from src.plotting import (
    plot_activity_cliff_fractions,
    plot_cliff_fraction_vs_pair_rank,
    plot_cumulative_cliff_fraction_Fn,
)
from src.utils import convert_to_native_types

try:
    from diptest import diptest as hartigan_diptest
except ImportError:
    hartigan_diptest = None  # type: ignore[misc, assignment]

# ============================================================================
# Helper Functions
# ============================================================================

def _is_cliff(data: dict[str, float], activity_threshold: float) -> bool:
    """
    Check if a pair is an activity cliff.

    Args:
        data: Dictionary containing pair data with 'log_ratio' key.
        activity_threshold: Threshold for log_ratio to consider a pair a cliff.

    Returns:
        True if the pair is a cliff, False otherwise.
    """
    log_ratio = data.get("log_ratio", 0)
    return not np.isnan(log_ratio) and log_ratio >= activity_threshold


def _filter_valid_pairs(
    pairwise_data: dict[str, dict[str, float]]
) -> list[tuple[str, dict[str, float]]]:
    """
    Filter out excluded pairs from pairwise data.

    Args:
        pairwise_data: Dictionary of pairwise data.

    Returns:
        List of (pair_key, data) tuples for valid (non-excluded) pairs.
    """
    return [
        (key, data) for key, data in pairwise_data.items()
        if data.get("status", "correct") != "excluded"
    ]


def _pair_key_to_inchi_keys(pair_key: str) -> list[str]:
    """
    Convert a pair key to a list of InChI keys.

    Args:
        pair_key: Pair key in format "inchi_key1,inchi_key2".

    Returns:
        List of two InChI keys [inchi_key1, inchi_key2].
    """
    return pair_key.split(",")


def _build_plot_title(
    base_title: str,
    dataset_name: str | None = None,
    embedding_method: str | None = None,
    distance_metric: str | None = None,
) -> str:
    """
    Build a plot title with optional dataset and method information.

    Args:
        base_title: Base title string.
        dataset_name: Optional dataset name.
        embedding_method: Optional embedding method.
        distance_metric: Optional distance metric.

    Returns:
        Formatted title string.
    """
    title = base_title
    if dataset_name:
        title = f"{title} - {dataset_name}"
    if embedding_method and distance_metric:
        title = f"{title} ({embedding_method}, {distance_metric})"
    return title


# ============================================================================
# Main Functions
# ============================================================================

def calculate_activity_cliff_fractions(
    pairwise_data: dict[str, dict[str, float]],
    activity_threshold: float,
) -> tuple[pd.DataFrame, dict[float, list[str]], int]:
    """
    Calculate the fraction of activity cliffs in the dataset as a function of similarity.

    Activity cliffs are pairs where log_ratio >= activity_threshold.
    Pairs with status='excluded' are filtered out from the analysis.

    Args:
        pairwise_data: Dictionary mapping "inchi_key_1,inchi_key_2" pairs to a dict
            containing 'similarity', 'log_ratio', 'distance', 'status', etc.
        activity_threshold: Threshold for log_ratio to consider a pair an activity cliff.

    Returns:
        Tuple of:
        - DataFrame with columns:
            - 'compound_similarity_threshold': similarity threshold
            - 'activity_cliff_fraction': fraction of activity cliffs in the whole dataset
            - 'n_pairs': number of pairs above the similarity threshold
        - Dictionary mapping similarity thresholds to lists of pair keys that are cliffs.
        - Number of excluded pairs.
    """
    similarity_thresholds = [round(x * 0.05, 2) for x in range(21)]

    # Filter out excluded pairs
    valid_pairwise_data = {
        key: data for key, data in pairwise_data.items()
        if data.get("status", "correct") != "excluded"
    }

    n_excluded = len(pairwise_data) - len(valid_pairwise_data)
    total_pairs = len(valid_pairwise_data)

    rows = []
    activity_cliff_pairs = {}

    for sim_threshold in similarity_thresholds:
        # Filter pairs with similarity >= threshold
        pairs_above_threshold = [
            (key, data) for key, data in valid_pairwise_data.items()
            if not np.isnan(data.get("similarity", 0)) and data["similarity"] >= sim_threshold
        ]

        n_pairs = len(pairs_above_threshold)

        # Identify activity cliffs
        activity_cliff_pairs[sim_threshold] = [
            key for key, data in pairs_above_threshold
            if _is_cliff(data, activity_threshold)
        ]
        n_cliffs = len(activity_cliff_pairs[sim_threshold])

        # Fraction of activity cliffs relative to valid pairs
        fraction = n_cliffs / total_pairs if total_pairs > 0 else 0.0

        rows.append({
            "compound_similarity_threshold": sim_threshold,
            "activity_cliff_fraction": fraction,
            "n_pairs": n_pairs,
        })

    return pd.DataFrame(rows), activity_cliff_pairs, n_excluded


def _calculate_cliff_fraction_vs_rank(
    pairwise_data: dict[str, dict[str, float]],
    activity_threshold: float,
    percent_thresholds: list[int],
) -> tuple[pd.DataFrame, dict[int, list[list[str]]]]:
    """
    Calculate cliff fraction as a function of pair rank percentage.

    Args:
        pairwise_data: Dictionary of pairwise data.
        activity_threshold: Threshold for log_ratio to consider a pair a cliff.
        percent_thresholds: List of percentage thresholds to evaluate.

    Returns:
        Tuple of:
        - DataFrame with columns: 'percent_rank', 'n_pairs', 'n_cliffs', 'cliff_fraction'
        - Dictionary mapping percent thresholds (up to 50%) to lists of InChI key pairs.
    """
    # Filter and sort pairs by similarity (descending - most similar first)
    valid_pairs = _filter_valid_pairs(pairwise_data)
    valid_pairs_sorted = sorted(
        valid_pairs,
        key=lambda x: x[1].get("similarity", 0.0),
        reverse=True
    )

    total_valid_pairs = len(valid_pairs_sorted)
    rows = []
    cliff_sets_by_percent = {}

    for n_percent in percent_thresholds:
        # Calculate number of pairs in top n%
        n_pairs_to_take = min(
            int(np.ceil(total_valid_pairs * n_percent / 100)),
            total_valid_pairs
        )

        # Take top n% of pairs
        top_pairs = valid_pairs_sorted[:n_pairs_to_take]

        # Identify cliffs in this set
        cliff_pairs = [
            (key, data) for key, data in top_pairs
            if _is_cliff(data, activity_threshold)
        ]

        n_cliffs = len(cliff_pairs)
        fraction = n_cliffs / n_pairs_to_take if n_pairs_to_take > 0 else 0.0

        rows.append({
            "percent_rank": n_percent,
            "n_pairs": n_pairs_to_take,
            "n_cliffs": n_cliffs,
            "cliff_fraction": fraction,
        })

        # Store cliff sets as pairs of inchi keys for thresholds up to 50%
        if n_percent <= 50:
            cliff_inchi_pairs = [
                _pair_key_to_inchi_keys(pair_key)
                for pair_key, _ in cliff_pairs
            ]
            cliff_sets_by_percent[n_percent] = cliff_inchi_pairs

    return pd.DataFrame(rows), cliff_sets_by_percent


def _ordered_inchi_keys_from_pairwise(pairwise_data: dict[str, dict]) -> list[str]:
    """Collect all InChI keys appearing in pair keys and return a stable sorted order."""
    keys: set[str] = set()
    for pair_key in pairwise_data:
        parts = pair_key.split(",", 1)
        if len(parts) == 2:
            keys.add(parts[0].strip())
            keys.add(parts[1].strip())
    return sorted(keys)


def _upper_triangle_finite_distances(D: np.ndarray) -> np.ndarray:
    """All finite pairwise distances i<j from a symmetric distance matrix."""
    n = D.shape[0]
    i, j = np.triu_indices(n, k=1)
    vals = D[i, j]
    return vals[np.isfinite(vals)]


def compute_distance_geometry_statistics(
    pairwise_data: dict[str, dict[str, float]],
    *,
    metric: str = "distance_scaled",
    k_neighbors: int = 5,
    dip_max_samples: int = 100_000,
    dip_random_seed: int = 42,
) -> dict:
    """
    Distance-space geometry metrics (structural distances between compounds).

    1d) Coefficient of variation of distances: std/mean (and mean/std for reference).
    1e) Hartigan's dip test for unimodality on the distribution of pairwise distances.
    1f) Relative contrast per molecule: (d95 - d5) / d5 over distances to all others;
        summary: median, 25th and 75th percentile across molecules.
    1g) Hubness: for each molecule, k nearest neighbors by ``metric``; N5(j) counts
        how often j appears in someone's k-NN set; skewness, hub and anti-hub fractions.
    """
    from src.pairwise_data_preparation import build_distance_matrix

    inchi_keys = _ordered_inchi_keys_from_pairwise(pairwise_data)
    n = len(inchi_keys)
    if n < 2:
        return {
            "error": "need_at_least_two_compounds",
            "n_compounds": n,
            "metric": metric,
        }

    D, missing_pairs, _excluded = build_distance_matrix(
        pairwise_data,
        inchi_keys,
        metric=metric,
        warn_missing=False,
        preserve_nan=True,
    )

    dist_vec = _upper_triangle_finite_distances(D)
    if dist_vec.size == 0:
        return {
            "error": "no_finite_pairwise_distances",
            "n_compounds": n,
            "metric": metric,
        }

    mean_d = float(np.mean(dist_vec))
    std_d = float(np.std(dist_vec, ddof=0))
    cv = float(std_d / mean_d) if mean_d != 0 else None
    mean_over_std = float(mean_d / std_d) if std_d != 0 else None

    coefficient_of_variation = {
        "mean": mean_d,
        "std": std_d,
        "coefficient_of_variation_std_over_mean": cv,
        "mean_over_std": mean_over_std,
        "n_pairwise_distances": int(dist_vec.size),
    }

    # Dip test (subsample very large arrays for runtime)
    dip_result: dict = {"available": False}
    if hartigan_diptest is not None:
        rng = np.random.default_rng(dip_random_seed)
        x_dip = dist_vec
        if x_dip.size > dip_max_samples:
            x_dip = rng.choice(x_dip, size=dip_max_samples, replace=False)
        try:
            dip_stat, p_value = hartigan_diptest(x_dip.astype(float))
            dip_result = {
                "available": True,
                "dip_statistic": float(dip_stat),
                "p_value": float(p_value),
                "n_samples_used": int(x_dip.size),
                "subsampled": bool(dist_vec.size > dip_max_samples),
            }
        except Exception as e:
            dip_result = {"available": False, "error": str(e)}
    else:
        dip_result = {
            "available": False,
            "error": "diptest package not installed (pip install diptest)",
        }

    # Relative contrast (d95 - d5) / d5 per molecule
    rc_list: list[float] = []
    for i in range(n):
        row = D[i, :].astype(float)
        row[i] = np.nan
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        d_other = row[mask]
        d5, d95 = np.percentile(d_other, [5, 95])
        if d5 > 0 and np.isfinite(d5) and np.isfinite(d95):
            rc_list.append(float((d95 - d5) / d5))
    rc_arr = np.array(rc_list, dtype=float) if rc_list else np.array([], dtype=float)
    relative_contrast = {
        "per_molecule_values": rc_arr.tolist(),
        "median": float(np.median(rc_arr)) if rc_arr.size else None,
        "percentile_25": float(np.percentile(rc_arr, 25)) if rc_arr.size else None,
        "percentile_75": float(np.percentile(rc_arr, 75)) if rc_arr.size else None,
        "n_molecules": int(rc_arr.size),
    }

    # Hubness N5: count appearances in k-NN sets
    N5 = np.zeros(n, dtype=np.int64)
    k_use = min(k_neighbors, max(0, n - 1))
    for i in range(n):
        row = D[i, :].astype(float)
        row[i] = np.nan
        mask = np.isfinite(row)
        if not np.any(mask):
            continue
        indices = np.where(mask)[0]
        distances = row[indices]
        order = np.argsort(distances, kind="mergesort")
        take = min(k_use, len(order))
        if take == 0:
            continue
        chosen = indices[order[:take]]
        for j in chosen:
            N5[int(j)] += 1

    n5_float = N5.astype(float)
    hubness = {
        "k_neighbors": k_use,
        "N5_per_molecule": {inchi_keys[i]: int(N5[i]) for i in range(n)},
        "skewness": float(scipy_skew(n5_float, bias=False)) if n > 2 else None,
        "hub_fraction_N5_gt_10": float(np.mean(N5 > 10)),
        "anti_hub_fraction_N5_eq_0": float(np.mean(N5 == 0)),
        "n_compounds": n,
    }

    return {
        "metric": metric,
        "n_compounds": n,
        "n_missing_pairs": len(missing_pairs),
        "coefficient_of_variation": coefficient_of_variation,
        "hartigan_dip_test": dip_result,
        "relative_contrast": relative_contrast,
        "hubness": hubness,
    }


def identify_cliffs(
    pairwise_data: dict[str, dict[str, float]],
    activity_threshold: float,
    results_dir: Path,
    dataset_name: str | None = None,
    embedding_method: str | None = None,
    distance_metric: str | None = None,
) -> dict[float, list[str]]:
    """
    Identify activity cliffs in the dataset.

    Activity cliffs are pairs of molecules that are structurally similar
    (high similarity) but have significantly different activities (high log_ratio).

    Steps:
        2a: Cliff identification and fraction calculation
        2b: Fitting sigmoid and plotting
        2c: Cliff fraction vs pair rank (top n% most similar pairs)
        2d: Coefficient of variation of structural distances (std/mean; also mean/std)
        2e: Hartigan dip test for unimodality of pairwise distances
        2f: Relative contrast per molecule (d95−d5)/d5; summary quartiles across molecules
        2g: Hubness (k=5 NN): N5 counts, skewness, hub / anti-hub fractions
        2h: Cumulative cliff fraction F(n) = (cliffs in top n% most similar) / (total cliff pairs);
            fit F(n) ≈ a·n^s; report s, R²; Gini = (s−1)/(s+1)

    Args:
        pairwise_data: Dictionary mapping "inchi_key_1,inchi_key_2" pairs to a dict
            containing 'similarity', 'log_ratio', 'distance', etc.
        activity_threshold: Threshold for log_ratio to consider a pair an activity cliff.
        results_dir: Base directory for saving results.
        dataset_name: Name of the dataset for plot titles.
        embedding_method: Embedding method used for plot titles.
        distance_metric: Distance metric used for plot titles.

    Returns:
        Dictionary mapping similarity thresholds to lists of pair keys that are cliffs.
    """
    console = Console()

    # Create cliff_identification subdirectory
    cliff_dir = results_dir / "cliff_identification"
    cliff_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Step 2 - Identifying activity cliffs...[/bold cyan]")

    # -------------------------------------------------------------------------
    # Step 2a: Cliff identification and fraction calculation
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 2a: Calculating cliff fractions...[/cyan]")

    cliff_fractions_df, cliff_sets, n_excluded = calculate_activity_cliff_fractions(
        pairwise_data=pairwise_data,
        activity_threshold=activity_threshold
    )
    cliff_fractions_df.to_csv(cliff_dir / "activity_cliff_fractions.csv", index=False)

    n_valid_pairs = len(pairwise_data) - n_excluded
    console.print(f"[green]  ✓ Cliff fractions calculated[/green]")
    console.print(f"  Valid pairs: {n_valid_pairs:,} (excluded: {n_excluded:,})")
    console.print(f"  Cliffs at similarity >= 0.5: {len(cliff_sets.get(0.5, []))}")
    console.print(f"  Cliffs at similarity >= 0.7: {len(cliff_sets.get(0.7, []))}")
    console.print(f"  Cliffs at similarity >= 0.9: {len(cliff_sets.get(0.9, []))}")

    # -------------------------------------------------------------------------
    # Step 2b: Fitting sigmoid and plotting
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 2b: Fitting sigmoid and plotting...[/cyan]")

    # Fit sigmoid to both curves
    x = cliff_fractions_df["compound_similarity_threshold"].values
    y_fraction = cliff_fractions_df["activity_cliff_fraction"].values
    y_pairs = cliff_fractions_df["n_pairs"].values

    fit_params = {
        "activity_cliff_fraction": fit_sigmoid(x, y_fraction),
        "n_pairs": fit_sigmoid(x, y_pairs),
    }

    # Plot and save activity cliff fractions
    title = _build_plot_title(
        "Activity Cliff Fraction",
        dataset_name=dataset_name,
        embedding_method=embedding_method,
        distance_metric=distance_metric,
    )

    plot_activity_cliff_fractions(
        cliff_fractions_df,
        output_path=cliff_dir / "activity_cliff_fractions.png",
        title=title,
    )

    console.print("[green]  ✓ Sigmoid fitted and plot saved[/green]")

    # -------------------------------------------------------------------------
    # Step 2c: Cliff fraction vs pair rank
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 2c: Calculating cliff fraction vs pair rank...[/cyan]")

    # Define percentage thresholds (dense for small values, sparse for large)
    percent_thresholds = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

    rank_df, cliff_sets_by_percent = _calculate_cliff_fraction_vs_rank(
        pairwise_data=pairwise_data,
        activity_threshold=activity_threshold,
        percent_thresholds=percent_thresholds,
    )

    rank_df.to_csv(cliff_dir / "cliff_vs_pair_rank.csv", index=False)

    # Save cliff sets as pairs of inchi keys for percent thresholds up to 50%
    cliff_sets_serializable = {str(k): v for k, v in cliff_sets_by_percent.items()}
    with open(cliff_dir / "activity_cliff_sets.json", "w") as f:
        json.dump(cliff_sets_serializable, f, indent=2)

    # Fit sigmoid to cliff fraction vs percent rank
    x_rank = rank_df["percent_rank"].values
    y_cliff_fraction = rank_df["cliff_fraction"].values
    rank_sigmoid_fit = fit_sigmoid(x_rank, y_cliff_fraction, decreasing=False, max_x=100)

    # Plot cliff fraction vs pair rank percentage with sigmoid fit
    plot_cliff_fraction_vs_pair_rank(
        df=rank_df,
        sigmoid_fit=rank_sigmoid_fit,
        output_path=cliff_dir / "cliff_vs_pair_rank.png",
        title="Cliff Fraction vs Pair Rank",
        dataset_name=dataset_name,
        embedding_method=embedding_method,
        distance_metric=distance_metric,
    )

    console.print("[green]  ✓ Cliff fraction vs pair rank calculated and plotted[/green]")
    console.print(f"  Top 1%: {rank_df[rank_df['percent_rank'] == 1]['cliff_fraction'].values[0]:.4f}")
    console.print(f"  Top 10%: {rank_df[rank_df['percent_rank'] == 10]['cliff_fraction'].values[0]:.4f}")
    console.print(f"  Top 50%: {rank_df[rank_df['percent_rank'] == 50]['cliff_fraction'].values[0]:.4f}")

    # -------------------------------------------------------------------------
    # Step 2c (continued): Cumulative cliff fraction F(n), power-law fit, Gini
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 2c (continued): Cumulative F(n), fit a·n^s, Gini...[/cyan]")
    total_cliff_pairs = sum(
        1
        for _, d in _filter_valid_pairs(pairwise_data)
        if _is_cliff(d, activity_threshold)
    )
    fn_df = rank_df.copy()
    if total_cliff_pairs > 0:
        fn_df["F_n"] = fn_df["n_cliffs"].astype(float) / float(total_cliff_pairs)
    else:
        fn_df["F_n"] = 0.0
    fn_df.to_csv(cliff_dir / "cumulative_cliff_fraction_Fn.csv", index=False)

    pl_fit = fit_power_law_cumulative_cliff_fraction(
        fn_df["percent_rank"].values,
        fn_df["F_n"].values,
    )
    gini_from_s = gini_from_power_law_exponent(pl_fit["s"]) if pl_fit.get("success") and pl_fit.get("s") is not None else None

    cumulative_analysis = {
        "total_cliff_pairs": int(total_cliff_pairs),
        "description": "F(n) = fraction of all cliff pairs that lie in the top n% most similar pairs",
        "F_n_series": fn_df[["percent_rank", "n_cliffs", "F_n"]].to_dict(orient="records"),
        "power_law_fit": {
            "model": "F(n) ≈ a · n^s",
            "a": pl_fit.get("a"),
            "s": pl_fit.get("s"),
            "r_squared": pl_fit.get("r_squared"),
            "success": pl_fit.get("success"),
            "error": pl_fit.get("error"),
            "n_points_fitted": pl_fit.get("n_points_fitted"),
        },
        "gini_coefficient": gini_from_s,
        "gini_formula": "(s - 1) / (s + 1) from fitted exponent s",
    }
    with open(cliff_dir / "cumulative_cliff_fraction_analysis.json", "w") as f:
        json.dump(convert_to_native_types(cumulative_analysis), f, indent=2)

    title_fn = _build_plot_title(
        "Cumulative cliff fraction F(n)",
        dataset_name=dataset_name,
        embedding_method=embedding_method,
        distance_metric=distance_metric,
    )
    plot_cumulative_cliff_fraction_Fn(
        fn_df,
        pl_fit,
        output_path=cliff_dir / "cumulative_cliff_fraction_Fn.png",
        title=title_fn,
        dataset_name=dataset_name,
        embedding_method=embedding_method,
        distance_metric=distance_metric,
    )
    console.print("[green]  ✓ cumulative_cliff_fraction_Fn.csv / .png / .json saved[/green]")
    if pl_fit.get("success"):
        console.print(
            f"  Power-law: s={pl_fit.get('s'):.6g}, R²={pl_fit.get('r_squared'):.6g}, "
            f"Gini={(gini_from_s if gini_from_s is not None else float('nan')):.6g}"
        )
    fit_params["cumulative_Fn_power_law"] = {
        "a": pl_fit.get("a"),
        "s": pl_fit.get("s"),
        "r_squared": pl_fit.get("r_squared"),
        "gini_coefficient": gini_from_s,
        "success": pl_fit.get("success"),
        "total_cliff_pairs": int(total_cliff_pairs),
    }

    # -------------------------------------------------------------------------
    # Steps 2d–2g: Distance geometry (CV, dip test, relative contrast, hubness)
    # -------------------------------------------------------------------------
    console.print("[cyan]Steps 2d–2g: Distance geometry statistics (CV, dip test, relative contrast, hubness)...[/cyan]")
    distance_geometry = compute_distance_geometry_statistics(
        pairwise_data,
        metric="distance_scaled",
        k_neighbors=5,
    )
    with open(cliff_dir / "distance_geometry_statistics.json", "w") as f:
        json.dump(distance_geometry, f, indent=2)
    console.print("[green]  ✓ distance_geometry_statistics.json saved[/green]")
    if distance_geometry.get("coefficient_of_variation"):
        cv = distance_geometry["coefficient_of_variation"]
        console.print(f"  CV (std/mean): {cv.get('coefficient_of_variation_std_over_mean', 'N/A')}")
    dip_info = distance_geometry.get("hartigan_dip_test") or {}
    if dip_info.get("available"):
        console.print(
            f"  Dip test: D={dip_info.get('dip_statistic', float('nan')):.6g}, "
            f"p={dip_info.get('p_value', float('nan')):.6g}"
        )
    elif dip_info.get("error"):
        console.print(f"  [yellow]Dip test skipped: {dip_info.get('error')}[/yellow]")
    rc = distance_geometry.get("relative_contrast") or {}
    if rc.get("median") is not None:
        console.print(
            f"  Relative contrast median / Q25 / Q75: {rc.get('median'):.6g} / "
            f"{rc.get('percentile_25'):.6g} / {rc.get('percentile_75'):.6g}"
        )
    hub = distance_geometry.get("hubness") or {}
    if hub.get("skewness") is not None:
        console.print(
            f"  Hubness: skew(N5)={hub.get('skewness'):.6g}, "
            f"hub_frac={hub.get('hub_fraction_N5_gt_10'):.6g}, "
            f"anti_hub_frac={hub.get('anti_hub_fraction_N5_eq_0'):.6g}"
        )

    # -------------------------------------------------------------------------
    # Save summary and fit parameters
    # -------------------------------------------------------------------------
    # Update fit_parameters to include both step 2b and step 2c fits
    fit_params["cliff_fraction_vs_pair_rank"] = rank_sigmoid_fit

    # Save updated fit parameters
    with open(cliff_dir / "fit_parameters.json", "w") as f:
        json.dump(fit_params, f, indent=2)

    def _compact_distance_geometry_for_summary(geom: dict) -> dict:
        """Omit large per-molecule arrays from summary.json (full data in distance_geometry_statistics.json)."""
        if not geom or geom.get("error"):
            return geom
        out = {k: v for k, v in geom.items() if k not in ("hubness", "relative_contrast")}
        rc = geom.get("relative_contrast") or {}
        out["relative_contrast"] = {
            k: v for k, v in rc.items() if k != "per_molecule_values"
        }
        hub = geom.get("hubness") or {}
        out["hubness"] = {k: v for k, v in hub.items() if k != "N5_per_molecule"}
        return out

    summary = {
        "activity_threshold": activity_threshold,
        "total_pairs": len(pairwise_data),
        "excluded_pairs": n_excluded,
        "valid_pairs": n_valid_pairs,
        "cliffs_at_similarity_0.0": len(cliff_sets.get(0.0, [])),
        "cliffs_at_similarity_0.5": len(cliff_sets.get(0.5, [])),
        "cliffs_at_similarity_0.7": len(cliff_sets.get(0.7, [])),
        "cliffs_at_similarity_0.9": len(cliff_sets.get(0.9, [])),
        "fit_parameters": fit_params,
        "distance_geometry_statistics_file": "distance_geometry_statistics.json",
        "distance_geometry": _compact_distance_geometry_for_summary(distance_geometry),
        "cumulative_cliff_fraction_analysis_file": "cumulative_cliff_fraction_analysis.json",
        "total_cliff_pairs": int(total_cliff_pairs),
        "power_law_exponent_s_Fn": pl_fit.get("s") if pl_fit.get("success") else None,
        "power_law_r_squared_Fn": pl_fit.get("r_squared") if pl_fit.get("success") else None,
        "gini_coefficient_Fn": gini_from_s,
    }
    with open(cliff_dir / "summary.json", "w") as f:
        json.dump(convert_to_native_types(summary), f, indent=2)

    console.print("[green]  ✓ Activity cliff identification complete[/green]")
    console.print(f"  Results saved to: {cliff_dir}")
    console.print(f"  Activity threshold: {activity_threshold}")

    return cliff_sets
