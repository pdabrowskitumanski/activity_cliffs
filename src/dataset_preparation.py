"""
Dataset preparation module for activity cliffs analysis.

This module provides functions for preparing molecular datasets by calculating
embeddings and saving them as pickle files for later analysis.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from umap import UMAP

from src.datasets import UNIT_SHIFTS, add_log_activity_and_censoring, resolve_dataset
from src.embeddings import EMBEDDING_METHODS, save_all_embeddings
from src.fitting import fit_normal_uniform_mixture, fit_kww_decay, fit_normal
from src.plotting import (
    plot_activity_statistics,
    plot_embedding_center_statistics_multi,
    plot_log_ratio_statistics,
    plot_umap_projections_multi,
    plot_vector_length_statistics_multi,
)
from src.printing import create_dataset_preparation_summary_markdown
from src.utils import load_config


# ============================================================================
# Statistics Calculation Functions
# ============================================================================

def calculate_activity_statistics(activities: np.ndarray) -> dict:
    """
    Calculate statistics of log activity values.
    
    Args:
        activities: Array of log activity values (-log10 of raw activities).
    
    Returns:
        Dictionary with statistics of log activities.
    """
    return {
        "n_compounds": int(len(activities)),
        "min": float(np.min(activities)),
        "max": float(np.max(activities)),
        "mean": float(np.mean(activities)),
        "median": float(np.median(activities)),
        "std": float(np.std(activities)),
        "q25": float(np.percentile(activities, 25)),
        "q75": float(np.percentile(activities, 75)),
        "q90": float(np.percentile(activities, 90)),
        "q99": float(np.percentile(activities, 99)),
    }


def calculate_vector_length_statistics(vector_lengths: np.ndarray) -> dict:
    """
    Calculate statistics of embedding vector lengths (norms) and fit Gaussian distribution.
    
    Args:
        vector_lengths: Array of vector length values.
    
    Returns:
        Dictionary with statistics of vector lengths and Gaussian fit parameters.
    """
    # Fit Gaussian (Normal) distribution using fit_normal from src/fitting.py
    normal_fit = fit_normal(vector_lengths)
    
    return {
        "n_embeddings": len(vector_lengths),
        "min": float(np.min(vector_lengths)),
        "max": float(np.max(vector_lengths)),
        "mean": float(np.mean(vector_lengths)),
        "median": float(np.median(vector_lengths)),
        "std": float(np.std(vector_lengths)),
        "q25": float(np.percentile(vector_lengths, 25)),
        "q75": float(np.percentile(vector_lengths, 75)),
        "q90": float(np.percentile(vector_lengths, 90)),
        "q99": float(np.percentile(vector_lengths, 99)),
        "normal_fit": normal_fit,
    }


def calculate_embedding_center_statistics(embeddings: np.ndarray) -> dict:
    """
    Calculate statistics of embedding center (mean per dimension).
    
    Args:
        embeddings: Array of shape (n_samples, n_features) containing embeddings.
    
    Returns:
        Dictionary with statistics including:
        - mean_per_dim: Mean across compounds for each dimension
        - length_of_center: L2 norm of mean_per_dim
        - std_per_dim: Standard deviation across compounds for each dimension
        - length_of_std: L2 norm of std_per_dim
        - z_score_shift: mean_per_dim / std_per_dim (element-wise)
        - n_dimensions_z_score_gt_1: Number of dimensions where |z_score_shift| > 1
        - fraction_z_score_gt_1: Fraction of dimensions where |z_score_shift| > 1
    """
    # Calculate mean per dimension (across compounds)
    mean_per_dim = np.mean(embeddings, axis=0)
    
    # Calculate length of center vector
    length_of_center = float(np.linalg.norm(mean_per_dim))
    
    # Calculate std per dimension (across compounds)
    std_per_dim = np.std(embeddings, axis=0)
    
    # Calculate length of std vector
    length_of_std = float(np.linalg.norm(std_per_dim))
    
    # Calculate z_score_shift = mean_per_dim / std_per_dim (avoid division by zero)
    std_per_dim_safe = np.where(std_per_dim == 0, 1e-10, std_per_dim)
    z_score_shift = mean_per_dim / std_per_dim_safe
    
    # Count dimensions where |z_score_shift| > 1
    n_dimensions = len(z_score_shift)
    n_z_score_gt_1 = int(np.sum(np.abs(z_score_shift) > 1))
    fraction_z_score_gt_1 = float(n_z_score_gt_1 / n_dimensions) if n_dimensions > 0 else 0.0
    
    return {
        "mean_per_dim": mean_per_dim.tolist(),
        "length_of_center": length_of_center,
        "std_per_dim": std_per_dim.tolist(),
        "length_of_std": length_of_std,
        "n_dimensions": int(n_dimensions),
        "n_dimensions_z_score_gt_1": n_z_score_gt_1,
        "fraction_z_score_gt_1": fraction_z_score_gt_1,
        "_mean_per_dim_array": mean_per_dim,  # Store for plotting
    }


def calculate_pairwise_log_ratios(activities: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise log_ratio = abs(log_activity_1 - log_activity_2) for all pairs.
    
    Args:
        activities: Array of log_activity values.
    
    Returns:
        Array of log_ratio values for all pairs (upper triangle only, excluding diagonal).
    """
    n = len(activities)
    log_ratios = []
    
    for i in range(n):
        for j in range(i + 1, n):
            log_ratio = abs(activities[i] - activities[j])
            log_ratios.append(log_ratio)
    
    return np.array(log_ratios)


def calculate_log_ratio_statistics(log_ratios: np.ndarray) -> dict:
    """
    Calculate statistics of log_ratio values.
    
    Args:
        log_ratios: Array of log_ratio values.
    
    Returns:
        Dictionary with statistics and Kohlrausch-Williams-Watts fit parameters.
    """
    # Basic statistics
    statistics = {
        "n_pairs": int(len(log_ratios)),
        "min": float(np.min(log_ratios)),
        "max": float(np.max(log_ratios)),
        "mean": float(np.mean(log_ratios)),
        "median": float(np.median(log_ratios)),
        "std": float(np.std(log_ratios)),
        "q25": float(np.percentile(log_ratios, 25)),
        "q75": float(np.percentile(log_ratios, 75)),
        "q90": float(np.percentile(log_ratios, 90)),
        "q95": float(np.percentile(log_ratios, 95)),
        "q99": float(np.percentile(log_ratios, 99)),
    }
    
    # Fit Kohlrausch-Williams-Watts decay
    kww_fit, _, _ = fit_kww_decay(log_ratios, n_bins=50, percentile_cutoff=99.0)
    
    return {
        "statistics": statistics,
        "kww_fit": kww_fit,
    }


def prepare_dataset(config_name: str) -> pd.DataFrame:
    """
    Prepare a dataset by calculating the selected embeddings (all by default) and saving as a pickle file.
    
    This function loads a configuration file and performs eight steps:
    
    Step 1 - Calculate Embeddings:
        - Load the input dataset (CSV file)
        - Calculate all available embeddings for each molecule
        - Validate each molecule against the Embedding schema
    
    Step 2 - Process and Save:
        - Apply unit conversion to log_activity values
        - Mark censored entries based on censor_threshold
        - Save the result as a pickle file
    
    Step 3 - Activity Statistics:
        - Calculate activity statistics for all and uncensored compounds
        - Fit Normal-Uniform mixture distribution to uncensored activities
        - Generate activity statistics plot (3-panel)
    
    Step 4 - Pairwise Activity Differences:
        - Calculate log_ratio = |log_activity₁ - log_activity₂| for all pairs
        - Calculate statistics of log_ratio values
        - Fit Kohlrausch-Williams-Watts decay distribution to log_ratio values
        - Generate histogram plot with Kohlrausch-Williams-Watts fit overlay
    
    Step 5 - Vector Length Statistics:
        - Calculate vector length statistics for all 1D embeddings
        - Fit Gaussian distribution to each
        - Generate multi-panel plot
    
    Step 6 - Embedding Center Statistics:
        - Calculate mean per dimension (across compounds) for each embedding
        - Calculate length of center vector and std vector
        - Calculate z-score shift per dimension and fraction where |z_score| > 1
        - Fit Gaussian distribution to mean per dimension values
        - Generate multi-panel plot with histogram + Gaussian fit
    
    Step 7 - UMAP Projections:
        - Calculate UMAP projection for all 1D embeddings
        - Generate multi-panel plot colored by activity
    
    Step 8 - Generate Markdown Summary:
        - Create comprehensive markdown report summarizing all preparation steps
        - Include statistics, plots, and findings from Steps 3-7
        - Save report to DATASET_PREPARATION_REPORT.md
         
    Args:
        config_name: Name of the config file (without .json extension) in configs/.
    
    Returns:
        DataFrame with validated embedding data.
    """
    console = Console()
    
    # Load configuration
    console.print(f"[cyan]Loading config: {config_name}[/cyan]")
    config = load_config(config_name)
    
    # Resolve dataset (standard dataset key or local CSV path)
    spec = resolve_dataset(config)
    dataset_name = spec.name
    smiles_column = spec.smiles_column
    activity_column = spec.activity_column
    unit = spec.unit
    censor_threshold = spec.censor_threshold

    # Extract optional parameters with defaults
    n_bits = config.get("n_bits", 1024)
    radius = config.get("radius", 2)
    chirality = config.get("chirality", False)
    embedding_kwargs = {"n_bits": n_bits, "radius": radius, "chirality": chirality}
    save_csv = config.get("save_csv", False)

    # Embedding methods to compute (null or missing: all)
    embedding_methods = config.get("methods") or EMBEDDING_METHODS
    unknown = [m for m in embedding_methods if m not in EMBEDDING_METHODS]
    if unknown:
        raise ValueError(f"Unknown embedding methods {unknown}. Supported methods: {', '.join(EMBEDDING_METHODS)}")
    
    # UMAP parameters
    umap_n_neighbors = config.get("umap_n_neighbors", 15)
    umap_min_dist = config.get("umap_min_dist", 0.1)
    
    shift = UNIT_SHIFTS[unit]
    input_path = spec.csv_path
    
    output_path_pkl = Path(config.get("output_file") or f"data/{dataset_name}_embeddings.pkl")
    if output_path_pkl.suffix != ".pkl":
        raise ValueError(f"output_file must be a .pkl file, got: {output_path_pkl}")
    
    # Results directory for statistics
    results_dir = Path("results") / dataset_name / "dataset_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if pickle file already exists
    # Reuse an existing pickle only if it has all requested embeddings with the same fingerprint settings
    pkl_exists = output_path_pkl.exists()
    if pkl_exists:
        existing = pd.read_pickle(output_path_pkl)
        if existing.attrs.get("embedding_kwargs") != embedding_kwargs or not set(embedding_methods) <= set(existing.columns):
            console.print(f"[yellow]Existing {output_path_pkl} lacks requested embeddings or settings; rebuilding[/yellow]")
            pkl_exists = False
    
    # Log configuration
    console.print("[bold]Embedding Configuration:[/bold]")
    console.print(f"  Dataset: {dataset_name}")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_path_pkl}")
    console.print(f"  Results: {results_dir}")
    console.print(f"  SMILES column: {smiles_column}")
    console.print(f"  Activity column: {activity_column}")
    console.print(f"  Unit: {unit} (shift = {shift})")
    console.print(f"  Save CSV: {save_csv}")
    if censor_threshold is not None:
        console.print(f"  Censor threshold: {censor_threshold} {unit}")
    console.print(f"  n_bits: {n_bits}")
    console.print(f"  radius: {radius}")
    console.print(f"  chirality: {chirality}")
    console.print(f"  Embeddings: {', '.join(embedding_methods)}")
    if pkl_exists:
        console.print(f"  [yellow]Pickle file exists: {output_path_pkl}[/yellow]")
    console.print()

    if pkl_exists:
        # =====================================================================
        # Skip Steps 1 & 2: Load existing pickle file
        # =====================================================================
        console.print("[bold yellow]Steps 1 & 2: Skipped (pickle file already exists)[/bold yellow]")
        console.print(f"[cyan]  Loading existing data from {output_path_pkl}...[/cyan]")
        result_df = pd.read_pickle(output_path_pkl)
        console.print(f"[green]  ✓ Loaded {len(result_df)} molecules from existing pickle[/green]")
    else:
        # =====================================================================
        # Step 1: Calculate Embeddings
        # =====================================================================
        console.print("[bold cyan]Step 1: Calculating embeddings...[/bold cyan]")
        
        result_df = save_all_embeddings(
            output_path=str(output_path_pkl.with_suffix("")),
            methods=embedding_methods,
            input_path=str(input_path),
            smiles_column=smiles_column,
            activity_column=activity_column,
            save_csv=save_csv,
            **embedding_kwargs,
        )
        
        console.print(f"[green]  ✓ Embeddings calculated for {len(result_df)} molecules[/green]")
        
        # =====================================================================
        # Step 2: Process and Save
        # =====================================================================
        console.print("[bold cyan]Step 2: Processing and saving...[/bold cyan]")
        
        # Apply unit shift to log_activity and mark censored entries
        result_df = add_log_activity_and_censoring(result_df, spec)
        result_df.attrs["embedding_kwargs"] = embedding_kwargs
        console.print(f"[cyan]  Applied unit conversion: log_activity = -log10(activity) + {shift}[/cyan]")

        if censor_threshold is not None:
            n_censored = result_df["censored"].sum()
            console.print(
                f"[cyan]  Marked {n_censored} entries as censored "
                f"(activity >= {censor_threshold} {unit})[/cyan]"
            )
        else:
            console.print("[dim]  No censor threshold specified, all entries marked as uncensored[/dim]")
        
        # Save the post-processed DataFrame
        output_path_pkl.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_pickle(output_path_pkl)
        
        if save_csv:
            output_path_csv = output_path_pkl.with_suffix(".csv")
            result_df.to_csv(output_path_csv, index=False)
        
        console.print(f"[green]  ✓ Data saved to {output_path_pkl}[/green]")
    
    # Save preparation metadata after Step 2 (or after loading existing pickle)
    preparation_metadata = {
        "input_file": str(input_path),
        "output_file": str(output_path_pkl),
        "pkl_existed": pkl_exists,
        "pkl_generated": not pkl_exists,
        "n_molecules": len(result_df),
        "smiles_column": smiles_column,
        "activity_column": activity_column,
        "unit": unit,
        "shift": shift,
        "censor_threshold": float(censor_threshold) if censor_threshold is not None else None,
        "n_censored": int(result_df["censored"].sum()) if "censored" in result_df.columns else 0,
        "n_uncensored": int((~result_df["censored"]).sum()) if "censored" in result_df.columns else len(result_df),
        "n_bits": n_bits,
        "radius": radius,
        "chirality": chirality,
        "embedding_methods": embedding_methods,
    }
    with open(results_dir / "preparation_metadata.json", "w") as f:
        json.dump(preparation_metadata, f, indent=2)
    
    # Get uncensored mask
    uncensored_mask = ~result_df["censored"].values
    n_all = len(result_df)
    n_uncensored = uncensored_mask.sum()
    n_censored_count = n_all - n_uncensored

    # =========================================================================
    # Step 3: Activity Statistics
    # =========================================================================
    console.print("[bold cyan]Step 3: Calculating activity statistics...[/bold cyan]")
    
    all_activities = result_df["log_activity"].values
    uncensored_activities = all_activities[uncensored_mask]
    
    console.print(f"[cyan]  Total compounds: {n_all}[/cyan]")
    console.print(f"[cyan]  Uncensored: {n_uncensored}[/cyan]")
    console.print(f"[cyan]  Censored: {n_censored_count}[/cyan]")
    
    activity_stats = calculate_activity_statistics(all_activities)
    mixture_fit = fit_normal_uniform_mixture(uncensored_activities)
    
    activity_data = {
        "all_activities": activity_stats,
        "uncensored_activities": {
            "n_compounds": int(n_uncensored),
            "min": float(np.min(uncensored_activities)),
            "max": float(np.max(uncensored_activities)),
            "mean": float(np.mean(uncensored_activities)),
            "median": float(np.median(uncensored_activities)),
            "std": float(np.std(uncensored_activities)),
        },
        "mixture_fit": mixture_fit,
        "unit": unit,
        "shift": int(shift),
        "censor_threshold": float(censor_threshold) if censor_threshold is not None else None,
        "n_censored": int(n_censored_count),
    }
    
    with open(results_dir / "activity_statistics.json", "w") as f:
        json.dump(activity_data, f, indent=2)
    
    plot_activity_statistics(
        all_activities=all_activities,
        uncensored_activities=uncensored_activities,
        mixture_fit=mixture_fit,
        output_path=results_dir / "activity_statistics.png",
        title="Activity Statistics",
    )
    
    console.print(
        f"[green]  ✓ Activity statistics saved "
        f"(Mixture fit: p={mixture_fit['p']:.2f}, μ={mixture_fit['mu']:.2f}, σ={mixture_fit['sigma']:.2f}, "
        f"Uniform=[{mixture_fit['a']:.2f}, {mixture_fit['b']:.2f}])[/green]"
    )

    # =========================================================================
    # Step 4: Pairwise Activity Differences (log_ratio)
    # =========================================================================
    console.print("[bold cyan]Step 4: Calculating pairwise activity differences...[/bold cyan]")
    
    # Calculate log_ratio for all pairs of uncensored compounds
    log_ratios = calculate_pairwise_log_ratios(uncensored_activities)
    
    console.print(f"[cyan]  Calculated log_ratio for {len(log_ratios):,} pairs (uncensored compounds)[/cyan]")
    
    # Calculate statistics and fit Kohlrausch-Williams-Watts distribution
    log_ratio_stats = calculate_log_ratio_statistics(log_ratios)
    
    # Save statistics
    with open(results_dir / "log_ratio_statistics.json", "w") as f:
        json.dump(log_ratio_stats, f, indent=2)
    
    # Plot histogram with Kohlrausch-Williams-Watts fit
    plot_log_ratio_statistics(
        log_ratios=log_ratios,
        log_ratio_stats=log_ratio_stats,
        output_path=results_dir / "log_ratio_statistics.png",
        title="Pairwise Activity Differences (log_ratio)",
    )
    
    kww_fit = log_ratio_stats["kww_fit"]
    if kww_fit.get("a") is not None and kww_fit.get("a") > 0:
        console.print(
            f"[green]  ✓ Log ratio statistics saved, Kohlrausch-Williams-Watts decay fit: "
            f"a={kww_fit['a']:.3f}, k={kww_fit['k']:.3f}, b={kww_fit['b']:.3f}, "
            f"R²={kww_fit['r_squared']:.3f}[/green]"
        )
    else:
        console.print("[yellow]  ⚠ Log ratio statistics saved, Kohlrausch-Williams-Watts decay fit failed[/yellow]")

    # =========================================================================
    # Step 5: Vector Length Statistics for All Embeddings
    # =========================================================================
    console.print("[bold cyan]Step 5: Calculating vector length statistics for all selected embeddings...[/bold cyan]")
    
    vector_length_stats_all = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing embeddings...", total=len(embedding_methods))
        
        for emb_name in embedding_methods:
            if emb_name not in result_df.columns:
                progress.console.print(f"[yellow]  Skipping {emb_name} (not in DataFrame)[/yellow]")
                progress.update(task, advance=1)
                continue
            
            # Get embeddings for uncensored compounds
            embeddings = np.array(result_df.loc[uncensored_mask, emb_name].tolist())
            vector_lengths = np.linalg.norm(embeddings, axis=1)
            
            # Calculate statistics
            stats = calculate_vector_length_statistics(vector_lengths)
            stats["_vector_lengths"] = vector_lengths  # Store for plotting
            vector_length_stats_all[emb_name] = stats
            
            progress.update(task, advance=1)
    
    # Save statistics (without the raw data)
    stats_to_save = {
        emb: {k: v for k, v in s.items() if not k.startswith("_")}
        for emb, s in vector_length_stats_all.items()
    }
    with open(results_dir / "vector_length_statistics.json", "w") as f:
        json.dump(stats_to_save, f, indent=2)
    
    # Plot multi-panel figure
    plot_vector_length_statistics_multi(
        vector_length_stats_all,
        output_path=results_dir / "vector_length_statistics.png",
        title=f"Vector Length Statistics (n={n_uncensored:,} uncensored)",
    )
    
    console.print(f"[green]  ✓ Vector length statistics saved for {len(vector_length_stats_all)} embeddings[/green]")
    
    # =========================================================================
    # Step 6: Embedding Center Statistics
    # =========================================================================
    console.print("[bold cyan]Step 6: Calculating embedding center statistics...[/bold cyan]")
    
    center_stats_all = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing embeddings...", total=len(embedding_methods))
        
        for emb_name in embedding_methods:
            if emb_name not in result_df.columns:
                progress.console.print(f"[yellow]  Skipping {emb_name} (not in DataFrame)[/yellow]")
                progress.update(task, advance=1)
                continue
            
            # Get embeddings for uncensored compounds
            embeddings = np.array(result_df.loc[uncensored_mask, emb_name].tolist())
            
            # Calculate center statistics
            stats = calculate_embedding_center_statistics(embeddings)
            center_stats_all[emb_name] = stats
            
            progress.update(task, advance=1)
    
    # Save statistics (without raw arrays)
    stats_to_save = {
        emb: {k: v for k, v in s.items() if not k.startswith("_")}
        for emb, s in center_stats_all.items()
    }
    with open(results_dir / "embedding_center_statistics.json", "w") as f:
        json.dump(stats_to_save, f, indent=2)
    
    # Plot multi-panel figure
    plot_embedding_center_statistics_multi(
        center_stats_all,
        output_path=results_dir / "embedding_center_statistics.png",
        title=f"Embedding Center Statistics (n={n_uncensored:,} uncensored)",
    )
    
    console.print(f"[green]  ✓ Embedding center statistics saved for {len(center_stats_all)} embeddings[/green]")
    
    # =========================================================================
    # Step 7: UMAP Projections for All Embeddings
    # =========================================================================
    console.print("[bold cyan]Step 7: Calculating UMAP projections for all selected embeddings...[/bold cyan]")
    
    umap_projections = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Computing UMAP...", total=len(embedding_methods))
        
        for emb_name in embedding_methods:
            if emb_name not in result_df.columns:
                progress.update(task, advance=1)
                continue
            
            # Get embeddings for uncensored compounds
            embeddings = np.array(result_df.loc[uncensored_mask, emb_name].tolist())
            
            # Calculate UMAP projection
            reducer = UMAP(
                n_components=2,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                random_state=42,
                n_jobs=1,
            )
            projection = reducer.fit_transform(embeddings)
            umap_projections[emb_name] = projection
            
            progress.update(task, advance=1)
    
    # Save UMAP projections to CSV
    uncensored_df = result_df[uncensored_mask].copy()
    umap_df = pd.DataFrame({
        "inchi_key": uncensored_df["inchi_key"].values,
        "smiles": uncensored_df["smiles"].values,
        "log_activity": uncensored_activities,
    })
    for emb_name, proj in umap_projections.items():
        umap_df[f"{emb_name}_umap_1"] = proj[:, 0]
        umap_df[f"{emb_name}_umap_2"] = proj[:, 1]
    umap_df.to_csv(results_dir / "umap_projections.csv", index=False)
    
    # Plot multi-panel figure
    plot_umap_projections_multi(
        umap_projections,
        uncensored_activities,
        output_path=results_dir / "umap_projections.png",
        title=f"UMAP Projections (n={n_uncensored:,} uncensored)",
    )
    
    console.print(f"[green]  ✓ UMAP projections saved for {len(umap_projections)} embeddings[/green]")
    
    # =========================================================================
    # Step 8: Generate Markdown Summary
    # =========================================================================
    console.print("[bold cyan]Step 8: Generating markdown summary...[/bold cyan]")

    markdown_content = create_dataset_preparation_summary_markdown(results_dir)
    full_markdown = [
        "# Dataset Preparation Report",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "---",
        "## Configuration",
        f"- **Dataset**: {dataset_name}",
        f"- **Unit**: {unit}",
        f"- **Censor threshold**: {censor_threshold if censor_threshold is not None else 'None'}",
        f"- **Embedding methods**: {len(embedding_methods)} types",
        f"- **Results directory**: `{results_dir}`",
        "---",
        markdown_content,
        "---",
        "*Report generated by Activity Cliffs Analysis Pipeline*",
    ]
    
    markdown_file = results_dir / "DATASET_PREPARATION_REPORT.md"
    with open(markdown_file, "w") as f:
        f.write("\n".join(full_markdown))
    
    console.print(f"[green]  ✓ Markdown summary saved to {markdown_file.name}[/green]")

    # =========================================================================
    # Summary
    # =========================================================================
    summary = Table(title="Dataset Preparation Summary", title_style="bold green", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("Dataset", dataset_name)
    summary.add_row("Molecules", f"{n_all:,}")
    summary.add_row("Uncensored", f"{n_uncensored:,}")
    summary.add_row("Censored", f"{n_censored_count:,}")
    summary.add_row("Embeddings", f"{len(embedding_methods)}: {', '.join(embedding_methods)}")
    summary.add_row("Embeddings file", str(output_path_pkl))
    summary.add_row("Results directory", str(results_dir))
    summary.add_row("Files saved", "\n".join([
        "activity_statistics.json/png",
        "log_ratio_statistics.json/png",
        "vector_length_statistics.json/png",
        "embedding_center_statistics.json/png",
        "umap_projections.csv/png",
        "DATASET_PREPARATION_REPORT.md",
    ]))
    console.print()
    console.print(summary)
    
    return result_df
