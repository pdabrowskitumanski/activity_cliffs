"""
Printing module for activity cliffs analysis.

This module provides functions for generating markdown summaries
of analysis results, including statistics, plots, and findings.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console


def format_stat(value: float, precision: int = 4) -> str:
    """Format a statistic value for display."""
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    elif abs(value) >= 1:
        return f"{value:.{precision}f}"
    else:
        return f"{value:.{precision}g}"


def create_stats_table(stats: dict, title: str = "Statistics") -> str:
    """
    Create a markdown table from a statistics dictionary.
    
    Args:
        stats: Dictionary with statistic names and values.
        title: Title for the table.
    
    Returns:
        Markdown formatted table string.
    """
    lines = [
        f"**{title}**",
        "| Statistic | Value |",
        "|-----------|-------|",
    ]
    
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            formatted = format_stat(value) if isinstance(value, float) else str(value)
            # Convert snake_case to Title Case
            display_key = key.replace("_", " ").title()
            lines.append(f"| {display_key} | {formatted} |")
    
    return "\n".join(lines)


def create_pairwise_data_creation_markdown(
    results_dir: Path,
) -> str:
    """
    Create a markdown summary for the pairwise data creation step.

    Args:
        results_dir: Path to the results directory containing pairwise_data_creation/.

    Returns:
        Markdown formatted string for the pairwise data creation section.
    """
    analysis_dir = results_dir / "pairwise_data_creation"
    
    if not analysis_dir.exists():
        return "## Step 1: Pairwise Data Creation\n\n*No pairwise data creation results found.*\n"
    
    lines = [
        "## Step 1: Pairwise Data Creation",
    ]
    
    # Load summary data
    summary_file = analysis_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        
        # Overview section
        n_pairs = summary.get('n_pairs', 'N/A')
        n_correct = summary.get('n_pairs_correct', n_pairs)
        n_censored = summary.get('n_pairs_censored', 0)
        n_excluded = summary.get('n_pairs_excluded', 0)
        
        lines.extend([
            "### Overview",
            f"- **Number of compounds**: {summary.get('n_compounds', 'N/A'):,}",
            f"- **Uncensored compounds**: {summary.get('n_compounds_uncensored', 'N/A'):,}",
            f"- **Censored compounds**: {summary.get('n_compounds_censored', 0):,}",
            f"- **Total pairs**: {n_pairs:,}",
            f"  - Correct pairs: {n_correct:,}",
        ])
        if n_censored > 0:
            lines.append(f"  - Censored pairs: {n_censored:,}")
        if n_excluded > 0:
            lines.append(f"  - Excluded pairs: {n_excluded:,} (both compounds censored)")
        lines.extend([
            f"- **Distance metric**: {summary.get('distance_metric', 'N/A')}",
            f"- **Scaling function**: {summary.get('scaling_function', 'N/A')}",
        ])
        
        # Distance Statistics
        dist_stats = summary.get("distance_statistics", {})
        if dist_stats:
            lines.extend([
                "### Step 1b: Pairwise Distance Statistics",
                "#### Raw Structural Distance",
                f"Pairwise structural distances (raw) range from "
                f"**{format_stat(dist_stats.get('distance_min', 0))}** to "
                f"**{format_stat(dist_stats.get('distance_max', 0))}**.",
                f"- Mean: {format_stat(dist_stats.get('distance_mean', 0))}",
                f"- Median: {format_stat(dist_stats.get('distance_median', 0))}",
                f"- Std Dev: {format_stat(dist_stats.get('distance_std', 0))}",
                "#### Log Ratio (Activity Distance)",
                f"Log ratio values range from "
                f"**{format_stat(dist_stats.get('log_ratio_min', 0))}** to "
                f"**{format_stat(dist_stats.get('log_ratio_max', 0))}**.",
                f"- Mean: {format_stat(dist_stats.get('log_ratio_mean', 0))}",
                f"- Median: {format_stat(dist_stats.get('log_ratio_median', 0))}",
                f"- Std Dev: {format_stat(dist_stats.get('log_ratio_std', 0))}",
            ])
        
        # Distance Statistics Plot
        dist_plot = analysis_dir / "distance_statistics.png"
        if dist_plot.exists():
            rel_path = dist_plot.relative_to(results_dir)
            img_src = str(rel_path)
            lines.append(f"![Distance Statistics]({img_src})")

        # Scaled distance distribution fits (BIC)
        bic_file = analysis_dir / "scaled_distance_distribution_fits.json"
        if bic_file.exists():
            with open(bic_file) as f:
                bic_data = json.load(f)
            lines.extend([
                "",
                "#### Scaled distance distribution fits (BIC)",
                "",
                "Gaussian, Beta, Gamma, log-normal, and Uniform distributions were fit by maximum likelihood; "
                "**BIC = k·ln(n) − 2·log-likelihood** (lower is better). All models use 2 free parameters.",
                "",
            ])
            best = bic_data.get("best_by_bic")
            best_bic = bic_data.get("best_bic")
            if best is not None:
                lines.append(f"- **Selected distribution (lowest BIC)**: `{best}` (BIC={format_stat(best_bic) if best_bic is not None else 'N/A'})")
            lines.extend(["", "| Distribution | Log-likelihood | BIC | Success |", "|--------------|----------------|-----|---------|"])
            fits = bic_data.get("fits") or {}
            for dist_name in ("gaussian", "beta", "gamma", "lognormal", "uniform"):
                fd = fits.get(dist_name) or {}
                ll = fd.get("log_likelihood")
                bic = fd.get("bic")
                ok = fd.get("success", False)
                ll_s = format_stat(float(ll)) if ll is not None and np.isfinite(float(ll)) else "N/A"
                bic_s = format_stat(float(bic)) if bic is not None and np.isfinite(float(bic)) else "N/A"
                lines.append(f"| {dist_name} | {ll_s} | {bic_s} | {ok} |")
            lines.append("")
        
        # Distance Matrices
        matrix_plot = analysis_dir / "distance_matrices.png"
        if matrix_plot.exists():
            rel_path = matrix_plot.relative_to(results_dir)
            img_src = str(rel_path)
            lines.extend([
                "### Step 1c: Distance Matrices",
                f"![Distance Matrices]({img_src})",
            ])
        
        # Special Pairs
        struct_pairs = analysis_dir / "structurally_similar_pairs.csv"
        act_pairs = analysis_dir / "activity_similar_pairs.csv"
        
        if struct_pairs.exists() or act_pairs.exists():
            lines.append("### Special Pairs")
            
            if struct_pairs.exists():
                # Count lines (excluding header)
                with open(struct_pairs) as f:
                    n_struct = sum(1 for _ in f) - 1
                lines.append(f"- **Structurally identical pairs** (distance = 0): {n_struct:,}")
            
            if act_pairs.exists():
                with open(act_pairs) as f:
                    n_act = sum(1 for _ in f) - 1
                lines.append(f"- **Activity identical pairs** (log_ratio < 0.01): {n_act:,}")
    
    else:
        lines.append("*Summary file not found.*")
    
    # Output files table
    lines.extend([
        "### Output Files",
        "| File | Description |",
        "|------|-------------|",
        "| `summary.json` | Complete analysis summary |",
        "| `compound_distance_statistics.json` | Raw and scaled distance statistics; scaled fits include BIC comparison |",
        "| `scaled_distance_distribution_fits.json` | Log-likelihood and BIC for Gaussian, Beta, Gamma, log-normal, Uniform on scaled distances |",
        "| `distance_statistics.png` | Distance distribution plots (raw and scaled) |",
        "| `raw_distances.json` | All pairwise distances (excluding embeddings) |",
        "| `structural_distance_matrix.npy` | Structural distance matrix |",
        "| `activity_distance_matrix.npy` | Activity distance matrix |",
        "| `distance_matrices.png` | Matrix visualizations (1x2 panel) |",
        "| `structurally_similar_pairs.csv` | Pairs with distance = 0 |",
        "| `activity_similar_pairs.csv` | Pairs with log_ratio < 0.01 |",
    ])
    
    return "\n".join(lines)


def create_correlation_summary_markdown(
    results_dir: Path,
) -> str:
    """
    Create a markdown summary for the correlation analysis step.

    Args:
        results_dir: Path to the results directory containing correlation/.

    Returns:
        Markdown formatted string for the correlation section.
    """
    corr_dir = results_dir / "correlation"
    
    if not corr_dir.exists():
        return "## Step 3: Correlation Analysis\n\n*No correlation analysis results found.*\n"
    
    lines = [
        "## Step 3: Correlation Analysis",
    ]
    
    # Load summary
    summary_file = corr_dir / "summary.json"
    summary = {}
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        
        total_pairs = summary.get('total_pairs', 'N/A')
        valid_pairs = summary.get('valid_pairs', total_pairs)
        excluded_pairs = summary.get('excluded_pairs', 0)
        
        lines.extend([
            "### Overview",
            f"- **Total pairs**: {total_pairs:,}",
        ])
        if excluded_pairs > 0:
            lines.append(f"  - Valid pairs: {valid_pairs:,} (excluded: {excluded_pairs:,})")
        lines.extend([
            f"- **Activity threshold**: {summary.get('activity_threshold', 'N/A')}",
            f"- **Regularization factor**: {summary.get('regularization_factor', 'N/A')}",
        ])
    
    # Lipschitz Heatmap
    lip_heatmap = corr_dir / "lipschitz_heatmap.png"
    if lip_heatmap.exists():
        rel_path = lip_heatmap.relative_to(results_dir)
        img_src = str(rel_path)
        lines.extend([
            "### Step 3b: Lipschitz Heatmap",
            f"![Lipschitz Heatmap]({img_src})",
        ])
    
    # Lipschitz Statistics from summary
    if summary and "lipschitz_max" in summary:
        lines.extend([
            "### Step 3c: Lipschitz Statistics",
            f"- Maximum Lipschitz constant: {format_stat(summary.get('lipschitz_max', 0))}",
            f"- 90th percentile: {format_stat(summary.get('lipschitz_percentile_90', 0))}",
            f"- 99th percentile: {format_stat(summary.get('lipschitz_percentile_99', 0))}",
        ])
        
        # KWW fit from summary
        kww_fit = summary.get("kww_fit", {})
        if kww_fit:
            lines.extend([
                "#### Kohlrausch-Williams-Watts Decay Fit",
                "Function: `a · exp(-k · x^b)`",
                "| Parameter | Interpretation |",
                "|-----------|----------------|",
                "| **b (shape)** | Suppression of large gradients (smoothness) |",
                "| **k (scale)** | Typical local SAR steepness |",
                "**Better representation has:** larger b, larger k, lower tail mass beyond cliff threshold.",
                "#### Fitted Values",
                f"- a (amplitude): {format_stat(kww_fit.get('a', 0))}",
                f"- k (scale): {format_stat(kww_fit.get('k', 0))}",
                f"- b (shape): {format_stat(kww_fit.get('b', 0))}",
                f"- R²: {format_stat(kww_fit.get('r_squared', 0))}",
            ])
        kt = summary.get("kww_theoretical") or {}
        if kt.get("success") or kt.get("mean_Er") is not None:
            lines.extend([
                "#### KWW / Weibull theoretical quantities (from fitted *a*, *k*, *b*)",
                "Empirical percentiles/mean above are data; below follow the Weibull interpretation "
                "of the exponent in `a·exp(-k·x^b)` (same *k*, *b* as in survival `exp(-k·r^b)`).",
                f"- **Theoretical 95th percentile** *r95* = (−ln(0.05/*a*)/*k*)^(1/*b*): "
                f"{format_stat(kt['r95']) if kt.get('r95') is not None else 'N/A'}",
                f"- **Theoretical mean** *E[r]* = Γ(1+1/*b*) / *k*^(1/*b*): "
                f"{format_stat(kt['mean_Er']) if kt.get('mean_Er') is not None else 'N/A'}",
            ])
            if kt.get("r95_note"):
                lines.append(f"  - *Note (r95)*: {kt.get('r95_note')}")
            if kt.get("error"):
                lines.append(f"  - *Note*: {kt.get('error')}")
    
    lip_plot = corr_dir / "lipschitz_statistics.png"
    if lip_plot.exists():
        rel_path = lip_plot.relative_to(results_dir)
        img_src = str(rel_path)
        lines.append(f"![Lipschitz Statistics]({img_src})")
    
    # Binned Pearson Correlation
    if summary:
        binned_corr = summary.get("binned_correlation", {})
        if binned_corr:
            lines.extend([
                "### Step 3e: Pearson Correlation vs Similarity Bin",
                f"- Bin width: {binned_corr.get('bin_width', 'N/A')}",
                f"- Valid bins: {binned_corr.get('n_valid_bins', 'N/A')} / {binned_corr.get('n_bins', 'N/A')}",
                f"- Mean Pearson r: {format_stat(binned_corr.get('mean_pearson_r', 0))}",
                f"- Range: [{format_stat(binned_corr.get('min_pearson_r', 0))}, {format_stat(binned_corr.get('max_pearson_r', 0))}]",
            ])
        
        binned_corr_plot = corr_dir / "binned_pearson_correlation.png"
        if binned_corr_plot.exists():
            rel_path = binned_corr_plot.relative_to(results_dir)
            img_src = str(rel_path)
            lines.append(f"![Binned Pearson Correlation]({img_src})")
        
        # Cliff Probability
        cliff_prob = summary.get("cliff_probability", {})
        if cliff_prob:
            lines.extend([
                "### Step 3f: Cliff Probability vs Similarity Bin",
                f"- Activity threshold: {cliff_prob.get('activity_threshold', 'N/A')}",
                f"- Total cliffs: {cliff_prob.get('total_cliffs', 'N/A'):,} / {cliff_prob.get('total_pairs', 'N/A'):,}",
                f"- Overall cliff probability: {format_stat(cliff_prob.get('overall_cliff_probability', 0))}",
            ])
        
        cliff_prob_plot = corr_dir / "binned_cliff_probability.png"
        if cliff_prob_plot.exists():
            rel_path = cliff_prob_plot.relative_to(results_dir)
            img_src = str(rel_path)
            lines.append(f"![Cliff Probability]({img_src})")

        # Chiral compound pairs (Step 3g)
        chiral = summary.get("chiral_cliff_lipschitz")
        if chiral is not None:
            n_total_chiral = chiral.get("n_total_pairs", 0)
            n_chiral = chiral.get("n_pairs", 0)
            frac_chiral = chiral.get("fraction_cliffs", 0.0)
            lines.extend([
                "### Step 3g: Chiral Compound Pairs (Activity Cliffs)",
                "Pairs with same connectivity (first InChI block) and activity difference above threshold.",
                f"- **Total chiral pairs considered**: {n_total_chiral:,}",
                f"- **Cliff pairs (analyzed)**: {n_chiral:,}",
                f"- **Fraction of cliffs**: {format_stat(frac_chiral)}",
            ])
            if n_chiral > 0:
                if chiral.get("statistics"):
                    stats = chiral["statistics"]
                    lines.extend([
                        "#### Regularized Lipschitz (cliff pairs)",
                        f"- Max: {format_stat(stats.get('max', 0))}, 90th / 99th percentile: {format_stat(stats.get('percentile_90', 0))} / {format_stat(stats.get('percentile_99', 0))}",
                        f"- Mean: {format_stat(stats.get('mean', 0))}, Median: {format_stat(stats.get('median', 0))}",
                    ])
                for label, key in [("Similarity", "statistics_similarity"), ("Log ratio (|Δlog_activity|)", "statistics_log_ratio")]:
                    s = chiral.get(key)
                    if s:
                        lines.append(f"- **{label}** (across cliff pairs): mean={format_stat(s.get('mean', 0))}, median={format_stat(s.get('median', 0))}, std={format_stat(s.get('std', 0))}, min–max=[{format_stat(s.get('min', 0))}, {format_stat(s.get('max', 0))}]")
                kww_chiral = chiral.get("kww_fit") or {}
                if kww_chiral.get("a") is not None:
                    lines.extend([
                        "#### KWW fit (regularized Lipschitz, chiral pairs)",
                        f"- a, k, b, R²: {format_stat(kww_chiral.get('a', 0))}, {format_stat(kww_chiral.get('k', 0))}, {format_stat(kww_chiral.get('b', 0))}, {format_stat(kww_chiral.get('r_squared', 0))}",
                    ])
                beta_ch = chiral.get("distance_beta_fit") or {}
                if beta_ch.get("a") is not None:
                    _cv_ch = beta_ch.get("cv")
                    lines.extend([
                        "#### Beta fit (scaled structural distance `distance_scaled`, chiral pairs)",
                        f"- **a**, **b**: {format_stat(beta_ch.get('a', 0))}, {format_stat(beta_ch.get('b', 0))} (R²={format_stat(beta_ch.get('r_squared', 0))})",
                        f"- **m** = a/(a+b): {format_stat(beta_ch.get('m', 0))}, **f** = a+b: {format_stat(beta_ch.get('f', 0))}",
                        f"- **s²** = m(1−m)/(f+1): {format_stat(beta_ch.get('variance', 0))}, **s**: {format_stat(beta_ch.get('std', 0)) if beta_ch.get('std') is not None else '—'}",
                        f"- **CV** = s/m: {format_stat(_cv_ch) if _cv_ch is not None else '—'}",
                    ])
            chiral_three = corr_dir / "chiral_cliff_three_panels.png"
            if chiral_three.exists():
                rel_path = chiral_three.relative_to(results_dir)
                img_src = str(rel_path)
                lines.append(f"![Chiral cliff pairs — Similarity, Log ratio, Lipschitz]({img_src})")
            chiral_plot = corr_dir / "chiral_cliff_lipschitz_statistics.png"
            if chiral_plot.exists():
                rel_path = chiral_plot.relative_to(results_dir)
                img_src = str(rel_path)
                lines.append(f"![Chiral Cliff Lipschitz Statistics]({img_src})")
            chiral_dist = corr_dir / "chiral_distance_beta.png"
            if chiral_dist.exists():
                rel_path = chiral_dist.relative_to(results_dir)
                img_src = str(rel_path)
                lines.append(f"![Chiral pairs — scaled distance distance_scaled (Beta fit)]({img_src})")

        # Matched molecular pairs (Step 3h)
        mmp = summary.get("mmp_cliff_lipschitz")
        if mmp is not None:
            n_total_mmp = mmp.get("n_total_pairs", 0)
            n_mmp = mmp.get("n_pairs", 0)
            frac_mmp = mmp.get("fraction_cliffs", 0.0)
            lines.extend([
                "### Step 3h: Matched Molecular Pairs (Activity Cliffs)",
                "Pairs from the supplied MMP CSV (smiles1, smiles2) with activity difference above threshold.",
                f"- **Total MMP pairs matched**: {n_total_mmp:,}",
                f"- **Cliff pairs (analyzed)**: {n_mmp:,}",
                f"- **Fraction of cliffs**: {format_stat(frac_mmp)}",
            ])
            if mmp.get("mmp_file"):
                lines.append(f"- **MMP file**: `{mmp.get('mmp_file', '')}`")
            if n_mmp > 0:
                if mmp.get("statistics"):
                    stats = mmp["statistics"]
                    lines.extend([
                        "#### Regularized Lipschitz (cliff pairs)",
                        f"- Max: {format_stat(stats.get('max', 0))}, 90th / 99th percentile: {format_stat(stats.get('percentile_90', 0))} / {format_stat(stats.get('percentile_99', 0))}",
                        f"- Mean: {format_stat(stats.get('mean', 0))}, Median: {format_stat(stats.get('median', 0))}",
                    ])
                for label, key in [("Similarity", "statistics_similarity"), ("Log ratio (|Δlog_activity|)", "statistics_log_ratio")]:
                    s = mmp.get(key)
                    if s:
                        lines.append(f"- **{label}** (across cliff pairs): mean={format_stat(s.get('mean', 0))}, median={format_stat(s.get('median', 0))}, std={format_stat(s.get('std', 0))}, min–max=[{format_stat(s.get('min', 0))}, {format_stat(s.get('max', 0))}]")
                kww_mmp = mmp.get("kww_fit") or {}
                if kww_mmp.get("a") is not None:
                    lines.extend([
                        "#### KWW fit (regularized Lipschitz, MMP pairs)",
                        f"- a, k, b, R²: {format_stat(kww_mmp.get('a', 0))}, {format_stat(kww_mmp.get('k', 0))}, {format_stat(kww_mmp.get('b', 0))}, {format_stat(kww_mmp.get('r_squared', 0))}",
                    ])
                beta_mmp = mmp.get("distance_beta_fit") or {}
                if beta_mmp.get("a") is not None:
                    _cv_m = beta_mmp.get("cv")
                    lines.extend([
                        "#### Beta fit (scaled structural distance `distance_scaled`, MMP pairs)",
                        f"- **a**, **b**: {format_stat(beta_mmp.get('a', 0))}, {format_stat(beta_mmp.get('b', 0))} (R²={format_stat(beta_mmp.get('r_squared', 0))})",
                        f"- **m** = a/(a+b): {format_stat(beta_mmp.get('m', 0))}, **f** = a+b: {format_stat(beta_mmp.get('f', 0))}",
                        f"- **s²** = m(1−m)/(f+1): {format_stat(beta_mmp.get('variance', 0))}, **s**: {format_stat(beta_mmp.get('std', 0)) if beta_mmp.get('std') is not None else '—'}",
                        f"- **CV** = s/m: {format_stat(_cv_m) if _cv_m is not None else '—'}",
                    ])
            mmp_three = corr_dir / "mmp_cliff_three_panels.png"
            if mmp_three.exists():
                rel_path = mmp_three.relative_to(results_dir)
                img_src = str(rel_path)
                lines.append(f"![MMP cliff pairs — Similarity, Log ratio, Lipschitz]({img_src})")
            mmp_plot = corr_dir / "mmp_cliff_lipschitz_statistics.png"
            if mmp_plot.exists():
                rel_path = mmp_plot.relative_to(results_dir)
                img_src = str(rel_path)
                lines.append(f"![MMP Cliff Lipschitz Statistics]({img_src})")
            mmp_dist = corr_dir / "mmp_distance_beta.png"
            if mmp_dist.exists():
                rel_path = mmp_dist.relative_to(results_dir)
                img_src = str(rel_path)
                lines.append(f"![MMP pairs — scaled distance distance_scaled (Beta fit)]({img_src})")
    
    return "\n".join(lines)


def create_cliff_identification_summary_markdown(
    results_dir: Path,
) -> str:
    """
    Create a markdown summary for the cliff identification step.
    
    Args:
        results_dir: Path to the results directory.
    
    Returns:
        Markdown formatted string for the cliff identification section.
    """
    lines = [
        "## Step 2: Activity Cliff Identification",
    ]
    
    # Check for cliff-related files
    cliff_dir = results_dir / "cliff_identification"
    summary_file = cliff_dir / "summary.json"
    
    if not cliff_dir.exists():
        lines.append("*No activity cliff identification results found.*")
        return "\n".join(lines)
    
    # Load summary if exists
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        
        total_pairs = summary.get('total_pairs', 'N/A')
        valid_pairs = summary.get('valid_pairs', total_pairs)
        excluded_pairs = summary.get('excluded_pairs', 0)
        
        lines.extend([
            "### Overview",
            f"- **Activity threshold**: {summary.get('activity_threshold', 'N/A')}",
            f"- **Total pairs**: {total_pairs:,}",
        ])
        if excluded_pairs > 0:
            lines.append(f"  - Valid pairs: {valid_pairs:,} (excluded: {excluded_pairs:,})")
        lines.extend([
            f"- **Cliffs at similarity >= 0.5**: {summary.get('cliffs_at_similarity_0.5', 'N/A'):,}",
            f"- **Cliffs at similarity >= 0.7**: {summary.get('cliffs_at_similarity_0.7', 'N/A'):,}",
            f"- **Cliffs at similarity >= 0.9**: {summary.get('cliffs_at_similarity_0.9', 'N/A'):,}",
        ])

        # Distance geometry (CV, dip test, relative contrast, hubness) — full JSON: distance_geometry_statistics.json
        dg = summary.get("distance_geometry")
        if isinstance(dg, dict) and not dg.get("error"):
            lines.extend(["", "### Distance geometry (structural distances)"])
            cv = dg.get("coefficient_of_variation") or {}
            if cv:
                lines.extend([
                    "- **Coefficient of variation (distances)**",
                    f"  - Mean: {format_stat(cv.get('mean', 0))}",
                    f"  - Std: {format_stat(cv.get('std', 0))}",
                    f"  - CV (std/mean): {format_stat(cv['coefficient_of_variation_std_over_mean']) if cv.get('coefficient_of_variation_std_over_mean') is not None else 'N/A'}",
                    f"  - Mean/std: {format_stat(cv['mean_over_std']) if cv.get('mean_over_std') is not None else 'N/A'}",
                    f"  - Pairwise distances counted: {cv.get('n_pairwise_distances', 'N/A'):,}",
                ])
            dip = dg.get("hartigan_dip_test") or {}
            if dip.get("available"):
                lines.extend([
                    "- **Hartigan dip test (unimodality of pairwise distances)**",
                    f"  - Dip statistic: {format_stat(dip.get('dip_statistic', 0))}",
                    f"  - p-value: {format_stat(dip.get('p_value', 0))}",
                ])
            elif dip.get("error"):
                lines.append(f"- **Hartigan dip test**: *not available* ({dip.get('error')})")
            rc = dg.get("relative_contrast") or {}
            if rc.get("median") is not None:
                lines.extend([
                    "- **Relative contrast** (per molecule: (d95−d5)/d5 over distances to others; metric: distance_scaled)",
                    f"  - Median: {format_stat(rc.get('median', 0))}",
                    f"  - 25th percentile: {format_stat(rc.get('percentile_25', 0))}",
                    f"  - 75th percentile: {format_stat(rc.get('percentile_75', 0))}",
                    f"  - Molecules: {rc.get('n_molecules', 'N/A'):,}",
                ])
            hub = dg.get("hubness") or {}
            if hub.get("skewness") is not None:
                lines.extend([
                    "- **Hubness** (k=5 nearest neighbors by distance_scaled; N5 = times a molecule appears in others' 5-NN)",
                    f"  - Skewness of N5: {format_stat(hub.get('skewness', 0))}",
                    f"  - Hub fraction (N5 > 10): {format_stat(hub.get('hub_fraction_N5_gt_10', 0))}",
                    f"  - Anti-hub fraction (N5 = 0): {format_stat(hub.get('anti_hub_fraction_N5_eq_0', 0))}",
                ])
            dg_file = summary.get("distance_geometry_statistics_file", "distance_geometry_statistics.json")
            lines.extend([
                "",
                f"Full per-molecule values (relative contrast, N5): `{dg_file}` in `cliff_identification/`.",
            ])
    
    # Cliff fractions plot
    plot_path = cliff_dir / "activity_cliff_fractions.png"
    if plot_path.exists():
        rel_path = plot_path.relative_to(results_dir)
        img_src = str(rel_path)
        lines.extend([
            "### Activity Cliff Fractions",
            f"![Activity Cliff Fractions]({img_src})",
        ])
    
    # Fit parameters
    fit_file = cliff_dir / "fit_parameters.json"
    if fit_file.exists():
        with open(fit_file) as f:
            fit_params = json.load(f)
        
        lines.extend([
            "### Sigmoid Fit Parameters",
            "Sigmoid function: `c + a / (1 + exp(k * (x - s₀)))`",
            "#### Parameter Interpretation",
            "| Parameter | Interpretation |",
            "|-----------|----------------|",
            "| **c** | Irreducible cliff probability (true SAR discontinuities + noise) |",
            "| **a** | Removable cliff mass (cliffs due to global dissimilarity) |",
            "| **s₀** | Similarity threshold where local SAR begins |",
            "| **k** | Sharpness of SAR localization |",
            "**Better representation has:** smaller c, smaller s₀, larger k, and similar or smaller a.",
        ])
        
        if "activity_cliff_fraction" in fit_params:
            acf = fit_params["activity_cliff_fraction"]
            lines.extend([
                "#### Activity Cliff Fraction Fit",
                f"- c (irreducible cliff prob.): {format_stat(acf.get('c', 0))}",
                f"- a (removable cliff mass): {format_stat(acf.get('a', 0))}",
                f"- k (sharpness): {format_stat(acf.get('k', 0))}",
                f"- s₀ (threshold): {format_stat(acf.get('x0', 0))}",
            ])
        
        if "n_pairs" in fit_params:
            np_params = fit_params["n_pairs"]
            lines.extend([
                "#### Number of Pairs Fit",
                f"- c: {format_stat(np_params.get('c', 0))}",
                f"- a: {format_stat(np_params.get('a', 0))}",
                f"- k: {format_stat(np_params.get('k', 0))}",
                f"- s₀: {format_stat(np_params.get('x0', 0))}",
            ])
        
        if "cliff_fraction_vs_pair_rank" in fit_params:
            cfr_params = fit_params["cliff_fraction_vs_pair_rank"]
            lines.extend([
                "#### Cliff Fraction vs Pair Rank Fit",
                f"- c: {format_stat(cfr_params.get('c', 0))}",
                f"- a: {format_stat(cfr_params.get('a', 0))}",
                f"- k: {format_stat(cfr_params.get('k', 0))}",
                f"- s₀: {format_stat(cfr_params.get('x0', 0))}",
            ])

        cfn = fit_params.get("cumulative_Fn_power_law")
        if isinstance(cfn, dict) and cfn.get("success"):
            lines.extend([
                "#### Cumulative F(n) power-law (F(n) ≈ a·n^s)",
                f"- **s** (exponent): {format_stat(cfn.get('s', 0))}",
                f"- **a** (scale): {format_stat(cfn.get('a', 0))}",
                f"- **R²**: {format_stat(cfn.get('r_squared', 0))}",
                f"- **Gini** = (s−1)/(s+1): {format_stat(cfn.get('gini_coefficient', 0)) if cfn.get('gini_coefficient') is not None else 'N/A'}",
                f"- **Total cliff pairs**: {cfn.get('total_cliff_pairs', 'N/A'):,}"
                if cfn.get("total_cliff_pairs") is not None
                else "- **Total cliff pairs**: N/A",
            ])
    
    # Cliff vs pair rank plot
    cliff_rank_plot = cliff_dir / "cliff_vs_pair_rank.png"
    if cliff_rank_plot.exists():
        rel_path = cliff_rank_plot.relative_to(results_dir)
        img_src = str(rel_path)
        lines.extend([
            "### Cliff Fraction vs Pair Rank",
            f"![Cliff Fraction vs Pair Rank]({img_src})",
        ])

    # Cumulative cliff fraction F(n), power-law fit, Gini
    fn_plot = cliff_dir / "cumulative_cliff_fraction_Fn.png"
    if fn_plot.exists():
        rel_path = fn_plot.relative_to(results_dir)
        img_src = str(rel_path)
        lines.extend([
            "",
            "### Cumulative cliff fraction F(n)",
            r"**Definition:** $F(n) = \dfrac{\text{cliff pairs in top } n\% \text{ most similar pairs}}{\text{total cliff pairs}}$.",
            "Fitted model: **F(n) ≈ a · n^s** (nonlinear least squares). **Gini** = (s − 1) / (s + 1) from the fitted exponent *s*.",
        ])
        if summary_file.exists():
            with open(summary_file) as f:
                summ = json.load(f)
            s_val = summ.get("power_law_exponent_s_Fn")
            r2_val = summ.get("power_law_r_squared_Fn")
            gini_val = summ.get("gini_coefficient_Fn")
            tcp = summ.get("total_cliff_pairs")
            if tcp is not None:
                lines.append(f"- **Total cliff pairs**: {tcp:,}")
            if s_val is not None:
                lines.append(f"- **Power-law exponent *s***: {format_stat(float(s_val))}")
            if r2_val is not None:
                lines.append(f"- **R²** (fit of *a·n^s*): {format_stat(float(r2_val))}")
            if gini_val is not None:
                lines.append(f"- **Gini** (from *s*): {format_stat(float(gini_val))}")
        lines.extend([
            "",
            f"![Cumulative cliff fraction F(n)]({img_src})",
            "",
            "Details: `cumulative_cliff_fraction_analysis.json`, `cumulative_cliff_fraction_Fn.csv` in `cliff_identification/`.",
        ])
    
    return "\n".join(lines)


def create_persistent_homology_summary_markdown(
    results_dir: Path,
) -> str:
    """
    Create a markdown summary for the persistent homology analysis.

    Includes number of H0 components, long-lived components (persistence > threshold),
    average persistence, and persistence landscape figures (Bubenik / persim).

    Args:
        results_dir: Path to the results directory (contains persistent_homology/).

    Returns:
        Markdown formatted string for the persistent homology section.
    """
    results_dir = Path(results_dir)
    ph_dir = results_dir / "persistent_homology"
    lines = ["## Step 4: Persistent Homology"]

    if not ph_dir.exists():
        lines.append("*No persistent homology results found.*")
        return "\n".join(lines)

    meta_file = ph_dir / "persistence_metadata.json"
    clusters_file = ph_dir / "persistent_clusters.json"

    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        stats = meta.get("statistics", {})
        dim0 = stats.get("dim_0", {})
        threshold = meta.get("persistence_threshold", "N/A")
        n_all_components = dim0.get("n_features", "N/A")
        avg_persistence = dim0.get("mean_persistence")
        n_long_lived = meta.get("n_persistent_clusters", "N/A")
        lines.extend([
            "### Overview",
            f"- **Persistence threshold**: {threshold}",
            f"- **Number of all H0 components**: {n_all_components}",
            f"- **Number of long-lived components** (persistence ≥ threshold, incl. inf): {n_long_lived}",
        ])
        if avg_persistence is not None:
            lines.append(f"- **Average persistence** (H0, finite): {format_stat(avg_persistence)}")
        l2_map = meta.get("landscape_l2_norm") or {}
        l2_h0 = l2_map.get("dim_0")
        if l2_h0 is not None:
            lines.append(f"- **Persistence landscape L2 norm** (H0): {format_stat(l2_h0)}")
        pe0 = dim0.get("persistent_entropy")
        if pe0 is not None:
            lines.append(
                f"- **Persistent entropy** Σ (p_i log p_i) over finite lifetimes (H0): {format_stat(pe0)}"
            )
        lines.append("")

        # Statistics table for dimensions 0, 1, 2
        stat_keys = [
            ("n_features", "Number of features"),
            ("n_finite", "Finite persistence"),
            ("n_infinite", "Infinite persistence"),
            ("max_persistence", "Max persistence"),
            ("mean_persistence", "Mean persistence"),
            ("median_persistence", "Median persistence"),
            ("std_persistence", "Std persistence"),
            ("total_persistence", "Total persistence"),
            ("persistent_entropy", "Persistent entropy Σ p_i ln p_i"),
            ("persistent_entropy_shannon", "Shannon −Σ p_i ln p_i"),
            ("landscape_l2_norm", "Landscape L2 norm"),
            ("max_birth", "Max birth"),
            ("max_death", "Max death"),
        ]
        lines.append("### Persistence statistics by dimension")
        lines.append("")
        lines.append("| Statistic | H0 | H1 | H2 |")
        lines.append("|-----------|----|----|----|")
        for key, label in stat_keys:
            row = [label]
            for d in range(3):
                dim_key = f"dim_{d}"
                dim_stats = stats.get(dim_key, {})
                val = dim_stats.get(key)
                if val is None:
                    row.append("—")
                elif isinstance(val, float):
                    row.append(format_stat(val))
                else:
                    row.append(str(val))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    def _img(rel_name: str) -> str:
        return f"persistent_homology/{rel_name}"

    # Filenames unchanged; plots are persistence landscapes (tent functions).
    single_fig = ph_dir / "persistence_diagram.png"
    if single_fig.exists():
        lines.extend([
            "### Persistence landscape (H0)",
            f"![Persistence landscape]({_img('persistence_diagram.png')})",
        ])
    else:
        all_fig = ph_dir / "persistence_diagram_all.png"
        if all_fig.exists():
            lines.extend([
                "### Persistence landscapes (H0, H1, …)",
                f"![Persistence landscapes — all dimensions]({_img('persistence_diagram_all.png')})",
            ])
        h0_fig = ph_dir / "persistence_diagram_H0.png"
        if h0_fig.exists():
            lines.extend([
                "### Persistence landscape (H0 only)",
                f"![Persistence landscape — H0]({_img('persistence_diagram_H0.png')})",
            ])

    return "\n".join(lines)


def create_model_training_summary_markdown(results_dir: Path) -> str:
    """
    Create a markdown summary for the model training step.

    Args:
        results_dir: Path to the results directory containing model_training/.

    Returns:
        Markdown formatted string for the model training section.
    """
    model_dir = results_dir / "model_training"
    
    if not model_dir.exists():
        return "## Step 5: Model Training\n\n*No model training results found.*\n"
    
    lines = [
        "## Step 5: Model Training",
    ]
    
    # Load split statistics
    stats_file = model_dir / "split_statistics.json"
    split_names = ["butina", "scaffold"]
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        
        lines.extend([
            "### Overview",
            f"- **Activity threshold**: {stats.get('activity_threshold', 'N/A')}",
            f"- **Random seed**: {stats.get('seed', 'N/A')}",
        ])
        # New format: per-split stats under "butina" / "scaffold"
        if "butina" in stats and "scaffold" in stats:
            for split_name in split_names:
                s = stats.get(split_name, {})
                lines.extend([
                    f"### Data Splits ({split_name.capitalize()})",
                    f"- **Train / Val / Test compounds**: {s.get('n_train_compounds', 'N/A'):,} / {s.get('n_val_compounds', 'N/A'):,} / {s.get('n_test_compounds', 'N/A'):,}",
                    f"- **Train / Val / Test pairs (preliminary)**: {s.get('n_train_pairs_prelim', 'N/A'):,} / {s.get('n_val_pairs_prelim', 'N/A'):,} / {s.get('n_test_pairs_prelim', 'N/A'):,}",
                    f"- **Train / Val / Test pairs (balanced)**: {s.get('n_train_pairs_balanced', 'N/A'):,} / {s.get('n_val_pairs_balanced', 'N/A'):,} / {s.get('n_test_pairs_balanced', 'N/A'):,}",
                    f"- **Train cliffs / non-cliffs**: {s.get('train_n_cliffs', 'N/A'):,} / {s.get('train_n_non_cliffs', 'N/A'):,}",
                    f"- **Val cliffs / non-cliffs**: {s.get('val_n_cliffs', 'N/A'):,} / {s.get('val_n_non_cliffs', 'N/A'):,}",
                    f"- **Test cliffs / non-cliffs**: {s.get('test_n_cliffs', 'N/A'):,} / {s.get('test_n_non_cliffs', 'N/A'):,}",
                ])
        else:
            lines.extend([
                "### Data Splits",
                f"- **Train compounds**: {stats.get('n_train_compounds', 'N/A'):,}",
                f"- **Test compounds**: {stats.get('n_test_compounds', 'N/A'):,}",
                f"- **Train pairs (preliminary)**: {stats.get('n_train_pairs_prelim', 'N/A'):,}",
                f"- **Test pairs (preliminary)**: {stats.get('n_test_pairs_prelim', 'N/A'):,}",
                f"- **Train pairs (balanced)**: {stats.get('n_train_pairs_balanced', 'N/A'):,}",
                f"- **Test pairs (balanced)**: {stats.get('n_test_pairs_balanced', 'N/A'):,}",
                "### Balanced Dataset Composition",
                f"- **Train cliffs**: {stats.get('train_n_cliffs', 'N/A'):,}",
                f"- **Train non-cliffs**: {stats.get('train_n_non_cliffs', 'N/A'):,}",
                f"- **Test cliffs**: {stats.get('test_n_cliffs', 'N/A'):,}",
                f"- **Test non-cliffs**: {stats.get('test_n_non_cliffs', 'N/A'):,}",
            ])
    
    # Baseline metrics per split (butina/scaffold)
    for split_name in split_names:
        baseline_metrics_file = model_dir / f"baseline_model_metrics_{split_name}.json"
        if baseline_metrics_file.exists():
            with open(baseline_metrics_file) as f:
                baseline_metrics = json.load(f)
            title = f"Step 5c: Baseline Logistic Regression ({split_name.capitalize()})"
            lines.extend([
                f"### {title}",
                "#### Performance Metrics",
                f"- **Accuracy**: {format_stat(baseline_metrics.get('accuracy', 0))}",
                f"- **Precision**: {format_stat(baseline_metrics.get('precision', 0))}",
                f"- **Recall**: {format_stat(baseline_metrics.get('recall', 0))}",
                f"- **F1-score**: {format_stat(baseline_metrics.get('f1', 0))}",
                f"- **ROC-AUC**: {format_stat(baseline_metrics.get('roc_auc', 0))}",
                f"- **Binary Cross-Entropy**: {format_stat(baseline_metrics.get('binary_cross_entropy', 0))}",
            ])
            coef_name = f"baseline_model_coefficients_{split_name}.json"
            baseline_coefs_file = model_dir / coef_name
            if baseline_coefs_file.exists():
                with open(baseline_coefs_file) as f:
                    baseline_coefs = json.load(f)
                lines.extend([
                    "#### Model Coefficients",
                    f"- **Intercept**: {format_stat(baseline_coefs.get('intercept', 0))}",
                    f"- **Coefficient (distance_scaled)**: {format_stat(baseline_coefs.get('coefficient', 0))}",
                ])

    for split_name in split_names:
        xgboost_metrics_file = model_dir / f"xgboost_model_metrics_{split_name}.json"
        if xgboost_metrics_file.exists():
            with open(xgboost_metrics_file) as f:
                xgboost_metrics = json.load(f)
            lines.extend([
                f"### Step 5d: XGBoost ({split_name.capitalize()})",
                "Classifier on absolute embedding difference |emb₁ − emb₂| (same input representation as Siamese).",
                "#### Performance Metrics",
                f"- **Accuracy**: {format_stat(xgboost_metrics.get('accuracy', 0))}",
                f"- **Precision**: {format_stat(xgboost_metrics.get('precision', 0))}",
                f"- **Recall**: {format_stat(xgboost_metrics.get('recall', 0))}",
                f"- **F1-score**: {format_stat(xgboost_metrics.get('f1', 0))}",
                f"- **ROC-AUC**: {format_stat(xgboost_metrics.get('roc_auc', 0))}",
                f"- **Binary Cross-Entropy**: {format_stat(xgboost_metrics.get('binary_cross_entropy', 0))}",
            ])

    for split_name in split_names:
        siamese_metrics_file = model_dir / f"siamese_model_metrics_{split_name}.json"
        if siamese_metrics_file.exists():
            with open(siamese_metrics_file) as f:
                siamese_metrics = json.load(f)
            title = f"Step 5e: Siamese Network ({split_name.capitalize()})"
            lines.extend([
                f"### {title}",
                "#### Performance Metrics",
                f"- **Accuracy**: {format_stat(siamese_metrics.get('accuracy', 0))}",
                f"- **Precision**: {format_stat(siamese_metrics.get('precision', 0))}",
                f"- **Recall**: {format_stat(siamese_metrics.get('recall', 0))}",
                f"- **F1-score**: {format_stat(siamese_metrics.get('f1', 0))}",
                f"- **ROC-AUC**: {format_stat(siamese_metrics.get('roc_auc', 0))}",
                f"- **Binary Cross-Entropy**: {format_stat(siamese_metrics.get('binary_cross_entropy', 0))}",
            ])

    # Output files table
    lines.extend([
        "### Output Files",
        "| File | Description |",
        "|------|-------------|",
        "| `split_statistics.json` | Train/test split statistics and model metrics (both splits) |",
        "| `train/val/test_compounds_{butina,scaffold}.csv` | Compound splits per split type |",
        "| `train/val/test_pairs_{butina,scaffold}.csv` | Balanced train-train, val-val, test-test pair splits |",
        "| `baseline_model_metrics_butina.json` / `_scaffold.json` | Baseline LR metrics per split |",
        "| `baseline_model_coefficients_butina.json` / `_scaffold.json` | Baseline coefficients per split |",
        "| `xgboost_model_metrics_butina.json` / `_scaffold.json` | XGBoost metrics per split |",
        "| `xgboost_model_butina.json` / `xgboost_model_scaffold.json` | XGBoost model (native format) per split |",
        "| `siamese_model_metrics_butina.json` / `_scaffold.json` | Siamese metrics per split |",
        "| `siamese_model_butina.pt` / `siamese_model_scaffold.pt` | Trained Siamese state per split |",
    ])
    
    return "\n".join(lines)


def create_summary_markdown(
    results_dir: Path,
    dataset_name: str,
    embedding_method: str,
    distance_metric: str,
) -> str:
    """
    Create a complete markdown summary for the whole analysis pipeline.

    This function generates a comprehensive markdown document that includes:
    - Header with analysis metadata
    - Dataset analysis summary (Step 1)
    - Activity cliff identification summary (Step 2)
    - Correlation analysis summary (Step 3)
    - Final summary with key findings

    Args:
        results_dir: Path to the results directory.
        dataset_name: Name of the dataset being analyzed.
        embedding_method: Embedding method used (e.g., 'morgan', 'molformer').
        distance_metric: Distance metric used (e.g., 'tanimoto', 'l1', 'cosine').

    Returns:
        Complete markdown formatted string.
    """
    results_dir = Path(results_dir)

    lines = [
        "# Activity Cliffs Analysis Report",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "---",
        "## Analysis Configuration",
        f"- **Dataset**: {dataset_name}",
        f"- **Embedding Method**: {embedding_method}",
        f"- **Distance Metric**: {distance_metric}",
        f"- **Results Directory**: `{results_dir}`",
        "---",
    ]

    # Add pairwise data creation section
    lines.append(create_pairwise_data_creation_markdown(results_dir))
    lines.append("---")

    # Add cliff identification section
    lines.append(create_cliff_identification_summary_markdown(results_dir))
    lines.append("---")

    # Add correlation analysis section
    lines.append(create_correlation_summary_markdown(results_dir))
    lines.append("---")

    # Add persistent homology analysis section
    lines.append(create_persistent_homology_summary_markdown(results_dir))
    lines.append("---")
    
    # Add model training section
    lines.append(create_model_training_summary_markdown(results_dir))
    lines.append("---")
    
    # Final summary
    lines.extend([
        "## Summary",
        "This report summarizes the activity cliffs analysis pipeline results.",
        "### Key Steps Performed",
        "1. **Pairwise Data Creation**: Computed pairwise distances, scaled distances, "
        "and built distance matrices for both structural and activity dimensions.",
        "2. **Activity Cliff Identification**: Identified activity cliffs; cumulative fraction "
        "F(n) of cliff pairs in the top n% most similar pairs, power-law fit F(n)≈a·n^s (exponent *s*, R²), "
        "and Gini = (s−1)/(s+1); distance-space geometry (CV, dip test, relative contrast, hubness).",
        "3. **Correlation Analysis**: Analyzed the relationship between structural similarity "
        "and activity differences using conditional probability and Lipschitz statistics.",
        "4. **Persistent Homology**: Rips filtration by similarity; H0/H1/H2 persistence diagrams "
        "and long-lived component clusters.",
        "5. **Model Training**: Trained baseline logistic regression and Siamese network models "
        "to predict activity cliffs from molecular embeddings.",
        "---",
        "*Report generated by Activity Cliffs Analysis Pipeline*",
    ])
    
    return "\n".join(lines)


def create_dataset_preparation_summary_markdown(
    results_dir: Path,
) -> str:
    """
    Create a markdown summary for the dataset preparation steps.

    This function summarizes all steps performed during dataset preparation:
    - Step 3: Activity Statistics
    - Step 4: Pairwise Activity Differences (log_ratio)
    - Step 5: Vector Length Statistics
    - Step 6: Embedding Center Statistics
    - Step 7: UMAP Projections

    Args:
        results_dir: Path to the dataset_analysis directory containing preparation results.

    Returns:
        Markdown formatted string for the dataset preparation section.
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        return "## Dataset Preparation\n\n*No dataset preparation results found.*\n"
    
    lines = [
        "## Dataset Preparation Summary",
    ]
    
    # Load preparation metadata (Steps 1-2)
    metadata_file = results_dir / "preparation_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        lines.extend([
            "### Steps 1-2: Embedding Calculation and Data Processing",
            f"- **Input file**: `{metadata.get('input_file', 'N/A')}`",
            f"- **Output file**: `{metadata.get('output_file', 'N/A')}`",
            f"- **Pickle file status**: {'Generated' if metadata.get('pkl_generated', False) else 'Already existed (skipped generation)'}",
            f"- **Number of molecules**: {metadata.get('n_molecules', 'N/A'):,}",
            f"- **SMILES column**: {metadata.get('smiles_column', 'N/A')}",
            f"- **Activity column**: {metadata.get('activity_column', 'N/A')}",
            f"- **Unit**: {metadata.get('unit', 'N/A')} (shift = {metadata.get('shift', 'N/A')})",
            f"- **Censor threshold**: {metadata.get('censor_threshold', 'None')}",
            f"- **Censored compounds**: {metadata.get('n_censored', 0):,}",
            f"- **Uncensored compounds**: {metadata.get('n_uncensored', 0):,}",
            f"- **Fingerprint parameters**: n_bits = {metadata.get('n_bits', 'N/A')}, radius = {metadata.get('radius', 'N/A')}",
            f"- **Embedding methods**: {len(metadata.get('embedding_methods', []))} types ({', '.join(metadata.get('embedding_methods', []))})",
        ])
    
    # Step 3: Activity Statistics
    activity_stats_file = results_dir / "activity_statistics.json"
    if activity_stats_file.exists():
        with open(activity_stats_file) as f:
            activity_data = json.load(f)
        
        lines.extend([
            "### Step 3: Activity Statistics",
        ])
        
        all_activities = activity_data.get("all_activities", {})
        uncensored_activities = activity_data.get("uncensored_activities", {})
        mixture_fit = activity_data.get("mixture_fit", {})
        
        lines.extend([
            f"- **Total compounds**: {all_activities.get('n_compounds', 'N/A'):,}",
            f"- **Uncensored compounds**: {uncensored_activities.get('n_compounds', 'N/A'):,}",
            f"- **Censored compounds**: {activity_data.get('n_censored', 0):,}",
            f"- **Unit**: {activity_data.get('unit', 'N/A')}",
            f"- **Censor threshold**: {activity_data.get('censor_threshold', 'N/A')}",
            "#### Activity Distribution Statistics",
            "**All Compounds:**",
            f"- Mean: {format_stat(all_activities.get('mean', 0))}",
            f"- Median: {format_stat(all_activities.get('median', 0))}",
            f"- Std: {format_stat(all_activities.get('std', 0))}",
            f"- Range: [{format_stat(all_activities.get('min', 0))}, {format_stat(all_activities.get('max', 0))}]",
            "**Uncensored Compounds:**",
            f"- Mean: {format_stat(uncensored_activities.get('mean', 0))}",
            f"- Median: {format_stat(uncensored_activities.get('median', 0))}",
            f"- Std: {format_stat(uncensored_activities.get('std', 0))}",
            "#### Normal-Uniform Mixture Fit",
        ])
        
        if mixture_fit:
            lines.extend([
                f"- **Mixing proportion (p)**: {format_stat(mixture_fit.get('p', 0))}",
                f"- **Normal component**: μ = {format_stat(mixture_fit.get('mu', 0))}, σ = {format_stat(mixture_fit.get('sigma', 0))}",
                f"- **Uniform component**: a = {format_stat(mixture_fit.get('a', 0))}, b = {format_stat(mixture_fit.get('b', 0))}",
                f"- **Log-likelihood**: {format_stat(mixture_fit.get('log_likelihood', 0))}",
                f"- **AIC**: {format_stat(mixture_fit.get('aic', 0))}",
                f"- **BIC**: {format_stat(mixture_fit.get('bic', 0))}",
            ])
        
        img_ref = "activity_statistics.png"
        img_src = img_ref
        lines.extend([
            f"![Activity Statistics]({img_src})",
        ])

    # Step 4: Pairwise Activity Differences
    log_ratio_stats_file = results_dir / "log_ratio_statistics.json"
    if log_ratio_stats_file.exists():
        with open(log_ratio_stats_file) as f:
            log_ratio_data = json.load(f)
        
        lines.extend([
            "### Step 4: Pairwise Activity Differences (log_ratio)",
        ])
        
        stats = log_ratio_data.get("statistics", {})
        kww_fit = log_ratio_data.get("kww_fit", {})
        
        lines.extend([
            f"- **Number of pairs**: {stats.get('n_pairs', 'N/A'):,}",
            f"- **Mean log_ratio**: {format_stat(stats.get('mean', 0))}",
            f"- **Median log_ratio**: {format_stat(stats.get('median', 0))}",
            f"- **Std log_ratio**: {format_stat(stats.get('std', 0))}",
            f"- **Range**: [{format_stat(stats.get('min', 0))}, {format_stat(stats.get('max', 0))}]",
            "#### Kohlrausch-Williams-Watts Decay Fit",
        ])
        
        if kww_fit:
            lines.extend([
                f"- **a**: {format_stat(kww_fit.get('a', 0))}",
                f"- **k**: {format_stat(kww_fit.get('k', 0))}",
                f"- **b**: {format_stat(kww_fit.get('b', 0))}",
                f"- **R²**: {format_stat(kww_fit.get('r_squared', 0))}",
                f"- **Values used**: {kww_fit.get('n_values_used', 0):,}",
            ])
        
        img_ref = f"{log_ratio_stats_file.stem}.png"
        img_src = img_ref
        lines.extend([
            f"![Log Ratio Statistics]({img_src})",
        ])

    # Step 5: Vector Length Statistics
    vector_length_file = results_dir / "vector_length_statistics.json"
    if vector_length_file.exists():
        with open(vector_length_file) as f:
            vector_length_data = json.load(f)
        
        lines.extend([
            "### Step 5: Vector Length Statistics",
            f"- **Number of embedding methods analyzed**: {len(vector_length_data)}",
            "#### Embedding Methods:",
        ])
        
        for emb_name, stats in vector_length_data.items():
            normal_fit = stats.get("normal_fit", {})
            lines.extend([
                f"**{emb_name}**:",
                f"- Mean length: {format_stat(stats.get('mean', 0))}",
                f"- Std length: {format_stat(stats.get('std', 0))}",
                f"- Range: [{format_stat(stats.get('min', 0))}, {format_stat(stats.get('max', 0))}]",
            ])
            if normal_fit:
                lines.append(
                    f"- Gaussian fit: μ = {format_stat(normal_fit.get('mu', 0))}, σ = {format_stat(normal_fit.get('sigma', 0))}"
                )
        
        img_ref = f"{vector_length_file.stem}.png"
        img_src = img_ref
        lines.extend([
            f"![Vector Length Statistics]({img_src})",
        ])

    # Step 6: Embedding Center Statistics
    center_stats_file = results_dir / "embedding_center_statistics.json"
    if center_stats_file.exists():
        with open(center_stats_file) as f:
            center_stats_data = json.load(f)
        
        lines.extend([
            "### Step 6: Embedding Center Statistics",
            f"- **Number of embedding methods analyzed**: {len(center_stats_data)}",
            "#### Embedding Methods:",
        ])
        
        for emb_name, stats in center_stats_data.items():
            lines.extend([
                f"**{emb_name}**:",
                f"- Length of center vector: {format_stat(stats.get('length_of_center', 0))}",
                f"- Length of std vector: {format_stat(stats.get('length_of_std', 0))}",
                f"- Number of dimensions: {stats.get('n_dimensions', 'N/A')}",
                f"- Dimensions with |z_score| > 1: {stats.get('n_dimensions_z_score_gt_1', 0)}",
                f"- Fraction with |z_score| > 1: {format_stat(stats.get('fraction_z_score_gt_1', 0))}",
            ])
        
        img_ref = f"{center_stats_file.stem}.png"
        img_src = img_ref
        lines.extend([
            f"![Embedding Center Statistics]({img_src})",
        ])

    # Step 7: UMAP Projections
    umap_csv_file = results_dir / "umap_projections.csv"
    if umap_csv_file.exists():
        img_ref = f"{umap_csv_file.stem}.png"
        img_src = img_ref
        lines.extend([
            "### Step 7: UMAP Projections",
            f"- **UMAP projections computed for all embedding methods**",
            f"- **Projections saved to**: `{umap_csv_file.name}`",
            f"![UMAP Projections]({img_src})",
        ])
    
    return "\n".join(lines)


def save_summary_markdown(
    results_dir: Path,
    dataset_name: str,
    embedding_method: str,
    distance_metric: str,
    filename: str = "REPORT.md",
) -> Path:
    """
    Generate and save the summary markdown to a file.

    Args:
        results_dir: Path to the results directory.
        dataset_name: Name of the dataset being analyzed.
        embedding_method: Embedding method used (e.g., 'morgan', 'molformer').
        distance_metric: Distance metric used (e.g., 'tanimoto', 'l1', 'cosine').
        filename: Name of the output markdown file.

    Returns:
        Path to the saved markdown file.
    """
    results_dir = Path(results_dir)
    markdown_content = create_summary_markdown(
        results_dir=results_dir,
        dataset_name=dataset_name,
        embedding_method=embedding_method,
        distance_metric=distance_metric,
    )
    
    output_path = results_dir / filename
    with open(output_path, "w") as f:
        f.write(markdown_content)

    return output_path


def generate_comparison_markdown(
    comparison_name: str,
    description: str,
    sections: list[str],
    output_dir: Path,
    console: Console,
) -> Path:
    """
    Assemble comparison report from section strings and write to file.

    Each section is a markdown string produced by a comparison step. Image references
    in sections use filenames relative to output_dir (e.g. `](cliff_fitting_constants_comparison.png)`).

    Returns path to the written markdown file.
    """
    console.print("[cyan]Generating comparison markdown report...[/cyan]")

    # Build content: title, description, then all sections
    parts = [f"# {comparison_name}", ""]
    if description:
        parts.append(description)
        parts.append("")
    parts.append("\n\n".join(sections))

    content = "\n".join(parts)


    report_path = output_dir / "COMPARISON_REPORT.md"
    with open(report_path, "w") as f:
        f.write(content)
    console.print("  [green]✓ Markdown report generated[/green]")
    return report_path
