"""
Comparison module for comparing results from multiple analysis runs.

This module provides functions to compare results across different embedding methods,
distance metrics, and configurations, generating comparison plots and reports.

The compare() pipeline consists of separate steps, each producing a markdown section
and optionally figures; sections are gathered into one markdown report.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.datasets import resolve_dataset_name
from src.plotting import (
    plot_cliff_fitting_constants_comparison,
    plot_cliff_sets_dice_similarity_heatmap,
    plot_comparison_chiral_mmp_2x2,
    plot_comparison_cliffs_5pct,
    plot_comparison_lipschitz_90,
    plot_comparison_persistence_mean_max,
    plot_model_metrics_comparison_grouped,
    plot_persistence_landscape_l2_distance_heatmap,
    plot_persistence_landscape_l2_norm,
    plot_persistence_landscape_mean_and_variance,
)
from src.printing import format_stat, generate_comparison_markdown
from src.utils import load_config


def compare(config_name: str) -> None:
    """
    Compare the results of multiple analysis configurations from a config file.

    Pipeline:
    1. Configurations table
    2. Structural results (incl. p0, BIC scaled fit, distance geometry)
    3. Cliff identification (incl. F(n) power law, sigmoid fits)
    4. Correlation (Lipschitz, KWW + theoretical moments, chiral/MMP tables)
    5. Persistence homology (full dim_0 table + key descriptors, landscapes)
    6. Model performance (ROC AUC matrix + per-split metrics)
    7. Summary

    Loads config from configs/<config_name>.json.

    Args:
        config_name: Name of the config file (without .json extension) in configs/.
    """
    console = Console()
    config = load_config(config_name)

    comparison_name = config.get("name", "Unnamed Comparison")
    description = config.get("description", "")
    if config.get("dataset") is None:
        raise ValueError("Config must contain 'dataset' (a standard dataset name or a path to a CSV file)")
    dataset_name = resolve_dataset_name(config["dataset"])
    configs_to_compare = config.get("configs", [])

    # Print header panel
    console.print()
    console.print(Panel(
        f"[bold]{comparison_name}[/bold]\n\n{description}"
        if description
        else f"[bold]{comparison_name}[/bold]",
        title="🔬 Activity Cliffs Comparison",
        border_style="blue",
    ))

    # Load metadata for each config
    config_data = []
    for entry in configs_to_compare:
        # An entry is a results directory (e.g. an analyze output_dir) or a directory name under results/<dataset>/
        config_path = Path(entry) if Path(entry).is_dir() else Path("results") / dataset_name / entry
        metadata_file = config_path / "analysis_metadata.json"

        if not metadata_file.exists():
            console.print(f"[yellow]⚠ Warning: {metadata_file} not found, skipping[/yellow]")
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        config_data.append({
            "name": config_path.name,
            "path": config_path,
            "embedding_method": metadata.get("embedding_method", "unknown"),
            "distance_metric": metadata.get("distance_metric", "unknown"),
            "metadata": metadata,
        })

    if not config_data:
        console.print("[red]✗ No valid configurations found[/red]")
        return

    # Print configurations table to console
    table = Table(
        title="Configurations to Compare",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Directory", style="yellow")
    table.add_column("Embedding", style="green")
    table.add_column("Distance Metric", style="magenta")

    for i, cfg in enumerate(config_data, 1):
        table.add_row(
            str(i),
            cfg["name"],
            cfg["embedding_method"],
            cfg["distance_metric"],
        )

    console.print()
    console.print(table)
    console.print()

    # Create output directory
    output_dir = Path("results") / dataset_name / "comparison" / comparison_name
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Generating comparison results in: {output_dir}[/cyan]")

    # Run each step and collect markdown sections
    sections: list[str] = []

    sections.append(step_1_configurations_table(config_data, output_dir, console))
    sections.append(step_2_structural_results(config_data, output_dir, console))
    sections.append(step_3_cliff_identification(config_data, output_dir, console))
    sections.append(step_4_correlation(config_data, output_dir, console))
    sections.append(step_5_persistence_homology(config_data, output_dir, console))
    sections.append(step_6_model_performance(config_data, output_dir, console))
    # Generate extra comparison plots first so they exist for the summary
    extra_plots_section = generate_extra_comparison_plots(config_data, output_dir)
    if extra_plots_section:
        sections.append(extra_plots_section)
    sections.append(step_7_summary(config_data, output_dir, console))

    # Assemble full markdown report and write to file
    generate_comparison_markdown(
        comparison_name=comparison_name,
        description=description,
        sections=sections,
        output_dir=output_dir,
        console=console,
    )

    console.print(f"\n[bold green]✓ Comparison complete![/bold green]")
    console.print(f"  Results saved to: {output_dir}")


def _sort_configs(config_data: list[dict]) -> list[dict]:
    """Sort configs by embedding method, then by distance metric."""
    def _key(c: dict) -> tuple[str, str]:
        return (c.get("embedding_method", ""), c.get("distance_metric", ""))
    return sorted(config_data, key=_key)


def generate_extra_comparison_plots(config_data: list[dict], output_dir: Path) -> str:
    """
    Generate four additional comparison figures and return a markdown section that references them.
    """
    sorted_configs = _sort_configs(config_data)
    if not sorted_configs:
        return ""

    def _label(c: dict) -> str:
        return f"{c['embedding_method']} / {c['distance_metric']}"

    labels = [_label(c) for c in sorted_configs]
    embedding_per_label = [c["embedding_method"] for c in sorted_configs]
    n = len(sorted_configs)

    # --- Plot 1: Cliffs in 2% most similar pairs ---
    n_cliffs_5 = []
    for cfg in sorted_configs:
        csv_path = cfg["path"] / "cliff_identification" / "cliff_vs_pair_rank.csv"
        v5 = None
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                if row.get("percent_rank") == 2:
                    v5 = row.get("n_cliffs")
                    break
        n_cliffs_5.append(float(v5) if v5 is not None else np.nan)
    if any(not np.isnan(x) for x in n_cliffs_5):
        plot_comparison_cliffs_5pct(
            labels, n_cliffs_5, embedding_per_label,
            output_dir / "cliff_rank_5pct_comparison.png",
        )

    # --- Plot 2: Lipschitz 90th percentile ---
    lipschitz_90 = []
    for cfg in sorted_configs:
        p = cfg["path"] / "correlation" / "summary.json"
        v = None
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            v = d.get("lipschitz_percentile_90")
        lipschitz_90.append(float(v) if v is not None else np.nan)
    if any(not np.isnan(x) for x in lipschitz_90):
        plot_comparison_lipschitz_90(labels, lipschitz_90, embedding_per_label, output_dir / "lipschitz_p90_comparison.png")

    # --- Plot 3: Mean and max persistence ---
    mean_pers, max_pers = [], []
    for cfg in sorted_configs:
        p = cfg["path"] / "persistent_homology" / "persistence_metadata.json"
        m1 = m2 = None
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            dim0 = (d.get("statistics") or {}).get("dim_0") or {}
            m1 = dim0.get("mean_persistence")
            m2 = dim0.get("max_persistence")
        mean_pers.append(float(m1) if m1 is not None else np.nan)
        max_pers.append(float(m2) if m2 is not None else np.nan)
    if any(not np.isnan(x) for x in mean_pers + max_pers):
        plot_comparison_persistence_mean_max(labels, mean_pers, max_pers, embedding_per_label, output_dir / "persistence_mean_max_comparison.png")

    # --- Plot 4: Chiral / MMP 2x2 (Gaussian fit + std barplots) ---
    # Example config for the Gaussian-fit panels: first L2 config, else the first config
    example_config = next(
        (c for c in sorted_configs if (c.get("distance_metric") or "").lower() == "l2"),
        sorted_configs[0],
    )

    def _load_histogram_and_mean_std(cfg: dict, filename: str) -> tuple[dict | None, tuple[float, float] | None]:
        p = cfg["path"] / "correlation" / filename
        if not p.exists():
            return None, None
        with open(p) as f:
            d = json.load(f)
        hist = d.get("histogram")
        reg = d.get("statistics_regularized_lipschitz") or {}
        mu, sig = reg.get("mean"), reg.get("std")
        return hist, (float(mu), float(sig)) if mu is not None and sig is not None else None

    chiral_hist, chiral_ms = _load_histogram_and_mean_std(example_config, "chiral_cliff_lipschitz_statistics.json")
    mmp_hist, mmp_ms = _load_histogram_and_mean_std(example_config, "mmp_cliff_lipschitz_statistics.json")

    chiral_std_per_config = []
    mmp_std_per_config = []
    for cfg in sorted_configs:
        lab = _label(cfg)
        vc = vm = np.nan
        pc = cfg["path"] / "correlation" / "chiral_cliff_lipschitz_statistics.json"
        pm = cfg["path"] / "correlation" / "mmp_cliff_lipschitz_statistics.json"
        if pc.exists():
            with open(pc) as f:
                d = json.load(f)
            vc = (d.get("statistics_similarity") or d.get("statistics_regularized_lipschitz") or {}).get("std")
        if pm.exists():
            with open(pm) as f:
                d = json.load(f)
            vm = (d.get("statistics_similarity") or d.get("statistics_regularized_lipschitz") or {}).get("std")
        chiral_std_per_config.append((lab, float(vc) if vc is not None else np.nan))
        mmp_std_per_config.append((lab, float(vm) if vm is not None else np.nan))
    plot_comparison_chiral_mmp_2x2(
        chiral_hist, chiral_ms, chiral_std_per_config,
        mmp_hist, mmp_ms, mmp_std_per_config,
        embedding_per_label,
        output_dir / "chiral_mmp_gaussian_std_comparison.png",
    )

    # Markdown section
    lines = ["## Additional comparison plots", ""]
    for name, desc in [
        ("cliff_rank_5pct_comparison.png", "Cliffs in 5% most similar pairs (grouped by embedding)."),
        ("lipschitz_p90_comparison.png", "90th percentile of Lipschitz constant per config."),
        ("persistence_mean_max_comparison.png", "Mean and longest persistence per config (two panels)."),
        ("chiral_mmp_gaussian_std_comparison.png", "Top: chiral analysis — left Gaussian fit of pair distances (example L2 config), right std per config. Bottom: MMP analysis — same layout."),
    ]:
        if (output_dir / name).exists():
            lines.extend([f"**{desc}**", "", f"![{name}]({name})", ""])
    return "\n".join(lines)


def step_1_configurations_table(
    config_data: list[dict], output_dir: Path, console: Console
) -> str:
    """
    Step 1: Gather configurations and produce a table (and pairwise data comparison).

    Returns markdown section for the report.
    """
    console.print("[cyan]Step 1: Configurations table...[/cyan]")

    # Pairwise data comparison (existing logic)
    summaries = []
    for cfg in config_data:
        summary_file = cfg["path"] / "pairwise_data_creation" / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                summary["config_name"] = cfg["name"]
                summary["embedding_method"] = cfg["embedding_method"]
                summary["distance_metric"] = cfg["distance_metric"]
                summaries.append(summary)

    if summaries:
        comparison_data = {
            "n_configs": len(summaries),
            "summaries": summaries,
        }
        with open(output_dir / "pairwise_data_comparison.json", "w") as f:
            json.dump(comparison_data, f, indent=2)
        console.print(f"  [green]✓ Compared {len(summaries)} configurations (pairwise data)[/green]")
    else:
        console.print("  [yellow]⚠ No pairwise data summaries found[/yellow]")

    # Build markdown section: one flat table, sorted by embedding_method
    sorted_configs = sorted(config_data, key=lambda c: c["embedding_method"])
    lines = [
        "## 1. Configurations",
        "",
        "| # | Directory | Embedding Method | Distance Metric |",
        "|---|-----------|------------------|-----------------|",
    ]
    for i, cfg in enumerate(sorted_configs, 1):
        lines.append(
            f"| {i} | `{cfg['name']}` | {cfg['embedding_method']} | {cfg['distance_metric']} |"
        )
    lines.append("")
    if summaries:
        lines.append("See `pairwise_data_comparison.json` for pairwise data creation statistics.")
    else:
        lines.append("*No pairwise data comparison available.*")
    return "\n".join(lines)


def _count_csv_data_rows(csv_path: Path) -> int:
    """Return number of data rows (excluding header) in a CSV file. Returns 0 if missing or empty."""
    if not csv_path.exists():
        return 0
    with open(csv_path) as f:
        n_lines = sum(1 for _ in f)
    return max(0, n_lines - 1)


def _read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _scaled_bic_distribution_label(pairwise_dir: Path, summary: dict) -> str:
    """BIC-selected distribution for scaled distances (matches single-run pairwise report)."""
    name = summary.get("scaled_distance_best_bic_distribution")
    d = _read_json_if_exists(pairwise_dir / "scaled_distance_distribution_fits.json")
    best_bic = d.get("best_bic") if d else None
    if not name and best_bic is None:
        return "—"
    if best_bic is not None:
        try:
            bb = float(best_bic)
            if np.isfinite(bb):
                return f"{name or 'unknown'} (BIC={format_stat(bb)})"
        except (TypeError, ValueError):
            pass
    return str(name) if name else "—"


def _distance_geometry_for_config(cfg_path: Path, cliff_summary: dict | None) -> dict | None:
    if cliff_summary:
        dg = cliff_summary.get("distance_geometry")
        if dg:
            return dg
        rel = cliff_summary.get("distance_geometry_statistics_file")
        if rel:
            loaded = _read_json_if_exists(cfg_path / "cliff_identification" / rel)
            if loaded:
                return loaded
    return _read_json_if_exists(cfg_path / "cliff_identification" / "distance_geometry_statistics.json")


def _unimodality_result_label(dg: dict | None) -> str:
    if not dg:
        return "—"
    dip = dg.get("hartigan_dip_test") or {}
    if not dip.get("available"):
        return "—"
    p = dip.get("p_value")
    if p is None:
        return "—"
    pv = float(p)
    verdict = "reject unimodality (multimodal)" if pv < 0.05 else "unimodality not rejected"
    return f"{verdict}; p={format_stat(pv)}"


def _append_chiral_mmp_comparison_table(
    lines: list[str],
    config_data: list[dict],
    filename: str,
    title: str,
    *,
    include_zero_distance: bool,
) -> None:
    """Per-config table: pair counts, Beta-fit m, f, variance, CV; chiral adds scaled-distance-zero stats."""
    sorted_configs = sorted(config_data, key=lambda c: c["embedding_method"])
    if include_zero_distance:
        header = (
            "| Embedding | Distance metric | N total pairs | N cliff pairs | m | f | Variance | CV | "
            "N dist_scaled=0 | Fraction dist_scaled=0 (%) |"
        )
        sep = "|-----------|-----------------|---------------|---------------|---|---|----------|----|------------------|---------------------------|"
    else:
        header = (
            "| Embedding | Distance metric | N total pairs | N cliff pairs | m | f | Variance | CV |"
        )
        sep = "|-----------|-----------------|---------------|---------------|---|---|----------|----|"

    row_lines: list[str] = []
    for cfg in sorted_configs:
        p = cfg["path"] / "correlation" / filename
        d = _read_json_if_exists(p)
        if d is None:
            continue
        em, dm = cfg["embedding_method"], cfg["distance_metric"]
        n_tot = d.get("n_total_pairs")
        n_cl = d.get("n_pairs")
        beta = d.get("distance_beta_fit") or {}
        m = beta.get("m")
        fpar = beta.get("f")
        var = beta.get("variance")
        cv = beta.get("cv")
        n_tot_s = f"{int(n_tot):,}" if isinstance(n_tot, int) else ("—" if n_tot is None else str(n_tot))
        n_cl_s = f"{int(n_cl):,}" if isinstance(n_cl, int) else ("—" if n_cl is None else str(n_cl))
        cells = [
            em,
            dm,
            n_tot_s,
            n_cl_s,
            format_stat(m) if m is not None else "—",
            format_stat(fpar) if fpar is not None else "—",
            format_stat(var) if var is not None else "—",
            format_stat(cv) if cv is not None else "—",
        ]
        if include_zero_distance:
            nz = d.get("n_pairs_scaled_distance_zero")
            fz = d.get("fraction_scaled_distance_zero")
            if nz is None:
                cells.extend(["—", "—"])
            else:
                cells.append(f"{int(nz):,}")
                if fz is not None:
                    cells.append(format_stat(100.0 * float(fz)))
                else:
                    cells.append("—")
        row_lines.append("| " + " | ".join(cells) + " |")

    lines.extend([f"### {title}", ""])
    if not row_lines:
        lines.extend(["*No data available for this subsection.*", ""])
        return
    lines.append(header)
    lines.append(sep)
    lines.extend(row_lines)
    lines.append("")


def step_2_structural_results(
    config_data: list[dict], output_dir: Path, console: Console
) -> str:
    """
    Step 2: Comparison of the structural results.

    Three tables: (1) general dataset info from first config, (2) log_ratio stats + same-activity
    pair count from first config, (3) per-config distance stats + distance-0 pair count, sorted by embedding.
    """
    console.print("[cyan]Step 2: Structural results comparison...[/cyan]")

    lines = ["## 2. Structural Results", ""]

    if not config_data:
        lines.append("*No configurations available.*")
        return "\n".join(lines)

    # Use first config for tables 1 and 2 (same data across configs)
    first_path = config_data[0]["path"]
    summary_path = first_path / "pairwise_data_creation" / "summary.json"
    if not summary_path.exists():
        lines.append("*No pairwise data summary found (summary.json missing).*")
        return "\n".join(lines)

    with open(summary_path) as f:
        summary = json.load(f)
    dist_stats = summary.get("distance_statistics") or {}

    # Table 1: General dataset information
    lines.extend([
        "### General dataset information",
        "",
        "| Statistic | Value |",
        "|-----------|-------|",
        f"| Number of compounds | {summary.get('n_compounds', '—'):,} |",
        f"| Uncensored compounds | {summary.get('n_compounds_uncensored', '—'):,} |",
        f"| Censored compounds | {summary.get('n_compounds_censored', '—'):,} |",
        f"| Total pairs | {summary.get('n_pairs', '—'):,} |",
        f"| Correct pairs | {summary.get('n_pairs_correct', '—'):,} |",
        f"| Censored pairs | {summary.get('n_pairs_censored', '—'):,} |",
        f"| Excluded pairs | {summary.get('n_pairs_excluded', '—'):,} |",
        "",
    ])

    # Table 2: Log-ratio statistics + number of pairs with same activity (from first config)
    n_same_activity = _count_csv_data_rows(first_path / "pairwise_data_creation" / "activity_similar_pairs.csv")
    lines.extend([
        "### Log-ratio statistics",
        "",
        "| Statistic | Value |",
        "|-----------|-------|",
        f"| Min | {format_stat(dist_stats.get('log_ratio_min', 0))} |",
        f"| Max | {format_stat(dist_stats.get('log_ratio_max', 0))} |",
        f"| Mean | {format_stat(dist_stats.get('log_ratio_mean', 0))} |",
        f"| Median | {format_stat(dist_stats.get('log_ratio_median', 0))} |",
        f"| Std | {format_stat(dist_stats.get('log_ratio_std', 0))} |",
        f"| Number of pairs with same activity | {n_same_activity:,} |",
        "",
    ])

    # Structural comparison: p0, BIC on scaled distances, distance geometry (cliff identification step)
    sorted_for_struct = sorted(config_data, key=lambda c: c["embedding_method"])
    struct_rows: list[tuple[str, str, float | None, str, float | None, float | None, str, float | None]] = []
    for cfg in sorted_for_struct:
        path = cfg["path"]
        s_path = path / "pairwise_data_creation" / "summary.json"
        n_zero = _count_csv_data_rows(path / "pairwise_data_creation" / "structurally_similar_pairs.csv")
        ssum = _read_json_if_exists(s_path) or {}
        ds = ssum.get("distance_statistics") or {}
        n_valid = ds.get("n_pairs_valid")
        p0 = None
        if isinstance(n_valid, (int, float)) and n_valid > 0:
            p0 = 100.0 * n_zero / float(n_valid)
        pairwise_dir = path / "pairwise_data_creation"
        bic_lab = _scaled_bic_distribution_label(pairwise_dir, ssum)
        cliff_s = _read_json_if_exists(path / "cliff_identification" / "summary.json")
        dg = _distance_geometry_for_config(path, cliff_s)
        cv_blk = (dg or {}).get("coefficient_of_variation") or {}
        cv_val = cv_blk.get("coefficient_of_variation_std_over_mean")
        rc_val = ((dg or {}).get("relative_contrast") or {}).get("median")
        uni = _unimodality_result_label(dg)
        skew = ((dg or {}).get("hubness") or {}).get("skewness")
        struct_rows.append((
            cfg["embedding_method"],
            cfg["distance_metric"],
            p0,
            bic_lab,
            float(cv_val) if cv_val is not None else None,
            float(rc_val) if rc_val is not None else None,
            uni,
            float(skew) if skew is not None else None,
        ))

    lines.extend([
        "### Structural comparison (scaled distances and geometry)",
        "",
        "Column **p0 (%)** is the fraction of valid pairs with structural distance exactly zero. "
        "**BIC distribution (scaled)** is the likelihood-selected model for the scaled distance histogram "
        "(same as the single-run pairwise report). CV, RC (median relative contrast), Hartigan dip test, "
        "and hubness skewness refer to the cliff-identification distance-geometry block (typically on scaled distances).",
        "",
        "| Embedding | Distance metric | p0 (%) | BIC distribution (scaled) | CV | RC (median) | Unimodality (dip) | Skewness N5 (k=5) |",
        "|-----------|-----------------|--------|---------------------------|-----|-------------|-------------------|------------------|",
    ])
    for em, dm, p0, bic_lab, cv_val, rc_val, uni, skew in struct_rows:
        lines.append(
            "| "
            + " | ".join([
                em,
                dm,
                format_stat(p0) if p0 is not None else "—",
                bic_lab,
                format_stat(cv_val) if cv_val is not None else "—",
                format_stat(rc_val) if rc_val is not None else "—",
                uni,
                format_stat(skew) if skew is not None else "—",
            ])
            + " |"
        )
    lines.append("")

    # Table 3: Per-config distance statistics, sorted by embedding_method (no directory column)
    sorted_configs = sorted(config_data, key=lambda c: c["embedding_method"])
    rows = []
    for cfg in sorted_configs:
        path = cfg["path"]
        s_path = path / "pairwise_data_creation" / "summary.json"
        n_dist_zero = _count_csv_data_rows(path / "pairwise_data_creation" / "structurally_similar_pairs.csv")
        if not s_path.exists():
            rows.append((cfg["embedding_method"], cfg["distance_metric"], "—", "—", "—", "—", "—", n_dist_zero))
            continue
        with open(s_path) as f:
            s = json.load(f)
        ds = s.get("distance_statistics") or {}
        rows.append((
            cfg["embedding_method"],
            cfg["distance_metric"],
            format_stat(ds.get("distance_min", 0)),
            format_stat(ds.get("distance_max", 0)),
            format_stat(ds.get("distance_mean", 0)),
            format_stat(ds.get("distance_median", 0)),
            format_stat(ds.get("distance_std", 0)),
            n_dist_zero,
        ))

    lines.extend([
        "### Distance statistics (by config)",
        "",
        "| Embedding | Distance metric | Min | Max | Mean | Median | Std | Pairs with distance 0 |",
        "|-----------|-----------------|-----|-----|------|--------|-----|------------------------|",
    ])
    for row in rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]:,} |")
    lines.append("")

    console.print("  [green]✓ Structural results tables generated[/green]")
    return "\n".join(lines)


def step_3_cliff_identification(
    config_data: list[dict], output_dir: Path, console: Console
) -> str:
    """
    Step 3: Comparison of cliffs identified across configurations.

    Tables: (1) number of cliffs at similarity 0.0, 0.5, 0.7, 0.9; (2–4) fit parameters
    for activity_cliff_fraction, n_pairs, cliff_fraction_vs_pair_rank. All sorted by embedding_method.
    """
    console.print("[cyan]Step 3: Cliff identification comparison...[/cyan]")

    # Load cliff_identification/summary.json per config (for tables) and fit_parameters.json (for plot)
    sorted_configs = sorted(config_data, key=lambda c: c["embedding_method"])
    cliff_summaries: list[dict] = []
    for cfg in sorted_configs:
        summary_path = cfg["path"] / "cliff_identification" / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            s["_embedding_method"] = cfg["embedding_method"]
            s["_distance_metric"] = cfg["distance_metric"]
            cliff_summaries.append(s)

    fit_params_list = []
    for cfg in config_data:
        fit_file = cfg["path"] / "cliff_identification" / "fit_parameters.json"
        if fit_file.exists():
            with open(fit_file) as f:
                fit_params = json.load(f)
                fit_params["config_name"] = cfg["name"]
                fit_params["embedding_method"] = cfg["embedding_method"]
                fit_params["distance_metric"] = cfg["distance_metric"]
                fit_params_list.append(fit_params)

    if fit_params_list:
        comparison_data = {
            "n_configs": len(fit_params_list),
            "fit_parameters": fit_params_list,
        }
        with open(output_dir / "cliff_identification_comparison.json", "w") as f:
            json.dump(comparison_data, f, indent=2)
        plot_cliff_fitting_constants_comparison(fit_params_list, output_dir)
        console.print(f"  [green]✓ Compared {len(fit_params_list)} configurations[/green]")

    lines = ["## 3. Cliff Identification", ""]

    if not cliff_summaries:
        lines.append("*No cliff identification summaries found.*")
        if fit_params_list:
            lines.extend([
                "",
                "#### Fitting Constants Comparison",
                "![Cliff Fitting Constants Comparison](cliff_fitting_constants_comparison.png)",
                "",
                "See `cliff_identification_comparison.json` for detailed fit parameters.",
            ])
        return "\n".join(lines)

    # Table 1: Number of cliffs at similarity 0.0, 0.5, 0.7, 0.9 — bold lowest in each column
    sim_keys = ["cliffs_at_similarity_0.0", "cliffs_at_similarity_0.5", "cliffs_at_similarity_0.7", "cliffs_at_similarity_0.9"]
    sim_labels = ["0.0", "0.5", "0.7", "0.9"]
    values_per_col: list[list[int]] = [[] for _ in sim_keys]
    for s in cliff_summaries:
        for i, k in enumerate(sim_keys):
            values_per_col[i].append(s.get(k, 0))
    best_row_per_col = [values_per_col[i].index(min(values_per_col[i])) for i in range(len(sim_keys))]

    lines.extend([
        "### Number of identified cliffs by similarity threshold",
        "",
        "| Embedding | Distance metric | 0.0 | 0.5 | 0.7 | 0.9 |",
        "|-----------|-----------------|-----|-----|-----|-----|",
    ])
    for r, s in enumerate(cliff_summaries):
        em, dm = s["_embedding_method"], s["_distance_metric"]
        cells = [em, dm]
        for i, k in enumerate(sim_keys):
            val = s.get(k, 0)
            cell = f"**{val:,}**" if r == best_row_per_col[i] else f"{val:,}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("*Bold: lowest value in each similarity column.*")
    lines.append("")

    # Cumulative F(n) power-law (Lorentz-style enrichment curve in report): s, R², Gini
    lines.extend([
        "### Cumulative cliff fraction F(n) — power-law fit (exponent s, R², Gini)",
        "",
        "F(n) is the fraction of all cliff pairs that fall in the top n% most similar pairs; "
        "fit F(n) ≈ a·n^s. Gini = (s−1)/(s+1) from the fitted s.",
        "",
        "| Embedding | Distance metric | s | R² | Gini |",
        "|-----------|-----------------|---|---|------|",
    ])
    for s in cliff_summaries:
        pl = (s.get("fit_parameters") or {}).get("cumulative_Fn_power_law") or {}
        s_exp = pl.get("s")
        r2 = pl.get("r_squared")
        gini = pl.get("gini_coefficient")
        lines.append(
            "| "
            + " | ".join([
                s["_embedding_method"],
                s["_distance_metric"],
                format_stat(s_exp) if s_exp is not None else "—",
                format_stat(r2) if r2 is not None else "—",
                format_stat(gini) if gini is not None else "—",
            ])
            + " |"
        )
    lines.append("")

    # Dice similarity of cliff sets at 95% (key "5") — heatmap
    cliff_sets_list: list[set[frozenset[str]]] = []
    dice_config_labels: list[str] = []
    for cfg in sorted_configs:
        path = cfg["path"] / "cliff_identification" / "activity_cliff_sets.json"
        dice_config_labels.append(f"{cfg['embedding_method']} / {cfg['distance_metric']}")
        if not path.exists():
            cliff_sets_list.append(set())
            continue
        with open(path) as f:
            data = json.load(f)
        raw = data.get("5", [])  # 95% structural similarity threshold
        cliff_sets_list.append(
            {frozenset(pair) for pair in raw if len(pair) == 2}
        )
    n_cfg = len(cliff_sets_list)
    if n_cfg >= 2:
        dice_matrix = np.zeros((n_cfg, n_cfg))
        for i in range(n_cfg):
            for j in range(n_cfg):
                a, b = cliff_sets_list[i], cliff_sets_list[j]
                if len(a) + len(b) == 0:
                    dice_matrix[i, j] = np.nan
                else:
                    dice_matrix[i, j] = 2.0 * len(a & b) / (len(a) + len(b))
        plot_cliff_sets_dice_similarity_heatmap(
            dice_config_labels,
            dice_matrix,
            output_dir,
        )
        lines.extend([
            "### Dice similarity of cliff sets (95% structural similarity)",
            "",
            "![Cliff sets Dice similarity](cliff_sets_dice_similarity_heatmap.png)",
            "",
        ])

    # Tables 2–4: Sigmoid fits (same parameterization as single-run report)
    fit_names = [
        ("activity_cliff_fraction", "Sigmoid — activity cliff fraction vs similarity"),
        ("n_pairs", "Sigmoid — number of pairs vs similarity"),
        ("cliff_fraction_vs_pair_rank", "Sigmoid — cliff fraction vs top-n% pair rank"),
    ]
    for fit_key, fit_title in fit_names:
        rows_data: list[tuple[str, str, float, float, float, float]] = []
        for s in cliff_summaries:
            fp = (s.get("fit_parameters") or {}).get(fit_key) or {}
            c = fp.get("c")
            a = fp.get("a")
            k = fp.get("k")
            x0 = fp.get("x0")
            if c is None and a is None and k is None and x0 is None:
                rows_data.append((s["_embedding_method"], s["_distance_metric"], float("nan"), float("nan"), float("nan"), float("nan")))
            else:
                rows_data.append((
                    s["_embedding_method"],
                    s["_distance_metric"],
                    c if c is not None else float("nan"),
                    a if a is not None else float("nan"),
                    k if k is not None else float("nan"),
                    x0 if x0 is not None else float("nan"),
                ))
        # Indices of best row per column (cols 2–5 = c, a, k, x0); skip nan
        def _not_nan(x: float) -> bool:
            return x == x  # nan != nan

        valid_c = [(r, row[2]) for r, row in enumerate(rows_data) if _not_nan(row[2])]
        valid_a = [(r, row[3]) for r, row in enumerate(rows_data) if _not_nan(row[3])]
        valid_k = [(r, row[4]) for r, row in enumerate(rows_data) if _not_nan(row[4])]
        valid_x0 = [(r, row[5]) for r, row in enumerate(rows_data) if _not_nan(row[5])]
        best_c = min(valid_c, key=lambda x: x[1])[0] if valid_c else -1
        best_a = min(valid_a, key=lambda x: x[1])[0] if valid_a else -1
        best_k = max(valid_k, key=lambda x: x[1])[0] if valid_k else -1
        best_x0 = min(valid_x0, key=lambda x: x[1])[0] if valid_x0 else -1

        lines.extend([
            f"### {fit_title}",
            "",
            "| Embedding | Distance metric | c | a | k | x0 |",
            "|-----------|-----------------|---|---|---|-----|",
        ])
        for r, row in enumerate(rows_data):
            em, dm = row[0], row[1]
            cell_c = format_stat(row[2]) if row[2] == row[2] else "—"
            cell_a = format_stat(row[3]) if row[3] == row[3] else "—"
            cell_k = format_stat(row[4]) if row[4] == row[4] else "—"
            cell_x0 = format_stat(row[5]) if row[5] == row[5] else "—"
            if r == best_c:
                cell_c = f"**{cell_c}**"
            if r == best_a:
                cell_a = f"**{cell_a}**"
            if r == best_k:
                cell_k = f"**{cell_k}**"
            if r == best_x0:
                cell_x0 = f"**{cell_x0}**"
            lines.append(f"| {em} | {dm} | {cell_c} | {cell_a} | {cell_k} | {cell_x0} |")
        lines.append("")
        lines.append("*Best: smallest c, smallest a, largest k, smallest x0.*")
        lines.append("")

    if fit_params_list:
        lines.extend([
            "#### Fitting Constants Comparison",
            "![Cliff Fitting Constants Comparison](cliff_fitting_constants_comparison.png)",
            "",
            "See `cliff_identification_comparison.json` for detailed fit parameters.",
        ])
    return "\n".join(lines)


def step_4_correlation(
    config_data: list[dict], output_dir: Path, console: Console
) -> str:
    """
    Step 4: Comparison of correlation between activity and distance.

    Table: Lipschitz percentiles (90, 99) and KWW fit (a, k, b, r_squared) per config.
    Best: lowest lip_90, lip_99; highest k, b.
    """
    console.print("[cyan]Step 4: Correlation (activity vs distance) comparison...[/cyan]")

    summaries = []
    for cfg in config_data:
        summary_file = cfg["path"] / "correlation" / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                summary["config_name"] = cfg["name"]
                summary["embedding_method"] = cfg["embedding_method"]
                summary["distance_metric"] = cfg["distance_metric"]
                summaries.append(summary)

    if not summaries:
        console.print("  [yellow]⚠ No correlation summaries found[/yellow]")
        return "\n".join([
            "## 4. Correlation (Activity vs Distance)",
            "",
            "*No correlation comparison available.*",
        ])

    comparison_data = {
        "n_configs": len(summaries),
        "summaries": summaries,
    }
    with open(output_dir / "correlation_comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=2)
    console.print(f"  [green]✓ Compared {len(summaries)} configurations[/green]")

    # Sort by embedding_method and build table: Lipschitz percentiles + KWW fit
    sorted_summaries = sorted(summaries, key=lambda s: s["embedding_method"])
    rows: list[tuple[str, str, float | None, float | None, float | None, float | None, float | None, float | None]] = []
    for s in sorted_summaries:
        lip_90 = s.get("lipschitz_percentile_90")
        lip_99 = s.get("lipschitz_percentile_99")
        kww = s.get("kww_fit") or {}
        a, k, b, r2 = kww.get("a"), kww.get("k"), kww.get("b"), kww.get("r_squared")
        rows.append((
            s["embedding_method"],
            s["distance_metric"],
            float(lip_90) if lip_90 is not None else None,
            float(lip_99) if lip_99 is not None else None,
            float(a) if a is not None else None,
            float(k) if k is not None else None,
            float(b) if b is not None else None,
            float(r2) if r2 is not None else None,
        ))

    def _not_nan(x): return x is not None and (x == x if isinstance(x, float) else True)
    # Best row index: lowest lip_90, lowest lip_99; highest k, highest b
    valid_lip90 = [(i, r[2]) for i, r in enumerate(rows) if _not_nan(r[2])]
    valid_lip99 = [(i, r[3]) for i, r in enumerate(rows) if _not_nan(r[3])]
    valid_k = [(i, r[5]) for i, r in enumerate(rows) if _not_nan(r[5])]
    valid_b = [(i, r[6]) for i, r in enumerate(rows) if _not_nan(r[6])]
    valid_r2 = [(i, r[7]) for i, r in enumerate(rows) if _not_nan(r[7])]
    best_lip90 = min(valid_lip90, key=lambda x: x[1])[0] if valid_lip90 else -1
    best_lip99 = min(valid_lip99, key=lambda x: x[1])[0] if valid_lip99 else -1
    best_k = max(valid_k, key=lambda x: x[1])[0] if valid_k else -1
    best_b = max(valid_b, key=lambda x: x[1])[0] if valid_b else -1
    best_r2 = max(valid_r2, key=lambda x: x[1])[0] if valid_r2 else -1

    lines = [
        "## 4. Correlation (Activity vs Distance)",
        "",
        "### Lipschitz and KWW fit comparison",
        "",
        "| Embedding | Distance metric | Lip. 90% | Lip. 99% | KWW a | KWW k | KWW b | KWW R² |",
        "|-----------|-----------------|----------|----------|------|-------|-------|--------|",
    ]
    for r, row in enumerate(rows):
        em, dm = row[0], row[1]
        lip90_s = format_stat(row[2]) if row[2] is not None else "—"
        lip99_s = format_stat(row[3]) if row[3] is not None else "—"
        a_s = format_stat(row[4]) if row[4] is not None else "—"
        k_s = format_stat(row[5]) if row[5] is not None else "—"
        b_s = format_stat(row[6]) if row[6] is not None else "—"
        r2_s = format_stat(row[7]) if row[7] is not None else "—"
        if r == best_lip90:
            lip90_s = f"**{lip90_s}**"
        if r == best_lip99:
            lip99_s = f"**{lip99_s}**"
        if r == best_k:
            k_s = f"**{k_s}**"
        if r == best_b:
            b_s = f"**{b_s}**"
        if r == best_r2:
            r2_s = f"**{r2_s}**"
        lines.append(f"| {em} | {dm} | {lip90_s} | {lip99_s} | {a_s} | {k_s} | {b_s} | {r2_s} |")
    lines.extend([
        "",
        "*Best: lowest Lipschitz 90% and 99%; highest KWW k and b.*",
        "",
    ])

    # KWW / Weibull survival fit and theoretical moments (full valid-pair Lipschitz tail)
    lines.extend([
        "### KWW survival fit and theoretical moments (Lipschitz)",
        "",
        "Kohlrausch–Williams–Watts decay on the empirical survival of regularized Lipschitz values; "
        "theoretical mean and 95th percentile follow the Weibull interpretation (see single-run correlation report).",
        "",
        "| Embedding | Distance metric | a | k | b | Mean E[r] | r95 | R² |",
        "|-----------|-----------------|---|---|---|-----------|-----|-----|",
    ])
    for s in sorted_summaries:
        kww = s.get("kww_fit") or {}
        th = s.get("kww_theoretical") or {}
        a, k, b, r2 = kww.get("a"), kww.get("k"), kww.get("b"), kww.get("r_squared")
        mean_er = th.get("mean_Er")
        r95 = th.get("r95")
        lines.append(
            "| "
            + " | ".join([
                s["embedding_method"],
                s["distance_metric"],
                format_stat(a) if a is not None else "—",
                format_stat(k) if k is not None else "—",
                format_stat(b) if b is not None else "—",
                format_stat(mean_er) if mean_er is not None else "—",
                format_stat(r95) if r95 is not None else "—",
                format_stat(r2) if r2 is not None else "—",
            ])
            + " |"
        )
    lines.append("")

    _append_chiral_mmp_comparison_table(
        lines,
        config_data,
        "chiral_cliff_lipschitz_statistics.json",
        "Chiral pairs — counts and Beta fit on scaled distance",
        include_zero_distance=True,
    )
    _append_chiral_mmp_comparison_table(
        lines,
        config_data,
        "mmp_cliff_lipschitz_statistics.json",
        "Matched molecular pairs — counts and Beta fit on scaled distance",
        include_zero_distance=False,
    )

    lines.append("See `correlation_comparison.json` for detailed statistics.")
    return "\n".join(lines)


def step_5_persistence_homology(
    config_data: list[dict], output_dir: Path, console: Console
) -> str:
    """
    Step 5: Comparison of persistence homology results.

    Table of dim_0 statistics and n_persistent_clusters per config (sorted by embedding).
    """
    console.print("[cyan]Step 5: Persistence homology comparison...[/cyan]")

    # Keys from statistics.dim_0 to include (order preserved for table)
    dim_0_keys = [
        "n_features", "n_finite", "n_infinite",
        "max_persistence", "mean_persistence", "median_persistence", "std_persistence",
        "total_persistence", "max_birth", "max_death",
    ]
    sorted_configs = sorted(config_data, key=lambda c: c["embedding_method"])
    rows: list[tuple[str, str, list[float | int | None], int | None]] = []

    for cfg in sorted_configs:
        path = cfg["path"] / "persistent_homology" / "persistence_metadata.json"
        if not path.exists():
            rows.append((cfg["embedding_method"], cfg["distance_metric"], [None] * len(dim_0_keys), None))
            continue
        with open(path) as f:
            data = json.load(f)
        stats_dim0 = (data.get("statistics") or {}).get("dim_0") or {}
        vals = [stats_dim0.get(k) for k in dim_0_keys]
        vals = [v if isinstance(v, (int, float)) else None for v in vals]
        n_persistent = data.get("n_persistent_clusters")
        if n_persistent is not None and not isinstance(n_persistent, int):
            n_persistent = int(n_persistent) if isinstance(n_persistent, (float, str)) else None
        rows.append((cfg["embedding_method"], cfg["distance_metric"], vals, n_persistent))

    lines = ["## 5. Persistence Homology", ""]
    if not rows:
        lines.append("*No persistence metadata found.*")
        return "\n".join(lines)

    # Table header: Embedding | Distance metric | dim_0 stats... | n_persistent_clusters
    header_cells = ["Embedding", "Distance metric"] + [k.replace("_", " ") for k in dim_0_keys] + ["n_persistent_clusters"]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")

    for em, dm, vals, n_pers in rows:
        cells = [em, dm]
        for v in vals:
            cells.append(format_stat(v) if v is not None else "—")
        cells.append(str(n_pers) if n_pers is not None else "—")
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines.append("")

    lines.extend([
        "### Key persistence descriptors (H₀)",
        "",
        "| Embedding | Distance metric | Landscape L2 | Mean persistence | Max persistence | Persistence entropy |",
        "|-----------|-----------------|---------------|------------------|----------------|---------------------|",
    ])
    for cfg in sorted_configs:
        meta_path = cfg["path"] / "persistent_homology" / "persistence_metadata.json"
        pdata = _read_json_if_exists(meta_path)
        dim0 = ((pdata or {}).get("statistics") or {}).get("dim_0") or {}
        l2 = dim0.get("landscape_l2_norm")
        if l2 is None and pdata:
            l2 = ((pdata.get("landscape_l2_norm") or {}).get("dim_0"))
        mp = dim0.get("mean_persistence")
        mx = dim0.get("max_persistence")
        ent = dim0.get("persistent_entropy")
        lines.append(
            "| "
            + " | ".join([
                cfg["embedding_method"],
                cfg["distance_metric"],
                format_stat(l2) if l2 is not None else "—",
                format_stat(mp) if mp is not None else "—",
                format_stat(mx) if mx is not None else "—",
                format_stat(ent) if ent is not None else "—",
            ])
            + " |"
        )
    lines.append("")

    # Persistence landscape: Var(λ₁(t)) across configs using persim
    try:
        from persim import PersLandscapeApprox
    except ImportError:
        console.print("  [yellow]⚠ persim not installed; skipping persistence landscape variance[/yellow]")
    else:
        csv_name = "persistence_dim0.csv"
        num_steps = 500
        start = 0.0
        stop = 1.0  # death cannot exceed 1
        config_dgms: list[tuple[dict, np.ndarray]] = []

        for cfg in sorted_configs:
            csv_path = cfg["path"] / "persistent_homology" / csv_name
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if df.empty or "birth" not in df.columns or "death" not in df.columns:
                continue
            birth = df["birth"].to_numpy(dtype=float)
            death = df["death"].to_numpy(dtype=float)
            death = np.where(np.isfinite(death), death, 1.0)  # inf -> 1 (death cannot exceed 1)
            dgm = np.column_stack((birth, death))
            if dgm.size == 0:
                continue
            config_dgms.append((cfg, dgm))

        if config_dgms:
            lambda1_list = []
            l2_norms: list[float] = []
            configs_for_landscape: list[dict] = []
            landscapes_list: list = []  # PersLandscapeApprox per config, for pairwise L2 distance
            for cfg, dgm in config_dgms:
                pla = PersLandscapeApprox(dgms=[dgm], hom_deg=0, start=start, stop=stop, num_steps=num_steps)
                lambda1_list.append(pla[0])
                l2_norms.append(float(pla.p_norm(p=2)))
                configs_for_landscape.append(cfg)
                landscapes_list.append(pla)

            # L2 norm table and bar plot (for any number of configs)
            lines.extend([
                "### L2 norm of persistence landscape",
                "",
                "| Embedding | Distance metric | L2 norm |",
                "|-----------|-----------------|--------|",
            ])
            for cfg, norm in zip(configs_for_landscape, l2_norms):
                lines.append(f"| {cfg['embedding_method']} | {cfg['distance_metric']} | {format_stat(norm)} |")
            lines.append("")
            short_labels = [f"{c['embedding_method']} / {c['distance_metric']}" for c in configs_for_landscape]
            embedding_methods = [c["embedding_method"] for c in configs_for_landscape]
            plot_persistence_landscape_l2_norm(
                short_labels,
                l2_norms,
                embedding_methods,
                output_dir / "persistence_landscape_l2_norm.png",
            )
            lines.extend([
                "![L2 norm of persistence landscape](persistence_landscape_l2_norm.png)",
                "",
            ])
            console.print("  [green]✓ Persistence landscape L2 norm table and plot generated[/green]")

            # Pairwise L2 distance between landscapes (heatmap) when ≥2 configs
            if len(landscapes_list) >= 2:
                n_pl = len(landscapes_list)
                l2_dist_matrix = np.zeros((n_pl, n_pl))
                for i in range(n_pl):
                    for j in range(n_pl):
                        diff = landscapes_list[i] - landscapes_list[j]
                        l2_dist_matrix[i, j] = diff.p_norm(p=2)
                plot_persistence_landscape_l2_distance_heatmap(
                    short_labels,
                    l2_dist_matrix,
                    output_dir / "persistence_landscape_l2_distance_heatmap.png",
                )
                lines.extend([
                    "### L2 distance between persistence landscapes",
                    "",
                    "Pairwise L2 distance between landscape of each config:",
                    "",
                    "![L2 distance between landscapes](persistence_landscape_l2_distance_heatmap.png)",
                    "",
                ])
                console.print("  [green]✓ Persistence landscape L2 distance heatmap generated[/green]")

            # Mean and variance of λ₁(t) only when ≥2 configs
            if len(config_dgms) >= 2:
                matrix = np.array(lambda1_list)
                mean_lambda1 = np.mean(matrix, axis=0)
                var_lambda1 = np.var(matrix, axis=0)
                t_grid = np.linspace(start, stop, num_steps)
                plot_persistence_landscape_mean_and_variance(
                    t_grid, mean_lambda1, var_lambda1,
                    output_dir / "persistence_landscape_mean_var_lambda1.png",
                )
                lines.extend([
                    "### Mean and variance of first persistence landscape λ₁(t)",
                    "",
                    "Pointwise mean and variance of the first landscape function across configs:",
                    "",
                    "![Mean and Var(λ₁(t))](persistence_landscape_mean_var_lambda1.png)",
                    "",
                ])
                console.print("  [green]✓ Persistence landscape mean and variance plot generated[/green]")
        else:
            console.print("  [dim]No persistence_dim0 CSVs found; skipping landscape section[/dim]")

    console.print("  [green]✓ Persistence homology table generated[/green]")
    return "\n".join(lines)


def step_6_model_performance(
    config_data: list[dict], output_dir: Path, console: Console
) -> str:
    """
    Step 6: Comparison of trained model performance across configurations.

    Two subsections: Butina split and Scaffold split. Each has (1) split analysis table
    (common, from first config), (2) three model sub-subsections: baseline, xgboost, siamese
    with table per config (sorted by embedding) and bar plot (grouped by embedding, short labels).
    """
    console.print("[cyan]Step 6: Model performance comparison...[/cyan]")

    # Collect split_statistics from first config that has it (for common split table)
    first_stats = None
    for cfg in config_data:
        stats_file = cfg["path"] / "model_training" / "split_statistics.json"
        if stats_file.exists():
            with open(stats_file) as f:
                first_stats = json.load(f)
            break

    # Per-split, per-model: collect metrics from each config (from stats[split][key])
    split_names = ["butina", "scaffold"]
    model_keys = [
        ("baseline_model_metrics", "Baseline (logistic regression on scaled distance)"),
        ("xgboost_model_metrics", "XGBoost (absolute difference of embeddings)"),
        ("siamese_model_metrics", "Siamese network (absolute difference of embeddings)"),
    ]
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "binary_cross_entropy"]

    # Build per-split lists of metrics (sorted by embedding) for tables and plots
    split_metrics: dict[str, dict[str, list[dict]]] = {}
    for split_name in split_names:
        split_metrics[split_name] = {}
        for key, _ in model_keys:
            list_for_model = []
            for cfg in config_data:
                stats_file = cfg["path"] / "model_training" / "split_statistics.json"
                if not stats_file.exists():
                    continue
                with open(stats_file) as f:
                    stats = json.load(f)
                split_data = stats.get(split_name)
                if not split_data:
                    continue
                metrics = (split_data or {}).get(key)
                if not metrics:
                    continue
                rec = {
                    "config_name": cfg["name"],
                    "embedding_method": cfg["embedding_method"],
                    "distance_metric": cfg["distance_metric"],
                    **{k: v for k, v in metrics.items() if k in metric_names},
                }
                list_for_model.append(rec)
            list_for_model.sort(key=lambda m: m["embedding_method"])
            split_metrics[split_name][key] = list_for_model
        if not any(split_metrics[split_name].values()):
            continue

    # Persist comparison json (flat structure for backward compatibility)
    all_baseline = []
    all_xgboost = []
    all_siamese = []
    for split_name in split_names:
        for key in ["baseline_model_metrics", "xgboost_model_metrics", "siamese_model_metrics"]:
            lst = split_metrics.get(split_name, {}).get(key, [])
            if key == "baseline_model_metrics":
                all_baseline.extend([{**m, "split": split_name} for m in lst])
            elif key == "xgboost_model_metrics":
                all_xgboost.extend([{**m, "split": split_name} for m in lst])
            else:
                all_siamese.extend([{**m, "split": split_name} for m in lst])
    with open(output_dir / "model_training_comparison.json", "w") as f:
        json.dump(
            {"baseline_metrics": all_baseline, "xgboost_metrics": all_xgboost, "siamese_metrics": all_siamese},
            f,
            indent=2,
        )

    # Generate plots per split per model (grouped, short labels)
    for split_name in split_names:
        for key, _ in model_keys:
            lst = split_metrics.get(split_name, {}).get(key, [])
            if not lst:
                continue
            short_labels = [f"{m['embedding_method']} / {m['distance_metric']}" for m in lst]
            model_type = key.replace("_model_metrics", "")
            plot_model_metrics_comparison_grouped(
                lst,
                model_type,
                output_dir,
                split_name=split_name,
                short_labels=short_labels,
            )

    lines = ["## 6. Model Performance", ""]
    if not first_stats and not split_metrics:
        lines.append("*No model training data found.*")
        return "\n".join(lines)

    # ROC AUC only: logistic regression, XGBoost, Siamese × Scaffold / Butina
    roc_header = (
        "| Embedding | Distance metric | LR (Scaffold) | LR (Butina) | "
        "XGBoost (Scaffold) | XGBoost (Butina) | Siamese (Scaffold) | Siamese (Butina) |"
    )
    roc_sep = (
        "|-----------|-----------------|---------------|-------------|"
        "--------------------|----------------|--------------------|----------------|"
    )
    roc_rows: list[str] = []
    for cfg in sorted(config_data, key=lambda c: c["embedding_method"]):
        st = _read_json_if_exists(cfg["path"] / "model_training" / "split_statistics.json") or {}

        def _roc_auc(split_key: str, model_key: str) -> str:
            block = (st.get(split_key) or {}).get(model_key) or {}
            v = block.get("roc_auc")
            return format_stat(float(v)) if v is not None else "—"

        roc_rows.append(
            "| "
            + " | ".join([
                cfg["embedding_method"],
                cfg["distance_metric"],
                _roc_auc("scaffold", "baseline_model_metrics"),
                _roc_auc("butina", "baseline_model_metrics"),
                _roc_auc("scaffold", "xgboost_model_metrics"),
                _roc_auc("butina", "xgboost_model_metrics"),
                _roc_auc("scaffold", "siamese_model_metrics"),
                _roc_auc("butina", "siamese_model_metrics"),
            ])
            + " |"
        )
    lines.extend([
        "### ROC AUC (test sets)",
        "",
        roc_header,
        roc_sep,
        *roc_rows,
        "",
    ])

    for split_name in split_names:
        split_data = (first_stats or {}).get(split_name)
        lines.extend([f"### {split_name.capitalize()} split", ""])

        # Split analysis table (common to all configs)
        if split_data:
            def _fmt(v):
                return f"{v:,}" if isinstance(v, int) else (str(v) if v is not None else "—")
            lines.extend([
                "#### Split analysis",
                "",
                "| n_train_compounds | n_val_compounds | n_test_compounds | n_train_pairs_balanced | n_val_pairs_balanced | n_test_pairs_balanced |",
                "|-------------------|-----------------|------------------|------------------------|----------------------|-----------------------|",
                "| "
                + " | ".join([
                    _fmt(split_data.get("n_train_compounds")),
                    _fmt(split_data.get("n_val_compounds")),
                    _fmt(split_data.get("n_test_compounds")),
                    _fmt(split_data.get("n_train_pairs_balanced")),
                    _fmt(split_data.get("n_val_pairs_balanced")),
                    _fmt(split_data.get("n_test_pairs_balanced")),
                ])
                + " |",
                "",
            ])
        else:
            lines.append("*Split statistics not available.*")
            lines.append("")

        # Three model sub-subsections: table + plot
        for key, model_title in model_keys:
            lst = split_metrics.get(split_name, {}).get(key, [])
            model_type = key.replace("_model_metrics", "")
            if not lst:
                continue
            lines.extend([f"#### {model_title}", ""])
            # Table: Embedding | Distance metric | accuracy | precision | ... (sorted by embedding)
            headers = ["Embedding", "Distance metric"] + [m.replace("_", " ").title() for m in metric_names]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            for m in lst:
                cells = [
                    m["embedding_method"],
                    m["distance_metric"],
                    *[format_stat(m.get(k)) if m.get(k) is not None else "—" for k in metric_names],
                ]
                lines.append("| " + " | ".join(str(c) for c in cells) + " |")
            lines.append("")
            img_name = f"{split_name}_{model_type}_model_metrics_comparison.png"
            lines.extend([f"![{model_title}]({img_name})", ""])

    lines.append("See `model_training_comparison.json` for detailed metrics.")
    return "\n".join(lines)


def step_7_summary(
    config_data: list[dict], output_dir: Path, console: Console
) -> str:
    """
    Step 7: Summary table of best and worst config per category.
    """
    console.print("[cyan]Step 7: Summary...[/cyan]")
    sorted_configs = sorted(config_data, key=lambda c: c["embedding_method"])

    def _label(cfg: dict) -> str:
        return f"{cfg['embedding_method']} / {cfg['distance_metric']}"

    # Collect (category_name, list of (label, value), best_is_lowest)
    # best_is_lowest True => best = min value, worst = max; False => best = max, worst = min
    rows: list[tuple[str, list[tuple[str, float]], bool]] = []

    # 1) Structurally identical pairs (pairs with distance 0) - best=lowest, worst=highest
    vals = []
    for cfg in sorted_configs:
        n = _count_csv_data_rows(cfg["path"] / "pairwise_data_creation" / "structurally_similar_pairs.csv")
        vals.append((_label(cfg), float(n)))
    if vals:
        rows.append(("Structurally identical pairs", vals, True))

    # 2) Identified cliffs at similarity 0.9 - best=lowest, worst=highest
    vals = []
    for cfg in sorted_configs:
        p = cfg["path"] / "cliff_identification" / "summary.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            v = d.get("cliffs_at_similarity_0.9")
            if v is not None:
                vals.append((_label(cfg), float(v)))
    if vals:
        rows.append(("Identified cliffs at similarity 0.9", vals, True))

    # 3) Irreducible cliff probability (c) - best=smallest c in activity_cliff_fraction
    vals = []
    for cfg in sorted_configs:
        p = cfg["path"] / "cliff_identification" / "summary.json"
        if p.exists():
            d = json.load(open(p))
            c = (d.get("fit_parameters") or {}).get("activity_cliff_fraction") or {}
            v = c.get("c")
            if v is not None:
                vals.append((_label(cfg), float(v)))
    if vals:
        rows.append(("Irreducible cliff probability (c)", vals, True))

    # 4) Similarity threshold where local SAR begins (x0) - best=smallest x0
    vals = []
    for cfg in sorted_configs:
        p = cfg["path"] / "cliff_identification" / "summary.json"
        if p.exists():
            d = json.load(open(p))
            acf = (d.get("fit_parameters") or {}).get("activity_cliff_fraction") or {}
            v = acf.get("x0")
            if v is not None:
                vals.append((_label(cfg), float(v)))
    if vals:
        rows.append(("Similarity threshold where local SAR begins (x0)", vals, True))

    # 5) Lipschitz 99th percentile - best=smallest, worst=highest
    vals = []
    for cfg in sorted_configs:
        p = cfg["path"] / "correlation" / "summary.json"
        if p.exists():
            d = json.load(open(p))
            v = d.get("lipschitz_percentile_99")
            if v is not None:
                vals.append((_label(cfg), float(v)))
    if vals:
        rows.append(("Lipschitz 99th percentile", vals, True))

    # 6) Lipschitz KWW shape (b) - best=largest b
    vals = []
    for cfg in sorted_configs:
        p = cfg["path"] / "correlation" / "summary.json"
        if p.exists():
            d = json.load(open(p))
            kww = (d.get("kww_fit") or {})
            v = kww.get("b")
            if v is not None:
                vals.append((_label(cfg), float(v)))
    if vals:
        rows.append(("Lipschitz KWW shape (b)", vals, False))

    # 7) Lipschitz KWW scale (k) - best=largest k
    vals = []
    for cfg in sorted_configs:
        p = cfg["path"] / "correlation" / "summary.json"
        if p.exists():
            d = json.load(open(p))
            v = (d.get("kww_fit") or {}).get("k")
            if v is not None:
                vals.append((_label(cfg), float(v)))
    if vals:
        rows.append(("Lipschitz KWW scale (k)", vals, False))

    # 8) Chiral analysis (mean) - statistics_similarity.mean, best=largest
    vals = []
    for cfg in sorted_configs:
        p = cfg["path"] / "correlation" / "chiral_cliff_lipschitz_statistics.json"
        if p.exists():
            d = json.load(open(p))
            sim = (d.get("statistics_similarity") or {})
            v = sim.get("mean")
            if v is not None:
                vals.append((_label(cfg), float(v)))
    if vals:
        rows.append(("Chiral analysis (mean)", vals, False))

    # 9) Chiral analysis (99th percentile) - best=highest
    vals = []
    for cfg in sorted_configs:
        p = cfg["path"] / "correlation" / "chiral_cliff_lipschitz_statistics.json"
        if p.exists():
            d = json.load(open(p))
            v = (d.get("statistics_similarity") or {}).get("percentile_99")
            if v is not None:
                vals.append((_label(cfg), float(v)))
    if vals:
        rows.append(("Chiral analysis (99th percentile)", vals, False))

    # 10) MMP analysis (mean) - best=largest
    vals = []
    for cfg in sorted_configs:
        p = cfg["path"] / "correlation" / "mmp_cliff_lipschitz_statistics.json"
        if p.exists():
            d = json.load(open(p))
            v = (d.get("statistics_similarity") or {}).get("mean")
            if v is not None:
                vals.append((_label(cfg), float(v)))
    if vals:
        rows.append(("MMP analysis (mean)", vals, False))

    # 11) MMP analysis (99th percentile) - best=highest
    vals = []
    for cfg in sorted_configs:
        p = cfg["path"] / "correlation" / "mmp_cliff_lipschitz_statistics.json"
        if p.exists():
            d = json.load(open(p))
            v = (d.get("statistics_similarity") or {}).get("percentile_99")
            if v is not None:
                vals.append((_label(cfg), float(v)))
    if vals:
        rows.append(("MMP analysis (99th percentile)", vals, False))

    # 12) Persistence homology energy (L2 norm of landscape) - best=highest
    try:
        from persim import PersLandscapeApprox
    except ImportError:
        pass
    else:
        vals = []
        start, stop, num_steps = 0.0, 1.0, 500
        for cfg in sorted_configs:
            csv_path = cfg["path"] / "persistent_homology" / "persistence_dim0.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if df.empty or "birth" not in df.columns or "death" not in df.columns:
                continue
            birth = df["birth"].to_numpy(dtype=float)
            death = df["death"].to_numpy(dtype=float)
            death = np.where(np.isfinite(death), death, 1.0)
            dgm = np.column_stack((birth, death))
            if dgm.size == 0:
                continue
            pla = PersLandscapeApprox(dgms=[dgm], hom_deg=0, start=start, stop=stop, num_steps=num_steps)
            vals.append((_label(cfg), float(pla.p_norm(p=2))))
        if vals:
            rows.append(("Persistence homology energy (landscape L2 norm)", vals, False))

    # 13) Persistence homology mean persistence - best=largest
    vals = []
    for cfg in sorted_configs:
        p = cfg["path"] / "persistent_homology" / "persistence_metadata.json"
        if p.exists():
            d = json.load(open(p))
            v = (d.get("statistics") or {}).get("dim_0") or {}
            mp = v.get("mean_persistence")
            if mp is not None:
                vals.append((_label(cfg), float(mp)))
    if vals:
        rows.append(("Persistence homology mean persistence", vals, False))

    # 14–16) Scaffold: baseline, xgboost, siamese accuracy - best=highest
    for split_name, model_key, title in [
        ("scaffold", "baseline_model_metrics", "Best scaffold logistic regression"),
        ("scaffold", "xgboost_model_metrics", "Best scaffold XGBoost"),
        ("scaffold", "siamese_model_metrics", "Best scaffold Siamese Network"),
    ]:
        vals = []
        for cfg in sorted_configs:
            p = cfg["path"] / "model_training" / "split_statistics.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            split_data = d.get(split_name)
            if not split_data:
                continue
            m = (split_data or {}).get(model_key)
            if not m:
                continue
            acc = m.get("accuracy")
            if acc is not None:
                vals.append((_label(cfg), float(acc)))
        if vals:
            rows.append((title, vals, False))

    # 17–19) Butina: baseline, xgboost, siamese accuracy - best=highest
    for split_name, model_key, title in [
        ("butina", "baseline_model_metrics", "Best butina logistic regression"),
        ("butina", "xgboost_model_metrics", "Best butina XGBoost"),
        ("butina", "siamese_model_metrics", "Best butina Siamese Network"),
    ]:
        vals = []
        for cfg in sorted_configs:
            p = cfg["path"] / "model_training" / "split_statistics.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            split_data = d.get(split_name)
            if not split_data:
                continue
            m = (split_data or {}).get(model_key)
            if not m:
                continue
            acc = m.get("accuracy")
            if acc is not None:
                vals.append((_label(cfg), float(acc)))
        if vals:
            rows.append((title, vals, False))

    lines = ["## 7. Summary", "", "### Best and worst config by category", ""]
    lines.append("| Category | Best | Worst |")
    lines.append("|----------|------|-------|")

    best_count: dict[str, int] = {}
    worst_count: dict[str, int] = {}

    for category_name, pairs, best_is_lowest in rows:
        if best_is_lowest:
            best_label = min(pairs, key=lambda x: x[1])[0]
            worst_label = max(pairs, key=lambda x: x[1])[0]
        else:
            best_label = max(pairs, key=lambda x: x[1])[0]
            worst_label = min(pairs, key=lambda x: x[1])[0]
        lines.append(f"| {category_name} | {best_label} | {worst_label} |")
        best_count[best_label] = best_count.get(best_label, 0) + 1
        worst_count[worst_label] = worst_count.get(worst_label, 0) + 1

    if not rows:
        lines.append("*No summary data available.*")
    else:
        # Configs sorted by number of categories where best (descending)
        lines.extend([
            "",
            "### Configs by number of categories where best",
            "",
            "| Config | Number of categories |",
            "|--------|----------------------|",
        ])
        for label in sorted(best_count.keys(), key=lambda L: -best_count[L]):
            lines.append(f"| {label} | {best_count[label]} |")
        lines.extend([
            "",
            "### Configs by number of categories where worst",
            "",
            "| Config | Number of categories |",
            "|--------|----------------------|",
        ])
        for label in sorted(worst_count.keys(), key=lambda L: -worst_count[L]):
            lines.append(f"| {label} | {worst_count[label]} |")

    # Additional comparison figures
    extra_figure_names = [
        "cliff_rank_5pct_comparison.png",
        "lipschitz_p90_comparison.png",
        "persistence_mean_max_comparison.png",
        "chiral_mmp_gaussian_std_comparison.png",
    ]
    lines.append("")
    lines.append("### Additional comparison figures")
    lines.append("")
    for name in extra_figure_names:
        if (output_dir / name).exists():
            lines.append(f"![{name}]({name})")
            lines.append("")

    console.print("  [green]✓ Summary table generated[/green]")
    return "\n".join(lines)


