"""
Cliff irreducibility and the geometry of the cliff subspace.

Two analyses share one scale parameter, the structural distance S at which molecules are joined.

Irreducibility (the primary result). At scale S the molecules form a graph: an edge wherever the
scaled structural distance is at most S. A cliff is an edge of that graph whose activity
difference reaches the cliff threshold. Its *barrier* is the gentlest activity step that any
detour through the same graph must still take, which is at most its own activity difference
because the direct edge is always available. A cliff is irreducible at level L when its barrier
is at least L: every structurally tolerable detour still contains a step of size L or more.
Sweeping S gives, for each level, the fraction of cliffs that remain irreducible, and the
half-dissolution scale at which that fraction falls to one half. Levels are the cliff threshold
itself, and two smaller values at the scale of assay reproducibility.

The barrier is never computed pair by pair. A pair has barrier < L exactly when its two
molecules are connected in the graph restricted to steps smaller than L, so each point of the
curve is one connected-components pass.

Supporting geometry. H0 of the Vietoris-Rips filtration of the same scaled structural distances
over all molecules: all births at 0, deaths the edge weights of a minimum spanning forest, and
zero-length bars omitted, so the diagram equals ripser's H0. Bar length is the scale over which
a structural cluster stays separate from the rest; we report its mean and maximum.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.plotting import plot_persistence_diagram

# Structural scales at which the curve is sampled, as the graph's mean degree: the number of
# structural neighbours per molecule. Degree rather than a percentile of pairs, because at a
# fixed percentile the degree grows with dataset size (edges go as p*n^2/2, degree as p*n), so
# the same percentile is a far denser neighbourhood on a large target and the curves of
# different datasets are sampled in different regimes. Dense at the low end, where the strictest
# barrier level dissolves, and extending until every level has saturated.
DEFAULT_SCALE_DEGREES = (
    0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0,
)
# Activity steps a detour may not avoid. The first is the cliff threshold (substituted at run
# time); the others sit at the scale of assay reproducibility.
DEFAULT_BARRIER_LEVELS = (1.0, 0.5, 0.3)
HALF_DISSOLUTION = 0.5


def _pair_arrays(
    pairwise_data: dict[str, dict],
    compound_to_idx: dict[str, int],
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Upper-triangle arrays of the scaled structural distance and the activity difference.

    Distances come from 'distance_scaled', falling back to 'distance_scaled_structure' for pairs
    excluded on activity grounds. The activity difference is NaN for excluded pairs, which marks
    them unusable.
    """
    size = n * (n - 1) // 2
    dist = np.full(size, np.nan, dtype=np.float32)
    dpa = np.full(size, np.nan, dtype=np.float32)
    for pair_key, data in pairwise_data.items():
        a, b = pair_key.split(",", 1)
        i, j = compound_to_idx[a], compound_to_idx[b]
        if i > j:
            i, j = j, i
        idx = i * n - i * (i + 1) // 2 + (j - i - 1)
        d = data.get("distance_scaled", np.nan)
        if d is None or np.isnan(d):
            d = data.get("distance_scaled_structure", np.nan)
        if d is not None and not np.isnan(d):
            dist[idx] = d
        if data.get("status") != "excluded":
            lr = data.get("log_ratio", np.nan)
            if lr is not None and not np.isnan(lr):
                dpa[idx] = lr
    return dist, dpa


def _triangle_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the upper triangle, as int32 to keep large datasets in memory."""
    counts = (n - 1 - np.arange(n - 1)).astype(np.int64)
    rows = np.repeat(np.arange(n - 1, dtype=np.int32), counts)
    cols = np.concatenate([np.arange(i + 1, n, dtype=np.int32) for i in range(n - 1)])
    return rows, cols


def _components(rows: np.ndarray, cols: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """Connected-component label per molecule for the subgraph selected by `mask`."""
    if not mask.any():
        return np.arange(n)
    graph = coo_matrix(
        (np.ones(int(mask.sum()), dtype=np.int8), (rows[mask], cols[mask])), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)
    return labels


def _scales_for_degrees(dist: np.ndarray, usable: np.ndarray, n: int, degrees) -> list[float]:
    """
    Distance threshold giving each target mean degree: the m-th smallest usable distance, where
    m = degree * n / 2 is the edge count that degree implies.
    """
    ordered = np.sort(dist[usable])
    out = []
    for degree in degrees:
        m = int(round(degree * n / 2))
        out.append(float(ordered[min(max(m, 1), len(ordered)) - 1]))
    return out


def irreducibility_curve(
    dist: np.ndarray,
    dpa: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    n: int,
    activity_threshold: float,
    scale_degrees=DEFAULT_SCALE_DEGREES,
    barrier_levels=DEFAULT_BARRIER_LEVELS,
) -> list[dict]:
    """
    Fraction of cliffs that stay irreducible, at each structural scale and barrier level.

    A cliff at scale S is a pair within distance S whose activity difference reaches the cliff
    threshold. It is reducible at level L when a detour exists through steps all smaller than L,
    which holds exactly when its two molecules share a connected component of the graph
    {distance <= S and activity difference < L}.
    """
    usable = np.isfinite(dist) & np.isfinite(dpa)
    if not usable.any():
        return []
    levels = tuple(dict.fromkeys((activity_threshold,) + tuple(barrier_levels)))
    scales = _scales_for_degrees(dist, usable, n, scale_degrees)

    curve = []
    seen: set[float] = set()
    for target, scale in zip(scale_degrees, scales):
        if scale in seen:
            continue   # ties in the distance distribution can collapse two targets onto one scale
        seen.add(scale)
        within = usable & (dist <= scale)
        cliff = within & (dpa >= activity_threshold)
        n_cliff = int(cliff.sum())
        n_pairs = int(within.sum())
        point = {
            "degree_target": float(target),
            "degree": 2.0 * n_pairs / n,
            "scale": float(scale),
            "n_pairs": n_pairs,
            "n_cliff_pairs": n_cliff,
        }
        for level in levels:
            if n_cliff == 0:
                point[f"fraction_irreducible_{level:g}"] = None
                continue
            labels = _components(rows, cols, within & (dpa < level), n)
            separated = labels[rows[cliff]] != labels[cols[cliff]]
            point[f"fraction_irreducible_{level:g}"] = float(separated.mean())
        curve.append(point)
    return curve


def half_dissolution_scales(curve: list[dict], levels) -> dict[str, dict | None]:
    """
    Mean degree at which each level's irreducible fraction falls to one half.

    Linear interpolation in log10(degree) between the two samples that bracket the crossing.
    None when the curve does not cross within the sampled range, with 'above'/'below' recording
    which side it stayed on.
    """
    out: dict[str, dict | None] = {}
    for level in levels:
        key = f"fraction_irreducible_{level:g}"
        pts = [(p["degree"], p["scale"], p[key]) for p in curve if p.get(key) is not None]
        if len(pts) < 2:
            out[f"{level:g}"] = None
            continue
        crossing = None
        for (p0, s0, f0), (p1, s1, f1) in zip(pts, pts[1:]):
            if (f0 - HALF_DISSOLUTION) * (f1 - HALF_DISSOLUTION) <= 0 and f0 != f1:
                t = (f0 - HALF_DISSOLUTION) / (f0 - f1)
                lp = np.log10(p0) + t * (np.log10(p1) - np.log10(p0))
                crossing = {"degree": float(10 ** lp), "scale": float(s0 + t * (s1 - s0))}
                break
        if crossing is None:
            side = "above" if pts[-1][2] > HALF_DISSOLUTION else "below"
            crossing = {"degree": None, "scale": None, "outside_sampled_range": side}
        out[f"{level:g}"] = crossing
    return out


def _minimum_spanning_forest(dist: np.ndarray) -> list[tuple[int, int, float]]:
    """
    Exact minimum spanning forest of a dense distance matrix (Prim's algorithm, O(m^2)).
    Zero distances are ordinary edges; +inf marks a missing edge. Returns edges (i, j, w) with
    i < j, sorted by (w, i, j). Its edge weights are exactly the H0 death values of the
    Vietoris-Rips filtration of dist.
    """
    m = len(dist)
    edges: list[tuple[int, int, float]] = []
    in_tree = np.zeros(m, dtype=bool)
    best = np.full(m, np.inf)
    parent = np.full(m, -1)
    for start in range(m):
        if in_tree[start]:
            continue
        best[start] = 0.0
        while True:
            candidates = np.where(~in_tree, best, np.inf)
            v = int(np.argmin(candidates))
            if not np.isfinite(candidates[v]):
                break
            in_tree[v] = True
            if parent[v] >= 0:
                u = int(parent[v])
                edges.append((min(u, v), max(u, v), float(best[v])))
            closer = ~in_tree & (dist[v] < best)
            best[closer] = dist[v][closer]
            parent[closer] = v
    edges.sort(key=lambda e: (e[2], e[0], e[1]))
    return edges


def _h0_diagram(edges: list[tuple[int, int, float]], n: int) -> np.ndarray:
    """
    Standard 0-dimensional persistence of an edge filtration, by union-find over the spanning
    forest edges in weight order. Every class is born at 0; an edge joining two components closes
    one class at its own weight, and zero-length bars are omitted, as in ripser. Components that
    never merge give bars [0, inf).
    """
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    points: list[tuple[float, float]] = []
    merged = 0
    for i, j, s in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        elif rank[ri] == rank[rj]:
            rank[ri] += 1
        parent[rj] = ri
        merged += 1
        if s > 0:
            points.append((0.0, s))
    points.extend([(0.0, np.inf)] * (n - merged))
    dgm = np.array(points, dtype=np.float64) if points else np.zeros((0, 2))
    if len(dgm):
        dgm = dgm[np.lexsort((dgm[:, 1], dgm[:, 0]))]
    return dgm


def _persistence_statistics(dgm: np.ndarray) -> dict:
    """Mean and maximum persistence over the finite bars, with bar counts."""
    if len(dgm) == 0:
        return {"n_features": 0, "n_finite": 0, "n_infinite": 0,
                "mean_persistence": None, "max_persistence": None}
    finite = dgm[np.isfinite(dgm[:, 1])]
    stats = {
        "n_features": int(len(dgm)),
        "n_finite": int(len(finite)),
        "n_infinite": int(len(dgm) - len(finite)),
        "mean_persistence": None,
        "max_persistence": None,
    }
    if len(finite):
        lifetimes = finite[:, 1] - finite[:, 0]
        stats["mean_persistence"] = float(np.mean(lifetimes))
        stats["max_persistence"] = float(np.max(lifetimes))
    return stats


def _plot_curve(curve: list[dict], levels, output_path: Path, title: str) -> None:
    """The irreducible fraction against the structural scale, one line per barrier level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for level in levels:
        key = f"fraction_irreducible_{level:g}"
        pts = [(p["degree"], p[key]) for p in curve if p.get(key) is not None]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, marker="o", markersize=4, label=f"barrier $\\geq$ {level:g}")
    ax.axhline(HALF_DISSOLUTION, color="0.6", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Mean degree (structural neighbours per molecule)")
    ax.set_ylabel("Fraction of cliffs irreducible")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calculate_cliff_irreducibility(
    pairwise_data: dict[str, dict],
    results_dir: Path,
    activity_threshold: float,
    *,
    scale_degrees=DEFAULT_SCALE_DEGREES,
    barrier_levels=DEFAULT_BARRIER_LEVELS,
) -> dict:
    """
    Cliff irreducibility across structural scales, with H0 of the same filtration as support.

    Args:
        pairwise_data: Dict mapping "inchi_key1,inchi_key2" to a dict with 'distance_scaled',
            'log_ratio' and 'status'.
        results_dir: Directory to write results into (cliff_irreducibility/).
        activity_threshold: Activity difference at which a pair counts as a cliff. Also used as
            the strictest barrier level.
        scale_degrees: Structural scales to sample, as the graph's mean degree.
        barrier_levels: Activity steps a detour may not avoid.

    Returns:
        Dict with 'curve', 'half_dissolution', 'diagram', 'statistics' and 'metadata'.
    """
    console = Console()
    console.print("[bold cyan]Step 4 - Cliff irreducibility...[/bold cyan]")
    out_dir = Path(results_dir) / "cliff_irreducibility"
    out_dir.mkdir(parents=True, exist_ok=True)

    compounds = sorted({k for pair in pairwise_data for k in pair.split(",", 1)})
    compound_to_idx = {key: i for i, key in enumerate(compounds)}
    n = len(compounds)
    levels = tuple(dict.fromkeys((activity_threshold,) + tuple(barrier_levels)))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("  Building pair arrays...", total=3)
        dist, dpa = _pair_arrays(pairwise_data, compound_to_idx, n)
        rows, cols = _triangle_indices(n)
        usable = np.isfinite(dist) & np.isfinite(dpa)
        cliff_any = np.zeros(n, dtype=bool)
        is_cliff = usable & (dpa >= activity_threshold)
        cliff_any[rows[is_cliff]] = True
        cliff_any[cols[is_cliff]] = True
        progress.update(task, advance=1, description="  Sweeping structural scales...")

        curve = irreducibility_curve(dist, dpa, rows, cols, n, activity_threshold,
                                     scale_degrees, barrier_levels)
        half = half_dissolution_scales(curve, levels)
        progress.update(task, advance=1, description="  H0 of the structural filtration...")

        matrix = np.full((n, n), np.inf, dtype=np.float32)
        matrix[rows, cols] = np.where(np.isfinite(dist), dist, np.inf)
        matrix[cols, rows] = matrix[rows, cols]
        np.fill_diagonal(matrix, 0.0)
        dgm = _h0_diagram(_minimum_spanning_forest(matrix), n)
        stats = _persistence_statistics(dgm)
        progress.update(task, advance=1, description="  Writing results...")

    pd.DataFrame(curve).to_csv(out_dir / "irreducibility_curve.csv", index=False)
    pd.DataFrame({"birth": dgm[:, 0], "death": dgm[:, 1],
                  "persistence": dgm[:, 1] - dgm[:, 0]}).to_csv(
        out_dir / "persistence_dim0.csv", index=False)

    metadata = {
        "activity_threshold": activity_threshold,
        "barrier_levels": list(levels),
        "scale_degrees": list(scale_degrees),
        "n_compounds": n,
        "n_compounds_in_any_cliff": int(cliff_any.sum()),
        "n_pairs_usable": int(usable.sum()),
        "n_cliff_pairs_total": int(is_cliff.sum()),
        "half_dissolution": half,
        "statistics": {"dim_0": stats},
    }
    with open(out_dir / "irreducibility_summary.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(out_dir / "persistence_metadata.json", "w") as f:
        json.dump({k: metadata[k] for k in ("activity_threshold", "n_compounds", "statistics")}, f, indent=2)

    _plot_curve(curve, levels, out_dir / "irreducibility_curve.png", "Cliff irreducibility")
    try:
        plot_persistence_diagram([dgm], output_path=out_dir / "persistence_diagram.png",
                                 title="H0 persistence (structural filtration)")
    except Exception as exc:  # plotting must never take the pipeline down
        console.print(f"[yellow]  ⚠ persistence diagram not written: {exc}[/yellow]")

    strict = half.get(f"{activity_threshold:g}") or {}
    console.print(
        f"  {n} molecules, {int(is_cliff.sum()):,} cliff pairs; "
        f"{int(cliff_any.sum())} molecules in at least one cliff"
    )
    for level in levels:
        deg = (half.get(f"{level:g}") or {}).get("degree")
        shown = f"degree {deg:.3g}" if deg else "outside the sampled range"
        console.print(f"  half-dissolution, barrier >= {level:g}: {shown}")
    console.print(f"[green]  ✓ Results saved to: {out_dir}[/green]")

    return {"curve": curve, "half_dissolution": half, "diagram": dgm,
            "statistics": stats, "metadata": metadata}
