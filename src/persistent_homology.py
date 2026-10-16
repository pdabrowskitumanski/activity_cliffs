"""
Persistent homology on molecular pairs.

Vietoris-Rips filtration of the cliff subspace: the vertices are the compounds involved in
at least one activity cliff (log_ratio >= activity_threshold, status != 'excluded'), and two
of them are joined by an edge at filtration value eps when their scaled structural distance
is <= eps (all pairs of cliff compounds, not only cliff pairs). H0 is the standard
0-dimensional persistence (all births 0; zero-length bars omitted), computed exactly from the
minimum spanning forest of the distance matrix; it equals ripser's H0. Persistence diagrams
and statistics are computed; the most persistent clusters (persistence > threshold) are
dumped for downstream use.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import ripser
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from scipy import sparse

from src.plotting import plot_persistence_landscape

# Grid for Bubenik-style persistence landscapes (tent functions via persim).
LANDSCAPE_NUM_STEPS = 500
LANDSCAPE_START = 0.0


def _extract_compounds_from_pairwise_data(
    pairwise_data: dict[str, dict],
) -> tuple[list[str], dict[str, int]]:
    """Extract unique compound inchi_keys and compound -> index mapping."""
    compounds = set()
    for pair_key in pairwise_data:
        a, b = pair_key.split(",", 1)
        compounds.add(a)
        compounds.add(b)
    compounds = sorted(compounds)
    compound_to_idx = {c: i for i, c in enumerate(compounds)}
    return compounds, compound_to_idx


def _cliff_compounds(
    pairwise_data: dict[str, dict],
    compound_to_idx: dict[str, int],
    activity_threshold: float,
) -> list[int]:
    """
    Indices of compounds involved in at least one activity cliff: a pair with
    log_ratio >= activity_threshold, status != 'excluded' and a finite similarity.
    """
    cliff = set()
    for pair_key, data in pairwise_data.items():
        if data.get("status") == "excluded":
            continue
        log_ratio = data.get("log_ratio", np.nan)
        if np.isnan(log_ratio) or log_ratio < activity_threshold:
            continue
        if np.isnan(data.get("similarity", np.nan)):
            continue
        a, b = pair_key.split(",", 1)
        cliff.add(compound_to_idx[a])
        cliff.add(compound_to_idx[b])
    return sorted(cliff)


def _cliff_distance_matrix(
    pairwise_data: dict[str, dict],
    compound_to_idx: dict[str, int],
    cliff_indices: list[int],
) -> np.ndarray:
    """
    Scaled structural distances between all pairs of cliff compounds (diagonal 0).
    Uses 'distance_scaled'; for excluded pairs (both compounds censored), whose activity-based
    fields are NaN, the structural distance 'distance_scaled_structure'. Pairs without a finite
    distance are +inf (no edge).
    """
    position = {idx: k for k, idx in enumerate(cliff_indices)}
    m = len(cliff_indices)
    dist = np.full((m, m), np.inf)
    np.fill_diagonal(dist, 0.0)
    for pair_key, data in pairwise_data.items():
        a, b = pair_key.split(",", 1)
        pa, pb = position.get(compound_to_idx[a]), position.get(compound_to_idx[b])
        if pa is None or pb is None:
            continue
        d = data.get("distance_scaled", np.nan)
        if np.isnan(d):
            d = data.get("distance_scaled_structure", np.nan)
        if not np.isnan(d):
            dist[pa, pb] = dist[pb, pa] = d
    return dist


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
        # grow one tree of the forest from `start`
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


def _build_rips_sparse(dist: np.ndarray):
    """Sparse distance matrix for ripser (finite entries only; missing entries are no edge)."""
    finite = np.isfinite(dist)
    rows, cols = np.nonzero(finite)
    return sparse.coo_matrix((dist[finite], (rows, cols)), shape=dist.shape).tocsr()


def _compute_persistence_statistics(dgms: list[np.ndarray]) -> dict:
    """Compute statistics for persistence diagrams (per dimension)."""
    stats = {}
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            stats[f"dim_{dim}"] = {
                "n_features": 0,
                "n_finite": 0,
                "n_infinite": 0,
            }
            continue
        finite_mask = ~np.isinf(dgm[:, 1])
        finite_dgm = dgm[finite_mask]
        n_infinite = int(np.sum(~finite_mask))

        if len(finite_dgm) > 0:
            persistence = finite_dgm[:, 1] - finite_dgm[:, 0]
            tp = float(np.sum(persistence))
            # E = Σ (pi/TP) ln(pi/TP) with pi lifetimes; equals Σ p_i ln p_i (≤ 0).
            # Shannon entropy H = −Σ p_i ln p_i is also reported for interpretation.
            if tp > 0:
                p = persistence / tp
                persistent_entropy = float(np.sum(p * np.log(p + 1e-300)))
                persistent_entropy_shannon = float(
                    -np.sum(p * np.log(p + 1e-300))
                )
            else:
                persistent_entropy = None
                persistent_entropy_shannon = None
            stats[f"dim_{dim}"] = {
                "n_features": int(len(dgm)),
                "n_finite": int(len(finite_dgm)),
                "n_infinite": n_infinite,
                "max_persistence": float(np.max(persistence)),
                "mean_persistence": float(np.mean(persistence)),
                "median_persistence": float(np.median(persistence)),
                "std_persistence": float(np.std(persistence)),
                "total_persistence": tp,
                "persistent_entropy": persistent_entropy,
                "persistent_entropy_shannon": persistent_entropy_shannon,
                "max_birth": float(np.max(finite_dgm[:, 0])),
                "max_death": float(np.max(finite_dgm[:, 1])),
            }
        else:
            stats[f"dim_{dim}"] = {
                "n_features": int(len(dgm)),
                "n_finite": 0,
                "n_infinite": n_infinite,
            }
    return stats


def _filtration_stop_for_landscape(dgms: list[np.ndarray]) -> float:
    """Upper bound for the landscape grid; the scaled-distance filtration lives in [0, 1]."""
    max_finite = 0.0
    for dgm in dgms:
        if len(dgm) == 0:
            continue
        deaths = dgm[:, 1]
        finite_d = deaths[np.isfinite(deaths)]
        if len(finite_d) > 0:
            max_finite = max(max_finite, float(np.max(finite_d)))
    return max(1.0, max_finite)


def _prepare_dgms_for_landscape(dgms: list[np.ndarray], stop: float) -> list[np.ndarray]:
    """Replace infinite death with `stop` for persim tent / landscape construction."""
    out: list[np.ndarray] = []
    for dgm in dgms:
        if len(dgm) == 0:
            out.append(dgm.copy())
            continue
        d = np.asarray(dgm, dtype=np.float64).copy()
        d[:, 1] = np.where(np.isfinite(d[:, 1]), d[:, 1], stop)
        out.append(d)
    return out


def _compute_landscape_l2_norms(
    dgms_prepared: list[np.ndarray],
    *,
    start: float,
    stop: float,
    num_steps: int,
) -> dict[str, float | None]:
    """L2 norm of the discrete persistence landscape (persim PersLandscapeApprox) per dimension."""
    try:
        from persim import PersLandscapeApprox
    except ImportError:
        return {f"dim_{i}": None for i in range(len(dgms_prepared))}

    norms: dict[str, float | None] = {}
    for dim, dgm in enumerate(dgms_prepared):
        if len(dgm) == 0:
            norms[f"dim_{dim}"] = None
            continue
        try:
            pla = PersLandscapeApprox(
                dgms=dgms_prepared,
                hom_deg=dim,
                start=start,
                stop=stop,
                num_steps=num_steps,
            )
            norms[f"dim_{dim}"] = float(pla.p_norm(p=2))
        except Exception:
            norms[f"dim_{dim}"] = None
    return norms


def _compute_H0_and_clusters(
    edges: list[tuple[int, int, float]],
    n: int,
    compounds: list[str],
    persistence_threshold: float,
) -> tuple[np.ndarray, list[dict]]:
    """
    Standard 0-dimensional persistence of an edge filtration (edges sorted by filtration value,
    e.g. the minimum spanning forest edges), computed with union-find (path compression, union by rank).

    Every vertex is born at 0. An edge (i, j, s) that joins two components ends exactly one
    H0 class: the bar [0, s). Components that never merge give bars [0, inf). Zero-length bars
    [0, 0) (edges at filtration value 0) are omitted, as in ripser, so the diagram equals
    ripser's H0 diagram of the same sparse filtration.

    Returns (dgm0, persistent_clusters):
    - dgm0: (N, 2) array of (birth, death) for the H0 persistence diagram.
    - persistent_clusters: components that end with persistence >= threshold (the member
      compounds of the component absorbed at that merge), and all components that never merge.
    """
    parent = list(range(n))
    rank = [0] * n
    root_to_nodes: list[set[int] | None] = [{i} for i in range(n)]

    diagram_points: list[tuple[float, float]] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> list[str] | None:
        """Merge the components of x and y; return the members of the absorbed component."""
        rx, ry = find(x), find(y)
        if rx == ry:
            return None
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        elif rank[rx] == rank[ry]:
            rank[rx] += 1
        inchi_keys = [compounds[idx] for idx in root_to_nodes[ry]]
        parent[ry] = rx
        root_to_nodes[rx] |= root_to_nodes[ry]
        root_to_nodes[ry] = None
        return inchi_keys

    eps = 1e-9
    persistent_clusters: list[dict] = []
    for i, j, s in edges:
        inchi_keys = union(i, j)
        if inchi_keys is None:
            continue
        if s > 0:
            diagram_points.append((0.0, s))
        if s >= persistence_threshold - eps:
            persistent_clusters.append({
                "persistence": round(s, 10),
                "inchi_keys": sorted(inchi_keys),
                "size": len(inchi_keys),
            })

    for idx in range(n):
        root = find(idx)
        if root_to_nodes[root] is None:
            continue
        inchi_keys = [compounds[i] for i in root_to_nodes[root]]
        persistent_clusters.append({
            "persistence": None,
            "inchi_keys": sorted(inchi_keys),
            "size": len(inchi_keys),
        })
        diagram_points.append((0.0, np.inf))
        root_to_nodes[root] = None

    dgm0 = np.array(diagram_points, dtype=np.float64) if diagram_points else np.zeros((0, 2))
    # Sort for deterministic output: by (birth, death); infinite death points last
    if len(dgm0) > 0:
        order = np.lexsort((dgm0[:, 1], dgm0[:, 0]))
        dgm0 = dgm0[order]
    return dgm0, persistent_clusters


def calculate_persistent_homology(
    pairwise_data: dict[str, dict],
    results_dir: Path,
    activity_threshold: float,
    *,
    persistence_threshold: float = 0.5,
    max_dim: int = 0,
) -> dict:
    """
    Compute persistent homology for the Rips complex defined by pairwise_data.

    Vertices are the compounds involved in at least one activity cliff
    (pairwise_data[pair]['log_ratio'] >= activity_threshold, status != 'excluded'); two of them
    are joined by an edge at filtration value eps when their scaled structural distance is <= eps.

    When max_dim=0 (default), H0 is computed exactly from the minimum spanning forest of the
    cliff compounds' distance matrix (Prim's algorithm) and a union-find pass over its edges.
    For max_dim>0, ripser is called with the same distance matrix.

    Saves persistence diagrams (CSV), statistics (JSON), persistence landscape
    plots (Bubenik tent landscapes via persim), and a dump of the most persistent
    clusters. Metadata includes landscape L2 norms and persistent entropy per
    dimension (from finite lifetimes).

    Args:
        pairwise_data: Dict mapping "inchi_key1,inchi_key2" to dict with
            'log_ratio', 'status', 'similarity'.
        results_dir: Directory to write results into (persistent_homology/).
        activity_threshold: Minimum log_ratio for an edge to be considered.
        persistence_threshold: Clusters with persistence >= this are saved (and inf).
        max_dim: Maximum homology dimension (0 = H0 only, fast; 1 = H0+H1; 2 = H0+H1+H2).

    Returns:
        Dict with 'diagrams', 'statistics', 'compounds', 'persistent_clusters'.
    """
    console = Console()
    results_dir = Path(results_dir)
    ph_dir = results_dir / "persistent_homology"
    ph_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Step 4 - Persistent homology...[/bold cyan]")


    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Persistent homology...", total=4)

        progress.console.print("[cyan]Step 4a: Cliff compounds and their distance matrix...[/cyan]")
        all_compounds, compound_to_idx = _extract_compounds_from_pairwise_data(pairwise_data)
        cliff_indices = _cliff_compounds(pairwise_data, compound_to_idx, activity_threshold)
        compounds = [all_compounds[i] for i in cliff_indices]
        n = len(compounds)
        dist = _cliff_distance_matrix(pairwise_data, compound_to_idx, cliff_indices)
        n_edges = int(np.isfinite(dist[np.triu_indices(n, k=1)]).sum())
        n_missing = n * (n - 1) // 2 - n_edges
        progress.console.print(
            f"  {n:,} of {len(all_compounds):,} compounds in at least one cliff; "
            f"{n_edges:,} pairwise distances" + (f" ({n_missing:,} undefined)" if n_missing else "")
        )
        progress.update(task, advance=1)

        progress.console.print("[cyan]Step 4b: Computing H0 and persistent clusters...[/cyan]")
        mst_edges = _minimum_spanning_forest(dist)
        dgm0, persistent_clusters = _compute_H0_and_clusters(
            mst_edges, n, compounds, persistence_threshold
        )

        if max_dim == 0:
            dgms = [dgm0]
            progress.console.print("[cyan]H0-only mode: using union-find (no ripser).[/cyan]")
        else:
            progress.console.print("[cyan]Step 4b (cont.): Computing H1/H2 via ripser (sparse)...[/cyan]")
            sparse_matrix = _build_rips_sparse(dist)
            result = ripser.ripser(
                sparse_matrix, maxdim=max_dim, distance_matrix=True
            )
            dgms = result["dgms"]
        progress.update(task, advance=1)

        progress.console.print("[cyan]Step 4c: Computing statistics and landscapes...[/cyan]")
        stats = _compute_persistence_statistics(dgms)
        landscape_stop = _filtration_stop_for_landscape(dgms)
        dgms_landscape = _prepare_dgms_for_landscape(dgms, landscape_stop)
        l2_norms = _compute_landscape_l2_norms(
            dgms_landscape,
            start=LANDSCAPE_START,
            stop=landscape_stop,
            num_steps=LANDSCAPE_NUM_STEPS,
        )
        for key, l2 in l2_norms.items():
            if key in stats:
                stats[key]["landscape_l2_norm"] = l2
        progress.update(task, advance=1)

        progress.console.print("[cyan]Saving results and figures...[/cyan]")
        # Save diagrams per dimension
        for dim, dgm in enumerate(dgms):
            df = pd.DataFrame(dgm, columns=["birth", "death"])
            df["persistence"] = df["death"] - df["birth"]
            df.to_csv(ph_dir / f"persistence_dim{dim}.csv", index=False)

        meta = {
            "activity_threshold": activity_threshold,
            "persistence_threshold": persistence_threshold,
            "max_dim": max_dim,
            "filtration": "Vietoris-Rips on scaled structural distance among cliff compounds",
            "n_compounds": len(all_compounds),
            "n_cliff_compounds": n,
            "n_pairs_total": len(pairwise_data),
            "n_edges_in_filtration": n_edges,
            "n_undefined_distances": n_missing,
            "statistics": stats,
            "n_persistent_clusters": len(persistent_clusters),
            "landscape_grid": {
                "start": LANDSCAPE_START,
                "stop": landscape_stop,
                "num_steps": LANDSCAPE_NUM_STEPS,
                "method": "PersLandscapeApprox (tent functions, persim)",
            },
            "landscape_l2_norm": l2_norms,
        }
        with open(ph_dir / "persistence_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        with open(ph_dir / "persistent_clusters.json", "w") as f:
            json.dump(
                {
                    "persistence_threshold": persistence_threshold,
                    "clusters": persistent_clusters,
                },
                f,
                indent=2,
            )

        # Persistence landscapes (tent functions); same filenames as before but landscapes
        if max_dim == 0:
            if len(dgms) > 0 and len(dgms[0]) > 0:
                plot_persistence_landscape(
                    dgms_landscape,
                    output_path=ph_dir / "persistence_diagram.png",
                    title="Persistence landscape (H0) — Rips on cliff compounds",
                    start=LANDSCAPE_START,
                    stop=landscape_stop,
                    num_steps=LANDSCAPE_NUM_STEPS,
                    figsize=(12, 6),
                )
        else:
            plot_persistence_landscape(
                dgms_landscape,
                output_path=ph_dir / "persistence_diagram_all.png",
                title="Persistence landscapes (H0, H1, …) — Rips on cliff compounds",
                start=LANDSCAPE_START,
                stop=landscape_stop,
                num_steps=LANDSCAPE_NUM_STEPS,
            )
            if len(dgms) > 0 and len(dgms[0]) > 0:
                plot_persistence_landscape(
                    [dgms_landscape[0]],
                    output_path=ph_dir / "persistence_diagram_H0.png",
                    title="Persistence landscape (H0 only) — Rips on cliff compounds",
                    start=LANDSCAPE_START,
                    stop=landscape_stop,
                    num_steps=LANDSCAPE_NUM_STEPS,
                    figsize=(12, 6),
                )
        progress.update(task, advance=1)
        progress.console.print("[green]✓ Persistent homology saved[/green]")

    return {
        "diagrams": dgms,
        "statistics": stats,
        "compounds": compounds,
        "persistent_clusters": persistent_clusters,
        "n_edges_in_filtration": n_edges,
    }
