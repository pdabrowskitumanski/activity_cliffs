import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from scipy.optimize import curve_fit

from src.chemistry import get_inchi_key
from src.cliff_identification import identify_cliffs
from src.correlation import correlate_similarity_vs_activity
from src.datasets import DatasetSpec, add_log_activity_and_censoring, resolve_dataset
from src.embeddings import get_embedding
from src.model_training import train_model
from src.pairwise_data_preparation import create_pairwise_data
from src.persistent_homology import calculate_persistent_homology
from src.printing import save_summary_markdown
from src.utils import load_config

# Add project root to path for direct script execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))


FINGERPRINT_DEFAULTS = {"n_bits": 1024, "radius": 2, "chirality": False}


def resolve_embeddings(config: dict, embedding_method: str) -> tuple[pd.DataFrame | None, dict]:
    """
    Resolve the embeddings pickle and the fingerprint settings for an analysis.

    With config["embeddings_file"], that pickle (from `embed`) must exist and contain
    embedding_method. Its fingerprint settings (n_bits, radius, chirality) are used, so they
    need not be repeated in the config; for Morgan fingerprints, settings given in the config
    must agree with the file. Without embeddings_file, the settings come from the config.

    Returns:
        (embeddings dataframe or None, fingerprint settings).
    """
    config_kwargs = {key: config.get(key, default) for key, default in FINGERPRINT_DEFAULTS.items()}
    if not config.get("embeddings_file"):
        return None, config_kwargs

    path = Path(config["embeddings_file"])
    if not path.exists():
        raise FileNotFoundError(f"embeddings_file not found: {path}")
    df = pd.read_pickle(path)
    if embedding_method not in df.columns:
        available = [c for c in df.columns if c not in ("smiles", "inchi_key", "activity", "log_activity", "censored")]
        raise ValueError(f"embeddings_file {path} has no '{embedding_method}' embeddings (available: {', '.join(available)})")

    file_kwargs = df.attrs.get("embedding_kwargs")
    if file_kwargs is None:
        # Pickle without recorded settings: trust the config
        return df, config_kwargs
    if embedding_method == "morgan":
        conflicts = [
            f"{key}={config[key]!r} (file: {file_kwargs.get(key)!r})"
            for key in FINGERPRINT_DEFAULTS
            if key in config and config[key] != file_kwargs.get(key)
        ]
        if conflicts:
            raise ValueError(
                f"Config settings {', '.join(conflicts)} contradict embeddings_file {path}; "
                "remove them from the config or use a matching embeddings file"
            )
    return df, dict(file_kwargs)


def load_dataset(
    spec: DatasetSpec,
    embedding_method: str = "morgan",
    n_compounds: int | None = None,
    embeddings_df: pd.DataFrame | None = None,
    **embedding_kwargs,
) -> pd.DataFrame:
    """
    Load a dataset with embeddings for the given method.

    Uses embeddings_df if given (see resolve_embeddings). Otherwise uses the default pickle
    from `embed` (data/<name>_embeddings.pkl) when it exists, contains the requested method
    and matches the fingerprint settings, and else computes the embeddings from the CSV.
    """
    if embeddings_df is None:
        default_pkl = Path("data") / f"{spec.name}_embeddings.pkl"
        if default_pkl.exists():
            df = pd.read_pickle(default_pkl)
            # Only Morgan fingerprints depend on embedding_kwargs
            settings_match = embedding_method != "morgan" or df.attrs.get("embedding_kwargs") == embedding_kwargs
            if embedding_method in df.columns and settings_match:
                embeddings_df = df

    if embeddings_df is not None:
        df = embeddings_df.rename(columns={embedding_method: "embedding"})
        if n_compounds is not None:
            df = df.head(n_compounds)
        return df[["smiles", "inchi_key", "activity", "log_activity", "embedding", "censored"]]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task_load = progress.add_task("[cyan]Loading dataset...", total=1)
        df = pd.read_csv(spec.csv_path)
        progress.update(task_load, completed=1)

        # Trim to n_compounds if specified
        if n_compounds is not None:
            df = df.head(n_compounds)

        # Rename columns to standardize
        df = df.rename(columns={spec.smiles_column: "smiles", spec.activity_column: "activity"})

        # log_activity with unit shift, and censored flag (same as `embed`)
        df = add_log_activity_and_censoring(df, spec)

        # Add InChI keys
        n_molecules = len(df)
        task_inchi = progress.add_task("[cyan]Calculating InChI keys...", total=n_molecules)
        inchi_keys = []
        for smiles in df["smiles"]:
            inchi_keys.append(get_inchi_key(smiles))
            progress.update(task_inchi, advance=1)
        df["inchi_key"] = inchi_keys

        # Deduplicate on inchi_key
        n_before = len(df)
        df = df.drop_duplicates(subset="inchi_key", keep="first")
        n_molecules = len(df)
        if n_before != n_molecules:
            progress.console.print(
                f"[yellow]Removed {n_before - n_molecules} duplicate molecules based on InChI key[/yellow]"
            )

        # Calculate embeddings
        task_embed = progress.add_task("[cyan]Calculating embeddings...", total=n_molecules)
        embeddings = []
        for smiles in df["smiles"]:
            embeddings.append(get_embedding(smiles, embedding_method, **embedding_kwargs))
            progress.update(task_embed, advance=1)
        df["embedding"] = embeddings

    return df[["smiles", "inchi_key", "activity", "log_activity", "embedding", "censored"]]


def analyze_triple(config_name: str):
    """
    Run the activity cliffs analysis pipeline from a config file.

    Loads config from configs/<config_name>.json and infers all parameters
    (dataset, embedding_method, distance_metric, etc.).

    Args:
        config_name: Name of the config file (without .json extension) in configs/.

    Returns:
        Number of compound pairs processed.
    """
    config = load_config(config_name)

    embedding_method = config.get("embedding_method")
    distance_metric = config.get("distance_metric")
    if embedding_method is None:
        raise ValueError("Config must contain 'embedding_method'")
    if distance_metric is None:
        raise ValueError("Config must contain 'distance_metric'")

    spec = resolve_dataset(config)
    dataset_name = spec.name
    n_compounds = config.get("n_compounds", None)
    embeddings_df, embedding_kwargs = resolve_embeddings(config, embedding_method)
    activity_threshold = config.get("activity_threshold", 1.0)
    mmp_file = str(spec.mmp_path) if spec.mmp_path is not None else None
    similarity_threshold = config.get("similarity_threshold", None)
    center = config.get("center", False)
    large_coords_to_trim = config.get("large_coords_to_trim", None)
    scaling_function = config.get("scaling_function", "linear")
    scaling_cutoff_percentile = config.get("scaling_cutoff_percentile", 99)
    seed = config.get("seed", 42)

    # Create results directory and metadata (need metadata_hash for path)
    analysis_metadata = {
        "dataset_name": dataset_name,
        "embedding_method": embedding_method,
        "distance_metric": distance_metric,
        "smiles_column": spec.smiles_column,
        "activity_column": spec.activity_column,
        "unit": spec.unit,
        "censor_threshold": spec.censor_threshold,
        "n_compounds": n_compounds,
        "activity_threshold": activity_threshold,
        "mmp_file": mmp_file,
        "similarity_threshold": similarity_threshold,
        "center": center,
        "large_coords_to_trim": large_coords_to_trim,
        "scaling_function": scaling_function,
        "scaling_cutoff_percentile": scaling_cutoff_percentile,
        "embeddings_file": config.get("embeddings_file"),
        "embedding_kwargs": embedding_kwargs,
        "seed": seed,
    }
    metadata_hash = hashlib.sha256(json.dumps(analysis_metadata, sort_keys=True).encode()).hexdigest()[:8]
    results_dir = Path(
        config.get("output_dir") or Path("results") / dataset_name / f"{embedding_method}_{distance_metric}_{metadata_hash}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "analysis_metadata.json", "w") as f:
        json.dump(analysis_metadata, f, indent=2)

    df = load_dataset(
        spec,
        embedding_method=embedding_method,
        n_compounds=n_compounds,
        embeddings_df=embeddings_df,
        **embedding_kwargs,
    )

    # Step 1: Pairwise data creation
    pairwise_data = create_pairwise_data(
        df,
        results_dir,
        distance_metric=distance_metric,
        center=center,
        large_coords_to_trim=large_coords_to_trim,
        scaling_function=scaling_function,
        scaling_cutoff_percentile=scaling_cutoff_percentile,
    )

    # Step 2: Cliff identification
    identify_cliffs(
        pairwise_data=pairwise_data,
        activity_threshold=activity_threshold,
        results_dir=results_dir,
        dataset_name=dataset_name,
        embedding_method=embedding_method,
    )

    # Step 3: Correlation analysis
    correlate_similarity_vs_activity(
        pairwise_data=pairwise_data,
        raw_data=df,
        results_dir=results_dir,
        activity_threshold=activity_threshold,
        mmp_file=mmp_file,
    )

    # Step 4: Persistent homology analysis
    calculate_persistent_homology(
        pairwise_data=pairwise_data,
        results_dir=results_dir,
        activity_threshold=activity_threshold,
    )

    # Step 5: Model training
    train_model(
        compounds=df,
        pairwise_data=pairwise_data,
        activity_threshold=activity_threshold,
        results_dir=results_dir,
        seed=seed,
        config=config,
    )

    # Step 6: Save summary markdown
    save_summary_markdown(
        results_dir=results_dir,
        dataset_name=dataset_name,
        embedding_method=embedding_method,
        distance_metric=distance_metric,
    )

    return len(pairwise_data)