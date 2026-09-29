import os
import sys

# On macOS, torch, scikit-learn, diptest and XGBoost each bundle their own OpenMP runtime.
# Running several of them multithreaded in one process segfaults XGBoost or deadlocks torch,
# so default to a single OpenMP thread there. Must be set before those libraries are imported.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import click
from rich.console import Console
from rich.panel import Panel

# Install before the heavy imports below, so warnings raised while importing are rendered too
from src.utils import install_rich_warnings

install_rich_warnings()

from src.single_dataset_parser import analyze_triple as run_analyze
from src.comparison import compare as run_compare
from src.dataset_preparation import prepare_dataset


def _print_done(message: str) -> None:
    """Print a command's final message in a green panel."""
    Console().print(Panel(message, title="[bold]✓ Done[/bold]", title_align="left", border_style="green", expand=False))


@click.group()
def cli():
    """Activity Cliffs Analysis Tool."""
    pass


@cli.command()
@click.argument("config_name")
def analyze(config_name: str) -> None:
    """
    Run the activity cliffs analysis pipeline using a config file.

    CONFIG_NAME: Name of the config file (without .json extension) in the configs/ folder.

    The config file should contain:
    - dataset: (required) Standard dataset name (SARS_CoV2, ChEMBL_Dopamine_D2, ChEMBL_Factor_Xa)
      or path to a CSV file
    - embedding_method: (required) Embedding method to use
    - distance_metric: (required) Distance metric to use
    - smiles_column, activity_column, unit, censor_threshold: (optional) Dataset settings;
      defaults come from src/dataset_sources.json for standard datasets
    - mmp_file: (optional) CSV with matched molecular pairs (default: source MMPs for standard datasets)
    - n_compounds: (optional) Number of compounds to use (default: all)
    - embeddings_file: (optional) Pickle from `embed` to take embeddings and fingerprint settings from
    - output_dir: (optional) Results directory (default: results/<name>/<embedding>_<metric>_<hash>)
    - n_bits, radius, chirality: (optional) Morgan settings when computing fingerprints
      (default: 1024, 2, False); taken from embeddings_file if given
    - activity_threshold: (optional) Log ratio threshold for cliffs (default: 1.0)
    - similarity_threshold: (optional) Similarity threshold for filtering (default: None)
    - center: (optional) Whether to center embeddings by subtracting the mean (default: False)
    - large_coords_to_trim: (optional) If specified, trim embeddings to this number of coordinates (default: None)
    - scaling_function: (optional) Function for scaling distances to [0, 1] (default: 'linear')
    - scaling_cutoff_percentile: (optional) Percentile to use for clipping distances before scaling (default: 99)

    See README.md for full configuration documentation.
    """
    try:
        results = run_analyze(config_name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except ValueError as e:
        raise click.ClickException(str(e))

    _print_done(f"Processed [bold]{results:,}[/bold] compound pairs.")


@cli.command()
@click.argument("config_name")
def compare(config_name: str) -> None:
    """
    Compare results from multiple analysis runs.

    CONFIG_NAME: Name of the comparison config file (without .json extension) in the configs/ folder.

    Requires a config file specifying which results to compare.
    """
    try:
        run_compare(config_name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except ValueError as e:
        raise click.ClickException(str(e))

    _print_done("Comparison complete.")


@cli.command()
@click.argument("config_name")
def embed(config_name: str) -> None:
    """
    Calculate embeddings and dataset statistics using a config file.

    CONFIG_NAME: Name of the config file (without .json extension) in the configs/ folder.

    The config file should contain:
    - dataset: (required) Standard dataset name (SARS_CoV2, ChEMBL_Dopamine_D2, ChEMBL_Factor_Xa)
      or path to a CSV file
    - methods: (optional) List of embedding methods to compute (default: all)
    - output_file: (optional) Path of the output .pkl file (default: data/<name>_embeddings.pkl)
    - smiles_column, activity_column, unit, censor_threshold: (optional) Dataset settings;
      defaults come from src/dataset_sources.json for standard datasets
    - n_bits: (optional) Number of bits for fingerprints (default: 1024)
    - radius: (optional) Radius for Morgan fingerprint (default: 2)
    - chirality: (optional) Include stereochemistry in Morgan fingerprint (default: False)
    - save_csv: (optional) Whether to also save as CSV (default: False)

    See README.md for full configuration documentation.
    """
    try:
        result_df = prepare_dataset(config_name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except ValueError as e:
        raise click.ClickException(str(e))

    _print_done(f"Built embeddings for [bold]{len(result_df):,}[/bold] molecules.")


if __name__ == "__main__":
    cli()
