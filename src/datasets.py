"""
Dataset resolution for activity cliffs analysis.

A config's "dataset" entry is either the key of a standard dataset defined in
src/dataset_sources.json (downloaded from its permalink and cached in data/), or
a path to a local CSV file.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

DATASET_SOURCES_PATH = Path(__file__).parent / "dataset_sources.json"
DATA_DIR = Path("data")

# Shift added to -log10(activity) to express activities on a molar (pIC50-like) scale
UNIT_SHIFTS = {
    "nM": 9,
    "uM": 6,
    "mM": 3,
    "M": 0,
}


@dataclass
class DatasetSpec:
    """Resolved dataset: name used for outputs, local CSV path, and column/unit settings."""

    name: str
    csv_path: Path
    mmp_path: Path | None
    smiles_column: str
    activity_column: str
    unit: str
    censor_threshold: float | None


def load_dataset_sources() -> dict[str, dict]:
    """Load the standard dataset definitions from src/dataset_sources.json."""
    with open(DATASET_SOURCES_PATH) as f:
        return json.load(f)


def _to_raw_url(url: str) -> str:
    """Convert a GitHub file page URL (.../blob/<ref>/<path>) to its raw download URL."""
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    if match:
        owner, repo, rest = match.groups()
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{rest}"
    return url


def _download(url: str, target: Path) -> Path:
    """Download url to target unless it is already cached."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        urlretrieve(_to_raw_url(url), tmp)
        tmp.rename(target)
    return target


def resolve_dataset_name(dataset: str) -> str:
    """Name used for output directories: the standard dataset key, or the CSV file stem."""
    if dataset in load_dataset_sources():
        return dataset
    return Path(dataset).stem


def resolve_dataset(config: dict) -> DatasetSpec:
    """
    Resolve the dataset described by a config.

    config["dataset"] is either a key of src/dataset_sources.json or a path to a local CSV.
    For standard datasets the activity and MMP files are downloaded to data/ on first use and
    column names, unit and censor threshold default to the values in dataset_sources.json.
    Any of smiles_column, activity_column, unit, censor_threshold and mmp_file given in the
    config take precedence.
    """
    dataset = config.get("dataset")
    if dataset is None:
        raise ValueError("Config must contain 'dataset' (a standard dataset name or a path to a CSV file)")

    sources = load_dataset_sources()
    if dataset in sources:
        source = sources[dataset]
        name = dataset
        csv_path = _download(source["activity"], DATA_DIR / f"{name}_molecules.csv")
        mmp_path = _download(source["mmp"], DATA_DIR / f"{name}_MMPs.csv") if source.get("mmp") else None
        defaults = source
    else:
        csv_path = Path(dataset)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset}' is neither a standard dataset ({', '.join(sources)}) nor an existing file"
            )
        name = csv_path.stem
        mmp_path = None
        defaults = {}

    if config.get("mmp_file"):
        mmp_path = Path(config["mmp_file"])

    unit = config.get("unit", defaults.get("unit", "nM"))
    if unit not in UNIT_SHIFTS:
        raise ValueError(f"Invalid unit '{unit}'. Supported units: {list(UNIT_SHIFTS.keys())}")

    return DatasetSpec(
        name=name,
        csv_path=csv_path,
        mmp_path=mmp_path,
        smiles_column=config.get("smiles_column", defaults.get("smiles_column", "smiles")),
        activity_column=config.get("activity_column", defaults.get("activity_column", "activity")),
        unit=unit,
        censor_threshold=config.get("censor_threshold", defaults.get("censor_threshold")),
    )


def add_log_activity_and_censoring(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Add log_activity (-log10(activity) + unit shift) and the censored flag to a dataframe."""
    df["log_activity"] = -np.log10(df["activity"]) + UNIT_SHIFTS[spec.unit]
    if spec.censor_threshold is not None:
        df["censored"] = df["activity"] >= spec.censor_threshold
    else:
        df["censored"] = False
    return df
