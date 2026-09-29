from functools import cache
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from pydantic import BaseModel, Field, ValidationError
from rdkit import Chem
from rdkit.Chem import MACCSkeys, MolFromSmiles
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetRDKitFPGenerator,
    GetTopologicalTorsionGenerator,
)
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from transformers import AutoModel, AutoTokenizer, logging as transformers_logging

from src.chemistry import get_inchi_key


BIT_EMBEDDING_METHODS = ["morgan", "maccs", "rdkit", "atom_pair", "topological_torsion"]
CONTINUOUS_EMBEDDING_METHODS = ["molformer", "chemberta", "chemeleon"]
EMBEDDING_METHODS = BIT_EMBEDDING_METHODS + CONTINUOUS_EMBEDDING_METHODS
MOLFORMER_MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
# Pinned Hub revision: MolFormer ships remote code, and newer revisions require a newer transformers
MOLFORMER_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"
CHEMBERTA_MODEL_NAME = 'DeepChem/ChemBERTa-77M-MLM'
CHEMELEON_CHECKPOINT_URL = 'https://zenodo.org/records/15460715/files/chemeleon_mp.pt'

# Global CheMeleon model instance (lazy-loaded)
_chemeleon_model = None


class Embedding(BaseModel):
    smiles: str = Field(description="The SMILES string of the molecule.")
    inchi_key: str = Field(description="The InChI key of the molecule.")
    activity: float = Field(description="The activity of the molecule.")
    # Embeddings are optional: only the methods selected for `embed` are computed
    # Continuous embeddings
    molformer: list[float] | None = Field(default=None, description="The MolFormer embedding of the molecule.", embedding_type="continuous")
    chemberta: list[float] | None = Field(default=None, description="The ChemBERTa embedding of the molecule.", embedding_type="continuous")
    chemeleon: list[float] | None = Field(default=None, description="The CheMeleon embedding of the molecule.", embedding_type="continuous")
    # bit embeddings
    morgan: list[int] | None = Field(default=None, description="The Morgan fingerprint of the molecule.", embedding_type="bit")
    maccs: list[int] | None = Field(default=None, description="The MACCS keys fingerprint of the molecule.", embedding_type="bit")
    rdkit: list[int] | None = Field(default=None, description="The RDKit fingerprint of the molecule.", embedding_type="bit")
    atom_pair: list[int] | None = Field(default=None, description="The Atom Pair fingerprint of the molecule.", embedding_type="bit")
    topological_torsion: list[int] | None = Field(default=None, description="The Topological Torsion fingerprint of the molecule.", embedding_type="bit")


def _get_chemeleon_model(device: str | torch.device | None = None):
    """
    Get or initialize the CheMeleon model (lazy-loaded singleton).
    
    Args:
        device: Device to run the model on. If None, uses default device.
    
    Returns:
        CheMeleon model instance.
    """
    global _chemeleon_model
    
    if _chemeleon_model is None:
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        # Use models/ directory in project root instead of hidden directory
        project_root = Path(__file__).parent.parent
        ckpt_dir = project_root / "models"
        ckpt_dir.mkdir(exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        
        if not mp_path.exists():
            # Download checkpoint if not present
            print(f"Downloading CheMeleon checkpoint to {mp_path}...")
            urlretrieve(CHEMELEON_CHECKPOINT_URL, mp_path)
        
        chemeleon_mp = torch.load(mp_path, weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        mp.load_state_dict(chemeleon_mp["state_dict"])
        _chemeleon_model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim),  # not actually used
        )
        _chemeleon_model.eval()
        if device is not None:
            _chemeleon_model.to(device=device)
    
    return _chemeleon_model


def get_chemeleon_embedding(smiles: str, device: str | torch.device | None = None) -> list[float]:
    """
    Get the CheMeleon embedding of a molecule.
    
    Args:
        smiles: SMILES string of the molecule.
        device: Device to run the model on. If None, uses default device.
    
    Returns:
        List of floats representing the CheMeleon embedding.
    
    Raises:
        ImportError: If chemprop is not installed.
        ValueError: If SMILES string is invalid.
    """

    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    model = _get_chemeleon_model(device=device)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    
    bmg = BatchMolGraph([featurizer(mol)])
    bmg.to(device=model.device)
    
    with torch.no_grad():
        embedding = model.fingerprint(bmg).numpy(force=True)
    
    return embedding.flatten().tolist()


def _models_dir() -> Path:
    """models/ directory in the project root, used instead of the default Hugging Face cache."""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


@cache
def _load_molformer():
    """Load the MolFormer model and tokenizer once per process."""
    model = AutoModel.from_pretrained(
        MOLFORMER_MODEL_NAME,
        revision=MOLFORMER_REVISION,
        cache_dir=str(_models_dir()),
        deterministic_eval=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MOLFORMER_MODEL_NAME,
        revision=MOLFORMER_REVISION,
        cache_dir=str(_models_dir()),
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


@cache
def _load_chemberta():
    """Load the ChemBERTa model and tokenizer once per process."""
    # Suppress the warning about the untrained pooler weights (not used, see get_chemberta_embedding)
    original_verbosity = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity_error()
        model = AutoModel.from_pretrained(
            CHEMBERTA_MODEL_NAME,
            cache_dir=str(_models_dir()),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            CHEMBERTA_MODEL_NAME,
            cache_dir=str(_models_dir()),
        )
    finally:
        transformers_logging.set_verbosity(original_verbosity)
    model.eval()
    return model, tokenizer


def get_molformer_embedding(smiles: str) -> list[float]:
    """
    Get the MolFormer embedding of a molecule.
    """
    model, tokenizer = _load_molformer()
    inputs = tokenizer([smiles], padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.detach().numpy().flatten().tolist()


def get_chemberta_embedding(smiles: str) -> list[float]:
    """
    Get the ChemBERTa embedding of a molecule.
    """
    model, tokenizer = _load_chemberta()
    inputs = tokenizer([smiles], padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling of the last hidden state. The checkpoint has no trained pooler, so
    # pooler_output would come from randomly initialized weights (different on every load).
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy().flatten().tolist()


def get_rdkit_embedding(mol: Chem.Mol, n_bits: int = 1024) -> list[float]:
    """
    Get the RDKit embedding of a molecule.
    """
    generator = GetRDKitFPGenerator(fpSize=n_bits)
    return generator.GetFingerprintAsNumPy(mol).flatten().tolist()


def get_atom_pair_embedding(mol: Chem.Mol, n_bits: int = 1024) -> list[float]:
    """
    Get the Atom Pair embedding of a molecule.
    """
    generator = GetAtomPairGenerator(fpSize=n_bits)
    return generator.GetFingerprintAsNumPy(mol).flatten().tolist()


def get_topological_torsion_embedding(mol: Chem.Mol, n_bits: int = 1024) -> list[float]:
    """
    Get the Topological Torsion embedding of a molecule.
    """
    generator = GetTopologicalTorsionGenerator(fpSize=n_bits)
    return generator.GetFingerprintAsNumPy(mol).flatten().tolist()


def get_morgan_fingerprint(
    mol: Chem.Mol, n_bits: int = 1024, radius: int = 2, chirality: bool = False
) -> np.ndarray:
    """
    Get the Morgan fingerprint of a molecule.

    Args:
        mol: RDKit Mol object.
        n_bits: Length of the fingerprint vector. Default is 1024.
        radius: Radius of the Morgan fingerprint. Default is 2.
        chirality: Include stereochemistry (R/S) in atom invariants, so that
            stereoisomers get different fingerprints. Default is False.

    Returns:
        numpy array representing the Morgan fingerprint.
    """
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=chirality)
    return generator.GetFingerprintAsNumPy(mol)


def get_maccs_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """
    Get the MACCS keys fingerprint of a molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        numpy array representing the MACCS keys fingerprint.
    """
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)


def get_embedding(smiles: str, method: str, **kwargs) -> np.ndarray:
    """
    Get the embedding of a molecule using the specified method.

    Args:
        smiles: SMILES string of the molecule.
        method: Embedding method to use. Supported: 'morgan', 'maccs'.
        **kwargs: Additional arguments passed to the fingerprint function.

    Returns:
        numpy array representing the molecular embedding.
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    if method == "morgan":
        return get_morgan_fingerprint(mol, **kwargs)
    elif method == "maccs":
        return get_maccs_fingerprint(mol)
    elif method == "molformer":
        return get_molformer_embedding(smiles)
    elif method == "chemberta":
        return np.array(get_chemberta_embedding(smiles))
    elif method == "rdkit":
        return get_rdkit_embedding(mol)
    elif method == "atom_pair":
        return get_atom_pair_embedding(mol)
    elif method == "topological_torsion":
        return get_topological_torsion_embedding(mol)
    elif method == "chemeleon":
        return np.array(get_chemeleon_embedding(smiles))
    else:
        raise ValueError(f"Unknown embedding method: {method}. Supported methods: {', '.join(EMBEDDING_METHODS)}")


def get_all_embeddings(smiles: str, methods: list[str] | None = None, **kwargs) -> dict:
    """
    Get embeddings of a molecule for the given methods (all of EMBEDDING_METHODS if None).
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    embeddings_dict = {}
    for method in methods or EMBEDDING_METHODS:
        if method == "chemeleon":
            # CheMeleon failures don't fail the molecule (field is optional)
            try:
                embeddings_dict["chemeleon"] = get_chemeleon_embedding(smiles)
            except Exception as e:
                print(f"Warning: Failed to compute CheMeleon embedding for {smiles}: {e}")
                embeddings_dict["chemeleon"] = None
        else:
            embeddings_dict[method] = get_embedding(smiles, method, **kwargs)

    return embeddings_dict


def embed(
    method: str,
    df: pd.DataFrame | None = None,
    input_path: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Add embeddings to a DataFrame containing SMILES.

    Either provide a DataFrame directly, or load from a pickle file with
    pre-computed embeddings.

    Args:
        method: Embedding method to use (one of EMBEDDING_METHODS).
        df: DataFrame with a 'smiles' column. If None, input_path must be provided.
        input_path: Path to a pickle file with pre-computed embeddings.
            If provided, loads the file and selects 'smiles', 'activity', 
            and the column matching `method` (renamed to 'embedding').
        **kwargs: Additional arguments passed to the fingerprint function
            (only used when computing embeddings from df).

    Returns:
        DataFrame with 'smiles', 'activity' (if available), and 'embedding' columns.
    """
    if input_path is not None:
        # Load from pickle and select relevant columns
        df_loaded = pd.read_pickle(input_path)
        
        if method not in df_loaded.columns:
            raise ValueError(
                f"Embedding method '{method}' not found in pickle file. "
                f"Available columns: {list(df_loaded.columns)}"
            )
        
        # Select columns
        columns_to_keep = ["smiles"]
        if "activity" in df_loaded.columns:
            columns_to_keep.append("activity")
        columns_to_keep.append(method)
        
        result = df_loaded[columns_to_keep].copy()
        result = result.rename(columns={method: "embedding"})
        return result
    
    elif df is not None:
        # Compute embeddings from DataFrame
        df = df.copy()
        df["embedding"] = df["smiles"].apply(lambda s: get_embedding(s, method, **kwargs))
        return df
    
    else:
        raise ValueError("Either 'df' or 'input_path' must be provided.")


def save_all_embeddings(
    output_path: str,
    input_path: str | None = None,
    df: pd.DataFrame | None = None,
    smiles_column: str = "smiles",
    activity_column: str = "activity",
    save_csv: bool = False,
    methods: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate embeddings for each molecule and save the dataframe.

    For each SMILES in the dataframe, calculates the selected embeddings
    (all supported embeddings if methods is None) and adds them as new columns. Each row is validated against the Embedding model.
    Invalid rows are skipped with a warning. The result is saved as a pickle file.

    Args:
        output_path: Path to save the pickled dataframe.
        input_path: Path to a CSV file with the dataset.
        df: DataFrame with a SMILES column. If None, input_path must be provided.
        smiles_column: Name of the SMILES column. Default is 'smiles'.
        activity_column: Name of the activity column. Default is 'activity'.
        methods: Embedding methods to compute. Default (None) computes all of EMBEDDING_METHODS.
        **kwargs: Additional arguments passed to get_all_embeddings (e.g., n_bits, radius).

    Returns:
        DataFrame with validated embedding data matching the Embedding model schema.
    """
    
    if input_path is not None:
        df = pd.read_csv(input_path)
    elif df is not None:
        df = df.copy()
    else:
        raise ValueError("Either 'input_path' or 'df' must be provided.")

    # Rename smiles column if needed
    if smiles_column != "smiles":
        df = df.rename(columns={smiles_column: "smiles"})

    # Rename activity column if needed
    if activity_column != "activity":
        df = df.rename(columns={activity_column: "activity"})

    valid_rows: list[dict] = []
    n_total = len(df)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Calculating embeddings and validating...", total=n_total)

        for idx, row in df.iterrows():
            smiles = row["smiles"]
            activity = row["activity"]

            try:
                # Calculate all embeddings
                embeddings = get_all_embeddings(smiles, methods=methods, **kwargs)

                # Calculate derived fields
                inchi_key = get_inchi_key(smiles)

                # Build row data
                row_data = {
                    "smiles": smiles,
                    "activity": activity,
                    "inchi_key": inchi_key,
                    **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in embeddings.items()},
                }

                # Validate against Embedding model
                Embedding(**row_data)

                # If validation passes, add to valid rows
                valid_rows.append(row_data)

            except ValidationError as e:
                progress.console.print(f"[yellow]Validation failed for {smiles}: {e}[/yellow]")
            except Exception as e:
                progress.console.print(f"[red]Error processing {smiles}: {e}[/red]")

            progress.update(task, advance=1)

        # Create DataFrame from valid rows
        result_df = pd.DataFrame(valid_rows)

        # Deduplicate on inchi_key
        n_before = len(result_df)
        result_df = result_df.drop_duplicates(subset="inchi_key", keep="first")
        n_after = len(result_df)
        if n_before != n_after:
            progress.console.print(
                f"[yellow]Removed {n_before - n_after} duplicate molecules based on InChI key[/yellow]"
            )

        # Report validation summary
        n_invalid = n_total - len(valid_rows)
        if n_invalid > 0:
            progress.console.print(
                f"[yellow]Skipped {n_invalid} invalid molecules out of {n_total}[/yellow]"
            )

        # Save as pickle
        progress.console.print(f"[cyan]Saving to {output_path}...[/cyan]")
        output_path_pkl = Path(output_path + ".pkl")
        output_path_pkl.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_pickle(output_path_pkl)
        if save_csv:
            # Save as csv
            output_path_csv = Path(output_path + ".csv")
            output_path_csv.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path_csv, index=False)
        progress.console.print(f"[green]✓ Saved {len(result_df)} validated molecules with embeddings[/green]")

    return result_df


def load_embeddings(path: str) -> pd.DataFrame:
    """
    Load a dataframe with embeddings from a pickle file.

    Args:
        path: Path to the pickle file.

    Returns:
        DataFrame with embedding columns.
    """
    return pd.read_pickle(path)
