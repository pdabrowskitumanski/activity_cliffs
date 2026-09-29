"""
Model training module for activity cliff prediction.

This module provides functions for training models to predict activity cliffs
from molecular embeddings using train/test splits based on Butina clustering
and scaffold (Murcko) splitting.
"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from rich.console import Console
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb

class SetEncoder(nn.Module):
    """
    Encodes variable-length set of vectors (n, feat_dim) into a single vector (feat_dim).
    Uses learned attention over the set so the full set is used rather than fixed pooling.
    For 1D input (batch, feat_dim) passes through unchanged (identity).
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.attn = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        # x: (batch, n, feat_dim)
        scores = self.attn(x)
        alpha = torch.softmax(scores, dim=1)
        return (alpha * x).sum(dim=1)


class SiameseCliffNet(nn.Module):
    """
    Siamese network for predicting activity cliffs from molecular embeddings.

    Takes two embeddings (1D vectors or variable-length (n_atoms, emb_dim)) and
    predicts the probability of an activity cliff using the absolute difference
    between encoded representations.
    """
    def __init__(self, emb_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.encoder = SetEncoder(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z1: First embedding [batch_size, emb_dim] or [batch_size, n1, emb_dim]
            z2: Second embedding [batch_size, emb_dim] or [batch_size, n2, emb_dim]

        Returns:
            Logits [batch_size] (before sigmoid).
        """
        z1 = self.encoder(z1)
        z2 = self.encoder(z2)
        diff = torch.abs(z1 - z2)
        return self.mlp(diff).squeeze(-1)


class PairDataset(Dataset):
    """
    PyTorch Dataset for molecular pairs.
    
    Args:
        pairs: List of (pair_key, inchi_key1, inchi_key2, label) tuples.
        compounds: DataFrame with 'inchi_key' and 'embedding' columns.
    """
    def __init__(self, pairs: list[tuple[str, str, str, int]], compounds: pd.DataFrame):
        self.pairs = pairs
        # Create mapping from InChI key to embedding
        self.embeddings = {
            row["inchi_key"]: np.array(row["embedding"], dtype=np.float32)
            for _, row in compounds.iterrows()
        }
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair_key, inchi_key1, inchi_key2, label = self.pairs[idx]
        
        emb1 = torch.tensor(self.embeddings[inchi_key1], dtype=torch.float32)
        emb2 = torch.tensor(self.embeddings[inchi_key2], dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return emb1, emb2, label_tensor


def _collate_variable_embeddings(batch: list) -> tuple:
    """
    Collate batch of (emb1, emb2, label) where emb1/emb2 may have variable length
    (e.g. (n1, 384), (n2, 384)). Returns (list of emb1, list of emb2, tensor of labels)
    so the trainer can encode then stack, or process per pair.
    """
    emb1_list = [item[0] for item in batch]
    emb2_list = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])
    return emb1_list, emb2_list, labels


# ============================================================================
# Helper Functions
# ============================================================================

def _butina_split(
    compounds: pd.DataFrame,
    pairwise_data: dict[str, dict[str, float]],
    cutoff: float = 0.35,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split compounds into train, validation, and test sets using Butina clustering.

    Uses distances from pairwise_data['distance_scaled'] to build a distance matrix.

    Args:
        compounds: DataFrame with 'inchi_key' column.
        pairwise_data: Dictionary mapping "inchi_key_1,inchi_key_2" pairs to
            dict containing 'distance_scaled', etc.
        cutoff: Distance cutoff for Butina clustering. Default is 0.35.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices, test_indices) as lists of DataFrame indices.
    """
    random.seed(seed)
    np.random.seed(seed)

    inchi_keys = compounds["inchi_key"].tolist()
    n = len(inchi_keys)

    if n < 3:
        raise ValueError("Need at least 3 compounds for Butina train/val/test splitting")

    # Build distance matrix from pairwise_data
    # Create mapping from InChI key to index
    key_to_idx = {key: idx for idx, key in enumerate(inchi_keys)}
    
    # Initialize distance matrix with NaN
    dist_matrix = np.full((n, n), np.nan)
    np.fill_diagonal(dist_matrix, 0.0)  # Distance to self is 0

    # Fill distance matrix from pairwise_data
    for pair_key, data in pairwise_data.items():
        parts = pair_key.split(",")
        if len(parts) != 2:
            continue
        
        inchi_key1, inchi_key2 = parts
        idx1 = key_to_idx.get(inchi_key1)
        idx2 = key_to_idx.get(inchi_key2)
        
        if idx1 is not None and idx2 is not None:
            distance = data.get("distance_scaled", np.nan)
            if not np.isnan(distance):
                dist_matrix[idx1, idx2] = distance
                dist_matrix[idx2, idx1] = distance

    # Convert distance matrix to list format for Butina clustering
    # Butina.ClusterData expects a list of distances in upper triangular format
    dists = []
    for i in range(1, n):
        for j in range(i):
            dist = dist_matrix[i, j]
            if np.isnan(dist):
                # If distance is missing, use a large default value
                dist = 1.0
            dists.append(dist)

    # Perform Butina clustering
    # Butina.ClusterData expects a list of distances and returns cluster indices
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

    # Split clusters into train / val / test (3-way alternating assignment)
    train_indices = []
    val_indices = []
    test_indices = []

    for i, cluster in enumerate(clusters):
        if i % 3 == 0:
            train_indices.extend(cluster)
        elif i % 3 == 1:
            val_indices.extend(cluster)
        else:
            test_indices.extend(cluster)

    # If any set is empty, redistribute
    if len(train_indices) == 0:
        all_idx = val_indices + test_indices
        n = len(all_idx)
        train_indices = all_idx[: n // 3]
        val_indices = all_idx[n // 3 : 2 * n // 3]
        test_indices = all_idx[2 * n // 3 :]
    elif len(val_indices) == 0:
        all_idx = train_indices + test_indices
        n = len(all_idx)
        train_indices = all_idx[: n // 3]
        val_indices = all_idx[n // 3 : 2 * n // 3]
        test_indices = all_idx[2 * n // 3 :]
    elif len(test_indices) == 0:
        all_idx = train_indices + val_indices
        n = len(all_idx)
        train_indices = all_idx[: n // 3]
        val_indices = all_idx[n // 3 : 2 * n // 3]
        test_indices = all_idx[2 * n // 3 :]

    return train_indices, val_indices, test_indices


def _scaffold_split(
    compounds: pd.DataFrame,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split compounds into train, validation, and test sets using Murcko scaffold.

    Molecules sharing the same Bemis-Murcko scaffold are kept in the same set
    (train or test) to assess out-of-distribution generalization.

    Args:
        compounds: DataFrame with 'smiles' column.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices, test_indices) as lists of DataFrame indices.
    """
    random.seed(seed)
    np.random.seed(seed)

    if "smiles" not in compounds.columns:
        raise ValueError("Compounds must have 'smiles' column for scaffold split")

    n = len(compounds)
    if n < 3:
        raise ValueError("Need at least 3 compounds for scaffold train/val/test splitting")

    # Assign each compound to a scaffold (SMILES string)
    scaffold_to_indices: dict[str, list[int]] = {}
    for idx in range(n):
        smi = compounds.iloc[idx]["smiles"]
        if not isinstance(smi, str):
            scaffold_key = "__invalid__"
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scaffold_key = "__invalid__"
            else:
                try:
                    scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                    scaffold_key = scaffold_smi if scaffold_smi else "__empty__"
                except Exception:
                    scaffold_key = "__invalid__"
        scaffold_to_indices.setdefault(scaffold_key, []).append(idx)

    scaffold_keys_sorted = sorted(
        scaffold_to_indices.keys(),
        key=lambda k: len(scaffold_to_indices[k]),
        reverse=True,
    )

    train_indices = []
    val_indices = []
    test_indices = []
    for i, key in enumerate(scaffold_keys_sorted):
        indices = scaffold_to_indices[key]
        if i % 3 == 0:
            train_indices.extend(indices)
        elif i % 3 == 1:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

    if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
        all_idx = train_indices + val_indices + test_indices
        n_all = len(all_idx)
        train_indices = all_idx[: n_all // 3]
        val_indices = all_idx[n_all // 3 : 2 * n_all // 3]
        test_indices = all_idx[2 * n_all // 3 :]

    return train_indices, val_indices, test_indices


def _create_pair_splits(
    compounds: pd.DataFrame,
    train_indices: list[int],
    test_indices: list[int],
    pairwise_data: dict[str, dict[str, float]],
    activity_threshold: float,
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    """
    Create train and test pair splits based on compound splits.

    Args:
        compounds: DataFrame with 'inchi_key' column.
        train_indices: List of train compound indices.
        test_indices: List of test compound indices.
        pairwise_data: Dictionary of pairwise data.
        activity_threshold: Threshold for labeling cliffs.

    Returns:
        Tuple of (train_pairs, test_pairs) where each is a list of (pair_key, inchi_key1, inchi_key2, label).
    """
    # Create sets of InChI keys for fast lookup
    train_inchi_keys = set(compounds.iloc[train_indices]["inchi_key"].tolist())
    test_inchi_keys = set(compounds.iloc[test_indices]["inchi_key"].tolist())

    train_pairs = []
    test_pairs = []

    for pair_key, data in pairwise_data.items():
        # Skip excluded pairs
        if data.get("status") == "excluded":
            continue

        # Parse pair key
        inchi_key1, inchi_key2 = pair_key.split(",")

        # Determine which split this pair belongs to
        in_train1 = inchi_key1 in train_inchi_keys
        in_train2 = inchi_key2 in train_inchi_keys
        in_test1 = inchi_key1 in test_inchi_keys
        in_test2 = inchi_key2 in test_inchi_keys

        # Assign label: 1 if log_ratio > activity_threshold, 0 otherwise
        log_ratio = data.get("log_ratio", 0.0)
        if np.isnan(log_ratio):
            continue
        label = 1 if log_ratio > activity_threshold else 0

        # Add to appropriate split (no cross-pairs)
        if in_train1 and in_train2:
            train_pairs.append((pair_key, inchi_key1, inchi_key2, label))
        elif in_test1 and in_test2:
            test_pairs.append((pair_key, inchi_key1, inchi_key2, label))
        # Skip cross-pairs (train-test or test-train)

    return train_pairs, test_pairs


def _create_pair_splits_three_way(
    compounds: pd.DataFrame,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    pairwise_data: dict[str, dict[str, float]],
    activity_threshold: float,
) -> tuple[
    list[tuple[str, str, str, int]],
    list[tuple[str, str, str, int]],
    list[tuple[str, str, str, int]],
]:
    """
    Create train-train, val-val, and test-test pair splits (no cross-split pairs).

    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs).
    """
    train_inchi = set(compounds.iloc[train_indices]["inchi_key"].tolist())
    val_inchi = set(compounds.iloc[val_indices]["inchi_key"].tolist())
    test_inchi = set(compounds.iloc[test_indices]["inchi_key"].tolist())

    train_pairs = []
    val_pairs = []
    test_pairs = []

    for pair_key, data in pairwise_data.items():
        if data.get("status") == "excluded":
            continue
        inchi_key1, inchi_key2 = pair_key.split(",")
        log_ratio = data.get("log_ratio", 0.0)
        if np.isnan(log_ratio):
            continue
        label = 1 if log_ratio > activity_threshold else 0

        in_train1, in_train2 = inchi_key1 in train_inchi, inchi_key2 in train_inchi
        in_val1, in_val2 = inchi_key1 in val_inchi, inchi_key2 in val_inchi
        in_test1, in_test2 = inchi_key1 in test_inchi, inchi_key2 in test_inchi

        if in_train1 and in_train2:
            train_pairs.append((pair_key, inchi_key1, inchi_key2, label))
        elif in_val1 and in_val2:
            val_pairs.append((pair_key, inchi_key1, inchi_key2, label))
        elif in_test1 and in_test2:
            test_pairs.append((pair_key, inchi_key1, inchi_key2, label))

    return train_pairs, val_pairs, test_pairs


def _balance_dataset(pairs: list[tuple[str, str, str, int]], seed: int = 42) -> list[tuple[str, str, str, int]]:
    """
    Balance a dataset by randomly sampling to get the largest balanced dataset.

    Args:
        pairs: List of (pair_key, inchi_key1, inchi_key2, label) tuples.
        seed: Random seed for reproducibility.

    Returns:
        Balanced list of pairs.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Separate pairs by label
    label_0_pairs = [p for p in pairs if p[3] == 0]
    label_1_pairs = [p for p in pairs if p[3] == 1]

    # Find the minimum count
    min_count = min(len(label_0_pairs), len(label_1_pairs))

    if min_count == 0:
        return []

    # Randomly sample to balance
    balanced_label_0 = random.sample(label_0_pairs, min_count)
    balanced_label_1 = random.sample(label_1_pairs, min_count)

    # Combine and shuffle
    balanced_pairs = balanced_label_0 + balanced_label_1
    random.shuffle(balanced_pairs)

    return balanced_pairs


def _pairs_to_absdiff_features(
    pairs: list[tuple[str, str, str, int]],
    compounds: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X (absolute difference of embeddings) and labels y from pairs.

    Same input representation as Siamese: |emb1 - emb2|. Embeddings are flattened;
    if 2D (n, dim), mean-pool to (dim,) first.

    Args:
        pairs: List of (pair_key, inchi_key1, inchi_key2, label).
        compounds: DataFrame with 'inchi_key' and 'embedding' columns.

    Returns:
        X: (n_pairs, n_features) float array.
        y: (n_pairs,) int array of labels.
    """
    inchi_to_emb = {
        row["inchi_key"]: np.asarray(row["embedding"], dtype=np.float32)
        for _, row in compounds.iterrows()
    }
    X_list = []
    y_list = []
    for _pk, k1, k2, label in pairs:
        e1 = inchi_to_emb.get(k1)
        e2 = inchi_to_emb.get(k2)
        if e1 is None or e2 is None:
            continue
        e1 = np.atleast_2d(e1)
        e2 = np.atleast_2d(e2)
        if e1.shape[0] > 1:
            e1 = e1.mean(axis=0, keepdims=True)
        if e2.shape[0] > 1:
            e2 = e2.mean(axis=0, keepdims=True)
        diff = np.abs(e1.ravel() - e2.ravel())
        X_list.append(diff)
        y_list.append(label)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Calculate classification metrics given predictions and ground-truth labels.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for the positive class (optional).
            If None, ROC-AUC cannot be calculated.

    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1, roc_auc, binary_cross_entropy.
        Missing metrics (e.g., roc_auc if y_pred_proba is None) will be NaN.
    """
    metrics = {}

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Precision
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0.0))

    # Recall
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0.0))

    # F1 score
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0.0))

    # ROC-AUC (requires probability predictions)
    if y_pred_proba is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            metrics["roc_auc"] = float(auc(fpr, tpr))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    # Binary cross-entropy (requires probability predictions)
    if y_pred_proba is not None:
        try:
            # Convert to tensors for F.binary_cross_entropy
            y_true_tensor = np.array(y_true, dtype=np.float32)
            y_pred_proba_tensor = np.array(y_pred_proba, dtype=np.float32)
            # Clip probabilities to avoid log(0)
            y_pred_proba_tensor = np.clip(y_pred_proba_tensor, 1e-7, 1 - 1e-7)
            bce = -np.mean(
                y_true_tensor * np.log(y_pred_proba_tensor)
                + (1 - y_true_tensor) * np.log(1 - y_pred_proba_tensor)
            )
            metrics["binary_cross_entropy"] = float(bce)
        except (ValueError, RuntimeError):
            metrics["binary_cross_entropy"] = float("nan")
    else:
        metrics["binary_cross_entropy"] = float("nan")

    return metrics


# ============================================================================
# Main Function
# ============================================================================

# Default XGBoost hyperparameters (can be overridden via config["xgboost"] or config["model_training"]["xgboost"])
DEFAULT_XGBOOST_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def train_model(
    compounds: pd.DataFrame,
    pairwise_data: dict[str, dict[str, float]],
    activity_threshold: float,
    results_dir: Path,
    seed: int = 42,
    config: dict | None = None,
) -> dict:
    """
    Train a model to predict activity cliffs from molecular embeddings.

    Steps:
        5a: Split compounds using Butina and scaffold (Murcko) clustering
        5b: Create train/test pair splits and balance datasets for both splits
        5c: Train baseline logistic regression on both splits and evaluate
        5d: Train XGBoost on absolute embedding difference (|emb1 - emb2|) on both splits
        5e: Train Siamese network on both splits and evaluate

    Args:
        compounds: DataFrame with columns: 'smiles', 'inchi_key', 'activity',
            'log_activity', 'embedding', 'vector_length'.
        pairwise_data: Dictionary mapping "inchi_key_1,inchi_key_2" pairs to
            dict containing 'similarity', 'log_ratio', 'status', etc.
        activity_threshold: Threshold for log_ratio to consider a pair an activity cliff.
        results_dir: Path to save results.
        seed: Random seed for reproducibility.
        config: Optional config dict. XGBoost params can be under config["xgboost"] or
            config["model_training"]["xgboost"] (max_depth, learning_rate, subsample, colsample_bytree, etc.).

    Returns:
        Dictionary containing train/test splits and statistics.
    """
    console = Console()

    console.print("[bold cyan]Step 5 - Training model for cliff prediction...[/bold cyan]")

    model_dir = results_dir / "model_training"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # -------------------------------------------------------------------------
    # Step 5a: Butina and scaffold splits of compounds (train / val / test)
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 5a: Splitting compounds into train/val/test (Butina and scaffold)...[/cyan]")

    tr_b, val_b, te_b = _butina_split(compounds, pairwise_data, cutoff=0.3, seed=seed)
    tr_s, val_s, te_s = _scaffold_split(compounds, seed=seed)

    compound_splits: dict[str, tuple[list[int], list[int], list[int]]] = {
        "butina": (tr_b, val_b, te_b),
        "scaffold": (tr_s, val_s, te_s),
    }

    for split_name, (tr_idx, val_idx, te_idx) in compound_splits.items():
        compounds.iloc[tr_idx][["inchi_key", "smiles"]].to_csv(
            model_dir / f"train_compounds_{split_name}.csv", index=False
        )
        compounds.iloc[val_idx][["inchi_key", "smiles"]].to_csv(
            model_dir / f"val_compounds_{split_name}.csv", index=False
        )
        compounds.iloc[te_idx][["inchi_key", "smiles"]].to_csv(
            model_dir / f"test_compounds_{split_name}.csv", index=False
        )
        console.print(f"[green]  ✓ {split_name.capitalize()}: {len(tr_idx)} train, {len(val_idx)} val, {len(te_idx)} test[/green]")

    # -------------------------------------------------------------------------
    # Step 5b: Create train-train, val-val, test-test pair splits and balance
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 5b: Creating train/val/test pair splits and balancing...[/cyan]")

    # pair_splits[split_name] = (train_pairs, val_pairs, test_pairs) each balanced
    pair_splits: dict[str, tuple[list[tuple[str, str, str, int]], list[tuple[str, str, str, int]], list[tuple[str, str, str, int]]]] = {}
    stats: dict = {
        "activity_threshold": activity_threshold,
        "seed": seed,
        "butina": {},
        "scaffold": {},
    }

    for split_name, (train_indices, val_indices, test_indices) in compound_splits.items():
        train_pairs_prelim, val_pairs_prelim, test_pairs_prelim = _create_pair_splits_three_way(
            compounds=compounds,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            pairwise_data=pairwise_data,
            activity_threshold=activity_threshold,
        )
        train_pairs_balanced = _balance_dataset(train_pairs_prelim, seed=seed)
        val_pairs_balanced = _balance_dataset(val_pairs_prelim, seed=seed)
        test_pairs_balanced = _balance_dataset(test_pairs_prelim, seed=seed)
        pair_splits[split_name] = (train_pairs_balanced, val_pairs_balanced, test_pairs_balanced)

        def _counts(pairs: list) -> tuple[int, int]:
            labels = [p[3] for p in pairs]
            return sum(labels), len(labels) - sum(labels)

        train_n_cliffs, train_n_non = _counts(train_pairs_balanced)
        val_n_cliffs, val_n_non = _counts(val_pairs_balanced)
        test_n_cliffs, test_n_non = _counts(test_pairs_balanced)

        stats[split_name] = {
            "n_train_compounds": len(train_indices),
            "n_val_compounds": len(val_indices),
            "n_test_compounds": len(test_indices),
            "n_train_pairs_prelim": len(train_pairs_prelim),
            "n_val_pairs_prelim": len(val_pairs_prelim),
            "n_test_pairs_prelim": len(test_pairs_prelim),
            "n_train_pairs_balanced": len(train_pairs_balanced),
            "n_val_pairs_balanced": len(val_pairs_balanced),
            "n_test_pairs_balanced": len(test_pairs_balanced),
            "train_n_cliffs": train_n_cliffs,
            "train_n_non_cliffs": train_n_non,
            "val_n_cliffs": val_n_cliffs,
            "val_n_non_cliffs": val_n_non,
            "test_n_cliffs": test_n_cliffs,
            "test_n_non_cliffs": test_n_non,
        }

        console.print(
            f"  {split_name.capitalize()}: train ({train_n_cliffs}/{train_n_non}), val ({val_n_cliffs}/{val_n_non}), test ({test_n_cliffs}/{test_n_non}) cliffs/non"
        )

        for name, pairs in [
            ("train", train_pairs_balanced),
            ("val", val_pairs_balanced),
            ("test", test_pairs_balanced),
        ]:
            pd.DataFrame(pairs, columns=["pair_key", "inchi_key_1", "inchi_key_2", "label"]).to_csv(
                model_dir / f"{name}_pairs_{split_name}.csv", index=False
            )

    console.print("[green]  ✓ Pair splits saved for both Butina and scaffold[/green]")

    with open(model_dir / "split_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    # -------------------------------------------------------------------------
    # Step 5c: Train baseline logistic regression on both splits
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 5c: Training baseline logistic regression (Butina and scaffold)...[/cyan]")

    baseline_models: dict[str, LogisticRegression] = {}
    baseline_metrics: dict[str, dict[str, float]] = {}

    for split_name, (train_pairs_balanced, val_pairs_balanced, test_pairs_balanced) in pair_splits.items():
        # 5c: train on train + val combined, test on test
        train_val_pairs = train_pairs_balanced + val_pairs_balanced
        train_distances = []
        train_labels_list = []
        for pair_key, _, _, label in train_val_pairs:
            data = pairwise_data.get(pair_key, {})
            distance = data.get("distance_scaled", np.nan)
            if not np.isnan(distance):
                train_distances.append(distance)
                train_labels_list.append(label)

        test_distances = []
        test_labels_list = []
        for pair_key, _, _, label in test_pairs_balanced:
            data = pairwise_data.get(pair_key, {})
            distance = data.get("distance_scaled", np.nan)
            if not np.isnan(distance):
                test_distances.append(distance)
                test_labels_list.append(label)

        if len(train_distances) == 0 or len(test_distances) == 0:
            console.print(f"[yellow]  ⚠ {split_name.capitalize()}: no valid distances, skipping baseline[/yellow]")
            stats[split_name]["baseline_model_metrics"] = None
            stats[split_name]["baseline_model_coefficients"] = None
            continue

        X_train = np.array(train_distances).reshape(-1, 1)
        y_train = np.array(train_labels_list)
        X_test = np.array(test_distances).reshape(-1, 1)
        y_test = np.array(test_labels_list)

        model = LogisticRegression(random_state=seed, max_iter=1000)
        model.fit(X_train, y_train)
        baseline_models[split_name] = model

        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
        baseline_metrics[split_name] = test_metrics

        stats[split_name]["baseline_model_metrics"] = test_metrics
        stats[split_name]["baseline_model_coefficients"] = {
            "intercept": float(model.intercept_[0]),
            "coefficient": float(model.coef_[0][0]),
        }

        with open(model_dir / f"baseline_model_metrics_{split_name}.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        with open(model_dir / f"baseline_model_coefficients_{split_name}.json", "w") as f:
            json.dump(stats[split_name]["baseline_model_coefficients"], f, indent=2)

        console.print(
            f"[green]  ✓ {split_name.capitalize()} baseline: "
            f"accuracy={test_metrics['accuracy']:.4f}, roc_auc={test_metrics['roc_auc']:.4f}, f1={test_metrics['f1']:.4f}[/green]"
        )

    with open(model_dir / "split_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    # -------------------------------------------------------------------------
    # Step 5d: Train XGBoost on |emb1 - emb2| on both splits
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 5d: Training XGBoost on absolute embedding difference (Butina and scaffold)...[/cyan]")

    xgb_params = {**DEFAULT_XGBOOST_PARAMS}
    if config:
        xgb_cfg = (config.get("model_training") or {}).get("xgboost") if isinstance(config.get("model_training"), dict) else None
        if isinstance(xgb_cfg, dict):
            xgb_params.update(xgb_cfg)
        if isinstance(config.get("xgboost"), dict):
            xgb_params.update(config["xgboost"])

    xgboost_models: dict[str, xgb.XGBClassifier] = {}
    xgboost_metrics: dict[str, dict[str, float]] = {}

    early_stopping_rounds = 10
    if config:
        early_stopping_rounds = (config.get("model_training") or {}).get("early_stopping_rounds", early_stopping_rounds)
        if isinstance(config.get("xgboost"), dict) and "early_stopping_rounds" in config["xgboost"]:
            early_stopping_rounds = config["xgboost"]["early_stopping_rounds"]

    if "embedding" in compounds.columns:
        for split_name, (train_pairs_balanced, val_pairs_balanced, test_pairs_balanced) in pair_splits.items():
            if len(train_pairs_balanced) == 0 or len(test_pairs_balanced) == 0:
                console.print(f"[yellow]  ⚠ {split_name.capitalize()}: no pairs, skipping XGBoost[/yellow]")
                stats[split_name]["xgboost_model_metrics"] = None
                continue
            X_train, y_train = _pairs_to_absdiff_features(train_pairs_balanced, compounds)
            X_val, y_val = _pairs_to_absdiff_features(val_pairs_balanced, compounds)
            X_test, y_test = _pairs_to_absdiff_features(test_pairs_balanced, compounds)
            if X_train.size == 0 or X_test.size == 0:
                console.print(f"[yellow]  ⚠ {split_name.capitalize()}: no valid features, skipping XGBoost[/yellow]")
                stats[split_name]["xgboost_model_metrics"] = None
                continue
            xgb_kwargs = {**xgb_params, "random_state": seed, "eval_metric": "logloss"}
            if X_val.size > 0:
                xgb_kwargs["early_stopping_rounds"] = early_stopping_rounds
            clf = xgb.XGBClassifier(**xgb_kwargs)
            eval_set = [(X_val, y_val)] if X_val.size > 0 else None
            clf.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            test_metrics = calculate_metrics(y_test, y_pred, y_proba)
            xgboost_models[split_name] = clf
            xgboost_metrics[split_name] = test_metrics
            stats[split_name]["xgboost_model_metrics"] = test_metrics
            with open(model_dir / f"xgboost_model_metrics_{split_name}.json", "w") as f:
                json.dump(test_metrics, f, indent=2)
            clf.save_model(str(model_dir / f"xgboost_model_{split_name}.json"))
            console.print(
                f"[green]  ✓ {split_name.capitalize()} XGBoost: "
                f"accuracy={test_metrics['accuracy']:.4f}, roc_auc={test_metrics['roc_auc']:.4f}, f1={test_metrics['f1']:.4f}[/green]"
            )
    else:
        console.print("[yellow]  ⚠ No embeddings found, skipping XGBoost[/yellow]")
        for split_name in pair_splits:
            stats[split_name]["xgboost_model_metrics"] = None

    with open(model_dir / "split_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    # -------------------------------------------------------------------------
    # Step 5e: Train Siamese network on both splits
    # -------------------------------------------------------------------------
    console.print("[cyan]Step 5e: Training Siamese network (Butina and scaffold)...[/cyan]")

    siamese_models: dict[str, SiameseCliffNet] = {}
    siamese_metrics: dict[str, dict[str, float]] = {}

    if "embedding" not in compounds.columns:
        console.print("[yellow]  ⚠ No embeddings found, skipping Siamese network training[/yellow]")
        return {
            "compound_splits": compound_splits,
            "pair_splits": pair_splits,
            "statistics": stats,
            "baseline_models": baseline_models,
            "baseline_metrics": baseline_metrics,
            "xgboost_models": xgboost_models,
            "xgboost_metrics": xgboost_metrics,
            "siamese_models": {},
            "siamese_metrics": {},
        }

    first_emb = np.array(compounds.iloc[0]["embedding"])
    emb_dim = int(first_emb.shape[-1])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    n_epochs = 50
    criterion = nn.BCEWithLogitsLoss()

    early_stopping_patience = 10
    if config and isinstance(config.get("model_training"), dict):
        early_stopping_patience = config["model_training"].get("early_stopping_patience", early_stopping_patience)

    for split_name, (train_pairs_balanced, val_pairs_balanced, test_pairs_balanced) in pair_splits.items():
        if len(train_pairs_balanced) == 0 or len(test_pairs_balanced) == 0:
            console.print(f"[yellow]  ⚠ {split_name.capitalize()}: no pairs, skipping Siamese[/yellow]")
            stats[split_name]["siamese_model_metrics"] = None
            continue

        train_dataset = PairDataset(train_pairs_balanced, compounds)
        val_dataset = PairDataset(val_pairs_balanced, compounds) if val_pairs_balanced else None
        test_dataset = PairDataset(test_pairs_balanced, compounds)
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=_collate_variable_embeddings,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=32,
                shuffle=False,
                collate_fn=_collate_variable_embeddings,
            )
            if val_dataset and len(val_pairs_balanced) > 0
            else None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=_collate_variable_embeddings,
        )

        siamese_model = SiameseCliffNet(emb_dim=emb_dim, hidden_dim=32)
        optimizer = torch.optim.Adam(siamese_model.parameters(), lr=0.001)
        device = next(siamese_model.parameters()).device

        best_val_loss = float("inf")
        best_state: dict | None = None
        patience_counter = 0

        siamese_model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for emb1_list, emb2_list, labels in train_loader:
                optimizer.zero_grad()
                labels = labels.to(device)
                z1 = torch.cat(
                    [siamese_model.encoder(e.unsqueeze(0).to(device)) for e in emb1_list],
                    dim=0,
                )
                z2 = torch.cat(
                    [siamese_model.encoder(e.unsqueeze(0).to(device)) for e in emb2_list],
                    dim=0,
                )
                logits = siamese_model.mlp(torch.abs(z1 - z2)).squeeze(-1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0.0

            # Validation loss for early stopping
            if val_loader is not None:
                siamese_model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for emb1_list, emb2_list, labels in val_loader:
                        labels = labels.to(device)
                        z1 = torch.cat(
                            [siamese_model.encoder(e.unsqueeze(0).to(device)) for e in emb1_list],
                            dim=0,
                        )
                        z2 = torch.cat(
                            [siamese_model.encoder(e.unsqueeze(0).to(device)) for e in emb2_list],
                            dim=0,
                        )
                        logits = siamese_model.mlp(torch.abs(z1 - z2)).squeeze(-1)
                        val_loss += criterion(logits, labels).item()
                        val_batches += 1
                siamese_model.train()
                avg_val_loss = val_loss / val_batches if val_batches > 0 else float("inf")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.cpu().clone() for k, v in siamese_model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    console.print(f"  [{split_name}] Early stopping at epoch {epoch + 1}")
                    break
            if (epoch + 1) % 10 == 0:
                console.print(f"  [{split_name}] Epoch {epoch + 1}/{n_epochs}, train loss: {avg_train_loss:.4f}")

        if best_state is not None:
            siamese_model.load_state_dict(best_state)

        siamese_model.eval()
        all_test_labels = []
        all_test_probs = []
        with torch.no_grad():
            for emb1_list, emb2_list, labels in test_loader:
                labels = labels.to(device)
                z1 = torch.cat(
                    [siamese_model.encoder(e.unsqueeze(0).to(device)) for e in emb1_list],
                    dim=0,
                )
                z2 = torch.cat(
                    [siamese_model.encoder(e.unsqueeze(0).to(device)) for e in emb2_list],
                    dim=0,
                )
                logits = siamese_model.mlp(torch.abs(z1 - z2)).squeeze(-1)
                probs = torch.sigmoid(logits)
                all_test_labels.extend(labels.cpu().numpy())
                all_test_probs.extend(probs.cpu().numpy())

        y_test_siamese = np.array(all_test_labels)
        y_test_pred_siamese = (np.array(all_test_probs) > 0.5).astype(int)
        y_test_pred_proba_siamese = np.array(all_test_probs)
        siamese_test_metrics = calculate_metrics(
            y_test_siamese, y_test_pred_siamese, y_test_pred_proba_siamese
        )
        siamese_models[split_name] = siamese_model
        siamese_metrics[split_name] = siamese_test_metrics
        stats[split_name]["siamese_model_metrics"] = siamese_test_metrics

        with open(model_dir / f"siamese_model_metrics_{split_name}.json", "w") as f:
            json.dump(siamese_test_metrics, f, indent=2)
        torch.save(siamese_model.state_dict(), model_dir / f"siamese_model_{split_name}.pt")

        console.print(
            f"[green]  ✓ {split_name.capitalize()} Siamese: "
            f"accuracy={siamese_test_metrics['accuracy']:.4f}, roc_auc={siamese_test_metrics['roc_auc']:.4f}, f1={siamese_test_metrics['f1']:.4f}[/green]"
        )

    with open(model_dir / "split_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    console.print(f"[green]  ✓ Results saved to: {model_dir}[/green]")

    return {
        "compound_splits": compound_splits,
        "pair_splits": pair_splits,
        "statistics": stats,
        "baseline_models": baseline_models,
        "baseline_metrics": baseline_metrics,
        "xgboost_models": xgboost_models,
        "xgboost_metrics": xgboost_metrics,
        "siamese_models": siamese_models,
        "siamese_metrics": siamese_metrics,
    }
