"""
Molecular similarity and distance metrics.

This module provides functions for calculating similarity and distance metrics
between molecular embeddings (vectors), as well as scaling functions for
converting distances to similarity values.
"""

import numpy as np
from typing import Callable
from scipy.spatial.distance import cdist


# ============================================================================
# Similarity and Distance Metrics
# ============================================================================

def tanimoto_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Tanimoto (Jaccard) similarity between two vectors.

    For binary vectors, this is |A ∩ B| / |A ∪ B|.
    For continuous vectors, uses the generalized formula.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Similarity value in [0, 1]. Returns 1.0 if both vectors are zero.
    """
    dot_product = np.dot(v1, v2)
    denominator = np.dot(v1, v1) + np.dot(v2, v2) - dot_product
    if denominator == 0:
        return 1.0
    return float(dot_product / denominator)


def dice_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Dice similarity between two vectors.

    For binary vectors, this is 2|A ∩ B| / (|A| + |B|).

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Similarity value in [0, 1]. Returns 1.0 if both vectors are zero.
    """
    dot_product = np.dot(v1, v2)
    denominator = np.dot(v1, v1) + np.dot(v2, v2)
    if denominator == 0:
        return 1.0
    return float(2 * dot_product / denominator)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    cos(θ) = (A · B) / (||A|| ||B||)

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Similarity value in [-1, 1]. Returns 1.0 if both vectors are zero,
        or 0.0 if only one is zero.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0 if norm1 == norm2 else 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Euclidean (L2) distance between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Distance value (≥ 0). Lower values indicate more similar vectors.
    """
    return float(np.linalg.norm(v1 - v2))


def manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Manhattan (L1) distance between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Distance value (≥ 0). Lower values indicate more similar vectors.
    """
    return float(np.sum(np.abs(v1 - v2)))


def hamming_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Hamming distance between two binary vectors.

    Returns the fraction of positions that differ.

    Args:
        v1: First binary vector.
        v2: Second binary vector.

    Returns:
        Distance value in [0, 1]. Lower values indicate more similar vectors.
    """
    return float(np.mean(v1 != v2))


def sokal_michener_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Sokal-Michener (simple matching) similarity for binary vectors.

    SM = (a + d) / (a + b + c + d)
    where:
        a = both 1
        d = both 0
        b = v1=1 & v2=0
        c = v1=0 & v2=1

    Args:
        v1: First binary vector.
        v2: Second binary vector.

    Returns:
        Similarity value in [0, 1].
    """
    v1_bin = v1.astype(bool)
    v2_bin = v2.astype(bool)
    matches = np.sum(v1_bin == v2_bin)
    return float(matches / len(v1))


def tensor_distance(X1: list[np.ndarray], X2: list[np.ndarray] = None, power: float = 0.5) -> np.ndarray:
    def energy_distance_pow_euclidean(s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Generalized Energy Distance with metric d(x,y) = (||x-y||_2)^p.
        Standard Energy Distance uses p=1 (Euclidean).
        p < 1 makes it more robust to outliers (heavy tails).
        Currently p=0.5 works best.
        """
        # Cross terms
        D12 = cdist(s1, s2, metric='euclidean')
        D12 = np.power(D12, power)
        term1 = 2 * np.mean(D12)
        
        # Self terms
        D11 = cdist(s1, s1, metric='euclidean')
        D11 = np.power(D11, power)
        term2 = np.mean(D11)
        
        D22 = cdist(s2, s2, metric='euclidean')
        D22 = np.power(D22, power)
        term3 = np.mean(D22)
        
        # Result should be >= 0 theoretically
        val = term1 - term2 - term3
        # Return sqrt of energy to maintain scaling roughly linear with p
        return np.sqrt(max(0.0, val))

    if X2 is None:
        X2 = X1
        
    n1 = len(X1)
    n2 = len(X2)
    dist_mat = np.zeros((n1, n2))
    
    # Check if we are computing self-distance (symmetric)
    is_symmetric = (X1 is X2)
        
    for i in range(n1):
        for j in range(n2):
            if is_symmetric and j < i:
                dist_mat[i, j] = dist_mat[j, i]
                continue
                
            d = energy_distance_pow_euclidean(X1[i], X2[j])
            dist_mat[i, j] = d
    return dist_mat


# Registry of available similarity metrics
SIMILARITY_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "tanimoto": tanimoto_similarity,
    "dice": dice_similarity,
    "cosine": cosine_similarity,
    "euclidean": euclidean_distance,
    "l2": euclidean_distance,
    "manhattan": manhattan_distance,
    "l1": manhattan_distance,
    "hamming": hamming_distance,
    "sokal_michener": sokal_michener_similarity,
    "tensor": tensor_distance,
}

# Metrics that return similarity (need to be converted to distance via 1-value)
SIMILARITY_BASED_METRICS = {"tanimoto", "dice", "cosine", "sokal_michener"}


def register_similarity_metric(
    name: str, func: Callable[[np.ndarray, np.ndarray], float]
) -> None:
    """
    Register a new similarity metric.

    Args:
        name: Name of the metric.
        func: Function that takes two numpy arrays and returns a float.
    """
    SIMILARITY_METRICS[name] = func


def _validate_metric(metric: str) -> None:
    """
    Validate that a metric name exists in the registry.

    Args:
        metric: Metric name to validate.

    Raises:
        ValueError: If the metric is not found.
    """
    if metric not in SIMILARITY_METRICS:
        available = ", ".join(sorted(SIMILARITY_METRICS.keys()))
        raise ValueError(f"Unknown metric: {metric}. Available metrics: {available}")


def calculate_vector_similarity(
    v1: np.ndarray, v2: np.ndarray, metric: str = "tanimoto"
) -> float:
    """
    Calculate similarity between two vectors using the specified metric.

    Args:
        v1: First vector.
        v2: Second vector.
        metric: Similarity metric to use. Available metrics:
            - 'tanimoto': Tanimoto (Jaccard) similarity
            - 'dice': Dice similarity
            - 'cosine': Cosine similarity
            - 'euclidean' / 'l2': Euclidean distance
            - 'manhattan' / 'l1': Manhattan distance
            - 'hamming': Hamming distance (for binary vectors)
            - 'sokal_michener': Sokal-Michener similarity (for binary vectors)

    Returns:
        Similarity (or distance) value as a float.
    """
    _validate_metric(metric)
    return SIMILARITY_METRICS[metric](np.asarray(v1), np.asarray(v2))


def calculate_vector_distance(
    v1: np.ndarray, v2: np.ndarray, metric: str = "tanimoto"
) -> float:
    """
    Calculate distance between two vectors (or matrix embeddings) using the specified metric.

    For 1D vectors, all metrics are supported. For 2D (variable-shape) embeddings
    (e.g. per-atom shape (n_atoms, dim)), only the 'tensor' metric is supported,
    which uses the generalized energy distance between sets of points.

    For similarity-based metrics (tanimoto, dice, cosine, sokal_michener),
    returns 1 - similarity to convert to distance.
    For distance-based metrics (euclidean, manhattan, hamming, tensor),
    returns the distance directly.

    Args:
        v1: First vector or matrix (n_points, n_features).
        v2: Second vector or matrix (n_points, n_features).
        metric: Distance metric to use. For 2D embeddings only 'tensor' is valid.

    Returns:
        Distance value as a float (lower = more similar).
    """
    _validate_metric(metric)
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # Variable-shape (matrix) embeddings: only tensor metric is defined
    if v1.ndim == 2 or v2.ndim == 2:
        if metric != "tensor":
            raise ValueError(
                "Variable-shape (matrix) embeddings require metric='tensor'. "
                "Other metrics expect 1D vectors. Use 'tensor' for per-atom or "
                "variable-length embeddings."
            )
        # tensor_distance expects lists of arrays; returns (1,1) for single pair
        return float(tensor_distance([v1], [v2])[0, 0])

    value = SIMILARITY_METRICS[metric](v1, v2)

    # Convert similarity to distance for similarity-based metrics
    if metric in SIMILARITY_BASED_METRICS:
        return 1.0 - value

    return value


# ============================================================================
# Scaling Functions Registry
# ============================================================================

SCALING_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {}


def register_scaling_function(name: str):
    """
    Decorator to register a scaling function.

    Args:
        name: Name to register the function under.

    Returns:
        Decorator function.

    Example:
        @register_scaling_function("my_scaler")
        def my_scaling_function(distances: np.ndarray) -> np.ndarray:
            ...
    """
    def decorator(func: Callable[[np.ndarray], np.ndarray]):
        SCALING_FUNCTIONS[name] = func
        return func
    return decorator


def _normalize_and_renormalize(
    distances: np.ndarray, transformed: np.ndarray
) -> np.ndarray:
    """
    Helper function to normalize and renormalize transformed distances.

    Args:
        distances: Original distance array.
        transformed: Transformed distance array.

    Returns:
        Renormalized array in [0, 1] range, or zeros if normalization fails.
    """
    s_min = np.nanmin(transformed)
    s_max = np.nanmax(transformed)

    if s_max == s_min or np.isnan(s_max) or np.isnan(s_min):
        return np.zeros_like(distances)

    return (transformed - s_min) / (s_max - s_min)


@register_scaling_function("linear")
def linear_scaling(distances: np.ndarray) -> np.ndarray:
    """
    Linear scaling of distances to [0, 1] range.

    Scales distances linearly such that:
    - 0 -> 0
    - max(distances) -> 1

    NaN values are preserved (for excluded pairs).

    Args:
        distances: Array of distance values (assumed to start from 0).

    Returns:
        Scaled distances in [0, 1] range.
    """
    d_max = np.nanmax(distances)

    if d_max == 0 or np.isnan(d_max):
        return np.zeros_like(distances)

    return distances / d_max


@register_scaling_function("sigmoid")
def sigmoid_scaling(distances: np.ndarray, k: float = 10.0) -> np.ndarray:
    """
    Sigmoid scaling of distances to [0, 1] range.

    Uses a sigmoid function centered at the midpoint of the distance range.
    Provides smoother transitions than linear scaling.

    NaN values are preserved (for excluded pairs).

    Args:
        distances: Array of distance values (assumed to start from 0).
        k: Steepness parameter for the sigmoid (default: 10.0).

    Returns:
        Scaled distances in [0, 1] range.
    """
    d_max = np.nanmax(distances)

    if d_max == 0 or np.isnan(d_max):
        return np.zeros_like(distances)

    # Normalize to [0, 1] linearly (assuming min is 0)
    normalized = distances / d_max

    # Apply sigmoid centered at 0.5
    # sigmoid(k * (x - 0.5)) maps [0, 1] to approximately [0, 1]
    sigmoid = 1 / (1 + np.exp(-k * (normalized - 0.5)))

    # Renormalize to exactly [0, 1]
    return _normalize_and_renormalize(distances, sigmoid)


@register_scaling_function("exponential")
def exponential_scaling(distances: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Exponential scaling of distances to [0, 1] range.

    Uses exp(-gamma * d) transformation.

    NaN values are preserved (for excluded pairs).

    Args:
        distances: Array of distance values (assumed to start from 0).
        gamma: Decay parameter (default: 1.0).

    Returns:
        Scaled distances in [0, 1] range.
    """
    d_max = np.nanmax(distances)

    if d_max == 0 or np.isnan(d_max):
        return np.zeros_like(distances)

    # Normalize to [0, 1] linearly (assuming min is 0)
    normalized = distances / d_max

    # Apply exponential decay
    scaled = np.exp(-gamma * normalized)

    # Renormalize
    return _normalize_and_renormalize(distances, scaled)


def get_scaling_function(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get a scaling function by name.

    Args:
        name: Name of the scaling function.

    Returns:
        The scaling function.

    Raises:
        ValueError: If the scaling function is not found.
    """
    if name not in SCALING_FUNCTIONS:
        available = ", ".join(SCALING_FUNCTIONS.keys())
        raise ValueError(f"Unknown scaling function: {name}. Available: {available}")
    return SCALING_FUNCTIONS[name]


def scale_distances_to_similarity(
    distance_matrix: np.ndarray,
    scaling_function: str = "linear",
    scaling_cutoff_percentile: int = 99,
) -> np.ndarray:
    """
    Scale distance matrix to [0, 1] and invert to get similarity matrix.

    Args:
        distance_matrix: Matrix of distance values.
        scaling_function: Name of the scaling function to use
            ('linear', 'sigmoid', 'exponential').
        scaling_cutoff_percentile: Percentile to use for clipping distances before scaling.
            Distances above this percentile will be clipped to the percentile value.

    Returns:
        Similarity matrix where higher values indicate more similar pairs.
    """
    # Clip distances to percentile cutoff to handle outliers
    if scaling_cutoff_percentile < 100:
        valid_distances = distance_matrix[~np.isnan(distance_matrix)]
        if len(valid_distances) > 0:
            cutoff_value = np.percentile(valid_distances, scaling_cutoff_percentile)
            distance_matrix = np.clip(distance_matrix, None, cutoff_value)

    scaler = get_scaling_function(scaling_function)

    # Scale distances to [0, 1]
    scaled = scaler(distance_matrix)

    # Invert: similarity = 1 - scaled_distance
    similarity = 1 - scaled

    return similarity
