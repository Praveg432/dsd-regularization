"""
Kernel matrix construction and Nyström sampling utilities.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


def rbf_kernel_matrix(X: NDArray, gamma: float = 1.0) -> NDArray:
    """Compute RBF kernel matrix: K_ij = exp(-γ ||x_i - x_j||²)."""
    dists_sq = cdist(X, X, metric='sqeuclidean')
    return np.exp(-gamma * dists_sq)


def rbf_kernel_vector(x_query: NDArray, X_landmarks: NDArray, gamma: float = 1.0) -> NDArray:
    """Compute kernel vector between query point and landmarks."""
    dists_sq = np.sum((X_landmarks - x_query) ** 2, axis=1)
    return np.exp(-gamma * dists_sq)


def nystrom_sample(
    X: NDArray,
    m: int,
    method: str = "kmeans",
    random_state: int = 42,
) -> tuple[NDArray, NDArray]:
    """
    Select m landmark points for Nyström approximation.
    
    Parameters
    ----------
    X : ndarray of shape (n, d)
        Full dataset.
    m : int
        Number of landmarks.
    method : str
        "kmeans" (coverage-optimized) or "random" (uniform sampling).
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    landmarks : ndarray of shape (m, d)
        Selected landmark points.
    indices : ndarray of shape (m,)
        Indices of landmarks in X.
    """
    rng = np.random.default_rng(random_state)
    
    if method == "random":
        indices = rng.choice(len(X), size=m, replace=False)
        return X[indices], indices
    
    elif method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=m, random_state=random_state, n_init=3)
        km.fit(X)
        # Find closest data point to each centroid
        from scipy.spatial.distance import cdist as _cdist
        dists = _cdist(km.cluster_centers_, X)
        indices = np.argmin(dists, axis=1)
        return X[indices], indices
    
    else:
        raise ValueError(f"Unknown sampling method: {method}")
