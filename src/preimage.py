"""
Pre-image computation: projecting RKHS decision boundaries back to input space.

Uses the Nyström-based pre-image formulation:
    x̂ = (W⁻¹ · k_query) · X_landmarks

Weights are NOT normalized (allows extrapolation beyond convex hull of landmarks,
which is mathematically necessary for boundary points not inside the landmark set).
"""

import numpy as np
from numpy.typing import NDArray
from .dsd import dsd_regularized_inverse, tikhonov_inverse, truncated_svd_inverse, naive_pseudo_inverse
from .kernels import rbf_kernel_matrix, rbf_kernel_vector, nystrom_sample


def compute_preimage(
    x_query_kernel: NDArray,
    W_inverse: NDArray,
    X_landmarks: NDArray,
) -> NDArray:
    """
    Compute pre-image from kernel-space representation.
    
    x̂ = Σᵢ αᵢ · x_landmarks_i  where α = W⁻¹ · k_query
    
    Weights are used RAW (no normalization). This allows the pre-image
    to lie outside the convex hull of landmarks — necessary for accurate
    reconstruction of boundary points.
    """
    weights = W_inverse @ x_query_kernel
    return weights @ X_landmarks


def preimage_pipeline(
    X_train: NDArray,
    X_boundary: NDArray,
    gamma: float = 1.0,
    m: int = 200,
    method: str = "dsd",
    alpha: float | None = None,
    beta: float | None = None,
    tikhonov_gamma: float = 1e-3,
    tsvd_rank: int = 50,
) -> NDArray:
    """
    Full pre-image computation pipeline.
    
    Parameters
    ----------
    X_train : ndarray of shape (n, d)
        Training data (for landmark selection).
    X_boundary : ndarray of shape (b, d)
        Points to compute pre-images for.
    gamma : float
        RBF kernel bandwidth.
    m : int
        Number of Nyström landmarks.
    method : str
        One of "dsd", "tikhonov", "tsvd", "naive".
    alpha, beta : float, optional
        DSD hyperparameters (auto-initialized if None).
    tikhonov_gamma : float
        Regularization strength for Tikhonov method.
    tsvd_rank : int
        Rank for truncated SVD method.
    
    Returns
    -------
    preimages : ndarray of shape (b, d)
        Reconstructed input-space points.
    """
    # Step 1: Nyström landmark selection
    landmarks, _ = nystrom_sample(X_train, m=m, method="kmeans")
    
    # Step 2: Compute kernel submatrix
    W = rbf_kernel_matrix(landmarks, gamma=gamma)
    
    # Step 3: Compute regularized pseudo-inverse
    if method == "dsd":
        result = dsd_regularized_inverse(W, alpha=alpha, beta=beta)
        W_inv = result.pseudo_inverse
    elif method == "tikhonov":
        W_inv = tikhonov_inverse(W, gamma=tikhonov_gamma)
    elif method == "tsvd":
        W_inv = truncated_svd_inverse(W, rank_k=tsvd_rank)
    elif method == "naive":
        W_inv = naive_pseudo_inverse(W)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Step 4: Compute pre-images for each boundary point (fully vectorized)
    from scipy.spatial.distance import cdist
    K_query = np.exp(-gamma * cdist(X_boundary, landmarks, metric='sqeuclidean'))  # (b, m)
    preimages = (K_query @ W_inv) @ landmarks  # (b, d)
    
    return preimages
