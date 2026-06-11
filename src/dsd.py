"""
Differential Spectral Damping (DSD) — Core Algorithm.

Implements gap-adaptive regularization for kernel matrix pseudo-inversion,
preserving dominant eigenvector geometry while suppressing corrupted tail subspace.

Reference: DSD formula
    λ̃ᵢ⁻¹ = λᵢ / (λᵢ² + α · exp(-β · δᵢ))

Where:
    λᵢ = eigenvalue
    δᵢ = localized eigengap (single-sided at boundaries)
    α = penalty magnitude (calibrated to spectral transition point)
    β = gap sensitivity (normalized to median gap scale)
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class DSDResult:
    """Result of DSD-regularized pseudo-inverse computation."""
    pseudo_inverse: NDArray
    eigenvalues: NDArray
    eigenvectors: NDArray
    eigengaps: NDArray
    damping: NDArray
    reg_inverse_diag: NDArray
    alpha: float
    beta: float
    condition_number_original: float      # post-filter (retained positive eigenvalues)
    condition_number_regularized: float
    condition_number_prefilter: float = float('inf')  # true condition including near-zero eigenvalues


def compute_eigengaps(eigenvalues: NDArray) -> NDArray:
    """
    Compute localized eigengaps (vectorized).
    
    δᵢ = min(|λᵢ - λᵢ₋₁|, |λᵢ - λᵢ₊₁|)
    
    Boundary handling: single-sided gap at edges.
    First eigenvalue uses right gap only; last uses left gap only.
    """
    m = len(eigenvalues)
    if m <= 1:
        return np.zeros(m)
    
    diffs = np.abs(np.diff(eigenvalues))  # (m-1,)
    
    # Left gaps: [diffs[0], diffs[0], diffs[1], ..., diffs[m-2]]
    left = np.concatenate([[diffs[0]], diffs])
    # Right gaps: [diffs[0], diffs[1], ..., diffs[m-2], diffs[m-2]]
    right = np.concatenate([diffs, [diffs[-1]]])
    
    gaps = np.minimum(left, right)
    return gaps


def initialize_hyperparameters(eigenvalues: NDArray) -> tuple[float, float]:
    """
    Principled initialization for α and β.
    
    α = λ_transition² where λ_transition is the eigenvalue at the spectral
        "knee" — the point where eigengaps become small (unreliable regime).
        Specifically: the eigenvalue at the index where gaps fall below the
        10th percentile of all gaps.
        
    β = 1 / median(gaps) — normalizes the exponential sensitivity so that
        "typical" gaps produce moderate damping, while significantly larger
        gaps produce near-zero damping (preservation).
    """
    if len(eigenvalues) < 3:
        return float(np.median(eigenvalues ** 2)), 1.0
    
    gaps = np.abs(np.diff(eigenvalues))
    
    # α: find spectral transition point
    gap_10th = np.percentile(gaps, 10)
    small_gap_mask = gaps < gap_10th
    
    if small_gap_mask.any():
        transition_idx = np.where(small_gap_mask)[0][-1]
        alpha = float(eigenvalues[transition_idx] ** 2)
    else:
        # Fallback: use smallest eigenvalue squared
        alpha = float(eigenvalues[0] ** 2)
    
    # Ensure α is not zero
    alpha = max(alpha, 1e-15)
    
    # β: normalized to median gap scale
    median_gap = float(np.median(gaps))
    beta = 1.0 / median_gap if median_gap > 1e-15 else 1.0
    
    return alpha, beta


def dsd_regularized_inverse(
    W: NDArray,
    alpha: float | None = None,
    beta: float | None = None,
    rank_k: int | None = None,
) -> DSDResult:
    """
    Compute the DSD-regularized pseudo-inverse of kernel matrix W.
    
    Parameters
    ----------
    W : ndarray of shape (m, m)
        Symmetric positive semi-definite kernel submatrix.
    alpha : float, optional
        Penalty magnitude. If None, initialized from spectral transition analysis.
    beta : float, optional
        Eigengap sensitivity. If None, initialized from gap statistics.
    rank_k : int, optional
        If specified, only use top-k eigenvalues/vectors.
    
    Returns
    -------
    DSDResult with pseudo-inverse and diagnostic quantities.
    """
    # Input validation
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be a square matrix, got shape {W.shape}")
    if not np.allclose(W, W.T, atol=1e-10):
        raise ValueError("W must be symmetric")
    
    # Eigendecomposition
    eigenvalues_full, eigenvectors_full = np.linalg.eigh(W)
    
    # Track pre-filter condition number (true ill-conditioning measure)
    cond_prefilter = (
        eigenvalues_full[-1] / max(abs(eigenvalues_full[0]), 1e-15)
        if len(eigenvalues_full) > 0 else 1.0
    )
    
    # Retain only positive eigenvalues
    # NOTE: Negative eigenvalues after perturbation indicate the matrix
    # is no longer PSD. We discard them (they correspond to noise-corrupted
    # directions) but log the count for diagnostic awareness.
    n_negative = int(np.sum(eigenvalues_full <= 1e-12))
    pos_mask = eigenvalues_full > 1e-12
    eigenvalues = eigenvalues_full[pos_mask]
    eigenvectors = eigenvectors_full[:, pos_mask]
    
    if len(eigenvalues) == 0:
        raise ValueError(
            f"All {len(eigenvalues_full)} eigenvalues are <= 1e-12. "
            f"Matrix may be zero or entirely noise-corrupted."
        )
    
    # Optional rank truncation
    if rank_k is not None and rank_k < len(eigenvalues):
        eigenvalues = eigenvalues[-rank_k:]
        eigenvectors = eigenvectors[:, -rank_k:]
    
    # Compute eigengaps (fixed boundary handling)
    eigengaps = compute_eigengaps(eigenvalues)
    
    # Initialize hyperparameters
    if alpha is None or beta is None:
        alpha_init, beta_init = initialize_hyperparameters(eigenvalues)
        alpha = alpha if alpha is not None else alpha_init
        beta = beta if beta is not None else beta_init
    
    # DSD damping: α · exp(-β · δᵢ)
    # Clip eigengaps to prevent underflow in exp() (exp(-500) ≈ 0, harmless)
    max_gap_for_exp = 500.0 / max(beta, 1e-10)
    damping = alpha * np.exp(-beta * np.clip(eigengaps, 0, max_gap_for_exp))
    
    # DSD regularized inverse: λᵢ / (λᵢ² + damping_i)
    reg_inverse_diag = eigenvalues / (eigenvalues ** 2 + damping)
    
    # Reconstruct pseudo-inverse: W̃⁺ = U · diag(λ̃⁻¹) · Uᵀ
    pseudo_inverse = (eigenvectors * reg_inverse_diag) @ eigenvectors.T
    
    # Condition numbers:
    # - original: post-filter (of retained positive eigenvalues)
    # - prefilter: true condition including near-zero eigenvalues
    cond_original = eigenvalues[-1] / max(eigenvalues[0], 1e-15)
    cond_regularized = reg_inverse_diag.max() / max(reg_inverse_diag.min(), 1e-15)
    
    return DSDResult(
        pseudo_inverse=pseudo_inverse,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigengaps=eigengaps,
        damping=damping,
        reg_inverse_diag=reg_inverse_diag,
        alpha=alpha,
        beta=beta,
        condition_number_original=cond_original,
        condition_number_regularized=cond_regularized,
        condition_number_prefilter=cond_prefilter,
    )



def tikhonov_inverse(W: NDArray, gamma: float = 1e-3) -> NDArray:
    """
    Tikhonov regularization via eigendecomposition (fair comparison path).
    
    Computes (W + γI)⁻¹ = U · diag(1/(λᵢ + γ)) · Uᵀ
    """
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    pos_mask = eigenvalues > 1e-12
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]
    reg_inverse_diag = 1.0 / (eigenvalues + gamma)
    return (eigenvectors * reg_inverse_diag) @ eigenvectors.T


def tikhonov_inverse_heuristic(W: NDArray, gammas: NDArray | None = None) -> NDArray:
    """
    Tikhonov regularization with automatic γ selection via L-curve heuristic.
    
    NOTE: Despite the name, this uses a condition-number/influence trade-off
    heuristic rather than true GCV (Generalized Cross-Validation). For the
    fair-comparison baseline that uses actual reconstruction loss, see
    tikhonov_inverse_optimized().
    
    Selects γ that balances effective condition number reduction against
    signal preservation (influence matrix trace).
    
    Parameters
    ----------
    W : ndarray (m, m)
        Symmetric PSD kernel matrix.
    gammas : ndarray, optional
        Candidate γ values to search. If None, uses a logarithmic grid
        spanning the eigenvalue range.
    
    Returns
    -------
    W_inv : ndarray (m, m)
        Tikhonov-regularized inverse with optimally-selected γ.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    pos_mask = eigenvalues > 1e-12
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]
    m = len(eigenvalues)
    
    if gammas is None:
        # Logarithmic grid spanning eigenvalue range
        gamma_min = max(eigenvalues[0] * 0.01, 1e-12)
        gamma_max = eigenvalues[-1] * 0.1
        gammas = np.logspace(np.log10(gamma_min), np.log10(gamma_max), 50)
    
    # L-curve criterion: minimize curvature of (log||residual||, log||solution||)
    # Simplified: pick γ that balances bias (large γ → over-regularized)
    # against variance (small γ → unstable)
    # Practical criterion: minimize the effective condition number
    # cond_eff(γ) = max(1/(λᵢ+γ)) / min(1/(λᵢ+γ)) subject to not
    # over-damping (trace of influence matrix stays above m/2)
    best_gamma = gammas[len(gammas) // 2]  # default: middle of grid
    best_score = np.inf
    
    for gamma in gammas:
        reg_inv = 1.0 / (eigenvalues + gamma)
        cond_eff = reg_inv.max() / max(reg_inv.min(), 1e-15)
        
        # Influence: fraction of signal preserved
        influence = np.sum(eigenvalues / (eigenvalues + gamma)) / m
        
        # Penalty: over-regularization kills signal
        # We want influence > 0.3 (at least 30% signal preserved)
        if influence < 0.3:
            continue
        
        # Score: condition number (lower is more stable)
        # weighted by signal loss
        score = cond_eff * (1.0 / max(influence, 1e-10))
        
        if score < best_score:
            best_score = score
            best_gamma = gamma
    
    reg_inverse_diag = 1.0 / (eigenvalues + best_gamma)
    return (eigenvectors * reg_inverse_diag) @ eigenvectors.T


def truncated_svd_inverse(W: NDArray, rank_k: int = 50) -> NDArray:
    """Truncated SVD pseudo-inverse: keep top-k eigenvectors only."""
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    eigenvalues = eigenvalues[-rank_k:]
    eigenvectors = eigenvectors[:, -rank_k:]
    inv_diag = 1.0 / eigenvalues
    return (eigenvectors * inv_diag) @ eigenvectors.T


def tikhonov_inverse_optimized(
    W: NDArray,
    X_train: NDArray,
    landmarks: NDArray,
    gamma_kernel: float,
    gammas: NDArray | None = None,
) -> NDArray:
    """
    Tikhonov regularization with γ optimized on pre-image reconstruction loss.
    
    This is the FAIREST comparison to DSD: Tikhonov gets the same optimization
    opportunity (minimize pre-image error on training data) but is restricted
    to a single scalar parameter γ rather than DSD's gap-adaptive structure.
    
    Parameters
    ----------
    W : ndarray (m, m)
        Symmetric PSD kernel matrix (possibly noisy).
    X_train : ndarray (n, d)
        Training points for evaluating reconstruction loss.
    landmarks : ndarray (m, d)
        Nyström landmark points.
    gamma_kernel : float
        RBF kernel bandwidth.
    gammas : ndarray, optional
        Grid of γ candidates to search.
    
    Returns
    -------
    W_inv : ndarray (m, m)
        Tikhonov inverse with loss-optimal γ.
    """
    from .kernels import rbf_kernel_vector
    
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    pos_mask = eigenvalues > 1e-12
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]
    
    if gammas is None:
        gammas = np.logspace(-8, 0, 30)
    
    # Subsample training points for speed
    n_eval = min(30, len(X_train))
    indices = np.random.choice(len(X_train), size=n_eval, replace=False)
    X_eval = X_train[indices]
    
    # Pre-compute kernel vectors
    K_eval = np.array([
        rbf_kernel_vector(x, landmarks, gamma_kernel) for x in X_eval
    ])  # (n_eval, m)
    
    best_gamma = 1e-3
    best_loss = np.inf
    
    for gamma in gammas:
        reg_inv_diag = 1.0 / (eigenvalues + gamma)
        W_inv = (eigenvectors * reg_inv_diag) @ eigenvectors.T
        
        # Compute pre-images and loss
        preimages = (K_eval @ W_inv) @ landmarks
        loss = np.mean(np.sum((preimages - X_eval) ** 2, axis=1))
        
        if loss < best_loss:
            best_loss = loss
            best_gamma = gamma
    
    reg_inverse_diag = 1.0 / (eigenvalues + best_gamma)
    return (eigenvectors * reg_inverse_diag) @ eigenvectors.T


def naive_pseudo_inverse(W: NDArray) -> NDArray:
    """Moore-Penrose pseudo-inverse (numerically unstable for ill-conditioned W)."""
    return np.linalg.pinv(W)
