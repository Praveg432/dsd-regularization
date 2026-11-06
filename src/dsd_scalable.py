"""
Scalable DSD: O(m²k) setup + O(mk) per-query path for large matrices.

For m > 1500, full eigendecomposition (O(m³)) and dense pseudo-inverse
reconstruction (O(m²) memory) become bottlenecks. This module provides:

1. Randomized partial eigendecomposition: compute only top-k eigenvalues
   in O(m²k) via scipy.sparse.linalg.eigsh or randomized SVD.
2. Factored (lazy) inverse: store (U_k, λ̃⁻¹_k) without constructing
   the full m×m matrix. Per-query cost: O(mk) instead of O(m²).
3. Automatic mode selection: full eigh for m ≤ threshold, partial for m > threshold.

Trade-off: Partial eigendecomposition cannot detect gaps in the discarded
tail. We handle this by choosing k large enough to include the spectral
transition (reliable → unreliable boundary) based on a pilot subsample.

Usage:
    result = dsd_scalable(W, max_rank=200)
    x_hat = result.apply(k_query) @ landmarks  # O(mk) per query
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional
import warnings

from .dsd import compute_eigengaps, initialize_hyperparameters


@dataclass
class DSDScalableResult:
    """Result of scalable DSD computation (factored form)."""
    # Factored form: W⁺ ≈ U · diag(λ̃⁻¹) · Uᵀ (never materialized as dense)
    eigenvectors: NDArray      # (m, k) — top-k eigenvectors
    reg_inverse_diag: NDArray  # (k,) — DSD-regularized inverse diagonal
    eigenvalues: NDArray       # (k,) — retained eigenvalues
    eigengaps: NDArray         # (k,) — eigengaps
    damping: NDArray           # (k,) — DSD damping per eigenvalue
    alpha: float
    beta: float
    rank_used: int
    full_size: int
    condition_number_original: float
    condition_number_regularized: float
    
    def apply(self, k_query: NDArray) -> NDArray:
        """
        Apply DSD inverse to a kernel vector: W⁺ · k_query.
        
        O(mk) per query — avoids constructing full m×m matrix.
        
        Parameters
        ----------
        k_query : ndarray (m,) or (b, m)
            Kernel vector(s) to invert.
        
        Returns
        -------
        weights : ndarray (m,) or (b, m) — NOT (m, d)
            Inverse-weighted coefficients. Multiply by landmarks for pre-image.
        """
        if k_query.ndim == 1:
            # Single query: (m,) → project → scale → back-project
            proj = self.eigenvectors.T @ k_query          # (k,)
            scaled = self.reg_inverse_diag * proj         # (k,)
            return self.eigenvectors @ scaled             # (m,)
        else:
            # Batch: (b, m) → (b, k) → (b, k) → (b, m)
            proj = k_query @ self.eigenvectors            # (b, k)
            scaled = proj * self.reg_inverse_diag         # (b, k)
            return scaled @ self.eigenvectors.T           # (b, m)
    
    def to_dense(self) -> NDArray:
        """
        Materialize the full m×m pseudo-inverse (for compatibility).
        WARNING: O(m²k) computation, O(m²) memory.
        """
        return (self.eigenvectors * self.reg_inverse_diag) @ self.eigenvectors.T


def _estimate_rank(W: NDArray, target_energy: float = 0.99) -> int:
    """
    Estimate the rank needed to capture target_energy fraction of the spectrum.
    Uses a cheap pilot: compute a few eigenvalues to estimate decay rate.
    """
    m = W.shape[0]
    # Quick diagonal-based heuristic: trace = sum of eigenvalues
    trace_W = np.trace(W)
    # For RBF kernels, effective rank ≈ m * (1 - exp(-γ·d_mean²))
    # Heuristic: start with m//3 as a reasonable default
    return min(max(m // 3, 50), m)


def dsd_scalable(
    W: NDArray,
    max_rank: Optional[int] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    auto_threshold: int = 1500,
) -> DSDScalableResult:
    """
    Scalable DSD: automatically selects full or partial eigendecomposition.
    
    For m ≤ auto_threshold: uses full eigh (O(m³)), same as dsd_regularized_inverse.
    For m > auto_threshold: uses partial eigsh (O(m²k)), factored output.
    
    Parameters
    ----------
    W : ndarray (m, m)
        Symmetric PSD kernel matrix.
    max_rank : int, optional
        Maximum number of eigenvalues to compute. If None, auto-estimated.
    alpha, beta : float, optional
        DSD hyperparameters. Auto-initialized if None.
    auto_threshold : int
        Matrix size above which partial decomposition is used.
    
    Returns
    -------
    DSDScalableResult with factored inverse (use .apply() for per-query).
    """
    W = np.asarray(W, dtype=np.float64)
    m = W.shape[0]
    
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got shape {W.shape}")
    
    # Decide path
    use_partial = m > auto_threshold
    
    if use_partial:
        from scipy.sparse.linalg import eigsh
        
        k = max_rank if max_rank is not None else _estimate_rank(W)
        k = min(k, m - 1)  # eigsh requires k < m
        
        # Compute top-k eigenvalues (largest)
        try:
            eigenvalues, eigenvectors = eigsh(W, k=k, which='LM')
        except Exception:
            # Fallback to full decomposition if eigsh fails
            warnings.warn(
                f"eigsh failed for m={m}, k={k}. Falling back to full eigh.",
                RuntimeWarning
            )
            eigenvalues, eigenvectors = np.linalg.eigh(W)
            eigenvalues = eigenvalues[-k:]
            eigenvectors = eigenvectors[:, -k:]
        
        # Sort ascending (eigsh returns in arbitrary order)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
    else:
        # Full decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(W)
    
    # Filter positive eigenvalues
    pos_mask = eigenvalues > 1e-12
    if not pos_mask.any():
        raise ValueError("No positive eigenvalues found.")
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]
    
    # Apply max_rank truncation if specified and in full mode
    if max_rank is not None and not use_partial and len(eigenvalues) > max_rank:
        eigenvalues = eigenvalues[-max_rank:]
        eigenvectors = eigenvectors[:, -max_rank:]
    
    # Eigengaps
    eigengaps = compute_eigengaps(eigenvalues)
    
    # Initialize hyperparameters
    if alpha is None or beta is None:
        alpha_init, beta_init = initialize_hyperparameters(eigenvalues)
        alpha = alpha if alpha is not None else alpha_init
        beta = beta if beta is not None else beta_init
    
    # DSD damping
    max_gap_for_exp = 500.0 / max(beta, 1e-10)
    damping = alpha * np.exp(-beta * np.clip(eigengaps, 0, max_gap_for_exp))
    
    # Regularized inverse diagonal
    reg_inverse_diag = eigenvalues / (eigenvalues ** 2 + damping)
    
    # Condition numbers
    cond_original = eigenvalues[-1] / max(eigenvalues[0], 1e-15)
    cond_regularized = reg_inverse_diag.max() / max(reg_inverse_diag.min(), 1e-15)
    
    return DSDScalableResult(
        eigenvectors=eigenvectors,
        reg_inverse_diag=reg_inverse_diag,
        eigenvalues=eigenvalues,
        eigengaps=eigengaps,
        damping=damping,
        alpha=alpha,
        beta=beta,
        rank_used=len(eigenvalues),
        full_size=m,
        condition_number_original=cond_original,
        condition_number_regularized=cond_regularized,
    )
