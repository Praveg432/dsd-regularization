"""
Differentiable Differential Spectral Damping (DSD) in PyTorch.

Enables gradient-based optimization of α and β through backpropagation.
The entire DSD pipeline (eigendecomposition → eigengap → damping → inverse)
is expressed as differentiable tensor operations.

Key challenge: torch.linalg.eigh gradients are ill-conditioned when
eigenvalues are close (small gaps). We handle this by:
  1. Using float64 for numerical stability
  2. Detaching eigenvector gradients when gaps are degenerate
  3. Clamping eigenvalues to prevent division by zero
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class DSDTorchResult:
    """Result of differentiable DSD computation."""
    pseudo_inverse: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    eigengaps: torch.Tensor
    damping: torch.Tensor
    reg_inverse_diag: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


def compute_eigengaps_torch(eigenvalues: torch.Tensor) -> torch.Tensor:
    """
    Compute localized eigengaps (differentiable).

    δᵢ = min(|λᵢ - λᵢ₋₁|, |λᵢ - λᵢ₊₁|)
    Boundary handling: single-sided gaps at edges.
    """
    m = eigenvalues.shape[0]

    # Left gaps: |λᵢ - λᵢ₋₁| for i > 0
    left_gaps = torch.abs(eigenvalues[1:] - eigenvalues[:-1])  # (m-1,)

    # Build full gap array with boundary handling
    # Interior: min of left and right
    # First: right gap only, Last: left gap only
    gaps = torch.zeros(m, dtype=eigenvalues.dtype, device=eigenvalues.device)
    gaps[0] = left_gaps[0]           # right gap of first element
    gaps[-1] = left_gaps[-1]         # left gap of last element

    if m > 2:
        interior_min = torch.minimum(left_gaps[:-1], left_gaps[1:])  # (m-2,)
        gaps[1:-1] = interior_min

    return gaps


def initialize_hyperparameters_torch(
    eigenvalues: torch.Tensor,
) -> tuple[float, float]:
    """
    Principled initialization for α and β from spectrum.
    Returns Python floats for Parameter initialization.
    """
    if eigenvalues.shape[0] < 3:
        median_sq = float(torch.median(eigenvalues ** 2).item())
        return max(median_sq, 1e-15), 1.0

    gaps = torch.abs(eigenvalues[1:] - eigenvalues[:-1])

    # α: eigenvalue² at spectral transition (10th percentile gap)
    gap_10th = float(torch.quantile(gaps.float(), 0.1).item())
    small_mask = gaps < gap_10th
    if small_mask.any():
        transition_idx = int(torch.where(small_mask)[0][-1].item())
        alpha = float((eigenvalues[transition_idx] ** 2).item())
    else:
        alpha = float((eigenvalues[0] ** 2).item())
    alpha = max(alpha, 1e-15)

    # β: 1 / median(gaps)
    median_gap = float(torch.median(gaps).item())
    beta = 1.0 / median_gap if median_gap > 1e-15 else 1.0

    return alpha, beta


class DSDModule(nn.Module):
    """
    Differentiable DSD regularization module.

    α and β are learnable parameters optimized via backpropagation
    through the pre-image reconstruction loss.

    Parameters
    ----------
    alpha_init : float
        Initial penalty magnitude. If None, auto-initialized on first forward.
    beta_init : float
        Initial gap sensitivity. If None, auto-initialized on first forward.
    learn_alpha : bool
        Whether α is a learnable parameter.
    learn_beta : bool
        Whether β is a learnable parameter.
    lorentzian_eps : float
        Broadening parameter for stable eigendecomposition gradients.
        Prevents NaN when eigenvalues are degenerate. Larger values = more
        stable but less accurate gradients. Default 1e-6.
    """

    def __init__(
        self,
        alpha_init: Optional[float] = None,
        beta_init: Optional[float] = None,
        learn_alpha: bool = True,
        learn_beta: bool = True,
        lorentzian_eps: float = 1e-6,
    ):
        super().__init__()
        self._initialized = alpha_init is not None and beta_init is not None
        self.lorentzian_eps = lorentzian_eps

        # Store in log-space to ensure positivity via exp()
        if alpha_init is not None:
            log_alpha = torch.tensor(np.log(max(alpha_init, 1e-15)), dtype=torch.float64)
        else:
            log_alpha = torch.tensor(0.0, dtype=torch.float64)

        if beta_init is not None:
            log_beta = torch.tensor(np.log(max(beta_init, 1e-15)), dtype=torch.float64)
        else:
            log_beta = torch.tensor(0.0, dtype=torch.float64)

        if learn_alpha:
            self.log_alpha = nn.Parameter(log_alpha)
        else:
            self.register_buffer('log_alpha', log_alpha)

        if learn_beta:
            self.log_beta = nn.Parameter(log_beta)
        else:
            self.register_buffer('log_beta', log_beta)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    @property
    def beta(self) -> torch.Tensor:
        return torch.exp(self.log_beta)

    def _auto_initialize(self, eigenvalues: torch.Tensor):
        """Initialize α, β from spectrum on first forward pass."""
        if not self._initialized:
            alpha_init, beta_init = initialize_hyperparameters_torch(eigenvalues)
            with torch.no_grad():
                self.log_alpha.fill_(np.log(max(alpha_init, 1e-15)))
                self.log_beta.fill_(np.log(max(beta_init, 1e-15)))
            self._initialized = True

    def forward(self, W: torch.Tensor) -> DSDTorchResult:
        """
        Compute DSD-regularized pseudo-inverse of kernel matrix W.

        Parameters
        ----------
        W : torch.Tensor of shape (m, m), float64
            Symmetric positive semi-definite kernel submatrix.

        Returns
        -------
        DSDTorchResult with pseudo-inverse and diagnostics.
        """
        # Eigendecomposition (differentiable)
        # NOTE: torch.linalg.eigh backward involves terms 1/(λᵢ - λⱼ).
        # For degenerate eigenvalues this produces NaN. We mitigate by:
        # 1. Using float64 (minimizes accidental degeneracy)
        # 2. Detaching eigenvectors from gradient graph when gaps < eps
        #    (only differentiate through eigenvalue/damping path)
        eigenvalues, eigenvectors = torch.linalg.eigh(W)

        # Retain positive eigenvalues
        pos_mask = eigenvalues > 1e-12
        eigenvalues = eigenvalues[pos_mask]
        eigenvectors = eigenvectors[:, pos_mask]

        # Check for near-degenerate eigenvalues and detach eigenvectors
        # if the minimum gap is too small for stable gradients
        min_gap = torch.min(torch.abs(eigenvalues[1:] - eigenvalues[:-1]))
        if min_gap.item() < self.lorentzian_eps:
            # Detach eigenvectors to prevent NaN gradients from 1/(λᵢ-λⱼ)
            # Gradients still flow through eigenvalues → damping → inverse
            eigenvectors = eigenvectors.detach()

        # Auto-initialize if needed
        self._auto_initialize(eigenvalues.detach())

        # Eigengaps (differentiable through eigenvalues)
        eigengaps = compute_eigengaps_torch(eigenvalues)

        # DSD damping: α · exp(-β · δᵢ)
        alpha = self.alpha
        beta = self.beta

        # Clamp β·δ to prevent overflow in exp
        beta_delta = beta * eigengaps
        beta_delta_clamped = torch.clamp(beta_delta, max=500.0)
        damping = alpha * torch.exp(-beta_delta_clamped)

        # Regularized inverse: λᵢ / (λᵢ² + damping_i)
        reg_inverse_diag = eigenvalues / (eigenvalues ** 2 + damping)

        # Reconstruct pseudo-inverse: W̃⁺ = U · diag(λ̃⁻¹) · Uᵀ
        pseudo_inverse = (eigenvectors * reg_inverse_diag.unsqueeze(0)) @ eigenvectors.T

        return DSDTorchResult(
            pseudo_inverse=pseudo_inverse,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            eigengaps=eigengaps,
            damping=damping,
            reg_inverse_diag=reg_inverse_diag,
            alpha=alpha,
            beta=beta,
        )


def dsd_preimage_torch(
    k_query: torch.Tensor,
    W_inv: torch.Tensor,
    landmarks: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pre-image from kernel-space representation (differentiable).

    x̂ = (W⁻¹ · k_query)ᵀ · landmarks
    """
    weights = W_inv @ k_query
    return weights @ landmarks


def rbf_kernel_matrix_torch(X: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute RBF kernel matrix: K_ij = exp(-γ ||xᵢ - xⱼ||²)."""
    dists_sq = torch.cdist(X, X, p=2.0) ** 2
    return torch.exp(-gamma * dists_sq)


def rbf_kernel_vector_torch(
    x_query: torch.Tensor,
    landmarks: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute kernel vector between query point and landmarks."""
    dists_sq = torch.sum((landmarks - x_query.unsqueeze(0)) ** 2, dim=1)
    return torch.exp(-gamma * dists_sq)


# ============================================================
# MODEL SERIALIZATION
# ============================================================

def save_dsd_model(
    filepath: str,
    dsd_module: DSDModule,
    landmarks: np.ndarray,
    gamma: float,
    metadata: dict | None = None,
):
    """
    Save a fitted DSD model to disk.
    
    Parameters
    ----------
    filepath : str
        Path to save (e.g., 'model.pt').
    dsd_module : DSDModule
        Trained DSD module with optimized α, β.
    landmarks : ndarray (m, d)
        Nyström landmark points.
    gamma : float
        RBF kernel bandwidth.
    metadata : dict, optional
        Additional metadata (dataset name, training info, etc.)
    """
    state = {
        'dsd_state_dict': dsd_module.state_dict(),
        'landmarks': landmarks,
        'gamma': gamma,
        'alpha': float(dsd_module.alpha.item()),
        'beta': float(dsd_module.beta.item()),
        'metadata': metadata or {},
    }
    torch.save(state, filepath)


def load_dsd_model(filepath: str) -> tuple[DSDModule, np.ndarray, float, dict]:
    """
    Load a fitted DSD model from disk.
    
    Returns
    -------
    (dsd_module, landmarks, gamma, metadata)
    """
    state = torch.load(filepath, weights_only=False)
    
    dsd_module = DSDModule(
        alpha_init=state['alpha'],
        beta_init=state['beta'],
    )
    dsd_module.load_state_dict(state['dsd_state_dict'])
    
    return (
        dsd_module,
        state['landmarks'],
        state['gamma'],
        state.get('metadata', {}),
    )
