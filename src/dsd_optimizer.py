"""
DSD Hyperparameter Optimizer.

Optimizes α and β via backpropagation through the pre-image reconstruction
loss. The key insight: since DSD's forward pass is fully differentiable,
we can compute ∂L/∂α and ∂L/∂β where L = ||x̂ - x||² and use standard
gradient descent.

Training protocol:
  1. Construct kernel matrix W from landmarks
  2. For each training point x: compute k(x, landmarks), then x̂ = DSD_inv(W) · k
  3. Loss = mean ||x̂ - x||²
  4. Backprop to update α, β

This finds the (α, β) that minimizes reconstruction error on observed data,
which should generalize to unseen boundary points since the spectral structure
is a property of the kernel matrix, not the query points.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass

from .dsd_torch import (
    DSDModule,
    DSDTorchResult,
    dsd_preimage_torch,
    rbf_kernel_matrix_torch,
    rbf_kernel_vector_torch,
)


@dataclass
class OptimizationResult:
    """Result of DSD hyperparameter optimization."""
    alpha_final: float
    beta_final: float
    alpha_init: float
    beta_init: float
    loss_history: list[float]
    n_epochs: int
    converged: bool
    improvement: float  # relative improvement over initial


class DSDOptimizer:
    """
    Gradient-based optimizer for DSD hyperparameters.

    Parameters
    ----------
    lr : float
        Learning rate for Adam optimizer.
    n_epochs : int
        Maximum optimization epochs.
    patience : int
        Early stopping patience (epochs without improvement).
    min_improvement : float
        Minimum relative improvement to continue training.
    n_train_points : int
        Number of training points to use per epoch.
    """

    def __init__(
        self,
        lr: float = 0.01,
        n_epochs: int = 200,
        patience: int = 20,
        min_improvement: float = 1e-4,
        n_train_points: int = 50,
    ):
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self.min_improvement = min_improvement
        self.n_train_points = n_train_points

    def optimize(
        self,
        X_train: np.ndarray,
        landmarks: np.ndarray,
        gamma: float,
        noise_matrix: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> tuple[DSDModule, OptimizationResult]:
        """
        Optimize DSD parameters on training data.

        Parameters
        ----------
        X_train : ndarray (n, d)
            Training points for reconstruction loss.
        landmarks : ndarray (m, d)
            Nyström landmark points.
        gamma : float
            RBF kernel bandwidth.
        noise_matrix : ndarray (m, m), optional
            Symmetric noise to add to kernel matrix (simulates perturbation).
        verbose : bool
            Print progress.

        Returns
        -------
        (dsd_module, result) : optimized module and training diagnostics.
        """
        # Convert to torch tensors (float64 for numerical stability)
        X_t = torch.tensor(X_train, dtype=torch.float64)
        L_t = torch.tensor(landmarks, dtype=torch.float64)

        # Build kernel matrix
        W = rbf_kernel_matrix_torch(L_t, gamma)

        # Add noise if specified
        if noise_matrix is not None:
            W = W + torch.tensor(noise_matrix, dtype=torch.float64)

        # Initialize DSD module (auto-init α, β from spectrum)
        dsd = DSDModule()

        # Run one forward pass to trigger auto-initialization
        with torch.no_grad():
            dsd(W)

        alpha_init = float(dsd.alpha.item())
        beta_init = float(dsd.beta.item())

        if verbose:
            print(f"  Initial α={alpha_init:.6f}, β={beta_init:.4f}")

        # Select training subset
        n_train = min(self.n_train_points, len(X_train))
        indices = np.random.choice(len(X_train), size=n_train, replace=False)
        X_subset = X_t[indices]

        # Pre-compute kernel vectors for training points (fixed)
        k_vectors = []
        for i in range(n_train):
            k_vec = rbf_kernel_vector_torch(X_subset[i], L_t, gamma)
            k_vectors.append(k_vec)
        k_vectors = torch.stack(k_vectors)  # (n_train, m)

        # Optimizer
        optimizer = torch.optim.Adam(dsd.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-5
        )

        # Training loop
        loss_history = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()

            # Forward: compute DSD inverse
            result = dsd(W)
            W_inv = result.pseudo_inverse

            # Compute pre-images for all training points
            preimages = k_vectors @ W_inv.T @ L_t  # (n_train, d)

            # Loss: mean squared reconstruction error
            loss = torch.mean(torch.sum((preimages - X_subset) ** 2, dim=1))

            # Backward
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(dsd.parameters(), max_norm=10.0)

            optimizer.step()
            scheduler.step(loss.item())

            loss_val = loss.item()
            loss_history.append(loss_val)

            # Early stopping
            if loss_val < best_loss * (1 - self.min_improvement):
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if verbose:
                    print(f"  Converged at epoch {epoch+1} (patience exhausted)")
                break

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: loss={loss_val:.6f}, "
                      f"α={dsd.alpha.item():.6f}, β={dsd.beta.item():.4f}")

        alpha_final = float(dsd.alpha.item())
        beta_final = float(dsd.beta.item())

        # Compute improvement
        initial_loss = loss_history[0] if loss_history else float('inf')
        final_loss = loss_history[-1] if loss_history else float('inf')
        improvement = (initial_loss - final_loss) / max(initial_loss, 1e-15)

        result = OptimizationResult(
            alpha_final=alpha_final,
            beta_final=beta_final,
            alpha_init=alpha_init,
            beta_init=beta_init,
            loss_history=loss_history,
            n_epochs=len(loss_history),
            converged=patience_counter >= self.patience,
            improvement=improvement,
        )

        if verbose:
            print(f"  Final α={alpha_final:.6f} (was {alpha_init:.6f}), "
                  f"β={beta_final:.4f} (was {beta_init:.4f})")
            print(f"  Loss: {initial_loss:.6f} → {final_loss:.6f} "
                  f"({improvement*100:.1f}% improvement)")

        return dsd, result
