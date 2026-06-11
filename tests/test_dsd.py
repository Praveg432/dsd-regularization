"""Unit tests for DSD core implementation (post-fix)."""

import numpy as np
import pytest
from src.dsd import (
    compute_eigengaps,
    initialize_hyperparameters,
    dsd_regularized_inverse,
    tikhonov_inverse,
    truncated_svd_inverse,
)
from src.kernels import rbf_kernel_matrix


class TestEigengaps:
    def test_uniform_spacing(self):
        """Uniformly spaced eigenvalues should have constant eigengaps."""
        eigenvalues = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gaps = compute_eigengaps(eigenvalues)
        assert np.allclose(gaps, 1.0)
    
    def test_clustered_tail(self):
        """Clustered eigenvalues should have small gaps."""
        eigenvalues = np.array([0.001, 0.002, 0.003, 1.0, 5.0])
        gaps = compute_eigengaps(eigenvalues)
        assert gaps[0] < 0.01  # clustered
        assert gaps[-1] > 1.0   # well-separated (left gap = |5-1| = 4)
    
    def test_boundary_no_inf(self):
        """Boundary eigenvalues should NOT have inf gaps (fixed)."""
        eigenvalues = np.array([0.1, 0.5, 1.0, 3.0, 10.0])
        gaps = compute_eigengaps(eigenvalues)
        assert np.all(np.isfinite(gaps))
        # First eigenvalue uses right gap only
        assert gaps[0] == abs(eigenvalues[1] - eigenvalues[0])
        # Last eigenvalue uses left gap only
        assert gaps[-1] == abs(eigenvalues[-1] - eigenvalues[-2])


class TestInitialization:
    def test_alpha_at_transition(self):
        """α should be at the spectral transition point, not median."""
        # Spectrum with clear transition: dense tail + sparse head
        eigenvalues = np.concatenate([
            np.linspace(0.001, 0.01, 50),   # dense tail (small gaps)
            np.linspace(0.1, 10.0, 50),      # sparse head (large gaps)
        ])
        alpha, beta = initialize_hyperparameters(eigenvalues)
        
        # α should be near the tail eigenvalues (transition point), not the head
        assert alpha < 0.1  # should be O(0.01²) = O(1e-4), not O(10²) = O(100)
        assert alpha > 0
    
    def test_beta_positive_finite(self):
        eigenvalues = np.logspace(-5, 2, 100)
        alpha, beta = initialize_hyperparameters(eigenvalues)
        assert beta > 0
        assert np.isfinite(beta)


class TestDSDFormula:
    def test_preserves_dominant_eigenvalues(self):
        """For large eigengaps, DSD should approximate exact inverse."""
        eigenvalues = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        W = np.diag(eigenvalues)
        
        result = dsd_regularized_inverse(W)
        
        # Dominant eigenvalue (large gap to neighbor) should be nearly exact
        exact_inv_top = 1.0 / eigenvalues[-1]
        dsd_inv_top = result.reg_inverse_diag[-1]
        relative_error = abs(dsd_inv_top - exact_inv_top) / exact_inv_top
        assert relative_error < 0.05  # within 5% of exact
    
    def test_suppresses_clustered_tail(self):
        """For small eigengaps, DSD should suppress the inverse magnitude."""
        eigenvalues = np.array([0.001, 0.0011, 0.0012, 0.0013, 1.0, 10.0])
        W = np.diag(eigenvalues)
        
        result = dsd_regularized_inverse(W)
        
        # Tail inverse should be smaller than naive 1/λ
        naive_tail = 1.0 / eigenvalues[0]  # ~1000
        dsd_tail = result.reg_inverse_diag[0]
        assert dsd_tail < naive_tail  # DSD damps it
    
    def test_symmetric_output(self):
        """Pseudo-inverse of symmetric matrix should be symmetric."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        W = rbf_kernel_matrix(X, gamma=1.0)
        
        result = dsd_regularized_inverse(W)
        assert np.allclose(result.pseudo_inverse, result.pseudo_inverse.T, atol=1e-10)
    
    def test_condition_improvement(self):
        """DSD should reduce condition number vs original."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        W = rbf_kernel_matrix(X, gamma=0.5)
        
        result = dsd_regularized_inverse(W)
        assert result.condition_number_regularized < result.condition_number_original
    
    def test_noise_robustness(self):
        """DSD error should grow slowly with matrix perturbation."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        W = rbf_kernel_matrix(X, gamma=2.0)
        
        # Compute DSD on clean matrix
        result_clean = dsd_regularized_inverse(W)
        
        # Compute DSD on noisy matrix
        noise = 1e-4 * np.random.randn(100, 100)
        noise = (noise + noise.T) / 2
        result_noisy = dsd_regularized_inverse(W + noise)
        
        # Pre-image weights should be similar (robustness)
        diff = np.linalg.norm(result_clean.pseudo_inverse - result_noisy.pseudo_inverse, 'fro')
        clean_norm = np.linalg.norm(result_clean.pseudo_inverse, 'fro')
        relative_change = diff / clean_norm
        
        # DSD should change < 50% under this noise level
        assert relative_change < 0.5


class TestBaselines:
    def test_tikhonov_via_eigh(self):
        """Tikhonov via eigendecomposition should be symmetric and well-conditioned."""
        np.random.seed(42)
        X = np.random.randn(30, 3)
        W = rbf_kernel_matrix(X, gamma=1.0)
        inv = tikhonov_inverse(W, gamma=0.01)
        assert np.allclose(inv, inv.T, atol=1e-10)
    
    def test_tsvd_rank(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        W = rbf_kernel_matrix(X, gamma=1.0)
        inv = truncated_svd_inverse(W, rank_k=10)
        rank = np.linalg.matrix_rank(inv, tol=1e-8)
        assert rank <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
