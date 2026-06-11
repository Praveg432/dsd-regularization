"""Tests for differentiable DSD (PyTorch module, optimizer, serialization)."""

import numpy as np
import torch
import pytest
import tempfile
import os

from src.dsd_torch import (
    DSDModule,
    DSDTorchResult,
    compute_eigengaps_torch,
    rbf_kernel_matrix_torch,
    rbf_kernel_vector_torch,
    dsd_preimage_torch,
    save_dsd_model,
    load_dsd_model,
)
from src.dsd_optimizer import DSDOptimizer


class TestEigengapsTorch:
    def test_uniform_spacing(self):
        eigenvalues = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        gaps = compute_eigengaps_torch(eigenvalues)
        assert torch.allclose(gaps, torch.ones(5, dtype=torch.float64))

    def test_boundary_handling(self):
        eigenvalues = torch.tensor([0.1, 0.5, 1.0, 3.0, 10.0], dtype=torch.float64)
        gaps = compute_eigengaps_torch(eigenvalues)
        assert gaps[0].item() == pytest.approx(0.4, abs=1e-10)   # right gap
        assert gaps[-1].item() == pytest.approx(7.0, abs=1e-10)  # left gap

    def test_differentiable(self):
        eigenvalues = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64, requires_grad=True)
        gaps = compute_eigengaps_torch(eigenvalues)
        gaps.sum().backward()
        assert eigenvalues.grad is not None
        assert torch.all(torch.isfinite(eigenvalues.grad))


class TestDSDModule:
    def test_forward_shape(self):
        X = torch.randn(30, 3, dtype=torch.float64)
        W = rbf_kernel_matrix_torch(X, gamma=1.0)
        dsd = DSDModule()
        result = dsd(W)
        assert result.pseudo_inverse.shape == (30, 30)

    def test_symmetric_output(self):
        X = torch.randn(40, 3, dtype=torch.float64)
        W = rbf_kernel_matrix_torch(X, gamma=1.0)
        dsd = DSDModule()
        result = dsd(W)
        assert torch.allclose(result.pseudo_inverse, result.pseudo_inverse.T, atol=1e-10)

    def test_gradient_flow(self):
        X = torch.randn(25, 3, dtype=torch.float64)
        W = rbf_kernel_matrix_torch(X, gamma=1.0)
        dsd = DSDModule()
        result = dsd(W)
        loss = result.pseudo_inverse.sum()
        loss.backward()
        assert dsd.log_alpha.grad is not None
        assert dsd.log_beta.grad is not None
        assert torch.isfinite(dsd.log_alpha.grad)
        assert torch.isfinite(dsd.log_beta.grad)

    def test_auto_initialization(self):
        X = torch.randn(50, 3, dtype=torch.float64)
        W = rbf_kernel_matrix_torch(X, gamma=1.0)
        dsd = DSDModule()
        assert not dsd._initialized
        dsd(W)
        assert dsd._initialized
        assert dsd.alpha.item() > 0
        assert dsd.beta.item() > 0

    def test_explicit_initialization(self):
        dsd = DSDModule(alpha_init=0.5, beta_init=10.0)
        assert dsd._initialized
        assert dsd.alpha.item() == pytest.approx(0.5, rel=1e-5)
        assert dsd.beta.item() == pytest.approx(10.0, rel=1e-5)

    def test_degenerate_eigenvalues_no_nan(self):
        """DSD should not produce NaN even with near-degenerate eigenvalues."""
        # Matrix with repeated eigenvalues
        W = torch.eye(20, dtype=torch.float64) * 1.0
        W[0, 0] = 1.0001  # near-degenerate
        W[1, 1] = 1.0002
        dsd = DSDModule(alpha_init=0.01, beta_init=1.0)
        result = dsd(W)
        assert torch.all(torch.isfinite(result.pseudo_inverse))


class TestDSDOptimizer:
    def test_optimization_reduces_loss(self):
        np.random.seed(42)
        torch.manual_seed(42)
        from src.kernels import nystrom_sample
        from sklearn.datasets import make_swiss_roll
        from sklearn.preprocessing import StandardScaler

        X, _ = make_swiss_roll(n_samples=500, noise=0.3, random_state=42)
        X = StandardScaler().fit_transform(X)
        landmarks, _ = nystrom_sample(X[:400], m=50, method='kmeans')
        noise = 5e-3 * np.random.randn(50, 50)
        noise = (noise + noise.T) / 2

        optimizer = DSDOptimizer(lr=0.01, n_epochs=50, patience=10, n_train_points=20)
        dsd_module, result = optimizer.optimize(
            X_train=X[:400], landmarks=landmarks, gamma=2.0, noise_matrix=noise
        )
        assert result.improvement > 0  # loss decreased
        assert result.alpha_final > 0
        assert result.beta_final > 0
        assert len(result.loss_history) > 1
        assert result.loss_history[-1] < result.loss_history[0]

    def test_convergence_flag(self):
        np.random.seed(0)
        torch.manual_seed(0)
        from src.kernels import nystrom_sample

        X = np.random.randn(100, 3)
        landmarks, _ = nystrom_sample(X[:80], m=30, method='random')

        # Very high patience so it runs all epochs
        optimizer = DSDOptimizer(lr=0.01, n_epochs=20, patience=100, n_train_points=10)
        _, result = optimizer.optimize(X_train=X[:80], landmarks=landmarks, gamma=1.0)
        assert result.n_epochs == 20  # ran all epochs without early stop


class TestSerialization:
    def test_save_load_roundtrip(self):
        dsd = DSDModule(alpha_init=0.5, beta_init=10.0)
        landmarks = np.random.randn(50, 3)
        gamma = 2.0

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            filepath = f.name

        try:
            save_dsd_model(filepath, dsd, landmarks, gamma, metadata={'test': True})
            dsd_loaded, lm_loaded, g_loaded, meta = load_dsd_model(filepath)

            assert g_loaded == gamma
            assert np.allclose(lm_loaded, landmarks)
            assert dsd_loaded.alpha.item() == pytest.approx(0.5, rel=1e-5)
            assert dsd_loaded.beta.item() == pytest.approx(10.0, rel=1e-5)
            assert meta['test'] is True
        finally:
            os.unlink(filepath)


class TestPreimage:
    def test_preimage_reconstruction(self):
        """Pre-image of a landmark should approximately recover that landmark."""
        np.random.seed(42)
        landmarks = np.random.randn(30, 3)
        landmarks_t = torch.tensor(landmarks, dtype=torch.float64)

        W = rbf_kernel_matrix_torch(landmarks_t, gamma=1.0)
        dsd = DSDModule()
        result = dsd(W)

        # Query: kernel vector of landmark[0]
        k_q = rbf_kernel_vector_torch(landmarks_t[0], landmarks_t, gamma=1.0)
        x_hat = dsd_preimage_torch(k_q, result.pseudo_inverse, landmarks_t)

        error = torch.norm(x_hat - landmarks_t[0]).item()
        assert error < 0.5  # reasonable reconstruction


class TestScalable:
    def test_scalable_factored_matches_dense(self):
        from src.dsd_scalable import dsd_scalable

        np.random.seed(42)
        X = np.random.randn(100, 3)
        from src.kernels import rbf_kernel_matrix
        W = rbf_kernel_matrix(X, gamma=1.0)

        result = dsd_scalable(W, auto_threshold=200)  # force full path
        dense = result.to_dense()

        # Apply to a vector
        k_q = np.random.randn(100)
        factored_result = result.apply(k_q)
        dense_result = dense @ k_q

        assert np.allclose(factored_result, dense_result, atol=1e-10)

    def test_scalable_partial_path(self):
        from src.dsd_scalable import dsd_scalable

        np.random.seed(42)
        X = np.random.randn(100, 3)
        from src.kernels import rbf_kernel_matrix
        W = rbf_kernel_matrix(X, gamma=1.0)

        # Force partial path with low threshold
        result = dsd_scalable(W, max_rank=30, auto_threshold=50)
        assert result.rank_used <= 30
        assert result.full_size == 100

        # Should still produce valid output
        k_q = np.random.randn(100)
        out = result.apply(k_q)
        assert out.shape == (100,)
        assert np.all(np.isfinite(out))

    def test_batch_apply(self):
        from src.dsd_scalable import dsd_scalable
        from src.kernels import rbf_kernel_matrix

        np.random.seed(42)
        X = np.random.randn(50, 3)
        W = rbf_kernel_matrix(X, gamma=1.0)

        result = dsd_scalable(W, auto_threshold=200)
        K_batch = np.random.randn(10, 50)
        out = result.apply(K_batch)
        assert out.shape == (10, 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
