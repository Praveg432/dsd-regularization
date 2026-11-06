"""Gaussian-Process regression with pluggable posterior-matrix inversion.

Standard GP regression inverts the posterior matrix (K + sigma^2 I). When that matrix is
ill-conditioned (sharp/smooth RBF, dense or near-duplicate inputs), the inversion corrupts not only
the predictive MEAN but also the predictive VARIANCE, producing mis-calibrated intervals. This module
lets us swap the inversion method and measure the effect on calibration (NLL, interval coverage),
which is the new axis the DSD follow-up studies.

Predictive equations (zero-mean prior):
    A       = K_XX + sigma^2 I                      (posterior matrix, n x n)
    mean(x*) = k_*^T A^{-1} y
    var(x*)  = k(x*,x*) - k_*^T A^{-1} k_*          (clipped at >=0, + sigma^2 for noisy obs)

The ONLY thing that varies across methods is how A^{-1} is formed:
    - "jitter":    plain inverse of (K + sigma^2 I)      (the standard GP practice baseline)
    - "tikhonov":  inverse of (K + (sigma^2 + gamma) I)  (extra ridge gamma, tuned)
    - "tsvd":      truncated-SVD pseudo-inverse of A      (rank k, tuned)
    - "dsd":       DSD gap-adaptive pseudo-inverse of A   (alpha,beta; auto-init or tuned)

Reuses src/dsd.py for the DSD and TSVD inverses so the follow-up builds directly on the preprint.
"""
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from .kernels import rbf_kernel_matrix
from .dsd import dsd_regularized_inverse, truncated_svd_inverse


def _rbf_cross(X1: NDArray, X2: NDArray, gamma: float) -> NDArray:
    """RBF cross-kernel K(X1, X2), shape (n1, n2)."""
    from scipy.spatial.distance import cdist
    d2 = cdist(X1, X2, metric="sqeuclidean")
    return np.exp(-gamma * d2)


@dataclass
class GPConfig:
    kernel_gamma: float = 1.0      # RBF bandwidth
    sigma2: float = 1e-4           # observation-noise variance (the GP "jitter"/nugget)
    method: str = "jitter"         # jitter | tikhonov | tsvd | dsd
    # method-specific hyperparameters (None -> use default / auto-init)
    tikhonov_gamma: float | None = None
    tsvd_rank: int | None = None
    dsd_alpha: float | None = None
    dsd_beta: float | None = None


class GPRegressor:
    """GP regression with a pluggable inverse of the posterior matrix A = K + sigma^2 I."""

    def __init__(self, config: GPConfig):
        self.cfg = config
        self.X_train_: NDArray | None = None
        self.y_train_: NDArray | None = None
        self.A_inv_: NDArray | None = None
        self.y_mean_: float = 0.0

    def _invert(self, A: NDArray) -> NDArray:
        m = self.cfg.method
        if m == "jitter":
            # plain inverse of the already-jittered posterior matrix
            return np.linalg.inv(A)
        if m == "tikhonov":
            g = self.cfg.tikhonov_gamma if self.cfg.tikhonov_gamma is not None else 1e-3
            n = A.shape[0]
            return np.linalg.inv(A + g * np.eye(n))
        if m == "tsvd":
            k = self.cfg.tsvd_rank if self.cfg.tsvd_rank is not None else A.shape[0]
            k = int(max(1, min(k, A.shape[0])))
            return truncated_svd_inverse(A, rank_k=k)
        if m == "dsd":
            res = dsd_regularized_inverse(A, alpha=self.cfg.dsd_alpha, beta=self.cfg.dsd_beta)
            return res.pseudo_inverse
        raise ValueError(f"unknown method: {m}")

    def fit(self, X: NDArray, y: NDArray) -> "GPRegressor":
        self.X_train_ = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.y_mean_ = float(y.mean())
        self.y_train_ = y - self.y_mean_                       # center target (zero-mean prior)
        K = rbf_kernel_matrix(self.X_train_, gamma=self.cfg.kernel_gamma)
        A = K + self.cfg.sigma2 * np.eye(len(self.X_train_))
        self.A_inv_ = self._invert(A)
        self._alpha_vec_ = self.A_inv_ @ self.y_train_          # precompute for mean
        return self

    def predict(self, X: NDArray, return_std: bool = True):
        """Return (mean, std) predictive distribution at X."""
        Xq = np.asarray(X, dtype=np.float64)
        k_star = _rbf_cross(Xq, self.X_train_, self.cfg.kernel_gamma)   # (nq, n)
        mean = k_star @ self._alpha_vec_ + self.y_mean_
        if not return_std:
            return mean
        # predictive variance: k(x*,x*) - k_*^T A^{-1} k_*, + sigma^2 for noisy predictive
        # k(x*,x*) = 1 for RBF
        v = np.einsum("ij,jk,ik->i", k_star, self.A_inv_, k_star)      # k_*^T A^{-1} k_*
        var_latent = np.clip(1.0 - v, 0.0, None)                       # numerical floor at 0
        var = var_latent + self.cfg.sigma2
        std = np.sqrt(var)
        return mean, std


# ---- metrics ---------------------------------------------------------------

def rmse(y_true: NDArray, y_pred: NDArray) -> float:
    return float(np.sqrt(np.mean((y_true.ravel() - y_pred.ravel()) ** 2)))


def gaussian_nll(y_true: NDArray, mean: NDArray, std: NDArray) -> float:
    """Mean negative log predictive density under a Gaussian predictive. Lower = better calibrated."""
    yt = y_true.ravel(); mu = mean.ravel(); sd = np.clip(std.ravel(), 1e-8, None)
    nll = 0.5 * np.log(2 * np.pi * sd ** 2) + (yt - mu) ** 2 / (2 * sd ** 2)
    return float(np.mean(nll))


def interval_coverage(y_true: NDArray, mean: NDArray, std: NDArray, level: float = 0.90) -> float:
    """Empirical coverage of the central `level` Gaussian credible interval. Target = level."""
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2)
    yt = y_true.ravel(); mu = mean.ravel(); sd = np.clip(std.ravel(), 1e-12, None)
    inside = np.abs(yt - mu) <= z * sd
    return float(np.mean(inside))
