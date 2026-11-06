"""Additional classifiers for the DSD breadth study: Kernel Fisher Discriminant (KFD) and
Kernel/linear Ridge Classification (RC), each with pluggable system-matrix inversion.

Both invert an ill-conditioned, kernel/feature-derived system matrix, the same regime where DSD was
shown to help LSTSVM in the preprint (arXiv:2606.23407). We use the LINEAR feature-space form in the
high-dimensional regime (d large), because that is where the product-matrix spectrum exhibits the
severe eigengap tail-clustering DSD targets (mirrors the LSTSVM experiments).

Inversion methods (all plug into the same solve):
  "dsd"       - DSD gap-adaptive inverse, auto-init (no tuning)
  "tikhonov"  - ridge gamma (grid-searched externally)
  "tsvd"      - truncated-SVD pseudo-inverse, rank k (tuned externally)
  "spectral"  - general spectral filter g(lambda)/lambda with a tunable exponent p (Wiener-style),
                i.e. filter_i = lambda_i^p / (lambda_i^(p+1) + eps); p and eps tuned externally.
                This is a genuinely different regularizer family (magnitude-based soft filter),
                NOT a Tikhonov clone, so it is a real competitor.
  "naive"     - Moore-Penrose pinv
This matches the preprint's fairness protocol (DSD auto-init; every competitor tuned).
"""
import numpy as np
from numpy.typing import NDArray
from .dsd import dsd_regularized_inverse, naive_pseudo_inverse, truncated_svd_inverse


def _spectral_filter_inverse(M: NDArray, p: float, eps: float) -> NDArray:
    """General magnitude-based spectral filter on the eigenbasis.

    Effective inverse eigenvalue: lambda^p / (lambda^(p+1) + eps).
    p=1, eps=gamma^2-ish -> close to a smoothed Tikhonov; p<1 damps small eigenvalues harder
    (more TSVD-like); p>1 preserves them more. A real tunable filter family distinct from DSD
    (which keys off the eigenGAP, not the magnitude).
    """
    ev, U = np.linalg.eigh((M + M.T) / 2.0)
    pos = ev > 1e-12
    ev = ev[pos]; U = U[:, pos]
    filt = (ev ** p) / (ev ** (p + 1.0) + eps)
    return (U * filt) @ U.T


def _solve(M: NDArray, rhs: NDArray, method: str, tik_gamma: float = 1e-3,
           tsvd_rank: int | None = None, spec_p: float = 1.0, spec_eps: float = 1e-3) -> NDArray:
    M = (M + M.T) / 2.0
    if method == "dsd":
        return dsd_regularized_inverse(M).pseudo_inverse @ rhs
    if method == "tikhonov":
        return np.linalg.solve(M + tik_gamma * np.eye(M.shape[0]), rhs)
    if method == "tsvd":
        k = tsvd_rank if tsvd_rank is not None else M.shape[0]
        k = int(max(1, min(k, M.shape[0])))
        return truncated_svd_inverse(M, rank_k=k) @ rhs
    if method == "spectral":
        return _spectral_filter_inverse(M, spec_p, spec_eps) @ rhs
    if method == "naive":
        return naive_pseudo_inverse(M) @ rhs
    raise ValueError(f"unknown method: {method}")


class RidgeClassifier:
    """Linear ridge (regularized least-squares) classifier in the high-d feature space.

    Solves (X^T X + ...) w = X^T y via the chosen inversion of the Gram-like system matrix
    M = X_aug^T X_aug (d+1 x d+1), which is ill-conditioned / tail-clustered at high d.
    Prediction: sign(w^T x_aug). Labels assumed in {-1,+1} internally.
    """

    def __init__(self, inversion_method: str = "dsd", tikhonov_gamma: float = 1e-3,
                 tsvd_rank: int | None = None, spec_p: float = 1.0, spec_eps: float = 1e-3):
        self.inversion_method = inversion_method
        self.tikhonov_gamma = tikhonov_gamma
        self.tsvd_rank = tsvd_rank
        self.spec_p = spec_p
        self.spec_eps = spec_eps
        self.w_ = None
        self.classes_ = None

    def fit(self, X: NDArray, y: NDArray) -> "RidgeClassifier":
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = np.unique(y)
        yb = np.where(y == self.classes_[0], -1.0, 1.0)
        Xa = np.hstack([X, np.ones((len(X), 1))])        # augment with bias column
        M = Xa.T @ Xa                                     # (d+1, d+1) tail-clustered at high d
        rhs = Xa.T @ yb
        self.w_ = _solve(M, rhs, self.inversion_method, self.tikhonov_gamma,
                         self.tsvd_rank, self.spec_p, self.spec_eps)
        return self

    def predict(self, X: NDArray) -> NDArray:
        Xa = np.hstack([np.asarray(X, dtype=np.float64), np.ones((len(X), 1))])
        s = Xa @ self.w_
        return np.where(s < 0, self.classes_[0], self.classes_[1])

    def score(self, X: NDArray, y: NDArray) -> float:
        return float(np.mean(self.predict(X) == y))


class FisherDiscriminant:
    """Linear Fisher Discriminant (LDA direction) in the high-d feature space.

    Solves S_w w = (m1 - m2) where S_w is the within-class scatter matrix (d x d), which is severely
    ill-conditioned / tail-clustered when d is large relative to n (the DSD regime). The inversion of
    S_w is where DSD acts. Threshold set at the midpoint of projected class means.
    """

    def __init__(self, inversion_method: str = "dsd", tikhonov_gamma: float = 1e-3,
                 tsvd_rank: int | None = None, spec_p: float = 1.0, spec_eps: float = 1e-3):
        self.inversion_method = inversion_method
        self.tikhonov_gamma = tikhonov_gamma
        self.tsvd_rank = tsvd_rank
        self.spec_p = spec_p
        self.spec_eps = spec_eps
        self.w_ = None
        self.threshold_ = None
        self.classes_ = None
        self.sign_ = 1.0

    def fit(self, X: NDArray, y: NDArray) -> "FisherDiscriminant":
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = np.unique(y)
        A = X[y == self.classes_[0]]
        B = X[y == self.classes_[1]]
        mA = A.mean(0); mB = B.mean(0)
        # within-class scatter
        Sw = (A - mA).T @ (A - mA) + (B - mB).T @ (B - mB)   # (d, d), tail-clustered at high d
        rhs = (mA - mB)
        self.w_ = _solve(Sw, rhs, self.inversion_method, self.tikhonov_gamma,
                         self.tsvd_rank, self.spec_p, self.spec_eps)
        # threshold at midpoint of projected means; orient so class0 < threshold
        pA = float(A @ self.w_ @ np.ones(1)) if A.ndim == 1 else float(np.mean(A @ self.w_))
        pB = float(np.mean(B @ self.w_))
        self.threshold_ = 0.5 * (pA + pB)
        self.sign_ = 1.0 if pA < pB else -1.0
        return self

    def predict(self, X: NDArray) -> NDArray:
        proj = np.asarray(X, dtype=np.float64) @ self.w_
        # class0 if on the 'A' side of threshold given orientation
        is_class0 = (self.sign_ * (proj - self.threshold_)) < 0
        return np.where(is_class0, self.classes_[0], self.classes_[1])

    def score(self, X: NDArray, y: NDArray) -> float:
        return float(np.mean(self.predict(X) == y))
