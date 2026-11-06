"""
Least-Squares Twin SVM (LSTSVM) with pluggable matrix inversion.

Correct formulation (Kumar & Gopal, 2009):
    Plane 1 (close to class A): M1 = E1ᵀE1 + (1/c1)·E2ᵀE2,  solve M1·u1 = -E2ᵀe2
    Plane 2 (close to class B): M2 = E2ᵀE2 + (1/c2)·E1ᵀE1,  solve M2·u2 = E1ᵀe1

Where E1 = [A, e1], E2 = [B, e2] are augmented feature matrices.

The matrices M1 and M2 contain CROSS-CLASS terms (E2ᵀE2 in M1, E1ᵀE1 in M2).
These product matrices have squared spectral decay → DSD advantage amplified.

DSD replaces np.linalg.solve(M, rhs) with M_inv @ rhs where M_inv is the
DSD-regularized pseudo-inverse — addressing eigenvector corruption that
flat Tikhonov (epsilon·I) cannot fix.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from .dsd import dsd_regularized_inverse, tikhonov_inverse, naive_pseudo_inverse


class LSTSVM(BaseEstimator, ClassifierMixin):
    """
    Least-Squares Twin SVM with configurable matrix inversion.
    
    Parameters
    ----------
    c1, c2 : float
        Regularization for class 1 and class 2 planes.
    inversion_method : str
        "dsd", "tikhonov", "naive", or "solve" (np.linalg.solve baseline).
    tikhonov_gamma : float
        Ridge factor for tikhonov/solve methods.
    """
    
    def __init__(
        self,
        c1: float = 1.0,
        c2: float = 1.0,
        inversion_method: str = "dsd",
        tikhonov_gamma: float = 1e-7,
    ):
        self.c1 = c1
        self.c2 = c2
        self.inversion_method = inversion_method
        self.tikhonov_gamma = tikhonov_gamma
        
        self.w1_ = None
        self.b1_ = None
        self.w2_ = None
        self.b2_ = None
        self.classes_ = None
    
    def _solve_system(self, M: NDArray, rhs: NDArray) -> NDArray:
        """Solve M·x = rhs using configured inversion method."""
        if self.inversion_method == "dsd":
            result = dsd_regularized_inverse(M)
            return result.pseudo_inverse @ rhs
        elif self.inversion_method == "tikhonov":
            M_reg = M + self.tikhonov_gamma * np.eye(M.shape[0])
            return np.linalg.solve(M_reg, rhs)
        elif self.inversion_method == "naive":
            return np.linalg.pinv(M) @ rhs
        elif self.inversion_method == "solve":
            # Baseline: direct solve with ridge (standard LSTSVM implementation)
            M_reg = M + self.tikhonov_gamma * np.eye(M.shape[0])
            return np.linalg.solve(M_reg, rhs)
        else:
            raise ValueError(f"Unknown method: {self.inversion_method}")
    
    def fit(self, X: NDArray, y: NDArray) -> "LSTSVM":
        """
        Fit LSTSVM.
        
        Parameters
        ----------
        X : array (n, d)
        y : array (n,) with exactly 2 unique class labels
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError("LSTSVM requires exactly 2 classes.")
        
        # Split by class
        A = X[y == self.classes_[0]]  # Class 0
        B = X[y == self.classes_[1]]  # Class 1
        m1, m2 = len(A), len(B)
        
        # Augmented matrices (append bias column of ones)
        E1 = np.hstack([A, np.ones((m1, 1))])  # (m1, d+1)
        E2 = np.hstack([B, np.ones((m2, 1))])  # (m2, d+1)
        
        # ─── Plane 1: close to class 0 ───
        # M1 = E1ᵀE1 + (1/c1)·E2ᵀE2
        M1 = E1.T @ E1 + (1.0 / self.c1) * (E2.T @ E2)
        rhs1 = -(E2.T @ np.ones(m2))
        
        sol1 = self._solve_system(M1, rhs1)
        self.w1_ = sol1[:-1].flatten()
        self.b1_ = float(sol1[-1])
        
        # ─── Plane 2: close to class 1 ───
        # M2 = E2ᵀE2 + (1/c2)·E1ᵀE1
        M2 = E2.T @ E2 + (1.0 / self.c2) * (E1.T @ E1)
        rhs2 = E1.T @ np.ones(m1)
        
        sol2 = self._solve_system(M2, rhs2)
        self.w2_ = sol2[:-1].flatten()
        self.b2_ = float(sol2[-1])
        
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels based on proximity to each plane."""
        X = np.asarray(X, dtype=float)
        
        # Distance to each plane (normalized by weight norm)
        dist1 = np.abs(X @ self.w1_ + self.b1_) / (np.linalg.norm(self.w1_) + 1e-10)
        dist2 = np.abs(X @ self.w2_ + self.b2_) / (np.linalg.norm(self.w2_) + 1e-10)
        
        # Closer to plane 1 → class 0; closer to plane 2 → class 1
        return np.where(dist1 < dist2, self.classes_[0], self.classes_[1])
    
    def score(self, X: NDArray, y: NDArray) -> float:
        """Classification accuracy."""
        return float(np.mean(self.predict(X) == y))
