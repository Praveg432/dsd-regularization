"""
Kernel Least-Squares Twin SVM (Kernel-LSTSVM) with DSD inversion.

Non-linear extension of LSTSVM: replaces linear dot products with
RBF kernel evaluations. The system matrices become actual kernel matrices
with exponential spectral decay — exactly DSD's strongest regime.

Formulation (Kumar & Gopal, 2009 — kernel variant):
    Plane 1: (K₁₁ + (1/c₁)·K₂₁ᵀK₂₁ + εI) · α₁ = -K₂₁ᵀe₂
    Plane 2: (K₂₂ + (1/c₂)·K₁₂ᵀK₁₂ + εI) · α₂ = K₁₂ᵀe₁

Where:
    K₁₁ = kernel(A, A) — class 1 self-kernel (m₁ × m₁)
    K₂₂ = kernel(B, B) — class 2 self-kernel (m₂ × m₂)
    K₁₂ = kernel(A, B) — cross-class kernel (m₁ × m₂)
    K₂₁ = K₁₂ᵀ         — (m₂ × m₁)

The system matrices M₁ ∈ ℝ^(m₁×m₁) and M₂ ∈ ℝ^(m₂×m₂) are kernel
matrices with exponential eigenvalue decay (from RBF). This is where
DSD dominates: gap-adaptive damping on actual kernel spectra rather
than the squared-singular-value spectra of linear product matrices.

Prediction:
    f₁(x) = k(x, A)·α₁ + b₁
    f₂(x) = k(x, B)·α₂ + b₂
    ŷ = class whose plane is closer: argmin_k |f_k(x)| / ||α_k||
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from .dsd import dsd_regularized_inverse, tikhonov_inverse, naive_pseudo_inverse
from .kernels import rbf_kernel_matrix
from scipy.spatial.distance import cdist


class KernelLSTSVM(BaseEstimator, ClassifierMixin):
    """
    Kernel Least-Squares Twin SVM with configurable matrix inversion.
    
    Non-linear decision boundaries via RBF kernel. DSD inverts the
    kernel matrices directly — exponential spectral decay regime.
    
    Parameters
    ----------
    c1, c2 : float
        Regularization for class 1 and class 2 planes.
    gamma : float
        RBF kernel bandwidth: K(x,y) = exp(-γ||x-y||²).
    inversion_method : str
        "dsd", "tikhonov", "naive".
    tikhonov_gamma : float
        Ridge parameter for Tikhonov method.
    """
    
    def __init__(
        self,
        c1: float = 1.0,
        c2: float = 1.0,
        gamma: float = 1.0,
        inversion_method: str = "dsd",
        tikhonov_gamma: float = 1e-3,
    ):
        self.c1 = c1
        self.c2 = c2
        self.gamma = gamma
        self.inversion_method = inversion_method
        self.tikhonov_gamma = tikhonov_gamma
        
        # Fitted parameters
        self.alpha1_ = None
        self.alpha2_ = None
        self.b1_ = None
        self.b2_ = None
        self.A_ = None  # Class 1 training points
        self.B_ = None  # Class 2 training points
        self.classes_ = None

    def _rbf_kernel(self, X: NDArray, Y: NDArray) -> NDArray:
        """Compute RBF kernel matrix between X and Y."""
        dists_sq = cdist(X, Y, metric='sqeuclidean')
        return np.exp(-self.gamma * dists_sq)
    
    def _solve_system(self, M: NDArray, rhs: NDArray) -> NDArray:
        """Solve M·x = rhs using configured inversion method."""
        if self.inversion_method == "dsd":
            # Symmetrize M (should already be, but floating point)
            M_sym = (M + M.T) / 2
            result = dsd_regularized_inverse(M_sym)
            return result.pseudo_inverse @ rhs
        elif self.inversion_method == "tikhonov":
            M_reg = M + self.tikhonov_gamma * np.eye(M.shape[0])
            return np.linalg.solve(M_reg, rhs)
        elif self.inversion_method == "naive":
            return np.linalg.pinv(M) @ rhs
        else:
            raise ValueError(f"Unknown method: {self.inversion_method}")
    
    def fit(self, X: NDArray, y: NDArray) -> "KernelLSTSVM":
        """
        Fit Kernel-LSTSVM.
        
        Parameters
        ----------
        X : array (n, d)
            Training features.
        y : array (n,)
            Binary class labels.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError("KernelLSTSVM requires exactly 2 classes.")
        
        # Split by class
        self.A_ = X[y == self.classes_[0]].copy()  # Class 0
        self.B_ = X[y == self.classes_[1]].copy()  # Class 1
        m1, m2 = len(self.A_), len(self.B_)
        
        # Compute kernel matrices
        K11 = self._rbf_kernel(self.A_, self.A_)  # (m1, m1)
        K22 = self._rbf_kernel(self.B_, self.B_)  # (m2, m2)
        K12 = self._rbf_kernel(self.A_, self.B_)  # (m1, m2)
        K21 = K12.T                                # (m2, m1)
        
        # Plane 1: M1 = K11 + (1/c1) * K12 @ K21, shape (m1, m1)
        M1 = K11 + (1.0 / self.c1) * (K12 @ K21)
        rhs1 = -(K12 @ np.ones(m2))
        
        self.alpha1_ = self._solve_system(M1, rhs1)  # (m1,)
        
        # Bias: from KKT condition  Σ α₁ᵢ = ... 
        # Simple approach: b = mean(f₁) over class 0 support
        f1_train = K11 @ self.alpha1_
        self.b1_ = -float(np.mean(f1_train))
        
        # Plane 2: M2 = K22 + (1/c2) * K21 @ K12, shape (m2, m2)
        M2 = K22 + (1.0 / self.c2) * (K21 @ K12)
        rhs2 = K21 @ np.ones(m1)
        
        self.alpha2_ = self._solve_system(M2, rhs2)  # (m2,)
        
        # Bias
        f2_train = K22 @ self.alpha2_
        self.b2_ = -float(np.mean(f2_train))
        
        return self
    
    def decision_function(self, X: NDArray) -> tuple[NDArray, NDArray]:
        """Compute distance to each hyperplane for test points."""
        X = np.asarray(X, dtype=np.float64)
        
        # Kernel vectors to each class's support
        K_test_A = self._rbf_kernel(X, self.A_)  # (n_test, m1)
        K_test_B = self._rbf_kernel(X, self.B_)  # (n_test, m2)
        
        # Distance to plane 1 (normalized)
        f1 = K_test_A @ self.alpha1_ + self.b1_
        norm1 = np.linalg.norm(self.alpha1_) + 1e-10
        dist1 = np.abs(f1) / norm1
        
        # Distance to plane 2 (normalized)
        f2 = K_test_B @ self.alpha2_ + self.b2_
        norm2 = np.linalg.norm(self.alpha2_) + 1e-10
        dist2 = np.abs(f2) / norm2
        
        return dist1, dist2
    
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels: assign to nearer hyperplane."""
        dist1, dist2 = self.decision_function(X)
        return np.where(dist1 < dist2, self.classes_[0], self.classes_[1])
    
    def score(self, X: NDArray, y: NDArray) -> float:
        """Classification accuracy."""
        return float(np.mean(self.predict(X) == y))
