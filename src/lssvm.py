"""
Least-Squares SVM (LS-SVM) with pluggable kernel matrix inversion.

LS-SVM replaces SVM's inequality constraints with equalities, converting
the QP problem into a linear system:

    [0, 1ᵀ; 1, Ω + γ⁻¹I] · [b; α] = [0; y]

Where Ω is the kernel matrix. The solution requires INVERTING (Ω + γ⁻¹I).
When Ω is ill-conditioned (RBF kernel, high-dimensional data), this inversion
is numerically unstable — DSD directly addresses this.

Unlike standard SVM where DSD helps with POST-HOC explanation, in LS-SVM
DSD stabilizes the CLASSIFIER ITSELF. Better inverse → better predictions.
"""

import numpy as np
from numpy.typing import NDArray
from .dsd import dsd_regularized_inverse, tikhonov_inverse, naive_pseudo_inverse
from .kernels import rbf_kernel_matrix, rbf_kernel_vector


class LSSVM:
    """
    Least-Squares Support Vector Machine with configurable inversion method.
    
    Parameters
    ----------
    gamma : float
        RBF kernel bandwidth.
    reg_param : float
        Regularization parameter (γ in the LS-SVM formulation).
    inversion_method : str
        "dsd", "tikhonov", or "naive" — controls how the kernel matrix is inverted.
    tikhonov_gamma : float
        Regularization strength for Tikhonov method.
    """
    
    def __init__(
        self,
        gamma: float = 1.0,
        reg_param: float = 1.0,
        inversion_method: str = "dsd",
        tikhonov_gamma: float = 1e-3,
    ):
        self.gamma = gamma
        self.reg_param = reg_param
        self.inversion_method = inversion_method
        self.tikhonov_gamma = tikhonov_gamma
        
        # Fitted parameters
        self.alpha_ = None
        self.bias_ = None
        self.X_train_ = None
    
    def fit(self, X: NDArray, y: NDArray) -> "LSSVM":
        """
        Fit the LS-SVM classifier.
        
        Solves: [0, 1ᵀ; 1, Ω + γ⁻¹I] · [b; α] = [0; y]
        
        By inverting the augmented system or solving the reduced system:
            α = (Ω + γ⁻¹I)⁻¹ · (y - b·1)
        With b estimated from the constraint Σαᵢ = 0.
        """
        self.X_train_ = X.copy()
        n = len(X)
        
        # Compute kernel matrix
        Omega = rbf_kernel_matrix(X, gamma=self.gamma)
        
        # Regularized kernel matrix: Ω + (1/reg_param) · I
        reg_matrix = Omega + (1.0 / self.reg_param) * np.eye(n)
        
        # Invert using specified method
        if self.inversion_method == "dsd":
            result = dsd_regularized_inverse(reg_matrix)
            K_inv = result.pseudo_inverse
        elif self.inversion_method == "tikhonov":
            K_inv = tikhonov_inverse(reg_matrix, gamma=self.tikhonov_gamma)
        elif self.inversion_method == "naive":
            K_inv = naive_pseudo_inverse(reg_matrix)
        else:
            raise ValueError(f"Unknown method: {self.inversion_method}")
        
        # Solve the full LS-SVM system via block elimination:
        # From the KKT conditions:
        #   b = (1ᵀ K_inv 1)⁻¹ · (1ᵀ K_inv y)
        #   α = K_inv · (y - b·1)
        
        ones = np.ones(n)
        K_inv_ones = K_inv @ ones
        K_inv_y = K_inv @ y
        
        # Bias
        denom = ones @ K_inv_ones
        if abs(denom) > 1e-10:
            self.bias_ = (ones @ K_inv_y) / denom
        else:
            self.bias_ = 0.0
        
        # Dual coefficients
        self.alpha_ = K_inv @ (y - self.bias_ * ones)
        
        return self
    
    def decision_function(self, X: NDArray) -> NDArray:
        """Compute decision values for test points."""
        n_test = len(X)
        decision = np.zeros(n_test)
        
        for i in range(n_test):
            k_vec = rbf_kernel_vector(X[i], self.X_train_, gamma=self.gamma)
            decision[i] = k_vec @ self.alpha_ + self.bias_
        
        return decision
    
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels (-1 or +1)."""
        return np.sign(self.decision_function(X))
    
    def score(self, X: NDArray, y: NDArray) -> float:
        """Classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
