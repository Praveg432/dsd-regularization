"""DSD-SVM: Differential Spectral Damping for Kernel Matrix Pseudo-Inversion."""

from .dsd import (
    dsd_regularized_inverse,
    DSDResult,
    compute_eigengaps,
    initialize_hyperparameters,
    tikhonov_inverse,
    tikhonov_inverse_optimized,
    truncated_svd_inverse,
    naive_pseudo_inverse,
)
from .kernels import rbf_kernel_matrix, rbf_kernel_vector, nystrom_sample
from .preimage import compute_preimage, preimage_pipeline
from .lstsvm import LSTSVM
from .kernel_lstsvm import KernelLSTSVM
from .lssvm import LSSVM
from .dsd_scalable import dsd_scalable, DSDScalableResult

__all__ = [
    "dsd_regularized_inverse",
    "DSDResult",
    "compute_eigengaps",
    "initialize_hyperparameters",
    "tikhonov_inverse",
    "tikhonov_inverse_optimized",
    "truncated_svd_inverse",
    "naive_pseudo_inverse",
    "rbf_kernel_matrix",
    "rbf_kernel_vector",
    "nystrom_sample",
    "compute_preimage",
    "preimage_pipeline",
    "LSTSVM",
    "KernelLSTSVM",
    "LSSVM",
    "dsd_scalable",
    "DSDScalableResult",
]
