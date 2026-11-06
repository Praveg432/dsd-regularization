"""
Experiment 04: Ground-Truth Feature Attribution Validation

Creates synthetic datasets where the TRUE important features are KNOWN by construction.
Tests whether DSD pre-image explanations correctly recover the ground-truth feature ranking.

Design:
  - Dataset A: 3 relevant features + 7 noise features (d=10)
  - Dataset B: 2 relevant features + 18 noise features (d=20)
  - Dataset C: Non-linear boundary in relevant features (XOR-like)

For each: train RBF-SVM, compute DSD pre-images along boundary, extract feature
importances from pre-image geometry, compare against KNOWN ground truth.

This experiment is FULLY COMPARTMENTALIZED — imports only from src/, creates no
side effects on other experiments, and can be run/deleted independently.

Run: python experiments/04_feature_attribution_validation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr

from src.dsd import dsd_regularized_inverse, tikhonov_inverse, naive_pseudo_inverse
from src.kernels import rbf_kernel_matrix, rbf_kernel_vector, nystrom_sample
from src.preimage import compute_preimage


# ============================================================
# SYNTHETIC DATA GENERATORS (ground truth known by construction)
# ============================================================

def make_dataset_A(n=2000, noise_std=0.1, random_state=42):
    """
    Dataset A: Linear boundary in features 0-2, pure noise in features 3-9.
    
    True boundary: x₀ + x₁ + x₂ > 0
    Features 3-9: standard normal noise (uncorrelated with label)
    
    Ground truth importance: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    """
    rng = np.random.default_rng(random_state)
    d_relevant, d_noise = 3, 7
    d_total = d_relevant + d_noise
    
    X_relevant = rng.standard_normal((n, d_relevant))
    X_noise = rng.standard_normal((n, d_noise)) * noise_std
    X = np.hstack([X_relevant, X_noise])
    
    y = (X_relevant.sum(axis=1) > 0).astype(int)
    
    ground_truth = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return X, y, ground_truth, "Linear boundary in features 0-2, noise in 3-9"


def make_dataset_B(n=2000, noise_std=0.1, random_state=42):
    """
    Dataset B: Radial boundary in features 0-1, noise in features 2-19.
    
    True boundary: x₀² + x₁² > 1.0 (circle)
    Features 2-19: standard normal noise
    
    Ground truth importance: [1, 1, 0, 0, ..., 0]
    """
    rng = np.random.default_rng(random_state)
    d_relevant, d_noise = 2, 18
    d_total = d_relevant + d_noise
    
    X_relevant = rng.standard_normal((n, d_relevant)) * 1.5
    X_noise = rng.standard_normal((n, d_noise)) * noise_std
    X = np.hstack([X_relevant, X_noise])
    
    y = ((X_relevant[:, 0] ** 2 + X_relevant[:, 1] ** 2) > 1.0).astype(int)
    
    ground_truth = np.zeros(d_total)
    ground_truth[:2] = 1.0
    
    return X, y, ground_truth, "Circular boundary in features 0-1, noise in 2-19"


def make_dataset_C(n=2000, noise_std=0.1, random_state=42):
    """
    Dataset C: XOR-like non-linear interaction in features 0-1.
    
    True boundary: sign(x₀) ≠ sign(x₁) (XOR quadrants)
    Features 2-9: standard normal noise
    
    Ground truth importance: [1, 1, 0, 0, ..., 0]
    """
    rng = np.random.default_rng(random_state)
    d_relevant, d_noise = 2, 8
    d_total = d_relevant + d_noise
    
    X_relevant = rng.standard_normal((n, d_relevant))
    X_noise = rng.standard_normal((n, d_noise)) * noise_std
    X = np.hstack([X_relevant, X_noise])
    
    y = ((X_relevant[:, 0] * X_relevant[:, 1]) > 0).astype(int)
    
    ground_truth = np.zeros(d_total)
    ground_truth[:2] = 1.0
    
    return X, y, ground_truth, "XOR boundary in features 0-1, noise in 2-9"


# ============================================================
# FEATURE IMPORTANCE EXTRACTION FROM PRE-IMAGES
# ============================================================

def extract_feature_importance_from_preimages(
    X_boundary: np.ndarray,
    preimages: np.ndarray,
) -> np.ndarray:
    """
    Extract per-feature importance from pre-image displacement.
    
    Importance of feature j = mean absolute displacement between
    boundary point and its pre-image in dimension j.
    
    Rationale: features that are ACTIVE on the decision boundary
    will show larger pre-image displacement (the boundary "lives"
    in those dimensions). Noise features will show near-zero displacement.
    """
    displacement = np.abs(preimages - X_boundary)
    importance = displacement.mean(axis=0)
    # Normalize to [0, 1]
    if importance.max() > 0:
        importance = importance / importance.max()
    return importance


def extract_feature_importance_from_gradient(
    X_boundary: np.ndarray,
    W_inv: np.ndarray,
    landmarks: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Extract per-feature importance via sensitivity analysis.
    
    For each boundary point, compute how much the pre-image CHANGES
    when each input feature is perturbed by a small amount.
    
    Importance of feature j = mean |∂x̂/∂x_j| across boundary points.
    
    This is more principled than displacement — it measures the
    boundary's SENSITIVITY to each feature.
    """
    eps = 1e-4
    d = X_boundary.shape[1]
    importances = np.zeros(d)
    
    for j in range(d):
        grads = []
        for x in X_boundary[:50]:  # subsample for speed
            # Perturb feature j
            x_plus = x.copy()
            x_plus[j] += eps
            x_minus = x.copy()
            x_minus[j] -= eps
            
            # Compute pre-images
            k_plus = rbf_kernel_vector(x_plus, landmarks, gamma)
            k_minus = rbf_kernel_vector(x_minus, landmarks, gamma)
            
            preimage_plus = compute_preimage(k_plus, W_inv, landmarks)
            preimage_minus = compute_preimage(k_minus, W_inv, landmarks)
            
            # Gradient magnitude
            grad = np.linalg.norm(preimage_plus - preimage_minus) / (2 * eps)
            grads.append(grad)
        
        importances[j] = np.mean(grads)
    
    # Normalize
    if importances.max() > 0:
        importances = importances / importances.max()
    return importances


# ============================================================
# EVALUATION METRICS
# ============================================================

def evaluate_attribution(importance: np.ndarray, ground_truth: np.ndarray, method_name: str):
    """
    Evaluate feature attribution quality against ground truth.
    
    Metrics:
    1. Spearman rank correlation with ground truth
    2. Precision@K: are the top-K attributed features actually relevant?
    3. Signal-to-noise ratio: mean(relevant importance) / mean(noise importance)
    """
    n_relevant = int(ground_truth.sum())
    d = len(ground_truth)
    
    # Spearman correlation
    rho, pvalue = spearmanr(importance, ground_truth)
    
    # Precision@K (K = number of truly relevant features)
    top_k_indices = np.argsort(importance)[-n_relevant:]
    relevant_indices = np.where(ground_truth > 0)[0]
    precision_at_k = len(set(top_k_indices) & set(relevant_indices)) / n_relevant
    
    # Signal-to-noise ratio
    relevant_importance = importance[ground_truth > 0].mean()
    noise_importance = importance[ground_truth == 0].mean() if (ground_truth == 0).any() else 0
    snr = relevant_importance / max(noise_importance, 1e-10)
    
    print(f"    {method_name:<20s} | ρ={rho:+.3f} (p={pvalue:.4f}) | P@{n_relevant}={precision_at_k:.2f} | SNR={snr:.2f}")
    
    return {"rho": rho, "precision_at_k": precision_at_k, "snr": snr}


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_single_dataset(X, y, ground_truth, description, dataset_name):
    """Run full attribution pipeline on one dataset."""
    
    print(f"\n{'─' * 70}")
    print(f"  Dataset {dataset_name}: {description}")
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}, Relevant: {int(ground_truth.sum())}")
    print(f"{'─' * 70}")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train RBF-SVM
    gamma = 1.0 / X_scaled.shape[1]  # standard scaling
    svm = SVC(kernel='rbf', gamma=gamma, C=10.0)
    svm.fit(X_scaled, y)
    acc = accuracy_score(y, svm.predict(X_scaled))
    print(f"  SVM accuracy: {acc:.3f} (γ={gamma:.4f})")
    
    # Find boundary points (closest to decision surface)
    decision_vals = svm.decision_function(X_scaled)
    boundary_indices = np.argsort(np.abs(decision_vals))[:200]
    X_boundary = X_scaled[boundary_indices]
    
    # Nyström setup
    m = min(300, len(X_scaled) - 200)
    landmarks, _ = nystrom_sample(X_scaled, m=m, method="kmeans")
    W = rbf_kernel_matrix(landmarks, gamma=gamma)
    
    # Add realistic noise
    noise = 1e-5 * np.random.randn(m, m)
    noise = (noise + noise.T) / 2
    W_noisy = W + noise
    
    # Compute inverses
    dsd_result = dsd_regularized_inverse(W_noisy)
    W_inv_dsd = dsd_result.pseudo_inverse
    W_inv_tik = tikhonov_inverse(W_noisy, gamma=1e-3)
    W_inv_naive = naive_pseudo_inverse(W_noisy)
    
    # Compute pre-images for each method
    methods = {
        "DSD": W_inv_dsd,
        "Tikhonov": W_inv_tik,
        "Naive": W_inv_naive,
    }
    
    print(f"\n  Feature Attribution via Pre-Image Sensitivity (gradient-based):")
    print(f"    {'Method':<20s} | {'Spearman ρ':<18s} | {'Precision@K':<11s} | {'SNR':<6s}")
    print(f"    {'─' * 65}")
    
    results = {}
    for method_name, W_inv in methods.items():
        importance = extract_feature_importance_from_gradient(
            X_boundary, W_inv, landmarks, gamma
        )
        results[method_name] = evaluate_attribution(importance, ground_truth, method_name)
    
    # Also show raw importance vectors for inspection
    print(f"\n  Feature importance vectors (top method = DSD):")
    importance_dsd = extract_feature_importance_from_gradient(X_boundary, W_inv_dsd, landmarks, gamma)
    print(f"    Ground truth:  {np.array2string(ground_truth, precision=2, separator=', ')}")
    print(f"    DSD attribution: {np.array2string(importance_dsd, precision=3, separator=', ')}")
    
    return results


def run_experiment():
    print("=" * 70)
    print("DSD-SVM Experiment 04: Ground-Truth Feature Attribution Validation")
    print("=" * 70)
    print()
    print("Test: Do DSD pre-image explanations correctly identify which features")
    print("      define the SVM decision boundary?")
    print()
    print("Method: Create datasets where TRUE important features are KNOWN.")
    print("        Extract feature importance from pre-image sensitivity.")
    print("        Compare against ground truth via Spearman ρ, Precision@K, SNR.")
    
    np.random.seed(42)
    
    datasets = [
        ("A", *make_dataset_A()),
        ("B", *make_dataset_B()),
        ("C", *make_dataset_C()),
    ]
    
    all_results = {}
    for name, X, y, gt, desc in datasets:
        all_results[name] = run_single_dataset(X, y, gt, desc, name)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Average Attribution Quality Across All Datasets")
    print("=" * 70)
    
    for method in ["DSD", "Tikhonov", "Naive"]:
        avg_rho = np.mean([all_results[ds][method]["rho"] for ds in all_results])
        avg_pak = np.mean([all_results[ds][method]["precision_at_k"] for ds in all_results])
        avg_snr = np.mean([all_results[ds][method]["snr"] for ds in all_results])
        print(f"  {method:<12s}: avg ρ={avg_rho:+.3f}, avg P@K={avg_pak:.2f}, avg SNR={avg_snr:.2f}")
    
    print("\n  Interpretation:")
    print("    ρ > 0.5: attribution ranking correlates with ground truth")
    print("    P@K = 1.0: ALL top-K attributed features are truly relevant")
    print("    SNR > 2.0: relevant features get 2x+ more importance than noise")
    print("\n✓ Experiment 04 complete.")


if __name__ == "__main__":
    run_experiment()
