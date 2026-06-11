"""
Experiment 03: Sequential Fix Validation

Applies each fix one at a time and measures its individual impact,
proving that each correction addresses a real problem.

Run: python experiments/03_fix_validation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
from src.kernels import rbf_kernel_matrix, rbf_kernel_vector, nystrom_sample


# ============================================================
# ORIGINAL (BUGGY) IMPLEMENTATIONS — for comparison
# ============================================================

def compute_eigengaps_ORIGINAL(eigenvalues):
    """Original: boundary eigenvalues get inf gap."""
    m = len(eigenvalues)
    gaps = np.zeros(m)
    for i in range(m):
        left = abs(eigenvalues[i] - eigenvalues[i - 1]) if i > 0 else np.inf
        right = abs(eigenvalues[i] - eigenvalues[i + 1]) if i < m - 1 else np.inf
        gaps[i] = min(left, right)
    return gaps


def init_alpha_ORIGINAL(eigenvalues):
    """Original: α = median(λ²)."""
    return float(np.median(eigenvalues ** 2))


def preimage_ORIGINAL(weights, landmarks):
    """Original: normalize weights to sum=1 (convex hull constraint)."""
    weights_normalized = weights / (np.sum(weights) + 1e-10)
    return weights_normalized @ landmarks


# ============================================================
# FIXED IMPLEMENTATIONS
# ============================================================

def compute_eigengaps_FIXED(eigenvalues):
    """Fix #4: boundary eigenvalues use single-sided gap (no inf)."""
    m = len(eigenvalues)
    gaps = np.zeros(m)
    for i in range(m):
        left = abs(eigenvalues[i] - eigenvalues[i - 1]) if i > 0 else None
        right = abs(eigenvalues[i] - eigenvalues[i + 1]) if i < m - 1 else None
        
        if left is not None and right is not None:
            gaps[i] = min(left, right)
        elif left is not None:
            gaps[i] = left  # last eigenvalue: use left gap only
        elif right is not None:
            gaps[i] = right  # first eigenvalue: use right gap only
        else:
            gaps[i] = 0.0  # single eigenvalue edge case
    return gaps


def init_alpha_FIXED(eigenvalues):
    """Fix #3: α scaled relative to dominant eigenvalue."""
    return float(eigenvalues[-1] ** 2 * 0.01)


def preimage_FIXED(weights, landmarks):
    """Fix #1: raw weights, no normalization (allows extrapolation)."""
    return weights @ landmarks


def dsd_compute(eigenvalues, eigenvectors, alpha, beta, eigengap_fn):
    """Compute DSD pseudo-inverse with specified eigengap function."""
    gaps = eigengap_fn(eigenvalues)
    damping = alpha * np.exp(-beta * np.clip(gaps, 0, 500 / max(beta, 1e-10)))
    reg_inv = eigenvalues / (eigenvalues ** 2 + damping)
    W_inv = (eigenvectors * reg_inv) @ eigenvectors.T
    return W_inv


def tikhonov_via_eigh(eigenvalues, eigenvectors, gamma=1e-3):
    """Fix #6: Tikhonov via eigendecomposition for fair comparison."""
    reg_inv = 1.0 / (eigenvalues + gamma)
    return (eigenvectors * reg_inv) @ eigenvectors.T


# ============================================================
# TEST HARNESS
# ============================================================

def evaluate_preimage_quality(W_inv, landmarks, X_test, gamma, preimage_fn):
    """
    Fix #2: Evaluate on HELD-OUT test points (not landmarks).
    
    For a held-out point x_test, compute its kernel vector against landmarks,
    apply W_inv, then reconstruct. Compare reconstruction to actual x_test.
    """
    errors = []
    for x in X_test:
        k_query = rbf_kernel_vector(x, landmarks, gamma=gamma)
        weights = W_inv @ k_query
        x_hat = preimage_fn(weights, landmarks)
        errors.append(np.linalg.norm(x_hat - x))
    return np.mean(errors), np.std(errors)


def run_experiment():
    print("=" * 70)
    print("DSD-SVM Experiment 03: Sequential Fix Validation")
    print("=" * 70)
    
    # Setup
    np.random.seed(42)
    X_full, _ = make_swiss_roll(n_samples=5000, noise=0.3, random_state=42)
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)
    
    # Split: landmarks from first 4000, test from last 1000
    X_train = X_full[:4000]
    X_test = X_full[4000:4200]  # 200 held-out test points
    
    gamma = 2.0
    m = 500
    
    # Get landmarks
    landmarks, _ = nystrom_sample(X_train, m=m, method="kmeans")
    W = rbf_kernel_matrix(landmarks, gamma=gamma)
    
    # Add slight perturbation (realistic floating-point scenario)
    noise = 1e-7 * np.random.randn(m, m)
    noise = (noise + noise.T) / 2
    W_noisy = W + noise
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(W_noisy)
    pos_mask = eigenvalues > 1e-12
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]
    
    print(f"\nSetup: m={m}, γ={gamma}, test_points={len(X_test)}")
    print(f"Eigenvalue range: [{eigenvalues[0]:.2e}, {eigenvalues[-1]:.2e}]")
    print(f"Condition number: {eigenvalues[-1]/eigenvalues[0]:.2e}")
    
    beta_original = 1.0 / np.std(compute_eigengaps_ORIGINAL(eigenvalues)[:-1])
    beta_fixed = 1.0 / np.std(compute_eigengaps_FIXED(eigenvalues))
    
    # ================================================================
    # BASELINE: All original (buggy) implementations
    # ================================================================
    print("\n" + "=" * 70)
    print("BASELINE: All original implementations")
    print("=" * 70)
    
    alpha_orig = init_alpha_ORIGINAL(eigenvalues)
    W_inv_orig = dsd_compute(eigenvalues, eigenvectors, alpha_orig, beta_original, compute_eigengaps_ORIGINAL)
    mean_err, std_err = evaluate_preimage_quality(W_inv_orig, landmarks, X_test, gamma, preimage_ORIGINAL)
    print(f"  DSD (all original):     MSE = {mean_err:.6f} ± {std_err:.6f}")
    
    # Naive baseline
    W_inv_naive = np.linalg.pinv(W_noisy)
    mean_err_naive, std_err_naive = evaluate_preimage_quality(W_inv_naive, landmarks, X_test, gamma, preimage_ORIGINAL)
    print(f"  Naive (original preimg): MSE = {mean_err_naive:.6f} ± {std_err_naive:.6f}")
    
    # Tikhonov baseline
    W_inv_tik = tikhonov_via_eigh(eigenvalues, eigenvectors, gamma=1e-3)
    mean_err_tik, std_err_tik = evaluate_preimage_quality(W_inv_tik, landmarks, X_test, gamma, preimage_ORIGINAL)
    print(f"  Tikhonov (original):    MSE = {mean_err_tik:.6f} ± {std_err_tik:.6f}")
    
    # ================================================================
    # FIX 1: Remove weight normalization (allow extrapolation)
    # ================================================================
    print("\n" + "-" * 70)
    print("FIX 1: Remove convex hull constraint (raw weights)")
    print("  Expected: methods that produce good W_inv should improve dramatically")
    print("-" * 70)
    
    mean_err_f1, std_err_f1 = evaluate_preimage_quality(W_inv_orig, landmarks, X_test, gamma, preimage_FIXED)
    print(f"  DSD (fix1 only):        MSE = {mean_err_f1:.6f} ± {std_err_f1:.6f}")
    
    mean_err_naive_f1, _ = evaluate_preimage_quality(W_inv_naive, landmarks, X_test, gamma, preimage_FIXED)
    print(f"  Naive (fix1 only):      MSE = {mean_err_naive_f1:.6f}")
    
    mean_err_tik_f1, _ = evaluate_preimage_quality(W_inv_tik, landmarks, X_test, gamma, preimage_FIXED)
    print(f"  Tikhonov (fix1 only):   MSE = {mean_err_tik_f1:.6f}")
    
    # ================================================================
    # FIX 4: Boundary eigengap handling
    # ================================================================
    print("\n" + "-" * 70)
    print("FIX 4: Boundary eigengap — first eigenvalue uses right gap (no inf)")
    print("  Expected: more damping on smallest eigenvalue → better tail suppression")
    print("-" * 70)
    
    # Show the difference
    gaps_orig = compute_eigengaps_ORIGINAL(eigenvalues)
    gaps_fixed = compute_eigengaps_FIXED(eigenvalues)
    print(f"  First eigengap (original): {gaps_orig[0]:.2e} (inf → got 0 from loop)")
    print(f"  First eigengap (fixed):    {gaps_fixed[0]:.2e}")
    print(f"  Last eigengap (original):  {gaps_orig[-1]:.2e}")
    print(f"  Last eigengap (fixed):     {gaps_fixed[-1]:.2e}")
    
    W_inv_f4 = dsd_compute(eigenvalues, eigenvectors, alpha_orig, beta_fixed, compute_eigengaps_FIXED)
    mean_err_f4, _ = evaluate_preimage_quality(W_inv_f4, landmarks, X_test, gamma, preimage_FIXED)
    print(f"  DSD (fix1+fix4):        MSE = {mean_err_f4:.6f}")
    
    # ================================================================
    # FIX 3: α initialization — scale relative to dominant eigenvalue
    # ================================================================
    print("\n" + "-" * 70)
    print("FIX 3: α initialization — scale from dominant eigenvalue, not median")
    print(f"  α (original): {alpha_orig:.2e} (median λ²)")
    alpha_fixed = init_alpha_FIXED(eigenvalues)
    print(f"  α (fixed):    {alpha_fixed:.2e} (0.01 × λ_max²)")
    print("  Expected: stronger damping in tail → better stability")
    print("-" * 70)
    
    W_inv_f3 = dsd_compute(eigenvalues, eigenvectors, alpha_fixed, beta_fixed, compute_eigengaps_FIXED)
    mean_err_f3, std_err_f3 = evaluate_preimage_quality(W_inv_f3, landmarks, X_test, gamma, preimage_FIXED)
    print(f"  DSD (fix1+fix3+fix4):   MSE = {mean_err_f3:.6f} ± {std_err_f3:.6f}")
    
    # ================================================================
    # ALL FIXES COMBINED vs BASELINES
    # ================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: All fixes applied")
    print("=" * 70)
    
    # DSD with all fixes
    print(f"  DSD (all fixes):        MSE = {mean_err_f3:.6f} ± {std_err_f3:.6f}")
    
    # Naive with fixed preimage
    print(f"  Naive (fixed preimage):  MSE = {mean_err_naive_f1:.6f}")
    
    # Tikhonov with fixed preimage
    print(f"  Tikhonov (fixed preimg): MSE = {mean_err_tik_f1:.6f}")
    
    # Relative performance
    print(f"\n  DSD vs Naive:    {'DSD WINS' if mean_err_f3 < mean_err_naive_f1 else 'Naive wins'} ({mean_err_naive_f1/mean_err_f3:.2f}x)" if mean_err_f3 > 0 else "")
    print(f"  DSD vs Tikhonov: {'DSD WINS' if mean_err_f3 < mean_err_tik_f1 else 'Tikhonov wins'} ({mean_err_tik_f1/mean_err_f3:.2f}x)" if mean_err_f3 > 0 else "")
    
    # ================================================================
    # NOW STRESS IT: Higher noise where naive should break
    # ================================================================
    print("\n" + "=" * 70)
    print("STRESS TEST: Increasing perturbation noise (all fixes applied)")
    print("=" * 70)
    
    noise_levels = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 5e-3]
    
    print(f"{'Noise':<12} {'DSD':<14} {'Tikhonov':<14} {'Naive':<14} {'DSD/Naive':<10}")
    print("-" * 64)
    
    for nl in noise_levels:
        # Fresh noisy matrix
        if nl > 0:
            noise_mat = nl * np.random.randn(m, m)
            noise_mat = (noise_mat + noise_mat.T) / 2
            W_n = W + noise_mat
            ev, evec = np.linalg.eigh(W_n)
            pos = ev > 1e-12
            ev, evec = ev[pos], evec[:, pos]
        else:
            ev, evec = eigenvalues, eigenvectors
        
        # DSD (all fixes)
        alpha_f = init_alpha_FIXED(ev)
        beta_f = 1.0 / max(np.std(compute_eigengaps_FIXED(ev)), 1e-10)
        W_inv_dsd = dsd_compute(ev, evec, alpha_f, beta_f, compute_eigengaps_FIXED)
        dsd_err, _ = evaluate_preimage_quality(W_inv_dsd, landmarks, X_test, gamma, preimage_FIXED)
        
        # Tikhonov
        W_inv_t = tikhonov_via_eigh(ev, evec, gamma=1e-3)
        tik_err, _ = evaluate_preimage_quality(W_inv_t, landmarks, X_test, gamma, preimage_FIXED)
        
        # Naive
        W_temp = W + (nl * np.random.randn(m, m) if nl > 0 else np.zeros((m, m)))
        W_temp = (W_temp + W_temp.T) / 2
        W_inv_n = np.linalg.pinv(W_temp)
        naive_err, _ = evaluate_preimage_quality(W_inv_n, landmarks, X_test, gamma, preimage_FIXED)
        
        ratio = naive_err / dsd_err if dsd_err > 1e-10 else float('inf')
        nl_str = f"{nl:.0e}" if nl > 0 else "0"
        print(f"{nl_str:<12} {dsd_err:<14.6f} {tik_err:<14.6f} {naive_err:<14.6f} {ratio:<10.2f}")
    
    print("\n✓ Experiment 03 complete.")


if __name__ == "__main__":
    run_experiment()
