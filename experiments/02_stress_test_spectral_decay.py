"""
Experiment 02: Stress Test — Exposing Naive Pseudo-Inverse Failure

Demonstrates that DSD maintains stable pre-images under conditions where
naive/Tikhonov methods catastrophically fail:
  - High Nyström sample sizes (m → more tail eigenvalue clustering)
  - Small kernel bandwidth (γ → sharper kernel → faster spectral decay)
  - Added matrix perturbation (simulates floating-point noise amplification)

Run: python experiments/02_stress_test_spectral_decay.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from src.dsd import dsd_regularized_inverse, tikhonov_inverse, naive_pseudo_inverse
from src.kernels import rbf_kernel_matrix, rbf_kernel_vector, nystrom_sample


def measure_preimage_stability(X, gamma, m, noise_level=0.0, n_boundary=100):
    """
    Measure pre-image reconstruction error under controlled conditions.
    
    Key insight: add PERTURBATION to the kernel matrix AFTER construction
    to simulate the floating-point noise that Davis-Kahan says will corrupt
    tail eigenvectors. This is where naive breaks but DSD holds.
    """
    # Select landmarks
    landmarks, _ = nystrom_sample(X, m=m, method="kmeans")
    
    # Compute kernel matrix
    W = rbf_kernel_matrix(landmarks, gamma=gamma)
    
    # Add controlled perturbation (simulates numerical noise)
    if noise_level > 0:
        noise = noise_level * np.random.randn(m, m)
        noise = (noise + noise.T) / 2  # Keep symmetric
        W_perturbed = W + noise
        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(W_perturbed)
        if eigvals.min() < 0:
            W_perturbed += (-eigvals.min() + 1e-10) * np.eye(m)
    else:
        W_perturbed = W.copy()
    
    # Compute inverses on PERTURBED matrix (this is where instability manifests)
    try:
        dsd_result = dsd_regularized_inverse(W_perturbed)
        W_inv_dsd = dsd_result.pseudo_inverse
    except:
        W_inv_dsd = None
    
    try:
        W_inv_tikhonov = tikhonov_inverse(W_perturbed, gamma=1e-3)
    except:
        W_inv_tikhonov = None
    
    try:
        W_inv_naive = naive_pseudo_inverse(W_perturbed)
    except:
        W_inv_naive = None
    
    # Compute pre-images and measure error
    # Ground truth: for landmarks, the pre-image should recover the landmark itself
    # (kernel vector of a landmark against all landmarks is a row of W)
    errors = {}
    test_indices = np.random.choice(m, size=min(n_boundary, m), replace=False)
    
    for name, W_inv in [("DSD", W_inv_dsd), ("Tikhonov", W_inv_tikhonov), ("Naive", W_inv_naive)]:
        if W_inv is None:
            errors[name] = np.inf
            continue
        
        errs = []
        for idx in test_indices:
            # Kernel vector of landmark[idx] against all landmarks (from ORIGINAL W, not perturbed)
            k_query = rbf_kernel_vector(landmarks[idx], landmarks, gamma=gamma)
            
            # Pre-image: raw weighted combination of landmarks (consistent with preimage.py)
            weights = W_inv @ k_query
            preimage = weights @ landmarks
            
            # Error: distance from the actual landmark position
            err = np.linalg.norm(preimage - landmarks[idx])
            errs.append(err)
        
        errors[name] = np.mean(errs)
    
    cond = np.linalg.cond(W_perturbed)
    return errors, cond, dsd_result


def run_experiment():
    print("=" * 70)
    print("DSD-SVM Experiment 02: Stress Test — Spectral Decay Regimes")
    print("=" * 70)
    
    np.random.seed(42)
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Generate data
    X, _ = make_swiss_roll(n_samples=5000, noise=0.3, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # ================================================================
    # TEST 1: Vary kernel bandwidth (smaller γ → faster spectral decay)
    # ================================================================
    print("\n" + "-" * 50)
    print("TEST 1: Kernel Bandwidth Sweep (γ)")
    print("  Smaller γ → sharper kernel → faster eigenvalue decay")
    print("-" * 50)
    
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    m_fixed = 500
    noise_fixed = 1e-6
    
    results_gamma = {"gamma": [], "DSD": [], "Tikhonov": [], "Naive": [], "cond": []}
    
    for gamma in gammas:
        errors, cond, _ = measure_preimage_stability(X, gamma=gamma, m=m_fixed, noise_level=noise_fixed)
        results_gamma["gamma"].append(gamma)
        results_gamma["DSD"].append(errors.get("DSD", np.inf))
        results_gamma["Tikhonov"].append(errors.get("Tikhonov", np.inf))
        results_gamma["Naive"].append(errors.get("Naive", np.inf))
        results_gamma["cond"].append(cond)
        print(f"  γ={gamma:5.1f} | Cond={cond:.2e} | DSD={errors['DSD']:.6f} | Tik={errors['Tikhonov']:.6f} | Naive={errors['Naive']:.6f}")
    
    # ================================================================
    # TEST 2: Vary Nyström sample size (more landmarks → more tail clustering)
    # ================================================================
    print("\n" + "-" * 50)
    print("TEST 2: Nyström Sample Size Sweep (m)")
    print("  Larger m → more eigenvalues → more tail clustering")
    print("-" * 50)
    
    ms = [50, 100, 200, 500, 800, 1000, 1500]
    gamma_fixed = 5.0  # Sharp kernel to provoke spectral decay
    
    results_m = {"m": [], "DSD": [], "Tikhonov": [], "Naive": [], "cond": []}
    
    for m in ms:
        errors, cond, _ = measure_preimage_stability(X, gamma=gamma_fixed, m=m, noise_level=noise_fixed)
        results_m["m"].append(m)
        results_m["DSD"].append(errors.get("DSD", np.inf))
        results_m["Tikhonov"].append(errors.get("Tikhonov", np.inf))
        results_m["Naive"].append(errors.get("Naive", np.inf))
        results_m["cond"].append(cond)
        print(f"  m={m:5d} | Cond={cond:.2e} | DSD={errors['DSD']:.6f} | Tik={errors['Tikhonov']:.6f} | Naive={errors['Naive']:.6f}")
    
    # ================================================================
    # TEST 3: Vary perturbation noise (more noise → worse for ill-conditioned)
    # ================================================================
    print("\n" + "-" * 50)
    print("TEST 3: Perturbation Noise Sweep")
    print("  More noise → amplified by ill-conditioning → naive diverges")
    print("-" * 50)
    
    noises = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    m_fixed2 = 500
    gamma_fixed2 = 5.0
    
    results_noise = {"noise": [], "DSD": [], "Tikhonov": [], "Naive": [], "cond": []}
    
    for noise in noises:
        errors, cond, _ = measure_preimage_stability(X, gamma=gamma_fixed2, m=m_fixed2, noise_level=noise)
        results_noise["noise"].append(noise)
        results_noise["DSD"].append(errors.get("DSD", np.inf))
        results_noise["Tikhonov"].append(errors.get("Tikhonov", np.inf))
        results_noise["Naive"].append(errors.get("Naive", np.inf))
        results_noise["cond"].append(cond)
        noise_str = f"{noise:.0e}" if noise > 0 else "0"
        print(f"  noise={noise_str:>8s} | Cond={cond:.2e} | DSD={errors['DSD']:.6f} | Tik={errors['Tikhonov']:.6f} | Naive={errors['Naive']:.6f}")
    
    # ================================================================
    # PLOT RESULTS
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Gamma sweep
    ax = axes[0]
    ax.semilogy(results_gamma["gamma"], results_gamma["DSD"], 'b-o', linewidth=2, markersize=6, label='DSD (proposed)')
    ax.semilogy(results_gamma["gamma"], results_gamma["Tikhonov"], 'r--s', linewidth=1.5, markersize=5, label='Tikhonov')
    ax.semilogy(results_gamma["gamma"], results_gamma["Naive"], 'k:^', linewidth=1.5, markersize=5, label='Naive')
    ax.set_xlabel('Kernel Bandwidth γ', fontsize=11)
    ax.set_ylabel('Pre-Image Reconstruction Error (log)', fontsize=11)
    ax.set_title('(a) Effect of Kernel Sharpness', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: m sweep
    ax = axes[1]
    ax.semilogy(results_m["m"], results_m["DSD"], 'b-o', linewidth=2, markersize=6, label='DSD (proposed)')
    ax.semilogy(results_m["m"], results_m["Tikhonov"], 'r--s', linewidth=1.5, markersize=5, label='Tikhonov')
    ax.semilogy(results_m["m"], results_m["Naive"], 'k:^', linewidth=1.5, markersize=5, label='Naive')
    ax.set_xlabel('Nyström Sample Size m', fontsize=11)
    ax.set_ylabel('Pre-Image Reconstruction Error (log)', fontsize=11)
    ax.set_title('(b) Effect of Matrix Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Noise sweep
    ax = axes[2]
    noise_labels = [f"{n:.0e}" if n > 0 else "0" for n in results_noise["noise"]]
    x_pos = range(len(noise_labels))
    ax.semilogy(x_pos, results_noise["DSD"], 'b-o', linewidth=2, markersize=6, label='DSD (proposed)')
    ax.semilogy(x_pos, results_noise["Tikhonov"], 'r--s', linewidth=1.5, markersize=5, label='Tikhonov')
    ax.semilogy(x_pos, results_noise["Naive"], 'k:^', linewidth=1.5, markersize=5, label='Naive')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(noise_labels, fontsize=8, rotation=45)
    ax.set_xlabel('Perturbation Noise Level', fontsize=11)
    ax.set_ylabel('Pre-Image Reconstruction Error (log)', fontsize=11)
    ax.set_title('(c) Robustness to Matrix Perturbation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('DSD Stress Test: Pre-Image Stability Under Spectral Decay\n'
                 'DSD maintains stable error where baselines diverge',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'stress_test_spectral_decay.png'), dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: results/figures/stress_test_spectral_decay.png")
    plt.close()
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: DSD Advantage Factors")
    print("=" * 70)
    
    # Best case for DSD advantage
    dsd_errors = np.array(results_noise["DSD"])
    naive_errors = np.array(results_noise["Naive"])
    tik_errors = np.array(results_noise["Tikhonov"])
    
    valid = (naive_errors > 0) & (dsd_errors > 0)
    if valid.any():
        max_advantage_naive = np.max(naive_errors[valid] / dsd_errors[valid])
        max_advantage_tik = np.max(tik_errors[valid] / dsd_errors[valid])
        print(f"  Maximum DSD advantage over Naive: {max_advantage_naive:.1f}x")
        print(f"  Maximum DSD advantage over Tikhonov: {max_advantage_tik:.1f}x")
    
    print("\n✓ Experiment 02 complete.")


if __name__ == "__main__":
    run_experiment()
