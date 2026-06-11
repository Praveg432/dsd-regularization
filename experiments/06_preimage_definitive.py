"""
Experiment 06: Definitive Pre-Image Comparison

DSD-optimized vs Tikhonov-optimized vs Naive
50 seeds, Swiss Roll, multiple noise levels.

Both DSD and Tikhonov get their best parameters:
  - DSD: gradient-optimized α, β on training data
  - Tikhonov: γ grid-searched on pre-image loss (same training data)
  - Naive: raw pinv (unstable reference)

Run: python experiments/06_preimage_definitive.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler

from src.dsd import dsd_regularized_inverse, tikhonov_inverse_optimized, naive_pseudo_inverse
from src.dsd_optimizer import DSDOptimizer
from src.dsd_torch import DSDModule
from src.kernels import rbf_kernel_matrix, rbf_kernel_vector, nystrom_sample
from src.preimage import compute_preimage


def ci_95(data):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf(0.975, n - 1)
    return mean - h, mean + h


def evaluate_preimage(W_inv, X_test, landmarks, gamma):
    errs = [np.linalg.norm(compute_preimage(rbf_kernel_vector(x, landmarks, gamma), W_inv, landmarks) - x)
            for x in X_test]
    return np.mean(errs)


def run_noise_level(noise_level, n_seeds=50):
    print(f"\n{'─' * 60}")
    print(f"  σ = {noise_level:.0e}")
    print(f"{'─' * 60}")

    dsd_errors = []
    tik_errors = []
    naive_errors = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        X, _ = make_swiss_roll(n_samples=2000, noise=0.3, random_state=seed)
        X = StandardScaler().fit_transform(X)
        X_train, X_test = X[:1500], X[1500:1550]

        landmarks, _ = nystrom_sample(X_train, m=300, method="kmeans")
        W = rbf_kernel_matrix(landmarks, gamma=2.0)
        noise_matrix = noise_level * np.random.randn(300, 300)
        noise_matrix = (noise_matrix + noise_matrix.T) / 2
        W_noisy = W + noise_matrix

        # DSD-optimized
        optimizer = DSDOptimizer(lr=0.01, n_epochs=150, patience=20, n_train_points=40)
        dsd_module, _ = optimizer.optimize(
            X_train=X_train[500:1000], landmarks=landmarks,
            gamma=2.0, noise_matrix=noise_matrix, verbose=False,
        )
        W_t = torch.tensor(W_noisy, dtype=torch.float64)
        with torch.no_grad():
            W_inv_dsd = dsd_module(W_t).pseudo_inverse.numpy()
        dsd_errors.append(evaluate_preimage(W_inv_dsd, X_test, landmarks, 2.0))

        # Tikhonov-optimized
        W_inv_tik = tikhonov_inverse_optimized(
            W_noisy, X_train[500:1000], landmarks, gamma_kernel=2.0
        )
        tik_errors.append(evaluate_preimage(W_inv_tik, X_test, landmarks, 2.0))

        # Naive
        W_inv_naive = naive_pseudo_inverse(W_noisy)
        naive_errors.append(evaluate_preimage(W_inv_naive, X_test, landmarks, 2.0))

        if (seed + 1) % 25 == 0:
            print(f"    Seed {seed+1}/50: DSD={dsd_errors[-1]:.4f}, "
                  f"Tik={tik_errors[-1]:.4f}, Naive={naive_errors[-1]:.4f}")

    dsd_errors = np.array(dsd_errors)
    tik_errors = np.array(tik_errors)
    naive_errors = np.array(naive_errors)

    # Report
    print(f"\n    {'Method':<22s} {'Mean':<10s} {'Std':<10s} {'95% CI'}")
    print(f"    {'─' * 55}")
    for name, arr in [("DSD-optimized", dsd_errors), ("Tikhonov-optimized", tik_errors), ("Naive", naive_errors)]:
        ci = ci_95(arr)
        print(f"    {name:<22s} {arr.mean():<10.5f} {arr.std():<10.5f} [{ci[0]:.5f}, {ci[1]:.5f}]")

    # Paired tests
    t1, p1 = stats.ttest_rel(tik_errors, dsd_errors)
    wins_tik = np.sum(dsd_errors < tik_errors)
    t2, p2 = stats.ttest_rel(naive_errors, dsd_errors)
    wins_naive = np.sum(dsd_errors < naive_errors)

    var_ratio_naive = naive_errors.std()**2 / max(dsd_errors.std()**2, 1e-15)

    print(f"\n    DSD vs Tikhonov-opt: t={t1:.3f}, p={p1:.4f} "
          f"{'✓ DSD wins' if p1 < 0.05 and dsd_errors.mean() < tik_errors.mean() else '✗'}, "
          f"DSD wins {wins_tik}/50")
    print(f"    DSD vs Naive:        t={t2:.3f}, p={p2:.4f} "
          f"{'✓ DSD wins' if p2 < 0.05 and dsd_errors.mean() < naive_errors.mean() else '✗'}, "
          f"DSD wins {wins_naive}/50")
    print(f"    Variance ratio (Naive/DSD): {var_ratio_naive:.1f}×")

    return dsd_errors, tik_errors, naive_errors


def run_experiment():
    print("=" * 60)
    print("Experiment 06: Definitive Pre-Image Comparison")
    print("=" * 60)
    print("\nDSD-optimized vs Tikhonov-optimized vs Naive")
    print("Both methods get access to same training data for tuning.")

    run_noise_level(5e-3, n_seeds=50)
    run_noise_level(1e-3, n_seeds=50)
    run_noise_level(1e-4, n_seeds=50)

    print("\n✓ Experiment 06 complete.")


if __name__ == "__main__":
    run_experiment()
