"""
Experiment 08: Definitive Kernel LSTSVM (Non-Linear)

DSD-optimized vs Tikhonov-optimized on actual RBF kernel matrices.
DSD's strongest regime: exponential spectral decay.

Both methods get equal optimization:
  - DSD: α, β gradient-optimized on class kernel pre-image loss
  - Tikhonov: γ grid-searched on training accuracy (20-point log grid)

50 seeds, paired testing.

Run: python experiments/08_kernel_lstsvm_definitive.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.kernel_lstsvm import KernelLSTSVM
from src.dsd import dsd_regularized_inverse
from src.dsd_optimizer import DSDOptimizer


def ci_95(data):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf(0.975, n - 1)
    return mean - h, mean + h


def fit_dsd_optimized(X_tr, y_tr, X_te, y_te, gamma_kernel, seed):
    """Kernel-LSTSVM with gradient-optimized DSD."""
    torch.manual_seed(seed)
    classes = np.unique(y_tr)
    A = X_tr[y_tr == classes[0]]
    B = X_tr[y_tr == classes[1]]

    # Optimize DSD on larger class kernel
    opt_data = A if len(A) >= len(B) else B
    optimizer = DSDOptimizer(lr=0.01, n_epochs=100, patience=15,
                            n_train_points=min(30, len(opt_data)))
    dsd_module, opt_result = optimizer.optimize(
        X_train=opt_data, landmarks=opt_data, gamma=gamma_kernel, verbose=False,
    )
    alpha_opt = opt_result.alpha_final
    beta_opt = opt_result.beta_final

    # Fit kernel LSTSVM with optimized parameters
    m1, m2 = len(A), len(B)
    def rbf(X, Y):
        return np.exp(-gamma_kernel * cdist(X, Y, 'sqeuclidean'))

    K11, K22, K12 = rbf(A, A), rbf(B, B), rbf(A, B)
    K21 = K12.T

    M1 = (K11 + K12 @ K21); M1 = (M1 + M1.T) / 2
    alpha1 = dsd_regularized_inverse(M1, alpha=alpha_opt, beta=beta_opt).pseudo_inverse @ (-(K12 @ np.ones(m2)))
    b1 = -np.mean(K11 @ alpha1)

    M2 = (K22 + K21 @ K12); M2 = (M2 + M2.T) / 2
    alpha2 = dsd_regularized_inverse(M2, alpha=alpha_opt, beta=beta_opt).pseudo_inverse @ (K21 @ np.ones(m1))
    b2 = -np.mean(K22 @ alpha2)

    # Predict
    K_test_A, K_test_B = rbf(X_te, A), rbf(X_te, B)
    dist1 = np.abs(K_test_A @ alpha1 + b1) / (np.linalg.norm(alpha1) + 1e-10)
    dist2 = np.abs(K_test_B @ alpha2 + b2) / (np.linalg.norm(alpha2) + 1e-10)
    preds = np.where(dist1 < dist2, classes[0], classes[1])
    return float(np.mean(preds == y_te))


def fit_tikhonov_optimized(X_tr, y_tr, X_te, y_te, gamma_kernel):
    """Kernel-LSTSVM with γ grid-searched on training accuracy."""
    gammas = np.logspace(-6, 0, 20)
    best_gamma, best_acc = 1e-3, 0
    for g in gammas:
        try:
            m = KernelLSTSVM(c1=1.0, c2=1.0, gamma=gamma_kernel,
                           inversion_method='tikhonov', tikhonov_gamma=g)
            m.fit(X_tr, y_tr)
            acc = m.score(X_tr, y_tr)
            if acc > best_acc:
                best_acc, best_gamma = acc, g
        except Exception:
            continue
    m = KernelLSTSVM(c1=1.0, c2=1.0, gamma=gamma_kernel,
                    inversion_method='tikhonov', tikhonov_gamma=best_gamma)
    m.fit(X_tr, y_tr)
    return m.score(X_te, y_te)


def run_scenario(name, X, y, gamma, n_seeds=50, noise=0.05):
    print(f"\n{'━' * 60}")
    print(f"  {name}")
    print(f"  n={len(X)}, d={X.shape[1]}, γ={gamma:.4f}, noise={noise}")
    print(f"{'━' * 60}")

    dsd_accs, tik_accs = [], []

    for seed in range(n_seeds):
        np.random.seed(seed)
        X_scaled = StandardScaler().fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y, test_size=0.3, random_state=seed, stratify=y)
        if noise > 0:
            X_tr = X_tr + noise * np.random.randn(*X_tr.shape)

        try:
            dsd_accs.append(fit_dsd_optimized(X_tr, y_tr, X_te, y_te, gamma, seed))
        except Exception:
            dsd_accs.append(0.5)

        try:
            tik_accs.append(fit_tikhonov_optimized(X_tr, y_tr, X_te, y_te, gamma))
        except Exception:
            tik_accs.append(0.5)

        if (seed + 1) % 25 == 0:
            print(f"    Seed {seed+1}: DSD={dsd_accs[-1]:.3f}, Tik={tik_accs[-1]:.3f}")

    dsd_accs = np.array(dsd_accs)
    tik_accs = np.array(tik_accs)

    t, p = stats.ttest_rel(dsd_accs, tik_accs)
    wins = np.sum(dsd_accs > tik_accs)

    print(f"\n    DSD-opt:     {dsd_accs.mean():.4f} ± {dsd_accs.std():.4f}  {ci_95(dsd_accs)}")
    print(f"    Tik-opt:     {tik_accs.mean():.4f} ± {tik_accs.std():.4f}  {ci_95(tik_accs)}")
    print(f"    Diff:        {dsd_accs.mean()-tik_accs.mean():+.4f}")
    print(f"    Paired test: t={t:.3f}, p={p:.4f} "
          f"{'✓ DSD wins' if p < 0.05 and dsd_accs.mean() > tik_accs.mean() else '✗'}")
    print(f"    Wins:        {wins}/50")


def run_experiment():
    print("=" * 60)
    print("Experiment 08: Kernel LSTSVM — DSD-opt vs Tikhonov-opt")
    print("=" * 60)
    print("\nNon-linear classification with RBF kernel matrices.")
    print("Both methods optimized fairly. 50 seeds.")

    X, y = make_moons(n_samples=400, noise=0.15, random_state=42)
    run_scenario("Two Moons (d=2, non-linear)", X, y, gamma=2.0, noise=0.05)

    X, y = make_classification(n_samples=300, n_features=50, n_informative=8,
                               n_redundant=15, n_clusters_per_class=2, random_state=42)
    run_scenario("High-dim (d=50, clustered)", X, y, gamma=0.03, noise=0.1)

    X, y = make_classification(n_samples=200, n_features=100, n_informative=10,
                               n_redundant=20, n_clusters_per_class=1,
                               flip_y=0.03, random_state=42)
    run_scenario("Genomics-like (d=100, n=200)", X, y, gamma=0.02, noise=0.1)

    print("\n✓ Experiment 08 complete.")


if __name__ == "__main__":
    run_experiment()
