"""
Experiment 09: Real-World Dataset Validation (Boundary Characterization)

NOTE: For the paper's primary real-world result (Madelon d=500), see
experiment 10_extended_validation.py. This experiment characterizes the
operating boundary at d≤34 where DSD is NOT expected to outperform.

DSD-optimized vs Tikhonov-optimized on real-world datasets.
Both methods fully optimized. 50 seeds. Pre-image and LSTSVM.

Datasets:
  - German Credit (d=20, n=1000)
  - Breast Cancer (d=30, n=569)
  - Ionosphere (d=34, n=351)

Run: python experiments/09_real_world_definitive.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from src.dsd import tikhonov_inverse_optimized, naive_pseudo_inverse
from src.dsd_optimizer import DSDOptimizer
from src.dsd_torch import DSDModule
from src.kernels import rbf_kernel_matrix, rbf_kernel_vector, nystrom_sample
from src.preimage import compute_preimage
from src.lstsvm import LSTSVM
from src.kernel_lstsvm import KernelLSTSVM


def ci_95(data):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf(0.975, n - 1)
    return mean - h, mean + h


def load_german_credit():
    try:
        data = fetch_openml(name='credit-g', version=1, as_frame=False, parser='auto')
        X = data.data
        if X.dtype == object:
            X_enc = np.zeros_like(X, dtype=float)
            for col in range(X.shape[1]):
                try:
                    X_enc[:, col] = X[:, col].astype(float)
                except (ValueError, TypeError):
                    X_enc[:, col] = LabelEncoder().fit_transform(X[:, col].astype(str))
            X = X_enc
        y = (data.target == 'good').astype(int)
        return X, y, "German Credit (d=20, n=1000)"
    except Exception:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=8,
                                   n_redundant=4, weights=[0.7, 0.3], random_state=42)
        return X, y, "German Credit Proxy (d=20, n=1000)"


def load_ionosphere():
    try:
        data = fetch_openml(name='ionosphere', version=1, as_frame=False, parser='auto')
        X = data.data.astype(float)
        y = (data.target == 'g').astype(int)
        return X, y, "Ionosphere (d=34, n=351)"
    except Exception:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=351, n_features=34, n_informative=12,
                                   n_redundant=8, random_state=42)
        return X, y, "Ionosphere Proxy (d=34, n=351)"


def run_preimage_test(X, y, name, n_seeds=50, noise_level=5e-3):
    """Pre-image stability: DSD-opt vs Tikhonov-opt."""
    d = X.shape[1]
    m = min(200, len(X) // 4)
    gamma = 1.0 / d

    print(f"\n  Pre-Image (m={m}, γ={gamma:.4f}, σ={noise_level:.0e}):")

    dsd_errors, tik_errors = [], []
    for seed in range(n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=seed)
        X_test = X_test[:50]

        landmarks, _ = nystrom_sample(X_train, m=m, method="kmeans")
        W = rbf_kernel_matrix(landmarks, gamma=gamma)
        noise = noise_level * np.random.randn(m, m)
        noise = (noise + noise.T) / 2
        W_noisy = W + noise

        # DSD-optimized
        opt = DSDOptimizer(lr=0.01, n_epochs=100, patience=15, n_train_points=min(30, len(X_train)//4))
        dsd_mod, _ = opt.optimize(X_train[:len(X_train)//2], landmarks, gamma, noise, verbose=False)
        with torch.no_grad():
            W_inv_dsd = dsd_mod(torch.tensor(W_noisy, dtype=torch.float64)).pseudo_inverse.numpy()
        errs = [np.linalg.norm(compute_preimage(rbf_kernel_vector(x, landmarks, gamma), W_inv_dsd, landmarks) - x)
                for x in X_test]
        dsd_errors.append(np.mean(errs))

        # Tikhonov-optimized
        W_inv_tik = tikhonov_inverse_optimized(W_noisy, X_train[:len(X_train)//2], landmarks, gamma)
        errs = [np.linalg.norm(compute_preimage(rbf_kernel_vector(x, landmarks, gamma), W_inv_tik, landmarks) - x)
                for x in X_test]
        tik_errors.append(np.mean(errs))

    dsd_errors = np.array(dsd_errors)
    tik_errors = np.array(tik_errors)
    t, p = stats.ttest_rel(tik_errors, dsd_errors)
    wins = np.sum(dsd_errors < tik_errors)

    print(f"    DSD-opt: {dsd_errors.mean():.4f} ± {dsd_errors.std():.4f}")
    print(f"    Tik-opt: {tik_errors.mean():.4f} ± {tik_errors.std():.4f}")
    print(f"    t={t:.3f}, p={p:.4f} {'✓ DSD wins' if p<0.05 and dsd_errors.mean()<tik_errors.mean() else '✗'}, "
          f"DSD wins {wins}/50")


def run_lstsvm_test(X, y, name, n_seeds=50, noise=0.1):
    """LSTSVM classification: DSD vs Tikhonov-opt."""
    print(f"\n  Kernel-LSTSVM (noise={noise}):")
    gamma = 1.0 / X.shape[1]

    dsd_accs, tik_accs = [], []
    for seed in range(n_seeds):
        np.random.seed(seed)
        X_scaled = StandardScaler().fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.3, random_state=seed, stratify=y)
        if noise > 0:
            X_tr = X_tr + noise * np.random.randn(*X_tr.shape)

        # DSD
        try:
            m = KernelLSTSVM(c1=1.0, c2=1.0, gamma=gamma, inversion_method='dsd')
            m.fit(X_tr, y_tr)
            dsd_accs.append(m.score(X_te, y_te))
        except Exception:
            dsd_accs.append(0.5)

        # Tikhonov-opt
        gammas = np.logspace(-5, 0, 15)
        best_g, best_a = 1e-3, 0
        for g in gammas:
            try:
                m = KernelLSTSVM(c1=1.0, c2=1.0, gamma=gamma, inversion_method='tikhonov', tikhonov_gamma=g)
                m.fit(X_tr, y_tr)
                a = m.score(X_tr, y_tr)
                if a > best_a: best_a, best_g = a, g
            except Exception:
                continue
        m = KernelLSTSVM(c1=1.0, c2=1.0, gamma=gamma, inversion_method='tikhonov', tikhonov_gamma=best_g)
        m.fit(X_tr, y_tr)
        tik_accs.append(m.score(X_te, y_te))

    dsd_accs = np.array(dsd_accs)
    tik_accs = np.array(tik_accs)
    t, p = stats.ttest_rel(dsd_accs, tik_accs)
    wins = np.sum(dsd_accs > tik_accs)

    print(f"    DSD:     {dsd_accs.mean():.4f} ± {dsd_accs.std():.4f}")
    print(f"    Tik-opt: {tik_accs.mean():.4f} ± {tik_accs.std():.4f}")
    print(f"    Diff={dsd_accs.mean()-tik_accs.mean():+.4f}, t={t:.3f}, p={p:.4f} "
          f"{'✓' if p<0.05 and dsd_accs.mean()>tik_accs.mean() else '✗'}, wins={wins}/50")


def run_experiment():
    print("=" * 60)
    print("Experiment 09: Real-World Definitive Validation")
    print("=" * 60)
    print("\nDSD-optimized vs Tikhonov-optimized on real datasets.")

    datasets = [
        load_german_credit(),
        (load_breast_cancer().data, load_breast_cancer().target, "Breast Cancer (d=30, n=569)"),
        load_ionosphere(),
    ]

    for X, y, name in datasets:
        print(f"\n{'━' * 60}")
        print(f"  {name}")
        print(f"{'━' * 60}")
        run_preimage_test(X, y, name)
        run_lstsvm_test(X, y, name)

    print("\n✓ Experiment 09 complete.")


if __name__ == "__main__":
    run_experiment()
