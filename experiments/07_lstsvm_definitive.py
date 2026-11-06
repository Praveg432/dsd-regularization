"""
Experiment 07: Definitive Linear LSTSVM Classification

DSD-optimized vs Tikhonov-optimized vs Naive
50 seeds, multiple dimensionalities and noise levels.

Tikhonov-optimized: γ swept via grid search on training accuracy.
DSD: uses auto-init (already near-optimal for LSTSVM product matrices).

Run: python experiments/07_lstsvm_definitive.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.lstsvm import LSTSVM


def ci_95(data):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf(0.975, n - 1)
    return mean - h, mean + h


def fit_tikhonov_optimized(X_tr, y_tr, X_te, y_te):
    """Tikhonov with γ grid-searched on training accuracy."""
    gammas = np.logspace(-8, -1, 15)
    best_gamma = 1e-7
    best_acc = 0
    for g in gammas:
        try:
            m = LSTSVM(c1=1.0, c2=1.0, inversion_method='tikhonov', tikhonov_gamma=g)
            m.fit(X_tr, y_tr)
            acc = m.score(X_tr, y_tr)
            if acc > best_acc:
                best_acc = acc
                best_gamma = g
        except Exception:
            continue
    model = LSTSVM(c1=1.0, c2=1.0, inversion_method='tikhonov', tikhonov_gamma=best_gamma)
    model.fit(X_tr, y_tr)
    return model.score(X_te, y_te)


def run_scenario(name, n_samples, n_features, n_informative, n_redundant,
                 noise_level, n_seeds=50):
    print(f"\n{'━' * 60}")
    print(f"  {name}")
    print(f"  n={n_samples}, d={n_features}, noise={noise_level}")
    print(f"{'━' * 60}")

    dsd_accs = []
    tik_accs = []
    naive_accs = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=n_informative, n_redundant=n_redundant,
            n_clusters_per_class=1, flip_y=0.03, random_state=seed,
        )
        X = StandardScaler().fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        if noise_level > 0:
            X_tr = X_tr + noise_level * np.random.randn(*X_tr.shape)

        # DSD
        try:
            m = LSTSVM(c1=1.0, c2=1.0, inversion_method='dsd')
            m.fit(X_tr, y_tr)
            dsd_accs.append(m.score(X_te, y_te))
        except Exception:
            dsd_accs.append(0.5)

        # Tikhonov-optimized
        try:
            tik_accs.append(fit_tikhonov_optimized(X_tr, y_tr, X_te, y_te))
        except Exception:
            tik_accs.append(0.5)

        # Naive
        try:
            m = LSTSVM(c1=1.0, c2=1.0, inversion_method='naive')
            m.fit(X_tr, y_tr)
            naive_accs.append(m.score(X_te, y_te))
        except Exception:
            naive_accs.append(0.5)

    dsd_accs = np.array(dsd_accs)
    tik_accs = np.array(tik_accs)
    naive_accs = np.array(naive_accs)

    print(f"\n    {'Method':<22s} {'Mean':<8s} {'Std':<8s} {'95% CI'}")
    print(f"    {'─' * 50}")
    for label, arr in [("DSD", dsd_accs), ("Tikhonov-opt", tik_accs), ("Naive", naive_accs)]:
        ci = ci_95(arr)
        print(f"    {label:<22s} {arr.mean():<8.4f} {arr.std():<8.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    t, p = stats.ttest_rel(dsd_accs, tik_accs)
    wins = np.sum(dsd_accs > tik_accs)
    print(f"\n    DSD vs Tik-opt: diff={dsd_accs.mean()-tik_accs.mean():+.4f}, "
          f"t={t:.3f}, p={p:.4f} "
          f"{'✓' if p < 0.05 and dsd_accs.mean() > tik_accs.mean() else '✗'}, "
          f"wins={wins}/50")


def run_experiment():
    print("=" * 60)
    print("Experiment 07: Linear LSTSVM — DSD vs Tikhonov-optimized")
    print("=" * 60)

    run_scenario("d=100, n=200 (DSD target regime)", 200, 100, 10, 20, 0.1)
    run_scenario("d=200, n=300 (high-dim stress)", 300, 200, 15, 30, 0.1)
    run_scenario("d=50, n=300 (moderate)", 300, 50, 5, 15, 0.1)
    run_scenario("d=30, n=400 (low-dim check)", 400, 30, 8, 10, 0.1)

    print("\n✓ Experiment 07 complete.")


if __name__ == "__main__":
    run_experiment()
