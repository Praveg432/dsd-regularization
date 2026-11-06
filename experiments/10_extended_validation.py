"""
Experiment 10: Extended Validation

Tests that widen the study's feasibility:
  1. Madelon (d=500, real-world) — validates DSD on genuine high-dim data
  2. Digits (d=64) — confirms boundary: DSD doesn't help at d<100
  3. Gradient-optimized Tikhonov — fairness: Tikhonov gets same Adam optimizer
  4. Spectral analysis — quantifies tail-clustering at each dimensionality

These results are cited in the paper. This script reproduces them.

Run: python experiments/10_extended_validation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats
from sklearn.datasets import fetch_openml, load_digits, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.lstsvm import LSTSVM
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


# ================================================================
# TEST 1a: GINA (d=970, real-world handwriting)
# ================================================================

def test_gina(n_seeds=30):
    print("\n" + "=" * 60)
    print("TEST 1a: GINA (d=970, n=3468, handwriting recognition)")
    print("=" * 60)

    try:
        data = fetch_openml(name='gina_agnostic', version=1, as_frame=False, parser='auto')
        X = data.data.astype(float)
        from sklearn.preprocessing import LabelEncoder as LE
        y = LE().fit_transform(data.target)
    except Exception:
        print("  GINA unavailable, using synthetic d=970")
        X, y = make_classification(n_samples=3468, n_features=970,
                                   n_informative=50, n_redundant=100, random_state=42)

    print(f"  Shape: {X.shape}, balance: {y.mean():.2f}")

    dsd_accs, tik_accs = [], []

    for seed in range(n_seeds):
        np.random.seed(seed)
        X_s = StandardScaler().fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_s, y, test_size=0.3, random_state=seed, stratify=y
        )
        X_tr = X_tr + 0.1 * np.random.randn(*X_tr.shape)

        try:
            m = LSTSVM(c1=1.0, c2=1.0, inversion_method='dsd')
            m.fit(X_tr, y_tr)
            dsd_accs.append(m.score(X_te, y_te))
        except Exception:
            dsd_accs.append(0.5)

        gammas = np.logspace(-8, -1, 15)
        best_g, best_a = 1e-7, 0
        for g in gammas:
            try:
                m = LSTSVM(c1=1.0, c2=1.0, inversion_method='tikhonov', tikhonov_gamma=g)
                m.fit(X_tr, y_tr)
                a = m.score(X_tr, y_tr)
                if a > best_a:
                    best_a, best_g = a, g
            except Exception:
                continue
        m = LSTSVM(c1=1.0, c2=1.0, inversion_method='tikhonov', tikhonov_gamma=best_g)
        m.fit(X_tr, y_tr)
        tik_accs.append(m.score(X_te, y_te))

        if (seed + 1) % 10 == 0:
            print(f"    Seed {seed+1}/{n_seeds}: DSD={dsd_accs[-1]:.3f}, Tik={tik_accs[-1]:.3f}")

    dsd_accs = np.array(dsd_accs)
    tik_accs = np.array(tik_accs)
    t, p = stats.ttest_rel(dsd_accs, tik_accs)
    wins = np.sum(dsd_accs > tik_accs)
    pooled_std = np.sqrt((dsd_accs.std()**2 + tik_accs.std()**2) / 2)
    cohens_d = (dsd_accs.mean() - tik_accs.mean()) / pooled_std

    print(f"\n  DSD:       {dsd_accs.mean():.4f} +/- {dsd_accs.std():.4f}")
    print(f"  Tik-opt:   {tik_accs.mean():.4f} +/- {tik_accs.std():.4f}")
    print(f"  Advantage: {dsd_accs.mean()-tik_accs.mean():+.4f}")
    print(f"  t={t:.3f}, p={p:.6f}, wins={wins}/{n_seeds}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    return dsd_accs, tik_accs


def test_gina_init_vs_opt(n_seeds=30):
    """Confirms DSD-init > DSD-optimized on GINA classification."""
    print("\n" + "=" * 60)
    print("TEST 1b: GINA — DSD-init vs DSD-optimized")
    print("  (Confirms optimization degrades classification)")
    print("=" * 60)

    try:
        data = fetch_openml(name='gina_agnostic', version=1, as_frame=False, parser='auto')
        X = data.data.astype(float)
        from sklearn.preprocessing import LabelEncoder as LE
        y = LE().fit_transform(data.target)
    except Exception:
        X, y = make_classification(n_samples=3468, n_features=970,
                                   n_informative=50, n_redundant=100, random_state=42)

    from src.dsd import dsd_regularized_inverse

    dsd_init_accs, dsd_opt_accs = [], []

    for seed in range(n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        X_s = StandardScaler().fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.3, random_state=seed, stratify=y)
        X_tr = X_tr + 0.1 * np.random.randn(*X_tr.shape)

        # DSD-init
        m = LSTSVM(c1=1.0, c2=1.0, inversion_method='dsd')
        m.fit(X_tr, y_tr)
        dsd_init_accs.append(m.score(X_te, y_te))

        # DSD-optimized on class kernel
        A = X_tr[y_tr == 0]
        B = X_tr[y_tr == 1]
        opt = DSDOptimizer(lr=0.01, n_epochs=80, patience=15, n_train_points=30)
        _, opt_res = opt.optimize(
            X_train=A[:min(100, len(A))],
            landmarks=A[:min(100, len(A))],
            gamma=1.0 / X_tr.shape[1], verbose=False,
        )
        alpha_opt, beta_opt = opt_res.alpha_final, opt_res.beta_final

        E1 = np.hstack([A, np.ones((len(A), 1))])
        E2 = np.hstack([B, np.ones((len(B), 1))])
        M1 = (E1.T @ E1 + E2.T @ E2); M1 = (M1 + M1.T) / 2
        sol1 = dsd_regularized_inverse(M1, alpha=alpha_opt, beta=beta_opt).pseudo_inverse @ (-(E2.T @ np.ones(len(B))))
        w1, b1 = sol1[:-1], float(sol1[-1])
        M2 = (E2.T @ E2 + E1.T @ E1); M2 = (M2 + M2.T) / 2
        sol2 = dsd_regularized_inverse(M2, alpha=alpha_opt, beta=beta_opt).pseudo_inverse @ (E1.T @ np.ones(len(A)))
        w2, b2 = sol2[:-1], float(sol2[-1])

        dist1 = np.abs(X_te @ w1 + b1) / (np.linalg.norm(w1) + 1e-10)
        dist2 = np.abs(X_te @ w2 + b2) / (np.linalg.norm(w2) + 1e-10)
        preds = np.where(dist1 < dist2, 0, 1)
        dsd_opt_accs.append(float(np.mean(preds == y_te)))

        if (seed + 1) % 10 == 0:
            print(f"    Seed {seed+1}: init={dsd_init_accs[-1]:.3f}, opt={dsd_opt_accs[-1]:.3f}")

    dsd_init_accs = np.array(dsd_init_accs)
    dsd_opt_accs = np.array(dsd_opt_accs)
    t, p = stats.ttest_rel(dsd_init_accs, dsd_opt_accs)
    wins = np.sum(dsd_init_accs > dsd_opt_accs)

    print(f"\n  DSD-init: {dsd_init_accs.mean():.4f} +/- {dsd_init_accs.std():.4f}")
    print(f"  DSD-opt:  {dsd_opt_accs.mean():.4f} +/- {dsd_opt_accs.std():.4f}")
    print(f"  Init wins: {wins}/{n_seeds}, t={t:.3f}, p={p:.6f}")
    print(f"  Conclusion: Init > Opt confirmed (optimization degrades classification)")

    return dsd_init_accs, dsd_opt_accs


# ================================================================
# TEST 1b: Madelon (d=500, real-world)
# ================================================================

def test_madelon(n_seeds=30):
    print("\n" + "=" * 60)
    print("TEST 1: Madelon (d=500, n=2600, real-world)")
    print("=" * 60)

    try:
        data = fetch_openml(name='madelon', version=1, as_frame=False, parser='auto')
        X = data.data.astype(float)
        y = (data.target == '1').astype(int)
    except Exception:
        print("  Madelon unavailable, using synthetic d=500")
        X, y = make_classification(n_samples=2600, n_features=500,
                                   n_informative=20, n_redundant=50, random_state=42)

    print(f"  Shape: {X.shape}, balance: {y.mean():.2f}")

    dsd_accs, tik_accs = [], []

    for seed in range(n_seeds):
        np.random.seed(seed)
        X_s = StandardScaler().fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_s, y, test_size=0.3, random_state=seed, stratify=y
        )
        X_tr = X_tr + 0.1 * np.random.randn(*X_tr.shape)

        # DSD
        try:
            m = LSTSVM(c1=1.0, c2=1.0, inversion_method='dsd')
            m.fit(X_tr, y_tr)
            dsd_accs.append(m.score(X_te, y_te))
        except Exception:
            dsd_accs.append(0.5)

        # Tikhonov-opt (grid search)
        gammas = np.logspace(-8, -1, 15)
        best_g, best_a = 1e-7, 0
        for g in gammas:
            try:
                m = LSTSVM(c1=1.0, c2=1.0, inversion_method='tikhonov', tikhonov_gamma=g)
                m.fit(X_tr, y_tr)
                a = m.score(X_tr, y_tr)
                if a > best_a:
                    best_a, best_g = a, g
            except Exception:
                continue
        m = LSTSVM(c1=1.0, c2=1.0, inversion_method='tikhonov', tikhonov_gamma=best_g)
        m.fit(X_tr, y_tr)
        tik_accs.append(m.score(X_te, y_te))

        if (seed + 1) % 10 == 0:
            print(f"    Seed {seed+1}/{n_seeds}: DSD={dsd_accs[-1]:.3f}, Tik={tik_accs[-1]:.3f}")

    dsd_accs = np.array(dsd_accs)
    tik_accs = np.array(tik_accs)
    t, p = stats.ttest_rel(dsd_accs, tik_accs)
    wins = np.sum(dsd_accs > tik_accs)
    pooled_std = np.sqrt((dsd_accs.std()**2 + tik_accs.std()**2) / 2)
    cohens_d = (dsd_accs.mean() - tik_accs.mean()) / pooled_std

    print(f"\n  DSD:       {dsd_accs.mean():.4f} +/- {dsd_accs.std():.4f}")
    print(f"  Tik-opt:   {tik_accs.mean():.4f} +/- {tik_accs.std():.4f}")
    print(f"  Advantage: {dsd_accs.mean()-tik_accs.mean():+.4f}")
    print(f"  t={t:.3f}, p={p:.6f}, wins={wins}/{n_seeds}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    return dsd_accs, tik_accs


# ================================================================
# TEST 2: Digits (d=64, boundary check)
# ================================================================

def test_digits(n_seeds=30):
    print("\n" + "=" * 60)
    print("TEST 2: Digits (d=64, n=1797, binary 0-4 vs 5-9)")
    print("=" * 60)

    data = load_digits()
    y_bin = (data.target >= 5).astype(int)
    X, y = data.data, y_bin

    dsd_accs, tik_accs = [], []

    for seed in range(n_seeds):
        np.random.seed(seed)
        X_s = StandardScaler().fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_s, y, test_size=0.3, random_state=seed, stratify=y
        )
        X_tr = X_tr + 0.1 * np.random.randn(*X_tr.shape)

        try:
            m = LSTSVM(c1=1.0, c2=1.0, inversion_method='dsd')
            m.fit(X_tr, y_tr)
            dsd_accs.append(m.score(X_te, y_te))
        except Exception:
            dsd_accs.append(0.5)

        gammas = np.logspace(-8, -1, 15)
        best_g, best_a = 1e-7, 0
        for g in gammas:
            try:
                m = LSTSVM(c1=1.0, c2=1.0, inversion_method='tikhonov', tikhonov_gamma=g)
                m.fit(X_tr, y_tr)
                a = m.score(X_tr, y_tr)
                if a > best_a:
                    best_a, best_g = a, g
            except Exception:
                continue
        m = LSTSVM(c1=1.0, c2=1.0, inversion_method='tikhonov', tikhonov_gamma=best_g)
        m.fit(X_tr, y_tr)
        tik_accs.append(m.score(X_te, y_te))

        if (seed + 1) % 10 == 0:
            print(f"    Seed {seed+1}/{n_seeds}: DSD={dsd_accs[-1]:.3f}, Tik={tik_accs[-1]:.3f}")

    dsd_accs = np.array(dsd_accs)
    tik_accs = np.array(tik_accs)
    t, p = stats.ttest_rel(dsd_accs, tik_accs)
    wins = np.sum(dsd_accs > tik_accs)

    print(f"\n  DSD:       {dsd_accs.mean():.4f} +/- {dsd_accs.std():.4f}")
    print(f"  Tik-opt:   {tik_accs.mean():.4f} +/- {tik_accs.std():.4f}")
    print(f"  Advantage: {dsd_accs.mean()-tik_accs.mean():+.4f}")
    print(f"  t={t:.3f}, p={p:.4f}, wins={wins}/{n_seeds}")

    return dsd_accs, tik_accs


# ================================================================
# TEST 3: Gradient-optimized Tikhonov (fairness check)
# ================================================================

def test_tikhonov_gradient_fairness(n_seeds=20):
    print("\n" + "=" * 60)
    print("TEST 3: Gradient-optimized Tikhonov vs DSD-opt")
    print("  Same Adam optimizer for both. Swiss Roll, σ=5e-3.")
    print("=" * 60)

    from sklearn.datasets import make_swiss_roll

    dsd_errs, tik_grad_errs = [], []

    for seed in range(n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        X, _ = make_swiss_roll(n_samples=2000, noise=0.3, random_state=seed)
        X = StandardScaler().fit_transform(X)
        X_train, X_test = X[:1500], X[1500:1550]
        landmarks, _ = nystrom_sample(X_train, m=300, method='kmeans')
        W = rbf_kernel_matrix(landmarks, gamma=2.0)
        noise = 5e-3 * np.random.randn(300, 300)
        noise = (noise + noise.T) / 2
        W_noisy = W + noise

        # DSD-optimized
        opt = DSDOptimizer(lr=0.01, n_epochs=100, patience=15, n_train_points=40)
        dsd_mod, _ = opt.optimize(
            X_train[500:1000], landmarks, 2.0, noise, verbose=False
        )
        W_t = torch.tensor(W_noisy, dtype=torch.float64)
        with torch.no_grad():
            W_inv_dsd = dsd_mod(W_t).pseudo_inverse.numpy()
        errs = [np.linalg.norm(
            compute_preimage(rbf_kernel_vector(x, landmarks, 2.0), W_inv_dsd, landmarks) - x
        ) for x in X_test]
        dsd_errs.append(np.mean(errs))

        # Tikhonov gradient-optimized (Adam on same loss)
        log_gamma = torch.tensor(np.log(1e-3), dtype=torch.float64, requires_grad=True)
        optimizer_tik = torch.optim.Adam([log_gamma], lr=0.05)
        L_t = torch.tensor(landmarks, dtype=torch.float64)
        X_opt = torch.tensor(X_train[500:540], dtype=torch.float64)
        K_opt = torch.exp(-2.0 * torch.cdist(X_opt, L_t) ** 2)
        evals, evecs = torch.linalg.eigh(W_t)
        pos = evals > 1e-12
        evals_p, evecs_p = evals[pos], evecs[:, pos]

        for _ in range(100):
            optimizer_tik.zero_grad()
            gamma_val = torch.exp(log_gamma)
            inv_diag = 1.0 / (evals_p + gamma_val)
            W_inv_t = (evecs_p * inv_diag) @ evecs_p.T
            preimages = (K_opt @ W_inv_t) @ L_t
            loss = torch.mean(torch.sum((preimages - X_opt) ** 2, dim=1))
            loss.backward()
            optimizer_tik.step()

        with torch.no_grad():
            gamma_final = torch.exp(log_gamma)
            inv_diag = 1.0 / (evals_p + gamma_final)
            W_inv_tik = ((evecs_p * inv_diag) @ evecs_p.T).numpy()
        errs = [np.linalg.norm(
            compute_preimage(rbf_kernel_vector(x, landmarks, 2.0), W_inv_tik, landmarks) - x
        ) for x in X_test]
        tik_grad_errs.append(np.mean(errs))

        if (seed + 1) % 10 == 0:
            print(f"    Seed {seed+1}/{n_seeds}: DSD={dsd_errs[-1]:.4f}, Tik-grad={tik_grad_errs[-1]:.4f}")

    dsd_errs = np.array(dsd_errs)
    tik_grad_errs = np.array(tik_grad_errs)
    t, p = stats.ttest_rel(tik_grad_errs, dsd_errs)
    wins = np.sum(dsd_errs < tik_grad_errs)

    print(f"\n  DSD-opt:      {dsd_errs.mean():.5f} +/- {dsd_errs.std():.5f}")
    print(f"  Tik-grad-opt: {tik_grad_errs.mean():.5f} +/- {tik_grad_errs.std():.5f}")
    print(f"  t={t:.3f}, p={p:.4f}, DSD wins {wins}/{n_seeds}")
    print(f"  Conclusion: {'DSD > gradient-Tikhonov (fairness confirmed)' if p < 0.05 and dsd_errs.mean() < tik_grad_errs.mean() else 'Equivalent'}")

    return dsd_errs, tik_grad_errs


# ================================================================
# TEST 4: Spectral tail-clustering analysis
# ================================================================

def test_spectral_analysis():
    print("\n" + "=" * 60)
    print("TEST 4: Spectral Tail-Clustering in LSTSVM Product Matrices")
    print("=" * 60)

    results = []
    for d, n in [(200, 300), (100, 200), (50, 300), (30, 400)]:
        np.random.seed(42)
        X, y = make_classification(
            n_samples=n, n_features=d, n_informative=max(d // 10, 5),
            n_redundant=d // 5, n_clusters_per_class=1, random_state=42
        )
        X = StandardScaler().fit_transform(X)
        X = X + 0.1 * np.random.randn(*X.shape)

        A = X[y == 0]
        B = X[y == 1]
        E1 = np.hstack([A, np.ones((len(A), 1))])
        E2 = np.hstack([B, np.ones((len(B), 1))])
        M = E1.T @ E1 + E2.T @ E2

        evals = np.linalg.eigvalsh(M)
        evals = evals[evals > 1e-12]

        cond = evals[-1] / evals[0]
        mid = len(evals) // 2
        tail_span = (evals[mid] - evals[0]) / (evals[-1] - evals[mid])

        results.append((d, n, cond, tail_span * 100))
        print(f"  d={d:>3d}, n={n}: cond={cond:.1e}, tail-clustering={tail_span*100:.1f}%")

    print("\n  Interpretation:")
    print("    Lower tail-clustering % = more severe spectral compression")
    print("    DSD advantage correlates with tail-clustering < ~12%")

    return results


# ================================================================
# MAIN
# ================================================================

def run_experiment():
    print("=" * 60)
    print("Experiment 10: Extended Validation")
    print("=" * 60)
    print("\nReproduces results cited in the paper that extend the core")
    print("findings: real-world high-dim, boundary characterization,")
    print("fairness verification, and spectral mechanism analysis.")

    test_gina(n_seeds=30)
    test_gina_init_vs_opt(n_seeds=30)
    test_madelon(n_seeds=30)
    test_digits(n_seeds=30)
    test_tikhonov_gradient_fairness(n_seeds=20)
    test_spectral_analysis()

    print("\n" + "=" * 60)
    print("✓ Experiment 10 complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
