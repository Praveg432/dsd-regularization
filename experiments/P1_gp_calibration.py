"""Phase-1: does DSD-conditioning improve GP CALIBRATION over fairly-tuned baselines?

Compares four posterior-inversion methods inside an identical GP (same kernel bandwidth, same data):
    jitter    - plain (K + sigma^2 I)^-1, sigma^2 tuned on validation NLL  [the standard GP practice]
    tikhonov  - extra ridge gamma on top of jitter, gamma tuned on validation NLL
    tsvd      - truncated-SVD pseudo-inverse, rank k tuned on validation NLL
    dsd       - DSD gap-adaptive inverse (auto-init alpha,beta from the preprint)

FAIRNESS (non-negotiable, per the preprint's own 'naive tuning flips conclusions' warning):
  - Every baseline gets its hyperparameter tuned on a VALIDATION split by minimizing validation NLL,
    the same objective we report. DSD uses only its auto-initialization (no tuning) -- so if DSD wins,
    it wins DESPITE giving the baselines the tuning advantage, exactly as in the preprint.
  - jitter's sigma^2 is ALSO tuned (so the 'standard practice' baseline is as strong as possible).
  - The kernel bandwidth is fixed per regime (shared by all methods) so the comparison isolates the
    inversion, not the kernel.

Metrics on the TEST split: RMSE, Gaussian NLL (calibration), empirical 90% credible-interval coverage.
Regimes target the Phase-0 GP ill-conditioning regime (low/moderate dim, sharp/smooth RBF,
near-duplicate inputs). 30 seeds, paired t-test of DSD vs the best baseline per metric, Cohen's d.

Run: python experiments/P1_gp_calibration.py [quick]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats

from src.gp_regressor import GPRegressor, GPConfig, rmse, gaussian_nll, interval_coverage

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

METHODS = ["jitter", "tikhonov", "tsvd", "dsd"]

# tuning grids (baselines only; dsd is auto-init)
SIGMA2_GRID = np.logspace(-6, -1, 10)      # jitter sigma^2 AND shared base for others
TIK_GAMMA_GRID = np.logspace(-6, -1, 10)   # extra ridge
# tsvd rank grid set per-n at runtime


def make_regression(n, d, regime, seed):
    """Synthetic regression with a known smooth target + the Phase-0 ill-conditioning structure."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    if regime == "near_duplicate":
        k = max(1, int(0.3 * n))
        idx = rng.choice(n, size=k, replace=False)
        X[idx] = X[rng.choice(n, size=k, replace=True)] + 1e-4 * rng.standard_normal((k, d))
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    # smooth target: sum of sinusoids over first few dims + mild cross term
    f = np.sin(3 * X[:, 0]) + 0.5 * np.cos(2 * X[:, min(1, d - 1)]) + 0.3 * X[:, 0] * X[:, min(1, d - 1)]
    noise_sd = 0.1
    y = f + noise_sd * rng.standard_normal(n)
    return X, y


def split(X, y, seed):
    """60/20/20 train/val/test."""
    rng = np.random.default_rng(seed + 10_000)
    idx = rng.permutation(len(X))
    n = len(X); a = int(0.6 * n); b = int(0.8 * n)
    tr, va, te = idx[:a], idx[a:b], idx[b:]
    return (X[tr], y[tr], X[va], y[va], X[te], y[te])


def tune_and_eval(method, Xtr, ytr, Xva, yva, Xte, yte, gamma_k, tsvd_grid):
    """Tune the method's hyperparameter on validation NLL, then eval on test. Returns metrics dict."""
    best = None
    if method == "jitter":
        for s2 in SIGMA2_GRID:
            gp = GPRegressor(GPConfig(kernel_gamma=gamma_k, sigma2=s2, method="jitter")).fit(Xtr, ytr)
            mu, sd = gp.predict(Xva); v = gaussian_nll(yva, mu, sd)
            if best is None or v < best[0]:
                best = (v, {"sigma2": s2})
        cfg = GPConfig(kernel_gamma=gamma_k, sigma2=best[1]["sigma2"], method="jitter")
    elif method == "tikhonov":
        for s2 in (SIGMA2_GRID[::2]):           # coarse on sigma2
            for g in TIK_GAMMA_GRID:
                gp = GPRegressor(GPConfig(kernel_gamma=gamma_k, sigma2=s2, method="tikhonov",
                                          tikhonov_gamma=g)).fit(Xtr, ytr)
                mu, sd = gp.predict(Xva); v = gaussian_nll(yva, mu, sd)
                if best is None or v < best[0]:
                    best = (v, {"sigma2": s2, "gamma": g})
        cfg = GPConfig(kernel_gamma=gamma_k, sigma2=best[1]["sigma2"], method="tikhonov",
                       tikhonov_gamma=best[1]["gamma"])
    elif method == "tsvd":
        for s2 in (SIGMA2_GRID[::2]):
            for k in tsvd_grid:
                gp = GPRegressor(GPConfig(kernel_gamma=gamma_k, sigma2=s2, method="tsvd",
                                          tsvd_rank=int(k))).fit(Xtr, ytr)
                mu, sd = gp.predict(Xva); v = gaussian_nll(yva, mu, sd)
                if best is None or v < best[0]:
                    best = (v, {"sigma2": s2, "k": int(k)})
        cfg = GPConfig(kernel_gamma=gamma_k, sigma2=best[1]["sigma2"], method="tsvd",
                       tsvd_rank=best[1]["k"])
    elif method == "dsd":
        # DSD: auto-init, NO tuning. We only tune sigma^2 minimally (the GP's own noise term) so the
        # posterior matrix is comparable; alpha,beta stay auto. Use the SAME sigma2 grid-min on val to
        # avoid giving DSD a free extra knob -- pick sigma2 by val NLL but with fixed auto alpha,beta.
        for s2 in (SIGMA2_GRID[::2]):
            gp = GPRegressor(GPConfig(kernel_gamma=gamma_k, sigma2=s2, method="dsd")).fit(Xtr, ytr)
            mu, sd = gp.predict(Xva); v = gaussian_nll(yva, mu, sd)
            if best is None or v < best[0]:
                best = (v, {"sigma2": s2})
        cfg = GPConfig(kernel_gamma=gamma_k, sigma2=best[1]["sigma2"], method="dsd")
    else:
        raise ValueError(method)

    # refit on train (tuning used train->val); eval on test
    gp = GPRegressor(cfg).fit(Xtr, ytr)
    mu, sd = gp.predict(Xte)
    return {"rmse": rmse(yte, mu), "nll": gaussian_nll(yte, mu, sd),
            "cov90": interval_coverage(yte, mu, sd, 0.90)}


def run(quick=False):
    t0 = time.time()
    seeds = 5 if quick else 30
    n = 200
    # (label, d, rbf_gamma, input_regime) -- Phase-0 ill-conditioned GP regimes
    regimes = [
        ("d2_sharpRBF", 2, 5.0, "normal"),
        ("d10_smoothRBF", 10, 0.05, "normal"),
        ("d10_nearDup_sharp", 10, 5.0, "near_duplicate"),
    ]
    tsvd_grid = [int(x) for x in (0.5 * 0.6 * n, 0.7 * 0.6 * n, 0.9 * 0.6 * n, 0.6 * n)]  # ranks < n_train
    tsvd_grid = sorted(set(max(5, min(k, int(0.6 * n))) for k in tsvd_grid))

    out = {"experiment": "P1 GP calibration: DSD vs fairly-tuned baselines",
           "n": n, "n_seeds": seeds, "metrics": ["rmse", "nll", "cov90"],
           "fairness": "baselines tuned on validation NLL; DSD auto-init (alpha,beta) only",
           "results_by_regime": {}}

    for label, d, gk, ireg in regimes:
        print(f"\n===== {label} (d={d}, rbf_gamma={gk}, {ireg}) =====", flush=True)
        per_seed = {m: {"rmse": [], "nll": [], "cov90": []} for m in METHODS}
        for s in range(seeds):
            X, y = make_regression(n, d, ireg, seed=s)
            Xtr, ytr, Xva, yva, Xte, yte = split(X, y, s)
            for m in METHODS:
                r = tune_and_eval(m, Xtr, ytr, Xva, yva, Xte, yte, gk, tsvd_grid)
                for key in ("rmse", "nll", "cov90"):
                    per_seed[m][key].append(r[key])
            if s == 0 or (s + 1) % 10 == 0:
                msg = " | ".join(f"{m}: nll={np.mean(per_seed[m]['nll']):.3f} cov={np.mean(per_seed[m]['cov90']):.3f}"
                                 for m in METHODS)
                print(f"  seed {s+1}/{seeds}: {msg}", flush=True)

        agg = {}
        for m in METHODS:
            agg[m] = {key: [round(float(np.mean(per_seed[m][key])), 4),
                            round(float(np.std(per_seed[m][key])), 4)]
                      for key in ("rmse", "nll", "cov90")}
        # DSD vs best BASELINE per metric (lower better for rmse,nll; closeness-to-0.90 for cov)
        baselines = [m for m in METHODS if m != "dsd"]
        stats_out = {}
        for key in ("rmse", "nll"):
            best_base = min(baselines, key=lambda m: np.mean(per_seed[m][key]))
            dsd_v = np.array(per_seed["dsd"][key]); base_v = np.array(per_seed[best_base][key])
            tstat, p = stats.ttest_rel(dsd_v, base_v)
            diff = dsd_v - base_v
            cohen = float(np.mean(diff) / (np.std(diff) + 1e-30))
            stats_out[key] = {"best_baseline": best_base,
                              "dsd_mean": round(float(dsd_v.mean()), 4),
                              "best_baseline_mean": round(float(base_v.mean()), 4),
                              "paired_p": round(float(p), 4),
                              "cohens_d": round(cohen, 3),
                              "dsd_better": bool(dsd_v.mean() < base_v.mean())}
        # coverage: closeness to 0.90 (|cov-0.90|, lower better)
        def covdist(m): return np.abs(np.array(per_seed[m]["cov90"]) - 0.90)
        best_base_cov = min(baselines, key=lambda m: covdist(m).mean())
        tstat, p = stats.ttest_rel(covdist("dsd"), covdist(best_base_cov))
        stats_out["cov90"] = {"best_baseline": best_base_cov,
                              "dsd_mean_cov": round(float(np.mean(per_seed['dsd']['cov90'])), 4),
                              "best_baseline_mean_cov": round(float(np.mean(per_seed[best_base_cov]['cov90'])), 4),
                              "dsd_closer_to_target": bool(covdist("dsd").mean() < covdist(best_base_cov).mean()),
                              "paired_p": round(float(p), 4)}
        out["results_by_regime"][label] = {"d": d, "rbf_gamma": gk, "input": ireg,
                                            "aggregate": agg, "dsd_vs_best_baseline": stats_out}
        print(f"  -> NLL: dsd {agg['dsd']['nll'][0]} vs best base {stats_out['nll']['best_baseline']} "
              f"{stats_out['nll']['best_baseline_mean']} (p={stats_out['nll']['paired_p']}, "
              f"dsd_better={stats_out['nll']['dsd_better']})", flush=True)
        print(f"  -> cov90: dsd {stats_out['cov90']['dsd_mean_cov']} vs "
              f"{stats_out['cov90']['best_baseline']} {stats_out['cov90']['best_baseline_mean_cov']} "
              f"(dsd_closer={stats_out['cov90']['dsd_closer_to_target']})", flush=True)

    out["runtime_seconds"] = round(time.time() - t0, 1)
    path = os.path.join(RESULTS, "exp_P1_gp_calibration.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\nsaved -> {path}  ({out['runtime_seconds']}s)")
    return out


if __name__ == "__main__":
    run(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))
