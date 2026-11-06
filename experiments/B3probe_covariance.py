"""Strategy-B probe B3: does gap-aware eigenvalue shrinkage help high-d COVARIANCE estimation?

Covariance estimation from n<d samples is a large-audience problem (finance, genomics; Ledoit-Wolf
shrinkage is a standard tool). Sample covariance has an ill-conditioned, clustered spectrum -- the DSD
regime. Here DSD's "down-weight unreliable directions" objective is ALIGNED with estimation (unlike
preconditioning B2), so it is a fair test.

We compare eigenvalue-shrinkage estimators of the covariance (all operate on the sample-covariance
eigenbasis; the question is how to transform the eigenVALUES):
  sample        - raw sample covariance (no shrinkage)
  LW_linear     - Ledoit-Wolf linear shrinkage toward scaled identity (standard strong baseline)
  nonlinear     - a strong nonlinear-shrinkage proxy: shrink each eigenvalue toward the GRAND MEAN by
                  an amount increasing for small eigenvalues (captures the essence of QuEST/analytical
                  nonlinear shrinkage: overlarge top eigs shrink down, tiny eigs shrink up). Strong
                  modern incumbent.
  dsd_gap       - gap-aware shrinkage: shrink each eigenvalue toward the grand mean by a factor that
                  increases where the local eigenGAP is small (Davis-Kahan-unreliable directions),
                  i.e. trust well-separated eigenvalues, distrust clustered ones.

Metrics vs the TRUE covariance (known in simulation):
  - Stein loss (a standard covariance-estimation loss): tr(S Sigma^-1) - logdet(S Sigma^-1) - d
  - min-variance portfolio out-of-sample variance using each estimate (lower=better; the practical one)

Fair: LW has a closed-form intensity; nonlinear + dsd_gap each get ONE shrinkage-strength parameter
tuned on a validation resample (so no method is hand-advantaged).

GREEN = dsd_gap beats the BEST of {LW_linear, nonlinear} on >=1 metric with significance across seeds.
RED   = nonlinear shrinkage matches/beats dsd_gap (the niche is already occupied).

Run: python experiments/B3probe_covariance.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def true_cov(d, seed):
    """Clustered-spectrum true covariance (the ill-conditioned regime)."""
    rng = np.random.default_rng(seed)
    ev = np.concatenate([rng.uniform(4, 8, size=max(1, d // 10)),
                         rng.uniform(0.8, 1.2, size=d // 3),
                         rng.uniform(0.02, 0.1, size=d - max(1, d // 10) - d // 3)])
    ev = np.sort(ev)[::-1]
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    S = (Q * ev) @ Q.T
    return (S + S.T) / 2


def sample_cov(Sigma, n, seed):
    rng = np.random.default_rng(seed)
    d = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(d))
    X = rng.standard_normal((n, d)) @ L.T
    Xc = X - X.mean(0)
    return (Xc.T @ Xc) / n, X


def eig_shrink(S, mode, strength):
    """Return a shrunk covariance by transforming sample eigenvalues on the sample eigenbasis."""
    ev, U = np.linalg.eigh((S + S.T) / 2)
    ev = np.clip(ev, 1e-12, None)
    grand = float(np.mean(ev))
    if mode == "sample":
        new = ev
    elif mode == "LW_linear":
        # linear shrinkage toward grand-mean identity with intensity = strength
        new = (1 - strength) * ev + strength * grand
    elif mode == "nonlinear":
        # shrink more for small eigs: intensity scales with (grand/ (ev+grand))
        w = strength * (grand / (ev + grand))
        new = (1 - w) * ev + w * grand
    elif mode == "dsd_gap":
        # gap-aware: shrink toward grand-mean where local gap is small (unreliable directions)
        evs = np.sort(ev)
        d = len(ev)
        gaps = np.empty(d)
        se = np.sort(ev)
        diffs = np.abs(np.diff(se))
        left = np.concatenate([[diffs[0]], diffs]); right = np.concatenate([diffs, [diffs[-1]]])
        g = np.minimum(left, right)
        # map gaps back to original ev order
        order = np.argsort(ev)
        gap_by_orig = np.empty(d); gap_by_orig[order] = g
        med = float(np.median(g)) + 1e-12
        # small gap -> w near strength; large gap -> w near 0
        w = strength * np.exp(-gap_by_orig / med)
        new = (1 - w) * ev + w * grand
    else:
        raise ValueError(mode)
    new = np.clip(new, 1e-10, None)
    return (U * new) @ U.T


def stein_loss(S_hat, Sigma):
    d = Sigma.shape[0]
    Sinv = np.linalg.inv(Sigma)
    M = S_hat @ Sinv
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        return np.inf
    return float(np.trace(M) - logdet - d)


def mv_portfolio_var(S_hat, Sigma):
    """Out-of-sample variance of the global min-variance portfolio built from S_hat, evaluated on true."""
    d = Sigma.shape[0]
    try:
        inv = np.linalg.inv(S_hat)
    except Exception:
        inv = np.linalg.pinv(S_hat)
    ones = np.ones(d)
    w = inv @ ones
    w = w / (ones @ w)
    return float(w @ Sigma @ w)


def tune_strength(mode, Sigma, d, n, seed):
    """Tune shrinkage strength on a validation resample by Stein loss (fair single-parameter tuning)."""
    if mode in ("sample",):
        return 0.0
    best = None
    for strength in np.linspace(0.05, 0.95, 10):
        S_val, _ = sample_cov(Sigma, n, seed + 7777)
        S_hat = eig_shrink(S_val, mode, strength)
        loss = stein_loss(S_hat, Sigma)
        if best is None or loss < best[0]:
            best = (loss, strength)
    return best[1]


def run():
    t0 = time.time()
    d, n = 120, 60          # n < d: the ill-conditioned regime
    seeds = 25
    modes = ["sample", "LW_linear", "nonlinear", "dsd_gap"]
    out = {"experiment": "B3 probe: gap-aware covariance shrinkage", "d": d, "n": n, "seeds": seeds,
           "results": {}}
    stein = {m: [] for m in modes}; pvar = {m: [] for m in modes}
    for s in range(seeds):
        Sigma = true_cov(d, s)
        S, _ = sample_cov(Sigma, n, s)
        for m in modes:
            strength = tune_strength(m, Sigma, d, n, s)
            S_hat = eig_shrink(S, m, strength)
            stein[m].append(stein_loss(S_hat, Sigma))
            pvar[m].append(mv_portfolio_var(S_hat, Sigma))
    agg = {}
    for m in modes:
        agg[m] = {"stein_mean": round(float(np.mean(stein[m])), 4),
                  "portfolio_var_mean": round(float(np.mean(pvar[m])), 5)}
    out["results"] = agg
    print("mode         stein(lower=better)   mvp_var(lower=better)")
    for m in modes:
        print(f"  {m:11s} {agg[m]['stein_mean']:18.4f}   {agg[m]['portfolio_var_mean']:.5f}")

    # verdict: dsd_gap vs best of {LW_linear, nonlinear}
    best_incumbent_stein = min(["LW_linear", "nonlinear"], key=lambda m: np.mean(stein[m]))
    best_incumbent_pvar = min(["LW_linear", "nonlinear"], key=lambda m: np.mean(pvar[m]))
    ps = stats.ttest_rel(stein["dsd_gap"], stein[best_incumbent_stein]).pvalue
    pp = stats.ttest_rel(pvar["dsd_gap"], pvar[best_incumbent_pvar]).pvalue
    dsd_stein_better = np.mean(stein["dsd_gap"]) < np.mean(stein[best_incumbent_stein])
    dsd_pvar_better = np.mean(pvar["dsd_gap"]) < np.mean(pvar[best_incumbent_pvar])
    print("\n==== VERDICT ====")
    print(f"  stein: dsd_gap {np.mean(stein['dsd_gap']):.4f} vs best incumbent "
          f"{best_incumbent_stein} {np.mean(stein[best_incumbent_stein]):.4f} "
          f"(better={dsd_stein_better}, p={ps:.4f})")
    print(f"  mvp_var: dsd_gap {np.mean(pvar['dsd_gap']):.5f} vs best incumbent "
          f"{best_incumbent_pvar} {np.mean(pvar[best_incumbent_pvar]):.5f} "
          f"(better={dsd_pvar_better}, p={pp:.4f})")
    green = (dsd_stein_better and ps < 0.05) or (dsd_pvar_better and pp < 0.05)
    verdict = "GREEN" if green else "RED"
    print(f"\nAUTO-VERDICT (B3): {verdict}  (GREEN needs dsd_gap to beat the BEST of linear+nonlinear "
          f"shrinkage on >=1 metric, p<0.05)")
    out["verdict"] = verdict
    out["stein_p_vs_best"] = round(float(ps), 4); out["pvar_p_vs_best"] = round(float(pp), 4)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    path = os.path.join(RESULTS, "exp_B3probe_covariance.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\nsaved -> {path}")
    return out


if __name__ == "__main__":
    run()
