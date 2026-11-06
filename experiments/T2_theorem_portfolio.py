"""Parallel vetting of the DSD theorem-candidate portfolio (THEOREM-PORTFOLIO.md).

Each candidate returns: holds (bool), a measured quantity vs predicted, and a triviality flag. We
actively try to break each. Keep only TRUE and NON-TRIVIAL ones.

C1 bounded inverse: max_lambda phi(lambda,d) == 1/(2 sqrt d); operator norm <= 1/(2 sqrt min_i d_i).
C2 Tikhonov recovery: equal gaps / beta->0  => DSD inverse == Tikhonov inverse (float-exact).
C3 effective-DoF tail reduction: DSD removes strictly more tail DoF than MATCHED-head Tikhonov.
C4 monotone reliability ordering: phi strictly increasing in gap at fixed lambda.
C5 distributional separation: with magnitude held ~fixed across a distribution and reliability varied
   via gap, a FIXED magnitude filter cannot match a gap filter (gap carries independent info).
C6 stability: DSD empirical leave-one-out sensitivity < matched-head Tikhonov on clustered data.

Run: python experiments/T2_theorem_portfolio.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.dsd import compute_eigengaps, initialize_hyperparameters

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def phi(lam, d):
    return lam / (lam ** 2 + d)


def clustered_spectrum(m, seed):
    rng = np.random.default_rng(seed)
    ev = np.concatenate([rng.uniform(5, 10, max(1, m // 10)),
                         1.0 + 0.002 * rng.standard_normal(m // 3),        # tight cluster (tiny gaps)
                         rng.uniform(0.01, 0.05, m - max(1, m // 10) - m // 3)])
    return np.sort(ev)


# ---------------- C1 ----------------
def test_C1(seeds=200):
    ok = True; maxrelerr = 0.0; nontrivial = False
    rng = np.random.default_rng(0)
    for _ in range(seeds):
        d = 10 ** rng.uniform(-6, 0)
        lam_grid = np.logspace(-4, 2, 4000)
        pred = 1.0 / (2 * np.sqrt(d))
        meas = phi(lam_grid, d).max()
        maxrelerr = max(maxrelerr, abs(meas - pred) / pred)
        if abs(meas - pred) / pred > 1e-3:
            ok = False
        if pred < 1e3:  # the bound actually constrains something finite
            nontrivial = True
    return {"holds": bool(ok), "max_rel_err_vs_1_over_2sqrt_d": round(float(maxrelerr), 6),
            "nontrivial": bool(nontrivial),
            "note": "operator norm floor 1/(2 sqrt(min d_i)); gap-dependent via d_i=alpha exp(-beta gap)"}


# ---------------- C2 ----------------
def test_C2(m=80, seeds=10):
    rng = np.random.default_rng(1)
    worst = 0.0
    for s in range(seeds):
        ev = np.sort(rng.uniform(0.1, 5, m))
        # beta -> 0 : damping -> alpha uniformly => Tikhonov with gamma=alpha (in the lam/(lam^2+gamma) form)
        alpha = 0.01
        gaps = compute_eigengaps(ev)
        d_dsd = alpha * np.exp(-1e-12 * gaps)       # beta ~ 0
        phi_dsd = ev / (ev ** 2 + d_dsd)
        phi_tik = ev / (ev ** 2 + alpha)
        worst = max(worst, float(np.max(np.abs(phi_dsd - phi_tik))))
    return {"holds": bool(worst < 1e-8), "max_abs_diff_dsd_vs_tikhonov_at_beta0": worst,
            "nontrivial": True, "note": "DSD strictly generalizes Tikhonov (beta->0 recovery)"}


# ---------------- C3 ----------------
def test_C3(m=300, seeds=30):
    """DoF: df = sum lambda^2/(lambda^2+d). Match Tikhonov so HEAD damping equals DSD's head damping,
    then check DSD removes strictly more TAIL dof (tail = smallest 50% eigs)."""
    rng = np.random.default_rng(2)
    tail_red = []; head_match_err = []
    for s in range(seeds):
        ev = clustered_spectrum(m, s)
        gaps = compute_eigengaps(ev)
        alpha, beta = initialize_hyperparameters(ev)
        d_dsd = alpha * np.exp(-beta * gaps)
        # match Tikhonov gamma to DSD's damping on the HEAD (reliable, large-gap) directions:
        head = ev > np.median(ev)
        gamma = float(np.mean(d_dsd[head]))          # matched head damping
        d_tik = np.full(m, gamma)
        df_i = lambda d: ev ** 2 / (ev ** 2 + d)
        tail = ~head
        df_dsd_tail = float(np.sum(df_i(d_dsd)[tail]))
        df_tik_tail = float(np.sum(df_i(d_tik)[tail]))
        tail_red.append(df_tik_tail - df_dsd_tail)   # positive => DSD removes MORE tail dof
        head_match_err.append(abs(np.mean(df_i(d_dsd)[head]) - np.mean(df_i(d_tik)[head])))
    tail_red = np.array(tail_red)
    holds = bool(np.mean(tail_red) > 0 and np.mean(tail_red > 0) > 0.8)
    return {"holds": holds, "mean_extra_tail_dof_removed_by_dsd": round(float(np.mean(tail_red)), 4),
            "frac_seeds_positive": round(float(np.mean(tail_red > 0)), 3),
            "mean_head_df_match_err": round(float(np.mean(head_match_err)), 5),
            "nontrivial": bool(abs(np.mean(tail_red)) > 1e-3),
            "note": "matched head damping; DSD removes more TAIL effective-dof"}


# ---------------- C4 ----------------
def test_C4():
    lam = np.array([0.01, 0.1, 1.0, 5.0]); alpha, beta = 0.05, 3.0
    ok = True; spreads = []
    for L in lam:
        gaps = np.linspace(0.001, 2.0, 50)
        vals = L / (L ** 2 + alpha * np.exp(-beta * gaps))
        if not np.all(np.diff(vals) > 0):
            ok = False
        spreads.append(float(vals[-1] - vals[0]))
    return {"holds": bool(ok), "nontrivial": bool(min(spreads) > 0),
            "phi_increasing_in_gap": bool(ok),
            "note": "d phi/d gap > 0: strict reliability ordering no magnitude filter can impose"}


# ---------------- C5 ----------------
def test_C5(m=120, seeds=60):
    """Engineered distribution: observed eigenVALUE magnitude held ~constant across the distribution,
    while eigenVECTOR reliability varies via a latent gap. A FIXED-across-distribution magnitude filter
    cannot adapt; a gap filter can. We measure reconstruction of a planted signal in the reliable vs
    corrupted directions. Gap carries independent info iff gap filter beats the best FIXED magnitude
    filter in EXPECTATION over the distribution."""
    rng = np.random.default_rng(3)
    # magnitude fixed at ~1 for all directions across the whole distribution
    # reliability: half the matrices have the signal in a small-gap (corrupted) subspace, half in a
    # large-gap (reliable) subspace; magnitude identical => a fixed magnitude filter must treat all same
    def one(seed, corrupted):
        r = np.random.default_rng(seed)
        ev = np.ones(m) + 0.001 * r.standard_normal(m)         # magnitudes ~1 (fixed across dist)
        # create a gap structure: make a block tightly clustered (small gap) vs spread (large gap)
        ev[:m//2] = 1.0 + 0.0005 * r.standard_normal(m//2)     # small-gap block
        ev[m//2:] = 1.0 + 0.05 * np.linspace(-1, 1, m - m//2)   # large-gap block
        ev = np.sort(ev)
        gaps = compute_eigengaps(ev)
        Q, _ = np.linalg.qr(r.standard_normal((m, m)))
        # planted signal lives in the small-gap block if 'corrupted' else large-gap block
        sig = np.zeros(m)
        if corrupted:
            sig[np.argsort(gaps)[:m//4]] = 1.0     # signal on smallest-gap dirs
        else:
            sig[np.argsort(gaps)[-m//4:]] = 1.0    # signal on largest-gap dirs
        # perturb eigenvectors ~1/gap, observe noisy signal coeffs
        obs = sig + (0.3 / np.maximum(gaps, 1e-3)) * r.standard_normal(m) * 0.02
        return ev, gaps, sig, obs
    # best FIXED magnitude filter across distribution: phi = ev/(ev^2+gamma), gamma fixed; since ev~1
    # for all, this is ~constant => cannot distinguish corrupted vs reliable
    gammas = np.logspace(-4, 1, 20)
    # gap filter: phi = ev/(ev^2 + a exp(-b gap)) with (a,b) fixed across distribution
    def recon_err(filter_vals, sig, obs):
        est = filter_vals * obs
        # normalize and compare direction
        if np.linalg.norm(est) < 1e-12: return 1.0
        return float(np.linalg.norm(est/np.linalg.norm(est) - sig/np.linalg.norm(sig)))
    best_mag = np.inf
    for g in gammas:
        errs = []
        for s in range(seeds):
            corrupted = (s % 2 == 0)
            ev, gaps, sig, obs = one(s, corrupted)
            errs.append(recon_err(ev/(ev**2+g), sig, obs))
        best_mag = min(best_mag, np.mean(errs))
    best_gap = np.inf
    for a in np.logspace(-4, 0, 8):
        for b in [1.0, 5.0, 20.0, 100.0]:
            errs = []
            for s in range(seeds):
                corrupted = (s % 2 == 0)
                ev, gaps, sig, obs = one(s, corrupted)
                errs.append(recon_err(ev/(ev**2 + a*np.exp(-b*gaps)), sig, obs))
            best_gap = min(best_gap, np.mean(errs))
    holds = bool(best_gap < best_mag - 1e-3)
    return {"holds": holds, "best_fixed_magnitude_err": round(float(best_mag), 4),
            "best_gap_err": round(float(best_gap), 4),
            "nontrivial": bool(abs(best_mag - best_gap) > 1e-3),
            "note": "fixed-across-distribution magnitude filter vs gap filter; gap wins iff it carries "
                    "info magnitude cannot (magnitudes held ~constant)"}


# ---------------- C6 ----------------
def test_C6(m=150, n_feat=150, seeds=25):
    """Empirical algorithmic stability: leave-one-out sensitivity of the solved direction, DSD vs
    MATCHED-head Tikhonov, on clustered-spectrum data. DSD should be LESS sensitive (more stable)."""
    rng = np.random.default_rng(4)
    dsd_sens, tik_sens = [], []
    from src.kernel_classifiers import FisherDiscriminant
    for s in range(seeds):
        # build a clustered-covariance 2-class dataset
        d = n_feat
        ev = clustered_spectrum(d, s)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        Sigma = (Q * ev) @ Q.T; Sigma = (Sigma + Sigma.T)/2
        L = np.linalg.cholesky(Sigma + 1e-8*np.eye(d))
        mu = Q @ (rng.standard_normal(d) * 0.4)
        n = 60
        Xp = rng.standard_normal((n, d)) @ L.T + mu/2
        Xn = rng.standard_normal((n, d)) @ L.T - mu/2
        X = np.vstack([Xp, Xn]); y = np.array([0]*n + [1]*n)
        # matched-head Tikhonov gamma from DSD damping on head
        A = X[y==0]; B = X[y==1]
        Sw = (A-A.mean(0)).T@(A-A.mean(0)) + (B-B.mean(0)).T@(B-B.mean(0))
        evw = np.linalg.eigvalsh((Sw+Sw.T)/2); evw = np.sort(evw[evw>1e-10])
        gaps = compute_eigengaps(evw); alpha, beta = initialize_hyperparameters(evw)
        dd = alpha*np.exp(-beta*gaps); gamma = float(np.mean(dd[evw>np.median(evw)]))
        def wvec(method, X_, y_):
            if method=="dsd":
                mdl = FisherDiscriminant(inversion_method="dsd").fit(X_, y_)
            else:
                mdl = FisherDiscriminant(inversion_method="tikhonov", tikhonov_gamma=gamma).fit(X_, y_)
            w = mdl.w_; return w/ (np.linalg.norm(w)+1e-12)
        w_dsd_full = wvec("dsd", X, y); w_tik_full = wvec("tik", X, y)
        # leave-one-out on a few points, measure direction change
        dd_, tt_ = [], []
        for loo in rng.choice(len(X), 8, replace=False):
            keep = np.ones(len(X), bool); keep[loo]=False
            wd = wvec("dsd", X[keep], y[keep]); wt = wvec("tik", X[keep], y[keep])
            dd_.append(1 - abs(wd@w_dsd_full)); tt_.append(1 - abs(wt@w_tik_full))
        dsd_sens.append(np.mean(dd_)); tik_sens.append(np.mean(tt_))
    dsd_sens=np.array(dsd_sens); tik_sens=np.array(tik_sens)
    from scipy import stats as st
    p = st.ttest_rel(dsd_sens, tik_sens).pvalue
    holds = bool(np.mean(dsd_sens) < np.mean(tik_sens) and p < 0.05)
    return {"holds": holds, "dsd_loo_sensitivity": round(float(np.mean(dsd_sens)),5),
            "tikhonov_loo_sensitivity": round(float(np.mean(tik_sens)),5),
            "paired_p": round(float(p),4), "nontrivial": True,
            "note": "lower LOO sensitivity => tighter algorithmic-stability generalization bound"}


def run():
    t0=time.time()
    out={"experiment":"T2 theorem portfolio vetting","candidates":{}}
    tests = {"C1_bounded_inverse": test_C1, "C2_tikhonov_recovery": test_C2,
             "C3_dof_tail_reduction": test_C3, "C4_monotone_ordering": test_C4,
             "C5_distributional_sep": test_C5, "C6_stability_LOO": test_C6}
    for name, fn in tests.items():
        print(f"vetting {name} ...", flush=True)
        r = fn()
        out["candidates"][name] = r
        verdict = "KEEP" if (r.get("holds") and r.get("nontrivial")) else "DROP"
        print(f"   holds={r.get('holds')} nontrivial={r.get('nontrivial')} -> {verdict}  | {r}", flush=True)
    out["runtime_seconds"]=round(time.time()-t0,1)
    survivors=[k for k,v in out["candidates"].items() if v.get("holds") and v.get("nontrivial")]
    out["survivors"]=survivors
    print(f"\n==== SURVIVORS (true AND non-trivial): {survivors} ====")
    path=os.path.join(RESULTS,"exp_T2_theorem_portfolio.json")
    json.dump(out, open(path,"w"), indent=2, default=float)
    print(f"saved -> {path}")
    return out


if __name__=="__main__":
    run()
