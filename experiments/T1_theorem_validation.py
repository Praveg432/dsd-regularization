"""Theorem validation (T1): test the gap-vs-magnitude excess-risk separation, trying to BREAK it.

Checks the four falsifiable predictions from THEOREM-PLAN.md against Monte-Carlo simulation of the
model in THEOREM-DERIVATION.md, plus real GINA/Madelon spectra. A prediction that fails => the theorem
is revised or discarded, not forced.

P1 (relevance): equal-magnitude / different-gap eigenvalue pairs EXIST in real GINA/Madelon within-
                class scatter matrices.
P2 (mechanism): eigenvector rotation sin-angle scales ~ eps/gap in the model (Davis-Kahan holds).
P3 (claim):     the best GAP-aware filter achieves lower excess risk than the best MAGNITUDE-only
                filter, in a spectrum WITH equal-lambda/different-gap pairs.
P4 (boundary):  that advantage VANISHES when the spectrum has NO equal-lambda/different-gap structure
                (distinct magnitudes). If the advantage persists there, the theorem is wrong.

Run: python experiments/T1_theorem_validation.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


# ----------------------------- model --------------------------------------
def build_population(d, regime, seed):
    """Sigma (within-class cov), mu (mean diff). regime controls spectrum structure."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    if regime == "equal_mag_diff_gap":
        # construct eigenvalues with PAIRS of (near-)equal magnitude but very different local gaps:
        # one member sits inside a tight cluster (tiny gap), its twin is isolated (large gap).
        base = []
        # isolated large eigs (large gaps)
        base += [10.0, 7.0, 4.0]
        # a tight cluster around 1.0 (tiny gaps) -- and an ISOLATED eig also near 1.0 is impossible if
        # sorted adjacency defines gap; instead: cluster around 1.0, and a matched-magnitude isolated
        # eig at 1.0-ish is placed far in index by surrounding it with big gaps. We approximate the
        # "equal magnitude different gap" structure with a cluster near 1.0 plus singletons near 1.0
        # separated by large gaps on both sides is not possible on a line; so we use TWO clusters of
        # equal mean magnitude but different internal tightness -> members have equal ~magnitude,
        # different gaps.
        cluster_tight = list(1.0 + 0.001 * rng.standard_normal(12))   # magnitude ~1, gaps ~1e-3
        cluster_loose = list(1.0 + 0.20 * rng.standard_normal(12))    # magnitude ~1, gaps ~0.1
        base += cluster_tight + cluster_loose
        # fill the rest with a smooth tail
        rest = list(np.linspace(0.05, 0.9, d - len(base)))
        ev = np.array(sorted(base + rest))
    elif regime == "distinct_mag":
        # control: strictly distinct magnitudes, smoothly spaced -> gap is a function of magnitude,
        # NO equal-magnitude/different-gap pairs. Theorem predicts NO gap advantage here (P4).
        ev = np.sort(rng.uniform(0.05, 10.0, size=d))
        ev = np.sort(np.linspace(0.05, 10.0, d) + 0.01 * rng.standard_normal(d))
    else:
        raise ValueError(regime)
    ev = np.clip(ev, 1e-3, None)
    Sigma = (Q * ev) @ Q.T
    Sigma = (Sigma + Sigma.T) / 2
    # signal spread roughly uniformly over eigenbasis (so equal-magnitude dirs carry comparable signal)
    m = rng.standard_normal(d) * 0.5
    mu = Q @ m
    return Sigma, mu, Q, ev


def filters_on(evh, gaph):
    """Return a dict of candidate effective-inverse filters phi_i given observed (evh, gaph).
    Magnitude-only filters are functions of evh; the gap filter also uses gaph.
    We tune each filter's 1 scalar on the SAME oracle objective (fair) inside the risk evaluation,
    here we just return the SHAPE families parameterized to be swept.
    """
    return None  # filters are swept inline in excess_risk_best


def excess_risk(phi, evh, Uh, hat_mu, Sigma, mu):
    """Excess-risk surrogate L = 1 - (w^T mu)^2 / ((w^T Sigma w) J*), for w = sum_i phi_i (uh_i^T hatmu) uh_i."""
    coeff = phi * (Uh.T @ hat_mu)          # phi_i * (uh_i^T hat_mu)
    w = Uh @ coeff
    num = (w @ mu) ** 2
    den = (w @ Sigma @ w) * (mu @ np.linalg.solve(Sigma, mu))
    if den <= 0:
        return 1.0
    return float(max(0.0, 1.0 - num / den))


def best_magnitude_filter(evh, Uh, hat_mu, Sigma, mu):
    """Best magnitude-only filter: phi_i = evh_i / (evh_i^2 + gamma) swept over gamma (Wiener/DSD form
    but gap-FREE), plus TSVD rank sweep. Returns min excess risk over the magnitude family."""
    best = np.inf
    for gamma in np.logspace(-6, 1, 25):
        phi = evh / (evh ** 2 + gamma)
        best = min(best, excess_risk(phi, evh, Uh, hat_mu, Sigma, mu))
    # TSVD: keep top-k (by magnitude)
    order = np.argsort(evh)[::-1]
    for k in range(1, len(evh) + 1, max(1, len(evh) // 20)):
        phi = np.zeros_like(evh)
        keep = order[:k]
        phi[keep] = 1.0 / evh[keep]
        best = min(best, excess_risk(phi, evh, Uh, hat_mu, Sigma, mu))
    return best


def best_gap_filter(evh, gaph, Uh, hat_mu, Sigma, mu):
    """Best gap-aware filter: phi_i = evh_i/(evh_i^2 + alpha exp(-beta gaph_i)) swept over (alpha,beta).
    This is DSD's family. Includes the magnitude family as the beta->0 limit, so it can only do >=."""
    best = np.inf
    med_gap = np.median(gaph) + 1e-12
    for alpha in np.logspace(-6, 1, 12):
        for beta in [0.0] + list(1.0 / med_gap * np.array([0.25, 1.0, 4.0, 16.0])):
            damping = alpha * np.exp(-beta * gaph)
            phi = evh / (evh ** 2 + damping)
            best = min(best, excess_risk(phi, evh, Uh, hat_mu, Sigma, mu))
    return best


def eigengaps(ev_sorted):
    d = len(ev_sorted)
    diffs = np.abs(np.diff(ev_sorted))
    left = np.concatenate([[diffs[0]], diffs]); right = np.concatenate([diffs, [diffs[-1]]])
    return np.minimum(left, right)


# ----------------------------- P2: rotation ~ eps/gap ---------------------
def check_P2(d=60, seeds=20):
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for s in range(seeds):
        Sigma, mu, Q, ev = build_population(d, "equal_mag_diff_gap", s)
        evals, U = np.linalg.eigh(Sigma)
        gaps = eigengaps(np.sort(evals))
        order = np.argsort(evals)
        for eps in (0.01, 0.03, 0.1):
            G = rng.standard_normal((d, d)); E = eps * (G + G.T) / 2 / np.sqrt(2 * d)
            evh, Uh = np.linalg.eigh(Sigma + E)
            # match perturbed to population eigenvectors by greedy abs-overlap
            for idx in range(d):
                i_pop = order[idx]
                overlaps = np.abs(Uh.T @ U[:, i_pop])
                j = int(np.argmax(overlaps))
                sin_ang = np.sqrt(max(0.0, 1 - overlaps[j] ** 2))
                g = gaps[np.where(order == i_pop)[0][0]]
                xs.append(eps / max(g, 1e-6)); ys.append(sin_ang)
    xs = np.array(xs); ys = np.array(ys)
    # restrict to the valid first-order regime (eps/gap not too large)
    mask = xs < 2.0
    r = stats.pearsonr(xs[mask], ys[mask])
    return {"pearson_r": round(float(r.statistic), 3), "p": round(float(r.pvalue), 6),
            "n": int(mask.sum()), "holds": bool(r.statistic > 0.5 and r.pvalue < 1e-3)}


# ----------------------------- P3 & P4: risk ordering ---------------------
def check_P3_P4(d=60, seeds=40, eps=0.05, sigma_mu=0.1):
    rng = np.random.default_rng(1)
    out = {}
    for regime in ("equal_mag_diff_gap", "distinct_mag"):
        gap_wins, deltas = 0, []
        gap_risks, mag_risks = [], []
        for s in range(seeds):
            Sigma, mu, Q, ev = build_population(d, regime, s)
            G = rng.standard_normal((d, d)); E = eps * (G + G.T) / 2 / np.sqrt(2 * d)
            S = Sigma + E
            evh, Uh = np.linalg.eigh(S)
            gaph = eigengaps(np.sort(evh))
            # observed eigs are returned ascending by eigh; align gaph to evh order
            order = np.argsort(evh); gaph_byval = np.empty(d); gaph_byval[order] = eigengaps(np.sort(evh))
            hat_mu = mu + sigma_mu * rng.standard_normal(d)
            r_gap = best_gap_filter(evh, gaph_byval, Uh, hat_mu, Sigma, mu)
            r_mag = best_magnitude_filter(evh, Uh, hat_mu, Sigma, mu)
            gap_risks.append(r_gap); mag_risks.append(r_mag)
            deltas.append(r_mag - r_gap)
            if r_gap < r_mag - 1e-9:
                gap_wins += 1
        deltas = np.array(deltas)
        tp = stats.ttest_1samp(deltas, 0.0)
        out[regime] = {
            "gap_risk_mean": round(float(np.mean(gap_risks)), 5),
            "mag_risk_mean": round(float(np.mean(mag_risks)), 5),
            "mean_advantage(mag-gap)": round(float(np.mean(deltas)), 6),
            "gap_wins": int(gap_wins), "seeds": seeds,
            "p_advantage_gt_0": round(float(tp.pvalue), 5),
            "advantage_significant": bool(np.mean(deltas) > 0 and tp.pvalue < 0.05),
        }
    return out


# ----------------------------- P1: real-data relevance --------------------
def check_P1():
    res = {}
    for name in ("gina_agnostic", "madelon"):
        try:
            d = fetch_openml(name=name, version=1, as_frame=False, parser="auto")
            X = d.data.astype(float)
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(d.target) if name == "gina_agnostic" else (d.target == "1").astype(int)
            X = StandardScaler().fit_transform(X)
            A = X[y == np.unique(y)[0]]; B = X[y == np.unique(y)[1]]
            Sw = (A - A.mean(0)).T @ (A - A.mean(0)) + (B - B.mean(0)).T @ (B - B.mean(0))
            ev = np.sort(np.linalg.eigvalsh((Sw + Sw.T) / 2))
            ev = ev[ev > 1e-10]
            gaps = eigengaps(ev)
            # find equal-magnitude (within 2%) pairs with gap ratio > 5x
            pairs = 0; checked = 0
            for i in range(len(ev)):
                for j in range(i + 1, min(i + 40, len(ev))):
                    checked += 1
                    if abs(ev[i] - ev[j]) / max(ev[i], 1e-12) < 0.02:
                        gr = max(gaps[i], gaps[j]) / max(min(gaps[i], gaps[j]), 1e-12)
                        if gr > 5.0:
                            pairs += 1
            res[name] = {"n_eig": int(len(ev)),
                         "equal_mag_diff_gap_pairs(>5x gap, <2% mag)": int(pairs),
                         "relevant": bool(pairs > 0)}
        except Exception as e:
            res[name] = {"error": str(e)[:120]}
    return res


def run():
    t0 = time.time()
    out = {"experiment": "T1 theorem validation (gap vs magnitude excess-risk separation)"}
    print("P2: eigenvector rotation ~ eps/gap ...", flush=True)
    out["P2_mechanism"] = check_P2()
    print(f"   {out['P2_mechanism']}", flush=True)
    print("P3/P4: risk ordering + boundary ...", flush=True)
    out["P3_P4_risk"] = check_P3_P4()
    for reg, r in out["P3_P4_risk"].items():
        print(f"   {reg}: mag-gap adv={r['mean_advantage(mag-gap)']} wins={r['gap_wins']}/{r['seeds']} "
              f"p={r['p_advantage_gt_0']} sig={r['advantage_significant']}", flush=True)
    print("P1: real-data relevance (GINA/Madelon) ...", flush=True)
    out["P1_relevance"] = check_P1()
    print(f"   {out['P1_relevance']}", flush=True)

    # overall verdict
    P2 = out["P2_mechanism"].get("holds", False)
    emdg = out["P3_P4_risk"]["equal_mag_diff_gap"]
    dist = out["P3_P4_risk"]["distinct_mag"]
    P3 = emdg["advantage_significant"]                       # gap wins where theorem says it should
    P4 = not dist["advantage_significant"]                   # NO gap advantage where it shouldn't
    P1 = any(v.get("relevant") for v in out["P1_relevance"].values() if isinstance(v, dict))
    verdict = "VALIDATED" if (P1 and P2 and P3 and P4) else "FAILED"
    out["gates"] = {"P1_relevance": bool(P1), "P2_mechanism": bool(P2),
                    "P3_claim": bool(P3), "P4_boundary": bool(P4)}
    out["verdict"] = verdict
    out["runtime_seconds"] = round(time.time() - t0, 1)
    print("\n==== THEOREM VERDICT ====")
    print(f"  P1 relevance (real data): {P1}")
    print(f"  P2 mechanism (rot~eps/gap): {P2}")
    print(f"  P3 claim (gap<mag where expected): {P3}")
    print(f"  P4 boundary (no adv where forbidden): {P4}")
    print(f"  => {verdict}")
    path = os.path.join(RESULTS, "exp_T1_theorem_validation.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"saved -> {path}")
    return out


if __name__ == "__main__":
    run()
