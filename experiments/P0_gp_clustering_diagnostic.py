"""Phase-0 diagnostic for the DSD GP follow-up.

Question this answers (the gate for the whole GP direction):
  Do Gaussian-process posterior matrices (K + sigma^2 I) exhibit the eigengap tail-clustering that
  DSD is designed to exploit, in realistic regimes, and strongly enough that DSD's gap-adaptive
  inverse actually DIFFERS from Tikhonov on them? If they do not cluster, DSD reduces to Tikhonov and
  there is no GP paper.

We sweep regimes (RBF bandwidth, input dim, near-duplicate inputs, n, jitter sigma^2) and report, per
regime:
  - condition number of (K + sigma^2 I)  [prefilter]
  - tail-clustering metric: fraction of spectral RANGE occupied by the bottom 50% of eigenvalues
    (the preprint's LSTSVM finding was ~7% at d=200 -- lower = more clustered = more DSD headroom)
  - eigengap stats (median, and how many gaps are "tiny" relative to the spectrum)
  - DSD-vs-Tikhonov divergence: relative Frobenius difference between the two regularized inverses,
    AND the max per-eigenvalue difference in the effective inverse filter. If ~0, DSD == Tikhonov.

No training here -- purely spectral structure of the matrices a GP would invert.

Run: python experiments/P0_gp_clustering_diagnostic.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.kernels import rbf_kernel_matrix
from src.dsd import dsd_regularized_inverse, tikhonov_inverse, compute_eigengaps

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def tail_clustering_metric(eigvals):
    """Fraction of total spectral RANGE occupied by the bottom 50% of eigenvalues.

    eigvals ascending. Range = max-min. Bottom-50% span = eigvals[mid]-eigvals[0].
    Lower => more tail clustering => more DSD headroom. (Preprint LSTSVM: ~7% at d=200.)
    """
    ev = np.sort(eigvals)
    rng = ev[-1] - ev[0]
    if rng <= 0:
        return 1.0
    mid = len(ev) // 2
    return float((ev[mid] - ev[0]) / rng)


def dsd_vs_tikhonov_divergence(W):
    """How much does DSD's regularized inverse differ from a tuned Tikhonov inverse on W?

    Returns (rel_frobenius_diff, max_filter_diff). If ~0, DSD provides nothing beyond Tikhonov here.
    Tikhonov gamma is chosen to match DSD's effective damping scale (median of DSD damping) so the
    comparison is fair rather than rigged.
    """
    res = dsd_regularized_inverse(W)  # auto-init alpha,beta
    # pick a Tikhonov gamma on the same scale as DSD's damping so neither is unfairly handicapped
    gamma = float(np.median(res.damping)) ** 0.5
    gamma = max(gamma, 1e-10)
    W_tik = tikhonov_inverse(W, gamma=gamma)
    dsd_inv = res.pseudo_inverse
    num = np.linalg.norm(dsd_inv - W_tik, "fro")
    den = np.linalg.norm(dsd_inv, "fro") + 1e-30
    rel_fro = float(num / den)

    # per-eigenvalue filter comparison on DSD's own eigenbasis:
    # DSD filter: lambda/(lambda^2 + damping); Tikhonov filter at same gamma: 1/(lambda+gamma)
    lam = res.eigenvalues
    dsd_filter = res.reg_inverse_diag
    tik_filter = 1.0 / (lam + gamma)
    # normalize each filter by its max to compare SHAPES (not scale)
    dsd_n = dsd_filter / (dsd_filter.max() + 1e-30)
    tik_n = tik_filter / (tik_filter.max() + 1e-30)
    max_filter_diff = float(np.max(np.abs(dsd_n - tik_n)))
    return rel_fro, max_filter_diff, gamma


def make_inputs(n, d, regime, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    if regime == "near_duplicate":
        # force near-duplicate rows: duplicate 30% of points with tiny jitter
        k = max(1, int(0.3 * n))
        idx = rng.choice(n, size=k, replace=False)
        X[idx] = X[rng.choice(n, size=k, replace=True)] + 1e-4 * rng.standard_normal((k, d))
    # standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return X


def run():
    n = 300
    # sigma^2 jitter levels a GP practitioner actually uses
    jitters = [1e-6, 1e-4, 1e-2]
    regimes = []
    # (label, d, rbf_gamma, input_regime)
    for d in (2, 10, 50, 100):
        regimes.append((f"d{d}_sharpRBF", d, 5.0, "normal"))
        regimes.append((f"d{d}_smoothRBF", d, 0.05, "normal"))
    regimes.append(("d10_nearDup_sharp", 10, 5.0, "near_duplicate"))
    regimes.append(("d50_nearDup_sharp", 50, 5.0, "near_duplicate"))

    out = {"experiment": "P0 GP posterior clustering diagnostic", "n": n,
           "tail_metric_note": "fraction of spectral range in bottom 50% of eigenvalues; "
                               "lower=more clustered; preprint LSTSVM ~0.07 at d=200",
           "regimes": []}

    for label, d, gk, ireg in regimes:
        X = make_inputs(n, d, ireg, seed=0)
        K = rbf_kernel_matrix(X, gamma=gk)
        for s2 in jitters:
            W = K + s2 * np.eye(n)
            ev = np.linalg.eigvalsh(W)
            ev = np.clip(ev, 0, None)
            cond = float(ev[-1] / max(ev[0], 1e-15))
            tail = tail_clustering_metric(ev)
            gaps = compute_eigengaps(np.sort(ev))
            med_gap = float(np.median(gaps))
            # fraction of gaps that are "tiny" (< 1% of median) -> clustered eigenvalues
            tiny_frac = float(np.mean(gaps < 0.01 * med_gap)) if med_gap > 0 else 0.0
            try:
                rel_fro, maxfilt, gamma = dsd_vs_tikhonov_divergence(W)
            except Exception as e:
                rel_fro, maxfilt, gamma = float("nan"), float("nan"), float("nan")
            cell = {
                "regime": label, "d": d, "rbf_gamma": gk, "input": ireg, "jitter_sigma2": s2,
                "condition_number": round(cond, 2),
                "tail_clustering": round(tail, 4),
                "median_eigengap": med_gap,
                "tiny_gap_fraction": round(tiny_frac, 4),
                "dsd_vs_tikhonov_rel_frobenius": round(rel_fro, 5),
                "dsd_vs_tikhonov_max_filter_shape_diff": round(maxfilt, 5),
            }
            out["regimes"].append(cell)
            print(f"{label:22s} s2={s2:<7g} cond={cond:11.1f} tail={tail:.3f} "
                  f"tinygap={tiny_frac:.2f} DSDvsTik_fro={rel_fro:.4f} filtdiff={maxfilt:.4f}",
                  flush=True)

    path = os.path.join(RESULTS, "exp_P0_gp_clustering.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\nsaved -> {path}")

    # ---- automated verdict ----
    cells = out["regimes"]
    # GREEN condition: there EXISTS a realistic regime with strong clustering AND meaningful
    # DSD-vs-Tikhonov divergence (so DSD is not just Tikhonov in disguise).
    green = [c for c in cells if c["tail_clustering"] < 0.12
             and c["condition_number"] > 1e3
             and c["dsd_vs_tikhonov_rel_frobenius"] > 0.05]
    print("\n==== VERDICT INPUTS ====")
    print(f"regimes with strong clustering + real DSD/Tik divergence: {len(green)} / {len(cells)}")
    for c in green[:8]:
        print(f"  GREEN-ish: {c['regime']} s2={c['jitter_sigma2']} tail={c['tail_clustering']} "
              f"cond={c['condition_number']} fro={c['dsd_vs_tikhonov_rel_frobenius']}")
    verdict = "GREEN" if len(green) >= 1 else "RED"
    print(f"\nAUTO-VERDICT: {verdict}  (GREEN needs >=1 realistic regime where DSD meaningfully "
          f"differs from Tikhonov on a clustered, ill-conditioned GP matrix)")
    out["verdict"] = verdict
    out["green_regimes"] = green
    json.dump(out, open(path, "w"), indent=2, default=float)
    return out


if __name__ == "__main__":
    run()
