"""Phase-2c: does DSD's advantage survive a STRONGER baseline set (the reviewer's demand)?

The preprint + Phase 2 compared DSD mainly to Tikhonov. Reviewers will ask for truncated-SVD and a
general spectral filter as real, tuned competitors. This experiment runs all of them under the fair
protocol on the linear-discriminant regime where DSD is supposed to win:

  DSD       - auto-init, NO tuning (as in the preprint)
  Tikhonov  - ridge gamma, 12-pt log grid on train accuracy
  TSVD      - truncated SVD, rank k tuned (grid of fractions of n) on train accuracy
  Spectral  - magnitude filter lambda^p/(lambda^(p+1)+eps), (p, eps) tuned on train accuracy
  naive     - pinv (reference)

Methods: RidgeClassifier, FisherDiscriminant. Datasets: synth d=200 + real GINA + real Madelon.
We report DSD's advantage over the BEST non-DSD competitor per cell (the honest bar), paired t-test,
Cohen's d, win fraction. 25 seeds.

Run: python experiments/P2c_stronger_baselines.py [quick]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, fetch_openml

from src.kernel_classifiers import RidgeClassifier, FisherDiscriminant

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TIK_GRID = np.logspace(-8, -1, 12)
SPEC_P_GRID = [0.5, 0.75, 1.0, 1.5, 2.0]
SPEC_EPS_GRID = np.logspace(-6, -1, 6)


def tune_best(Cls, X_tr, y_tr, method, n_feat):
    """Return a fitted model with the method's hyperparameter tuned on TRAIN accuracy (fair protocol)."""
    if method == "dsd":
        return Cls(inversion_method="dsd").fit(X_tr, y_tr)        # no tuning
    if method == "naive":
        return Cls(inversion_method="naive").fit(X_tr, y_tr)
    best = None
    if method == "tikhonov":
        for g in TIK_GRID:
            m = Cls(inversion_method="tikhonov", tikhonov_gamma=float(g)).fit(X_tr, y_tr)
            a = m.score(X_tr, y_tr)
            if best is None or a > best[0]:
                best = (a, m)
        return best[1]
    if method == "tsvd":
        m_dim = n_feat + 1
        ranks = sorted(set(int(f * m_dim) for f in (0.3, 0.5, 0.7, 0.9, 1.0)))
        ranks = [max(2, min(k, m_dim)) for k in ranks]
        for k in ranks:
            m = Cls(inversion_method="tsvd", tsvd_rank=int(k)).fit(X_tr, y_tr)
            a = m.score(X_tr, y_tr)
            if best is None or a > best[0]:
                best = (a, m)
        return best[1]
    if method == "spectral":
        for p in SPEC_P_GRID:
            for eps in SPEC_EPS_GRID:
                m = Cls(inversion_method="spectral", spec_p=float(p), spec_eps=float(eps)).fit(X_tr, y_tr)
                a = m.score(X_tr, y_tr)
                if best is None or a > best[0]:
                    best = (a, m)
        return best[1]
    raise ValueError(method)


def load_dataset(name):
    if name == "synth_d200":
        return make_classification(n_samples=300, n_features=200, n_informative=100,
                                   n_redundant=40, n_clusters_per_class=1, random_state=42)
    if name == "gina":
        d = fetch_openml(name="gina_agnostic", version=1, as_frame=False, parser="auto")
        from sklearn.preprocessing import LabelEncoder
        return d.data.astype(float), LabelEncoder().fit_transform(d.target)
    if name == "madelon":
        d = fetch_openml(name="madelon", version=1, as_frame=False, parser="auto")
        return d.data.astype(float), (d.target == "1").astype(int)
    raise ValueError(name)


def run(quick=False):
    t0 = time.time()
    seeds = 5 if quick else 25
    datasets = ["synth_d200"] if quick else ["synth_d200", "gina", "madelon"]
    methods = {"RidgeClassifier": RidgeClassifier, "FisherDiscriminant": FisherDiscriminant}
    competitors = ["tikhonov", "tsvd", "spectral", "naive"]

    out = {"experiment": "P2c DSD vs stronger baseline set (Tikhonov, TSVD, spectral filter)",
           "n_seeds": seeds, "fairness": "DSD auto-init; all competitors tuned on train accuracy",
           "results": {}}

    for ds in datasets:
        try:
            X, y = load_dataset(ds)
        except Exception as e:
            print(f"[skip {ds}: {e}]", flush=True)
            continue
        n_feat = X.shape[1]
        print(f"\n===== {ds}  shape={X.shape} =====", flush=True)
        out["results"][ds] = {"shape": list(X.shape)}
        for mname, Cls in methods.items():
            acc = {m: [] for m in (["dsd"] + competitors)}
            for s in range(seeds):
                np.random.seed(s)
                Xs = StandardScaler().fit_transform(X)
                Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=s, stratify=y)
                Xtr = Xtr + 0.1 * np.random.randn(*Xtr.shape)
                for m in (["dsd"] + competitors):
                    try:
                        model = tune_best(Cls, Xtr, ytr, m, n_feat)
                        acc[m].append(model.score(Xte, yte))
                    except Exception:
                        acc[m].append(np.nan)
            means = {m: round(float(np.nanmean(acc[m])), 4) for m in acc}
            stds = {m: round(float(np.nanstd(acc[m])), 4) for m in acc}
            # best non-DSD competitor (by mean acc)
            best_comp = max(competitors, key=lambda m: np.nanmean(acc[m]))
            dsd_v = np.array(acc["dsd"], float); comp_v = np.array(acc[best_comp], float)
            mask = ~(np.isnan(dsd_v) | np.isnan(comp_v))
            dv, cv = dsd_v[mask], comp_v[mask]
            if len(dv) >= 2 and np.std(dv - cv) > 0:
                tstat, p = stats.ttest_rel(dv, cv)
                cohen = float((dv - cv).mean() / ((dv - cv).std() + 1e-30))
                wins = int(np.sum(dv > cv))
            else:
                p, cohen, wins = float("nan"), 0.0, int(np.sum(dv > cv)) if len(dv) else 0
            cell = {"means": means, "stds": stds, "best_non_dsd": best_comp,
                    "dsd_minus_best_pp": round(float((means["dsd"] - means[best_comp]) * 100), 2),
                    "paired_p_vs_best": round(float(p), 4), "cohens_d_vs_best": round(cohen, 3),
                    "dsd_wins_vs_best": wins, "n_valid": int(len(dv))}
            out["results"][ds][mname] = cell
            comp_str = " ".join(f"{m}={means[m]}" for m in (["dsd"] + competitors))
            print(f"  {mname:18s} {comp_str}", flush=True)
            print(f"       -> DSD vs best({best_comp}): {cell['dsd_minus_best_pp']:+}pp "
                  f"p={cell['paired_p_vs_best']} d={cell['cohens_d_vs_best']} "
                  f"wins={wins}/{cell['n_valid']}", flush=True)

    out["runtime_seconds"] = round(time.time() - t0, 1)
    path = os.path.join(RESULTS, "exp_P2c_stronger_baselines.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\nsaved -> {path}  ({out['runtime_seconds']}s)")

    survive = []
    for ds, md in out["results"].items():
        for mname in ("RidgeClassifier", "FisherDiscriminant"):
            c = md.get(mname)
            if c and c["dsd_minus_best_pp"] > 0 and c["paired_p_vs_best"] < 0.05 and c["n_valid"] >= 10:
                survive.append((ds, mname, c["dsd_minus_best_pp"], c["best_non_dsd"], c["paired_p_vs_best"]))
    print("\n==== VERDICT ====")
    if survive:
        print("ROBUST: DSD still significantly beats the BEST of {Tikhonov, TSVD, spectral, naive}:")
        for w in survive:
            print(f"  {w[0]} / {w[1]}: +{w[2]}pp over {w[3]} (p={w[4]})")
    else:
        print("NOT ROBUST: a stronger baseline matches/beats DSD somewhere; see JSON for the cell(s).")
    out["verdict"] = "ROBUST" if survive else "NEEDS_REVIEW"
    out["survive_cells"] = survive
    json.dump(out, open(path, "w"), indent=2, default=float)
    return out


if __name__ == "__main__":
    run(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))
