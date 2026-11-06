"""Phase-2 pivot gate: does DSD generalize to OTHER classification kernel methods under fair tuning?

Tests whether DSD (auto-init, no tuning) beats grid-searched Tikhonov for two additional classifiers
whose system matrices are ill-conditioned / tail-clustered at high d:
    - RidgeClassifier    : inverts X_aug^T X_aug  (d+1 x d+1)
    - FisherDiscriminant : inverts within-class scatter S_w  (d x d)

Fairness protocol identical to the preprint: DSD gets auto-init only; Tikhonov gets a 15-point log
grid (selected on TRAIN accuracy, as in experiments/10). We also log the system-matrix tail-clustering
to confirm the high-d regime holds for these methods (it should, same product-matrix family as LSTSVM).

Datasets: synthetic d=200 (fast, controllable), plus real GINA (d=970) and Madelon (d=500) via OpenML
if reachable. Metric: test classification accuracy. Paired t-test DSD vs Tikhonov, Cohen's d.

Run: python experiments/P2_breadth_classification.py [quick]
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
TIK_GRID = np.logspace(-8, -1, 15)


def tail_clustering(M):
    ev = np.linalg.eigvalsh((M + M.T) / 2)
    ev = ev[ev > 1e-12]
    if len(ev) < 3:
        return float("nan"), float("nan")
    cond = float(ev[-1] / ev[0])
    mid = len(ev) // 2
    rng = ev[-1] - ev[0]
    tail = float((ev[mid] - ev[0]) / rng) if rng > 0 else 1.0
    return cond, tail


def eval_method(ClassCls, X_tr, y_tr, X_te, y_te):
    """DSD (auto) vs Tikhonov (grid on train acc) vs naive. Returns (dsd_acc, tik_acc, naive_acc)."""
    # DSD auto-init
    try:
        dsd_acc = ClassCls(inversion_method="dsd").fit(X_tr, y_tr).score(X_te, y_te)
    except Exception:
        dsd_acc = np.nan
    # Tikhonov grid (select on TRAIN accuracy, same as preprint exp 10)
    best_g, best_a = TIK_GRID[0], -1
    for g in TIK_GRID:
        try:
            m = ClassCls(inversion_method="tikhonov", tikhonov_gamma=float(g)).fit(X_tr, y_tr)
            a = m.score(X_tr, y_tr)
            if a > best_a:
                best_a, best_g = a, g
        except Exception:
            continue
    tik_acc = ClassCls(inversion_method="tikhonov", tikhonov_gamma=float(best_g)).fit(X_tr, y_tr).score(X_te, y_te)
    # naive
    try:
        naive_acc = ClassCls(inversion_method="naive").fit(X_tr, y_tr).score(X_te, y_te)
    except Exception:
        naive_acc = np.nan
    return dsd_acc, tik_acc, naive_acc


def load_dataset(name):
    if name == "synth_d200":
        X, y = make_classification(n_samples=300, n_features=200, n_informative=100,
                                   n_redundant=40, n_clusters_per_class=1, random_state=42)
        return X, y
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

    out = {"experiment": "P2 DSD breadth across classification kernel methods",
           "n_seeds": seeds, "fairness": "DSD auto-init; Tikhonov 15-pt grid on train acc",
           "results": {}}

    for ds in datasets:
        try:
            X, y = load_dataset(ds)
        except Exception as e:
            print(f"[skip {ds}: {e}]", flush=True)
            continue
        print(f"\n===== {ds}  shape={X.shape} =====", flush=True)
        out["results"][ds] = {"shape": list(X.shape)}
        for mname, Cls in methods.items():
            dsd_a, tik_a, naive_a, tails, conds = [], [], [], [], []
            for s in range(seeds):
                np.random.seed(s)
                Xs = StandardScaler().fit_transform(X)
                Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=s, stratify=y)
                Xtr = Xtr + 0.1 * np.random.randn(*Xtr.shape)   # match preprint's noise injection
                # log clustering of the actual system matrix this method inverts (seed 0 only)
                if s == 0:
                    if mname == "RidgeClassifier":
                        Xa = np.hstack([Xtr, np.ones((len(Xtr), 1))]); M = Xa.T @ Xa
                    else:
                        A = Xtr[ytr == np.unique(ytr)[0]]; B = Xtr[ytr == np.unique(ytr)[1]]
                        M = (A - A.mean(0)).T @ (A - A.mean(0)) + (B - B.mean(0)).T @ (B - B.mean(0))
                    c, tl = tail_clustering(M); conds.append(c); tails.append(tl)
                da, ta, na = eval_method(Cls, Xtr, ytr, Xte, yte)
                dsd_a.append(da); tik_a.append(ta); naive_a.append(na)
            dsd_a = np.array(dsd_a, float); tik_a = np.array(tik_a, float)
            mask = ~(np.isnan(dsd_a) | np.isnan(tik_a))
            da_m, ta_m = dsd_a[mask], tik_a[mask]
            if len(da_m) >= 2:
                tstat, p = stats.ttest_rel(da_m, ta_m)
                diff = da_m - ta_m
                cohen = float(diff.mean() / (diff.std() + 1e-30))
                wins = int(np.sum(da_m > ta_m))
            else:
                p, cohen, wins = float("nan"), float("nan"), 0
            cell = {
                "system_matrix_condition": round(conds[0], 1) if conds else None,
                "system_matrix_tail_clustering": round(tails[0], 4) if tails else None,
                "dsd_acc_mean_std": [round(float(np.nanmean(dsd_a)), 4), round(float(np.nanstd(dsd_a)), 4)],
                "tikhonov_acc_mean_std": [round(float(np.nanmean(tik_a)), 4), round(float(np.nanstd(tik_a)), 4)],
                "naive_acc_mean": round(float(np.nanmean(naive_a)), 4),
                "advantage_pp": round(float((np.nanmean(dsd_a) - np.nanmean(tik_a)) * 100), 2),
                "paired_p": round(float(p), 4), "cohens_d": round(cohen, 3),
                "dsd_wins": wins, "n_valid": int(len(da_m)),
            }
            out["results"][ds][mname] = cell
            print(f"  {mname:18s} tail={cell['system_matrix_tail_clustering']} "
                  f"cond={cell['system_matrix_condition']} | DSD {cell['dsd_acc_mean_std'][0]} vs "
                  f"Tik {cell['tikhonov_acc_mean_std'][0]} (Δ={cell['advantage_pp']}pp, p={cell['paired_p']}, "
                  f"wins={wins}/{cell['n_valid']})", flush=True)

    out["runtime_seconds"] = round(time.time() - t0, 1)
    path = os.path.join(RESULTS, "exp_P2_breadth_classification.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\nsaved -> {path}  ({out['runtime_seconds']}s)")

    # verdict
    wins_any = []
    for ds, md in out["results"].items():
        for mname in ("RidgeClassifier", "FisherDiscriminant"):
            c = md.get(mname)
            if c and c["advantage_pp"] > 0 and c["paired_p"] < 0.05 and c["n_valid"] >= 10:
                wins_any.append((ds, mname, c["advantage_pp"], c["paired_p"]))
    print("\n==== VERDICT ====")
    if wins_any:
        print("GREEN: DSD significantly helps >=1 additional classification method under fair tuning:")
        for w in wins_any:
            print(f"  {w[0]} / {w[1]}: +{w[2]}pp (p={w[3]})")
    else:
        print("RED/AMBER: DSD does not significantly beat tuned Tikhonov on the new methods.")
    out["verdict"] = "GREEN" if wins_any else "RED_OR_AMBER"
    out["green_cells"] = wins_any
    json.dump(out, open(path, "w"), indent=2, default=float)
    return out


if __name__ == "__main__":
    run(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))
