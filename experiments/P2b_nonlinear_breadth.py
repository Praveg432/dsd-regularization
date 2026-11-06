"""Phase-2b: close the collinearity caveat with a GENUINELY DIFFERENT (non-linear) kernel method.

Ridge and Fisher (Phase 2) are nearly-collinear linear discriminants, so they are ~1.5 independent
pieces of evidence. This experiment adds non-linear methods whose system matrices are actual RBF-Gram
matrices (a different spectral regime from linear product matrices): the existing KernelLSTSVM and
LSSVM. A DSD win here is independent evidence that DSD's benefit is a property of the ill-conditioned
inversion regime, not of any one objective.

Fair protocol (identical to the preprint): DSD auto-init only; Tikhonov gets a 15-point log grid
selected on TRAIN accuracy; naive = pinv. Metric: test classification accuracy. We also log the
kernel system-matrix tail-clustering + condition number to characterise the regime.

Datasets: synthetic high-d (d=200) + real GINA (d=970) + real Madelon (d=500). Kernel methods build
n x n matrices, so we subsample each class to keep sizes tractable. RBF bandwidth gamma = 1/d
(standard default), which for high-d standardized data gives a moderately sharp kernel.

Run: python experiments/P2b_nonlinear_breadth.py [quick]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, fetch_openml

from src.kernel_lstsvm import KernelLSTSVM
from src.lssvm import LSSVM

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TIK_GRID = np.logspace(-8, -1, 15)
MAX_PER_CLASS = 300          # cap class size so kernel matrices stay tractable (<=600 total train)


def tail_clustering_rbf(X, gamma):
    """Tail-clustering + condition of an RBF Gram matrix on X (the matrix kernel methods invert)."""
    d2 = cdist(X, X, metric="sqeuclidean")
    K = np.exp(-gamma * d2)
    ev = np.linalg.eigvalsh((K + K.T) / 2)
    ev = ev[ev > 1e-12]
    if len(ev) < 3:
        return float("nan"), float("nan")
    cond = float(ev[-1] / ev[0])
    mid = len(ev) // 2
    rng = ev[-1] - ev[0]
    tail = float((ev[mid] - ev[0]) / rng) if rng > 0 else 1.0
    return cond, tail


def subsample(X, y, max_per_class, seed):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        take = min(len(ci), max_per_class)
        idx.extend(rng.choice(ci, size=take, replace=False))
    idx = np.array(idx)
    return X[idx], y[idx]


def eval_method(make_model, X_tr, y_tr, X_te, y_te, gamma):
    """DSD (auto) vs Tikhonov (grid on train acc) vs naive. make_model(method, tik_gamma)->estimator."""
    try:
        dsd_acc = make_model("dsd", 1e-3).fit(X_tr, y_tr).score(X_te, y_te)
    except Exception:
        dsd_acc = np.nan
    best_g, best_a = TIK_GRID[0], -1
    for g in TIK_GRID:
        try:
            a = make_model("tikhonov", float(g)).fit(X_tr, y_tr).score(X_tr, y_tr)
            if a > best_a:
                best_a, best_g = a, g
        except Exception:
            continue
    try:
        tik_acc = make_model("tikhonov", float(best_g)).fit(X_tr, y_tr).score(X_te, y_te)
    except Exception:
        tik_acc = np.nan
    try:
        naive_acc = make_model("naive", 1e-3).fit(X_tr, y_tr).score(X_te, y_te)
    except Exception:
        naive_acc = np.nan
    return dsd_acc, tik_acc, naive_acc


def load_dataset(name):
    if name == "synth_d200":
        X, y = make_classification(n_samples=700, n_features=200, n_informative=100,
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
    seeds = 4 if quick else 15
    datasets = ["synth_d200"] if quick else ["synth_d200", "gina", "madelon"]

    def kernel_lstsvm_factory(gamma):
        return lambda method, tg: KernelLSTSVM(c1=1.0, c2=1.0, gamma=gamma,
                                               inversion_method=method, tikhonov_gamma=tg)

    def lssvm_factory(gamma):
        return lambda method, tg: LSSVM(gamma=gamma, reg_param=1.0,
                                        inversion_method=method, tikhonov_gamma=tg)

    methods = {"KernelLSTSVM": kernel_lstsvm_factory, "LSSVM": lssvm_factory}

    out = {"experiment": "P2b DSD breadth on NON-LINEAR kernel methods (RBF Gram regime)",
           "n_seeds": seeds, "max_per_class": MAX_PER_CLASS,
           "fairness": "DSD auto-init; Tikhonov 15-pt grid on train acc; gamma=1/d",
           "results": {}}

    for ds in datasets:
        try:
            X, y = load_dataset(ds)
        except Exception as e:
            print(f"[skip {ds}: {e}]", flush=True)
            continue
        d = X.shape[1]
        gamma = 1.0 / d
        print(f"\n===== {ds}  shape={X.shape}  gamma=1/d={gamma:.4g} =====", flush=True)
        out["results"][ds] = {"shape": list(X.shape), "gamma": gamma}
        for mname, factory in methods.items():
            dsd_a, tik_a, naive_a, tails, conds = [], [], [], [], []
            for s in range(seeds):
                np.random.seed(s)
                Xs = StandardScaler().fit_transform(X)
                Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=s, stratify=y)
                Xtr, ytr = subsample(Xtr, ytr, MAX_PER_CLASS, s)
                # cap test too for speed on big sets
                if len(Xte) > 400:
                    Xte, yte = subsample(Xte, yte, 200, s + 999)
                if s == 0:
                    c, tl = tail_clustering_rbf(Xtr, gamma); conds.append(c); tails.append(tl)
                mk = factory(gamma)
                da, ta, na = eval_method(mk, Xtr, ytr, Xte, yte, gamma)
                dsd_a.append(da); tik_a.append(ta); naive_a.append(na)
            dsd_a = np.array(dsd_a, float); tik_a = np.array(tik_a, float)
            mask = ~(np.isnan(dsd_a) | np.isnan(tik_a))
            da_m, ta_m = dsd_a[mask], tik_a[mask]
            if len(da_m) >= 2 and np.std(da_m - ta_m) > 0:
                tstat, p = stats.ttest_rel(da_m, ta_m)
                diff = da_m - ta_m
                cohen = float(diff.mean() / (diff.std() + 1e-30))
                wins = int(np.sum(da_m > ta_m))
            else:
                p = float("nan") if len(da_m) < 2 else 1.0
                cohen = 0.0
                wins = int(np.sum(da_m > ta_m)) if len(da_m) else 0
            cell = {
                "kernel_matrix_condition": round(conds[0], 1) if conds else None,
                "kernel_matrix_tail_clustering": round(tails[0], 4) if tails else None,
                "dsd_acc_mean_std": [round(float(np.nanmean(dsd_a)), 4), round(float(np.nanstd(dsd_a)), 4)],
                "tikhonov_acc_mean_std": [round(float(np.nanmean(tik_a)), 4), round(float(np.nanstd(tik_a)), 4)],
                "naive_acc_mean": round(float(np.nanmean(naive_a)), 4),
                "advantage_pp": round(float((np.nanmean(dsd_a) - np.nanmean(tik_a)) * 100), 2),
                "paired_p": round(float(p), 4), "cohens_d": round(cohen, 3),
                "dsd_wins": wins, "n_valid": int(len(da_m)),
            }
            out["results"][ds][mname] = cell
            print(f"  {mname:14s} Ktail={cell['kernel_matrix_tail_clustering']} "
                  f"Kcond={cell['kernel_matrix_condition']} | DSD {cell['dsd_acc_mean_std'][0]} vs "
                  f"Tik {cell['tikhonov_acc_mean_std'][0]} (Δ={cell['advantage_pp']}pp, "
                  f"p={cell['paired_p']}, wins={wins}/{cell['n_valid']})", flush=True)

    out["runtime_seconds"] = round(time.time() - t0, 1)
    path = os.path.join(RESULTS, "exp_P2b_nonlinear_breadth.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\nsaved -> {path}  ({out['runtime_seconds']}s)")

    green = []
    for ds, md in out["results"].items():
        for mname in ("KernelLSTSVM", "LSSVM"):
            c = md.get(mname)
            if c and c["advantage_pp"] > 0 and (c["paired_p"] < 0.05) and c["n_valid"] >= 8:
                green.append((ds, mname, c["advantage_pp"], c["paired_p"]))
    print("\n==== VERDICT ====")
    if green:
        print("GREEN: DSD significantly helps a NON-LINEAR kernel method under fair tuning:")
        for w in green:
            print(f"  {w[0]} / {w[1]}: +{w[2]}pp (p={w[3]})")
    else:
        print("AMBER/RED: DSD does not significantly beat tuned Tikhonov on the non-linear methods.")
        print("(Check whether the RBF-Gram regime is actually tail-clustered at gamma=1/d; if not,")
        print(" scope the breadth claim to the ill-conditioned product-matrix / linear-discriminant family.)")
    out["verdict"] = "GREEN" if green else "AMBER_OR_RED"
    out["green_cells"] = green
    json.dump(out, open(path, "w"), indent=2, default=float)
    return out


if __name__ == "__main__":
    run(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))
