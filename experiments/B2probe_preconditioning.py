"""Strategy-B probe B2: does the eigengap-reliability idea help PRECONDITIONING (a big audience)?

Claim under test: a DSD-style spectral preconditioner (suppress small-gap / unreliable directions)
accelerates Conjugate Gradient on ill-conditioned, spectrally-clustered symmetric PD systems more than
matched-cost baselines (no precond, Jacobi/diagonal, Tikhonov-shift).

THE DECISIVE SUB-TEST (kill criterion): a preconditioner built from a FULL eigendecomposition is
pointless (if you have eigh(A) you can solve directly). So we test the APPROXIMATE form: build the
spectral preconditioner from a cheap TRUNCATED (top-r) eigendecomposition and check whether any
advantage SURVIVES. If the advantage needs exact full eigh, this is RED for the solver audience.

Fair protocol: Tikhonov shift and DSD use the SAME truncated spectral info; Jacobi uses the diagonal.
Metric: PCG iterations to reach relative residual tol (lower = better).

GREEN  = DSD-precond reduces PCG iterations vs the best matched-cost baseline on real clustered-spectrum
         systems AND the advantage survives with truncated (cheap) spectral info.
RED    = advantage only with full eigh, or no iteration reduction vs baselines.

Run: python experiments/B2probe_preconditioning.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from src.kernels import rbf_kernel_matrix
from src.dsd import compute_eigengaps, initialize_hyperparameters

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def make_system(kind, n, seed):
    """Return a symmetric PD matrix A with a clustered spectrum."""
    rng = np.random.default_rng(seed)
    if kind == "rbf":
        X = rng.standard_normal((n, 8)); X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        A = rbf_kernel_matrix(X, gamma=0.05) + 1e-6 * np.eye(n)
    elif kind == "clustered_synth":
        # explicit clustered spectrum: a few large eigs, a dense cluster of tiny eigs
        ev = np.concatenate([rng.uniform(5, 10, size=n // 10),
                             rng.uniform(0.9, 1.1, size=n // 3),
                             rng.uniform(1e-4, 5e-4, size=n - n // 10 - n // 3)])
        ev = np.sort(ev)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        A = (Q * ev) @ Q.T
        A = (A + A.T) / 2
    else:
        raise ValueError(kind)
    return A


def spectral_precond(A, mode, r, tik_shift=1e-3):
    """Build a preconditioner M^{-1} applied as a LinearOperator.

    mode:
      'none'      -> identity
      'jacobi'    -> 1/diag(A)
      'tik_trunc' -> truncated-eigsh Tikhonov: sum 1/(lam_i+shift) u u^T over top-r, + shift^-1 on rest
      'dsd_trunc' -> truncated-eigsh DSD: sum lam_i/(lam_i^2 + alpha exp(-beta gap_i)) u u^T over top-r
    r = number of eigenpairs used (truncated => cheap). 'dsd_full'/'tik_full' use all n (the impractical
    reference, to isolate whether the advantage needs full info).
    """
    n = A.shape[0]
    if mode == "none":
        return None, 0.0
    if mode == "jacobi":
        d = np.clip(np.diag(A), 1e-12, None)
        t0 = time.time()
        op = LinearOperator((n, n), matvec=lambda x: x / d)
        return op, time.time() - t0

    full = mode.endswith("_full")
    k = n if full else min(r, n)
    t0 = time.time()
    if full:
        ev, U = np.linalg.eigh(A)
    else:
        # cheap truncated: top-k by magnitude via eigh then slice (proxy for randomized/Lanczos;
        # we only USE k pairs, mimicking the cost of a truncated solver)
        ev_all, U_all = np.linalg.eigh(A)
        ev = ev_all[-k:]; U = U_all[:, -k:]
    ev = np.clip(ev, 1e-14, None)
    if mode.startswith("tik"):
        filt = 1.0 / (ev + tik_shift)
    elif mode.startswith("dsd"):
        gaps = compute_eigengaps(np.sort(ev))
        alpha, beta = initialize_hyperparameters(ev)
        damping = alpha * np.exp(-beta * np.clip(gaps, 0, 500.0 / max(beta, 1e-10)))
        filt = ev / (ev ** 2 + damping)
    else:
        raise ValueError(mode)
    # residual handling for truncated: apply mean filter value to the complement (so M^-1 is full-rank)
    if not full:
        resid = float(np.median(filt))
        U_full = U
        def mv(x):
            coeff = U_full.T @ x
            return U_full @ (filt * coeff) + resid * (x - U_full @ (U_full.T @ x))
    else:
        U_full = U
        def mv(x):
            return U_full @ (filt * (U_full.T @ x))
    build_t = time.time() - t0
    return LinearOperator((n, n), matvec=mv), build_t


def pcg_iters(A, b, M, tol=1e-6, maxiter=2000):
    count = {"n": 0}
    def cb(xk): count["n"] += 1
    Aop = LinearOperator(A.shape, matvec=lambda x: A @ x)
    x, info = cg(Aop, b, rtol=tol, maxiter=maxiter, M=M, callback=cb)
    return count["n"], info


def run():
    t0 = time.time()
    n = 400
    seeds = 10
    r = max(5, n // 10)   # truncated rank = 10% of n (the "cheap" regime)
    kinds = ["rbf", "clustered_synth"]
    modes = ["none", "jacobi", "tik_trunc", "dsd_trunc", "tik_full", "dsd_full"]

    out = {"experiment": "B2 probe: eigengap-aware preconditioning for CG", "n": n, "trunc_rank": r,
           "seeds": seeds, "note": "iterations to tol=1e-6; lower=better; _trunc uses top-10% eigs (cheap), "
                                   "_full uses all eigs (impractical reference)", "results": {}}

    for kind in kinds:
        print(f"\n===== system: {kind} (n={n}) =====", flush=True)
        iters = {m: [] for m in modes}
        conds = []
        for s in range(seeds):
            A = make_system(kind, n, s)
            ev = np.linalg.eigvalsh(A); ev = ev[ev > 1e-14]
            conds.append(float(ev[-1] / ev[0]))
            rng = np.random.default_rng(1000 + s)
            b = rng.standard_normal(n)
            for m in modes:
                M, _ = spectral_precond(A, m, r)
                it, info = pcg_iters(A, b, M)
                iters[m].append(it)
        res = {m: [int(np.median(iters[m])), int(np.mean(iters[m]))] for m in modes}
        out["results"][kind] = {"condition_number_median": round(float(np.median(conds)), 1),
                                "iters_median_mean": res}
        print(f"  cond(med)={np.median(conds):.1e}")
        for m in modes:
            print(f"    {m:12s} iters median={res[m][0]:5d} mean={res[m][1]:.1f}", flush=True)

    out["runtime_seconds"] = round(time.time() - t0, 1)
    path = os.path.join(RESULTS, "exp_B2probe_preconditioning.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\nsaved -> {path}")

    # verdict: DSD_trunc beats best matched-cost baseline (none/jacobi/tik_trunc) AND is close to dsd_full
    print("\n==== VERDICT ====")
    green = []
    for kind, md in out["results"].items():
        it = md["iters_median_mean"]
        dsd_tr = it["dsd_trunc"][0]
        baseline_best = min(it["none"][0], it["jacobi"][0], it["tik_trunc"][0])
        survives = dsd_tr < baseline_best          # cheap DSD beats best matched-cost baseline
        full_gap = it["dsd_full"][0]
        print(f"  {kind}: dsd_trunc={dsd_tr}  best_matched_baseline={baseline_best}  dsd_full={full_gap}  "
              f"{'BEATS baseline' if survives else 'no gain'}")
        if survives:
            green.append(kind)
    verdict = "GREEN" if green else "RED"
    print(f"\nAUTO-VERDICT (B2): {verdict}  "
          f"(GREEN needs cheap truncated DSD-precond to cut CG iters below the best matched-cost baseline)")
    out["verdict"] = verdict; out["green_systems"] = green
    json.dump(out, open(path, "w"), indent=2, default=float)
    return out


if __name__ == "__main__":
    run()
