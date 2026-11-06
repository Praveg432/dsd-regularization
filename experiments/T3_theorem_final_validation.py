"""Deep validation of the final DSD characterization theorem (THEOREM-FINAL.md), trying hard to break
parts (a) boundedness, (b) Tikhonov recovery, (c) monotone ordering. Includes tightness + adversarial
parameter sweeps + a counterexample search.

Run: python experiments/T3_theorem_final_validation.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.dsd import compute_eigengaps

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def run():
    t0 = time.time(); rng = np.random.default_rng(0)
    out = {"experiment": "T3 deep validation of DSD characterization theorem"}

    # ---- (a) boundedness: phi <= 1/(2 sqrt d), and operator norm bound on real matrices ----
    a_viol = 0; a_tight_hits = 0; worst_rel = 0.0
    for _ in range(20000):
        lam = 10 ** rng.uniform(-6, 3); d = 10 ** rng.uniform(-8, 2)
        val = lam / (lam ** 2 + d); bound = 1.0 / (2 * np.sqrt(d))
        if val > bound * (1 + 1e-9): a_viol += 1
        worst_rel = max(worst_rel, (val - bound) / bound)
        if abs(lam - np.sqrt(d)) / np.sqrt(d) < 1e-3:  # near the predicted argmax
            if abs(val - bound) / bound < 1e-3: a_tight_hits += 1
    # operator-norm bound on random clustered PSD matrices
    opnorm_ok = True
    for s in range(200):
        m = 80
        ev = np.sort(np.concatenate([rng.uniform(1, 10, 8), 1 + 1e-3 * rng.standard_normal(40),
                                     rng.uniform(1e-3, 0.1, 32)]))
        gaps = compute_eigengaps(ev); alpha = 10 ** rng.uniform(-4, 0); beta = 10 ** rng.uniform(-1, 1)
        dmin = float(np.min(alpha * np.exp(-beta * gaps)))
        phi = ev / (ev ** 2 + alpha * np.exp(-beta * gaps))
        if phi.max() > 1.0 / (2 * np.sqrt(dmin)) * (1 + 1e-8): opnorm_ok = False
    out["part_a_boundedness"] = {"violations": int(a_viol), "tight_at_sqrt_d_confirmed": bool(a_tight_hits > 0),
                                 "max_rel_overshoot": round(float(worst_rel), 8),
                                 "operator_norm_bound_holds": bool(opnorm_ok),
                                 "holds": bool(a_viol == 0 and opnorm_ok)}

    # ---- (b) Tikhonov recovery at equal gaps AND beta->0 across random spectra ----
    b_worst = 0.0
    for s in range(500):
        m = rng.integers(20, 120); ev = np.sort(rng.uniform(0.01, 10, m)); alpha = 10 ** rng.uniform(-4, 0)
        # equal-gap case: construct exactly equal gaps
        delta = rng.uniform(0.1, 1.0); ev_eq = np.cumsum(np.r_[rng.uniform(0.1, 1), np.full(m - 1, delta)])
        gaps_eq = compute_eigengaps(ev_eq)
        gamma = alpha * np.exp(-rng.uniform(0.1, 5) * delta)
        # here enforce equal gaps exactly -> all damping equal to gamma
        beta = rng.uniform(0.1, 5)
        d_eq = alpha * np.exp(-beta * gaps_eq)
        phi_dsd = ev_eq / (ev_eq ** 2 + d_eq)
        phi_tik = ev_eq / (ev_eq ** 2 + alpha * np.exp(-beta * delta))
        b_worst = max(b_worst, float(np.max(np.abs(phi_dsd - phi_tik))))
        # beta->0 case
        d0 = alpha * np.exp(-1e-14 * compute_eigengaps(ev))
        b_worst = max(b_worst, float(np.max(np.abs(ev / (ev ** 2 + d0) - ev / (ev ** 2 + alpha)))))
    out["part_b_tikhonov_recovery"] = {"max_abs_diff": b_worst, "holds": bool(b_worst < 1e-8)}

    # ---- (c) monotone ordering: dphi/ddelta>0 across extreme param ranges + counterexample search ----
    c_viol = 0; checked = 0
    for _ in range(50000):
        lam = 10 ** rng.uniform(-6, 3); alpha = 10 ** rng.uniform(-8, 2); beta = 10 ** rng.uniform(-3, 3)
        g1 = 10 ** rng.uniform(-6, 1); g2 = g1 * (1 + 10 ** rng.uniform(-4, 1))  # g2 > g1
        p1 = lam / (lam ** 2 + alpha * np.exp(-beta * g1))
        p2 = lam / (lam ** 2 + alpha * np.exp(-beta * g2))
        checked += 1
        if not (p2 >= p1 - 1e-15):  # larger gap must give >= phi
            c_viol += 1
    out["part_c_monotone_ordering"] = {"checked": checked, "violations": int(c_viol),
                                       "holds": bool(c_viol == 0)}

    # ---- overall ----
    holds = all(out[k]["holds"] for k in ("part_a_boundedness", "part_b_tikhonov_recovery",
                                          "part_c_monotone_ordering"))
    out["theorem_fully_validated"] = bool(holds)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    print("Part (a) boundedness:", out["part_a_boundedness"])
    print("Part (b) Tikhonov recovery:", out["part_b_tikhonov_recovery"])
    print("Part (c) monotone ordering:", out["part_c_monotone_ordering"])
    print(f"\n==== THEOREM FULLY VALIDATED: {holds} ====")
    path = os.path.join(RESULTS, "exp_T3_theorem_final_validation.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"saved -> {path}")
    return out


if __name__ == "__main__":
    run()
