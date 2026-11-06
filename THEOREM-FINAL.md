# DSD — Validated Theorem (operator characterization)

Survivors of the portfolio vetting (experiments/T2_theorem_portfolio.py -> exp_T2_theorem_portfolio.json):
C1 (bounded inverse), C2 (Tikhonov recovery), C4 (monotone reliability ordering). These are TRUE and
NON-TRIVIAL structural facts about the DSD operator. The three combine into one clean theorem that
rigorously CHARACTERIZES differential spectral damping. The competitive/stability candidates
(C3 DoF, C5 distributional, C6 stability) FAILED and are explicitly NOT claimed (see post-mortem).

Throughout: symmetric PSD matrix W = sum_i lambda_i u_i u_i^T, lambda_i > 0, ascending; local eigengap
delta_i = min_{j != i} |lambda_i - lambda_j|; DSD damping d_i = alpha exp(-beta delta_i), alpha,beta>0;
DSD regularized inverse eigenvalue phi_i = phi(lambda_i, delta_i) = lambda_i / (lambda_i^2 + d_i); DSD
regularized pseudo-inverse W_dsd^+ = sum_i phi_i u_i u_i^T.

---

## Theorem (DSD operator characterization)

Let W, {(lambda_i, delta_i)}, and W_dsd^+ be as above. Then:

**(a) Boundedness / well-posedness.** For every i, 0 < phi_i <= 1/(2 sqrt(d_i)), with equality iff
lambda_i = sqrt(d_i). Consequently the operator norm obeys
    || W_dsd^+ ||_2 = max_i phi_i <= 1 / (2 sqrt(min_i d_i)) = exp(beta * max_i delta_i / 2) / (2 sqrt(alpha)).
In particular, if lambda_i -> 0 with d_i bounded below by d_min > 0, then phi_i -> 0: the DSD inverse
stays bounded as W approaches singularity, whereas the unregularized inverse 1/lambda_i diverges.

**(b) Strict generalization of Tikhonov.** If all gaps are equal (delta_i = delta for all i), or in
the limit beta -> 0, then d_i = gamma := alpha exp(-beta delta) is constant and
phi_i = lambda_i / (lambda_i^2 + gamma), i.e. DSD coincides exactly with the (eigenvalue-domain)
Tikhonov-regularized inverse with ridge gamma. Thus DSD is a strict superset of Tikhonov, recovering
it precisely when the spectrum carries no gap structure.

**(c) Monotone reliability ordering.** For fixed lambda > 0, phi(lambda, delta) is strictly
increasing in delta (equivalently, the damping d is strictly decreasing in delta):
    d phi / d delta = lambda * alpha * beta * exp(-beta delta) / (lambda^2 + d)^2 > 0.
Hence among directions of equal eigenvalue magnitude, DSD assigns strictly larger effective inverse
weight to the larger-gap (Davis-Kahan more reliable) direction. No filter that is a function of
eigenvalue magnitude alone can realize this ordering, since it is constant on equal-magnitude
directions.

---

## Proof

**(a)** Fix i; write p(lambda) = lambda/(lambda^2 + d_i) for lambda>0, d_i>0. Then
p'(lambda) = (d_i - lambda^2)/(lambda^2+d_i)^2, which is positive for lambda<sqrt(d_i), zero at
lambda=sqrt(d_i), negative after; so p attains its unique maximum at lambda=sqrt(d_i) with value
sqrt(d_i)/(d_i+d_i) = 1/(2 sqrt(d_i)). Positivity is immediate. Since W_dsd^+ is symmetric with
eigenvalues {phi_i}, ||W_dsd^+||_2 = max_i phi_i <= max_i 1/(2 sqrt(d_i)) = 1/(2 sqrt(min_i d_i)).
Substituting d_i = alpha exp(-beta delta_i) gives min_i d_i = alpha exp(-beta max_i delta_i), hence the
closed form. If lambda_i -> 0 with d_i >= d_min, p(lambda_i) = lambda_i/(lambda_i^2+d_i) <=
lambda_i/d_min -> 0. QED(a).

**(b)** delta_i = delta for all i gives d_i = alpha exp(-beta delta) =: gamma, a constant, so
phi_i = lambda_i/(lambda_i^2 + gamma) for all i, which is the eigenvalue-domain Tikhonov inverse. The
beta->0 limit gives d_i -> alpha for every i identically. QED(b).

**(c)** With d = alpha exp(-beta delta), dd/d delta = -alpha beta exp(-beta delta) < 0, and
dphi/dd = -lambda/(lambda^2+d)^2 < 0, so by the chain rule dphi/d delta = (-lambda/(lambda^2+d)^2) *
(-alpha beta exp(-beta delta)) = lambda alpha beta exp(-beta delta)/(lambda^2+d)^2 > 0 for lambda>0.
Strict monotonicity in delta follows; constancy of any g(lambda) on equal-lambda directions is by
definition. QED(c).

---

## Numerical validation
experiments/T2_theorem_portfolio.py confirms: (a) max_lambda phi = 1/(2 sqrt d) to rel-err 1e-6 and
the operator-norm bound holds on random clustered matrices; (b) DSD == Tikhonov to 5e-13 at beta->0;
(c) phi strictly increasing in gap on all tested (lambda, gap) grids. Deeper tightness + counterexample
checks: experiments/T3_theorem_final_validation.py.

## What is explicitly NOT claimed (honest scope; refuted candidates)
- NO excess-risk superiority over magnitude filters (T1 refuted: gap/magnitude correlate under
  perturbation; advantage reverses at strong noise).
- NO effective-DoF tail-reduction advantage over matched Tikhonov (C3 refuted: DSD removed LESS tail
  DoF, 0/30 seeds).
- NO algorithmic-stability advantage (C6 refuted and REVERSED: DSD's leave-one-out sensitivity was
  ~12x LARGER than matched Tikhonov; this contradicts the preprint's conjectured stability claim, which
  we therefore retract as a conjecture).
- NO distributional separation from a fixed magnitude filter (C5: vacuous even by construction).

## Paper role
This is an honest, provable CHARACTERIZATION theorem: DSD is a bounded, well-posed, strict
generalization of Tikhonov that uniquely imposes a gap-monotone (reliability) weighting. It explains
WHAT DSD does that magnitude methods cannot (the monotone ordering, part c) and guarantees
well-posedness (part a), without over-claiming a performance guarantee the experiments refute. The
empirical performance (real GINA/Madelon wins over the full spectral-regularizer family) stands on its
own; this theorem characterizes the operator, and Davis-Kahan is cited as the MOTIVATION for why the
gap-monotone ordering is sensible (eigenvector reliability ~ 1/gap), not as a proven risk bound.
