# DSD Follow-up — Novelty Scan (pre-writeup gate, Nov 2026)

Scanned latest 2024-2026 work across four angles. Verdict: **the DSD breadth result and the
gap-monotone characterization theorem are NOT preempted**, with two prior works whose boundedness /
Tikhonov-recovery results overlap our C1/C2 parts and MUST be cited + differentiated (we do not claim
novelty on boundedness or Tikhonov-recovery alone; our novelty is the GAP-keyed ordering + the
combination + the high-d-classification breadth). Content rephrased for compliance.

## Closest prior works (must cite + differentiate)

1. **Chen, "A Spectral-Domain Pseudo-Inverse Construction Method..." (arXiv:2607.26951, 2026, math.NA).**
   Proves bounded stability and Moore-Penrose consistency of a spectral-domain pseudo-inverse, and is
   explicitly "numerically equivalent to zeroth-order Tikhonov." MAGNITUDE-based filter factors;
   geophysics/inverse-problems context; no eigengap, no reliability ordering, no classification.
   => Overlaps our Theorem parts (a) boundedness and (b) Tikhonov recovery IN SPIRIT (these are
   classical regularization-theory facts; cf. Engl-Hanke-Neubauer). We cite it and state plainly that
   our (a)/(b) are not claimed as novel; our novelty is part (c) the gap-monotone ordering and the
   composite operator, which this work does not have.

2. **SC-Net, "Interpretable Operator Learning... Adaptive Spectral Filtering" (arXiv:2603.20602, 2026).**
   Learns a pointwise spectral filter reweighting by SNR, "outperforms Oracle Tikhonov" on 1D inverse
   problems. MAGNITUDE/SNR-based, learned, inverse-problems context; not gap-based, not classification.
   => Same theme (adaptive spectral filter beats Tikhonov) in a different domain and on a different
   axis (magnitude/SNR vs gap). Cite as related adaptive-filtering work; differentiate on the gap axis.

3. **Velikanov-Panov-Yarotsky, "Generalization Error of Spectral Algorithms" (ICLR 2024, 2403.11696).**
   Generalization as a functional of a learning profile h(lambda) -- a MAGNITUDE filter class. Our
   gap-keyed filter sits OUTSIDE this class (part c is exactly a filter that is not a function of
   lambda alone). Cite as the magnitude-filter generalization framework we are complementary to.
   NB: this is also why our FAILED excess-risk theorem (T1) was right to be abandoned -- within the
   magnitude-filter world the separation collapses; we do not claim a result in their framework.

4. **Renewal Perspective on Spectral Regularization (arXiv:2605.30059, 2026).** Shows non-exponential
   renewal reset laws generate "alternative spectral filters" beyond scalar ridge. MAGNITUDE-domain,
   optimization-dynamics angle. Related framing (filters beyond ridge); not gap-based. Optional cite.

## Active twin-SVM / kernel-classification line (differentiate the breadth result)

M. Tanveer's group (IIT Indore) continues prolifically into 2025-2026: TRKM / Twin Restricted Kernel
Machines (2502.15759), Twin RKM Multiview (2512.15757), ADMM L1 LSTSVM (AIMS Math 2025), plus the
earlier Wave-TSVM / GL-TSVM / granular-ball LSTSVM. ALL target LOSS functions, robustness, data
representation, or multiview -- NONE address eigenGAP / eigenvector-reliability regularization of the
system-matrix inverse. => DSD remains orthogonal and composable with this entire line; no overlap with
our breadth claim (DSD helping Ridge/Fisher/LSTSVM via gap-adaptive inversion).

Also relevant to cite for high-d LDA: "Doubly Regularized LDA with Automatic Parameter Selection"
(R2LDA, 2004.13335) -- two magnitude ridge params; differentiate on the gap axis.

## Novelty verdict (honest)

- **Characterization theorem:** parts (a)+(b) are classical/known-in-spirit (cite Chen 2026 + EHN) --
  NOT claimed novel; part (c) gap-monotone reliability ordering + the composite characterization IS
  novel and unpreempted. Frame the theorem as "a characterization", with (c) as the new content.
- **Empirical breadth (DSD beats Tikhonov/TSVD/spectral on real GINA/Madelon across Ridge/Fisher/
  LSTSVM):** unpreempted; the active twin-SVM line does not touch gap-adaptive inversion.
- **Operating-envelope (4 honest boundaries):** unpreempted and distinctive (few papers map where
  their method fails).

## Consequence for the writeup
- Do NOT claim novelty on boundedness / Tikhonov-recovery; cite Chen 2026 and classical regularization
  theory, and present (a)/(b) as "we restate for completeness; the new element is (c)".
- Position against the magnitude-filter generalization framework (Velikanov et al.) explicitly: our
  filter is gap-keyed, outside h(lambda); we do NOT claim a generalization-error theorem there (and we
  say why, citing our own refuted attempt as honest scope).
- Keep the empirical breadth + operating envelope as the primary contribution; the theorem is the
  honest characterization, not an overclaimed guarantee.
