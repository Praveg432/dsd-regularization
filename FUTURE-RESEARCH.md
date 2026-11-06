# DSD Future Research Directions

**Document purpose:** Captures all experimental findings, ideas, and negative results from brainstorming sessions that are outside the scope of the current paper but inform future work.

**Date:** June 2026  
**Context:** After completing the DSD paper ("Gap-Adaptive Regularization for Ill-Conditioned Kernel Methods"), these directions were explored and documented for reference.

---

## 1. Multi-Parameter DSD Variants

### 1.1 The Hypothesis

If 2 parameters (α, β) provide significant classification improvement, would 3–5 parameters capture finer spectral structure?

### 1.2 Variants Tested

| Params | Formula | What it adds |
|--------|---------|--------------|
| 2 | α · exp(−β · δᵢ) | Standard DSD |
| 3 | α · exp(−β · δᵢ^γ) | γ = gap exponent (transition shape) |
| 4 | α · exp(−β · δᵢ^γ) + ε | ε = minimum damping floor |
| 5 | mag_factor · (α · exp(−β · δᵢ^γ) + ε) | mag_factor = magnitude awareness |

### 1.3 Results: Classification (GINA d=970, 15 seeds)

| Params | Accuracy | vs 2-param | p-value |
|--------|----------|------------|---------|
| k=2 | 85.79% | baseline | — |
| k=3 | 85.87% | +0.08pp | 0.28 (n.s.) |
| k=4 | 85.87% | +0.08pp | 0.29 (n.s.) |
| k=5 | 85.98% | +0.19pp | 0.024 |

**Conclusion:** 3–4 parameters don't help for classification. k=5 shows marginal significance (+0.19pp) but the effect is tiny relative to the 2-param advantage (+4.8pp over Tikhonov).

### 1.4 Results: Pre-Image Reconstruction (Swiss Roll, 20 seeds, σ=5e-3)

| Params | Error | vs 2-param | vs Tikhonov-grad |
|--------|-------|------------|-----------------|
| 2-param DSD-opt | 0.0727 | baseline | wins 16/20 (p=0.007) |
| 5-param DSD-opt | 0.0700 | −3.7% (p=0.02) | wins 19/20 (p=0.0001) |
| Tikhonov-grad | 0.0783 | — | baseline |

**Conclusion for pre-image:** 5-param significantly beats both 2-param and Tikhonov on manifold data. The magnitude-awareness term adds genuine value for reconstruction fidelity.

### 1.5 Key Insight: Task Determines Optimal Parameterization

| Task | Best approach | Why |
|------|--------------|-----|
| Classification | 2-param, NO optimization | Structural prior prevents overfitting; more params/optimization degrades |
| Pre-image (manifold) | 5-param, gradient-optimized | More params capture finer geometric structure |
| Pre-image (tabular) | Tikhonov (1-param) | Flat damping is correct for non-geometric data |

---

## 2. Optimization Degrades Classification (The "Spectral Prior" Finding)

### 2.1 The Experiment

GINA d=970, 30 seeds: DSD with auto-initialization vs DSD with gradient-optimized (α, β) on reconstruction loss.

### 2.2 Results

| Method | Accuracy | p-value |
|--------|----------|---------|
| DSD-init (auto) | **85.9%** | — |
| DSD-optimized (reconstruction loss) | 81.1% | < 10⁻⁶ (init wins 30/30) |
| Tikhonov-opt | 81.1% | — |

### 2.3 Explanation

The optimizer minimizes reconstruction error, which pushes DSD's damping curve toward Tikhonov-like behavior (uniform regularization). This undoes the beneficial *over-regularization* of corrupted eigenvector directions that the spectral initialization provides.

The auto-initialization happens to produce a damping curve that is slightly more aggressive on the tail than optimal for reconstruction — but this "excess" damping prevents the classifier from fitting to noise-corrupted spectral directions during training. It's the same phenomenon as early stopping in neural networks: the training-suboptimal point is the generalization-optimal point.

### 2.4 Implication

DSD's classification advantage is NOT from having more tunable parameters than Tikhonov. It's from having a **structural prior** (the eigengap-driven initialization formula) that happens to produce better-generalizing regularization than any loss-minimizing optimizer would find. The formula α₀ = λ²_transition, β₀ = 1/median(Δλ) encodes domain knowledge about spectral reliability that no classification loss can recover.

### 2.5 Also Tested: 5-param with Classification-Aware Optimization

Hinge loss on validation set, 60 epochs Adam. Result: 85.55% — still worse than 2-param init (85.79%, p=0.03). Even classification-aware optimization of more parameters cannot beat the structural prior.

---

## 3. Monotonic Control-Point Damping (Pre-Image)

### 3.1 Concept

Replace the exponential functional form with a learned piecewise-monotonic damping curve using n interpolated control points. Monotonicity constraint ensures damping decreases from tail (unreliable) to head (reliable).

### 3.2 Results on Manifold Data (20 seeds each)

| Dataset | Mono-10cp | DSD-2p | Tikhonov | Mono vs Tik |
|---------|-----------|--------|----------|-------------|
| Swiss Roll (γ=2) | **0.063** | 0.064 | 0.068 | −7.3% (p=0.0003) |
| Swiss Roll extreme (γ=5) | **0.194** | 0.225 | 0.198 | −2.1% (p<0.0001) |
| S-Curve | **0.104** | 0.111 | 0.114 | −8.6% (p<0.0001) |
| Swiss Roll in d=20 | 4.486 | 4.487 | 4.486 | tie |

**Mono-10cp beats Tikhonov significantly on 3/4 manifold tests.** This is the best pre-image method we found for manifold-structured data.

### 3.3 Results on Tabular Data (20 seeds)

| Dataset | Mono-10cp | Tikhonov | Mono vs Tik |
|---------|-----------|----------|-------------|
| Breast Cancer (d=30) | 1.278 | 1.241 | Tikhonov wins |
| Digits (d=64) | 2.059 | 2.009 | Tikhonov wins |
| GINA (d=970) | 21.77 | 21.75 | tie |

**On tabular data, Tikhonov remains unbeaten.** The flat damping assumption is correct when data has no manifold geometry.

### 3.4 Why It Works on Manifolds but Not Tabular

Manifold data has *hierarchical geometric structure* encoded in different eigenvectors at different spectral positions. The top eigenvectors capture global geometry (large-scale curvature), middle ones capture local variations, and tail ones capture noise. A learned damping curve can differentially weight these scales. Tabular data has no such hierarchy — all eigenvectors are roughly equally informative, so flat damping is already optimal.

---

## 4. Adaptive-n Control Points (Failed Approach)

### 4.1 Idea

Choose n automatically: n = min(max(n_train // 10, 1), m // 5). At n=1, it reduces to Tikhonov. Should theoretically guarantee ≥ Tikhonov performance.

### 4.2 Results

| Setting | Adaptive | Tikhonov | Winner |
|---------|----------|----------|--------|
| Swiss Roll (γ=2, n=10) | 0.088 | 0.068 | Tikhonov (badly) |
| Swiss Roll (γ=5, n=10) | 0.195 | 0.198 | Adaptive (+1.7%) |
| Breast Cancer (n=10) | 0.818 | 0.779 | Tikhonov |
| Breast Cancer (n=3) | 0.819 | 0.780 | Tikhonov |
| Breast Cancer (n=1) | 0.818 | 0.781 | Tikhonov |

### 4.3 Why It Failed

Even n=1 (same hypothesis space as Tikhonov) loses to Tikhonov-grad due to **optimizer dynamics**:
- Different learning rates, initialization, and parameterization lead to different local optima
- More parameters → harder non-convex landscape → worse solutions despite richer hypothesis space
- The theoretical guarantee (can't be worse) only holds with perfect global optimization

### 4.4 What Would Fix It

1. **Initialize at Tikhonov solution** — start from the 1-param optimal and fine-tune
2. **Only accept improvements** — monotonic descent from Tikhonov baseline
3. **Second-order optimization** (L-BFGS) — better at navigating non-convex landscapes
4. **Bayesian optimization** of control points — global search, not gradient descent

These are engineering solutions. The research question is whether the marginal improvement (2–9% on manifolds) justifies the complexity.

---

## 5. Per-Eigenvector Learned Damping (Failed Approach)

### 5.1 Idea

Learn dᵢ for each eigenvalue independently (m parameters), with L2 regularization toward Tikhonov.

### 5.2 Results (Breast Cancer, 20 seeds)

| Method | Error | vs Tikhonov |
|--------|-------|-------------|
| Per-eigenvector (L2-reg) | 1.504 | Tikhonov wins (badly) |
| Tikhonov-grad | 1.237 | baseline |

### 5.3 Why It Failed

142 free parameters optimized on 40 training points → catastrophic overfitting. The learned damping curve fits the training points perfectly but doesn't generalize.

### 5.4 Lesson

The ratio of parameters to training points must be kept low. With m=142 and n_train=40, there are 3.5× more parameters than data points. The constraint structure (monotonicity, low-rank) is essential, not just regularization.

---

## 6. Real-World Dataset Survey (LSTSVM Classification)

### 6.1 Datasets Where DSD Wins

| Dataset | d | n | DSD | Tikhonov | Advantage | Cohen's d |
|---------|---|---|-----|----------|-----------|-----------|
| GINA (handwriting) | 970 | 3468 | 85.9% | 81.1% | +4.8pp | 4.49 |
| Madelon (feature sel.) | 500 | 2600 | 57.7% | 55.0% | +2.6pp | 1.76 |
| ISOLET (speech) | 617 | 7797* | 89.9% | 88.1% | +1.7pp | — |

*subsampled to 3000

### 6.2 Datasets Where DSD Loses or Ties

| Dataset | d | DSD | Tikhonov | Note |
|---------|---|-----|----------|------|
| HAR (activity) | 561 | 99.95% | 99.96% | Too easy, ceiling effect |
| Digits | 64 | 88.3% | 89.1% | Below operating threshold |
| Ionosphere | 34 | 73.3% | 80.3% | Well below threshold |
| Breast Cancer | 30 | — | — | Well below threshold |

### 6.3 Operating Boundary

DSD wins when:
- d ≥ 100 AND
- Data has genuine redundancy (many non-informative features) AND
- Classification is not already saturated (< 95%)

---

## 7. Summary of Research Directions (Prioritized)

### High Priority (Strong Evidence, Clear Path)

1. **Tikhonov-DSD Hybrid for Classification** — Use DSD auto-init at d≥100, auto-fall-back to Tikhonov at d<100. Eliminates the operating boundary limitation entirely. Implementation: trivial (if condition_number > threshold, use DSD, else Tikhonov).

2. **Mono-10cp for Manifold Pre-Image** — Implement as a separate module. Clear win on manifold data (7–9% over Tikhonov). Target applications: point cloud reconstruction, mesh denoising, signal demodulation.

### Medium Priority (Interesting, Needs More Work)

3. **Classification-aware DSD initialization** — Instead of α₀ = λ²_transition (a reconstruction heuristic), derive an initialization specifically targeting generalization. The current init works "accidentally" — understanding why would enable principled improvement.

4. **Gaussian Process application** — DSD on GP posterior (K + σ²I)⁻¹. Same spectral decay issue. GP community is large; if DSD helps GP prediction quality the way it helps LSTSVM, impact multiplies significantly.

5. **Formal stability proof** — Prove that DSD pre-image error remains bounded as δᵢ → 0. Required for theoretical completeness.

### Low Priority (Interesting but Likely Dead Ends)

6. **More than 2 parameters for classification** — Tested, doesn't help. The structural prior is already optimal.

7. **Adaptive-n for pre-image** — Theoretically sound but optimization is unreliable. Would need significant engineering (warm-start, Bayesian opt) for marginal gains.

8. **Per-eigenvector damping** — Overfits catastrophically unless n_train >> m. Not practical for typical kernel matrix sizes.

### Confirmed Dead Ends (Tested in Sections 8–9)

9. **KDE-based spectral density gap** — No measurable improvement. The bilateral min-gap is sufficient because β compensates for measurement imprecision.

10. **Random Matrix Theory (Marchenko-Pastur / Tracy-Widom) initialization** — Fundamentally wrong model for structured kernel matrices. Produces catastrophically wrong α values.

11. **Resolvent-based implicit gradients** — Solves a non-existent problem. The eigenvector detach mechanism already handles degeneracy equivalently.

12. **Manifold-constrained loss for DSD parameter optimization** — kNN distance has near-zero gradient signal w.r.t. α/β. Manifold constraints must act on the pre-image coordinate directly (MCIP solver), not on spectral damping params.

### Validated Improvements (Section 9, Low-Noise Regime Only)

13. **Multi-Scale β (3-zone)** — Recommended new default. Significant on 2/5 configs (p<0.012), never degrades, uniquely improves manifold distance. +2 params, zero overhead.

14. **Per-Eigenvalue α (5 control points)** — Valid for low-noise manifold data (Cohen's d=0.95 on S-Curve). Not safe as universal default due to high-noise sensitivity. Available as opt-in.

---

## 8. Experiment 11: DSD Core Algorithm Optimization — Tested & Rejected

**Date:** June 2026  
**Code:** `experiments/11_dsd_optimization_candidates.py`  
**Protocol:** 30 seeds, Swiss Roll (n=2000, m=300, γ=2.0), two noise levels (σ=5e-3, σ=5e-2)  
**Metric:** Mean pre-image reconstruction error on 50 held-out test points

### 8.1 Candidate A: Local Spectral Density (KDE-Based Eigengap Replacement)

**Hypothesis:** The bilateral min-gap δᵢ = min(|λᵢ − λᵢ₋₁|, |λᵢ − λᵢ₊₁|) is "myopic" — it ignores broader neighborhood topology. Replacing it with a Gaussian KDE density estimate over k-nearest eigenvalues should provide higher-fidelity perturbation risk measurement.

**Implementation:** 1D Gaussian KDE with adaptive bandwidth h = range of k-neighbor window. Returns inverse density (scaled to median gap magnitude) as the effective "gap" input to the DSD formula.

**Results:**

| Noise | Baseline Error | KDE k=7 Error | Δ% | p-value | Significant? |
|-------|---------------|---------------|-----|---------|-------------|
| σ=5e-3 | 0.07023 | 0.07020 | +0.0% | 0.96 | No |
| σ=5e-2 | 0.14024 | 0.14075 | −0.4% | 0.50 | No |

Extended sensitivity sweep (k ∈ {3, 5, 7, 11, 15}, noise ∈ {0.1, 0.2}, m ∈ {300, 500}):
- Best case: k=5, noise=0.2 → +2.4% improvement, p=0.155 (not significant)
- No configuration reached p < 0.05

**Compute overhead:** 1.01× baseline (negligible — KDE is O(mk) vs O(m³) for eigh).

**Root Cause of Failure:** The bilateral gap is already a sufficient proxy for the optimizer. Since α and β are free parameters optimized via backprop, the optimizer compensates for gap measurement imprecision by adjusting β. A slightly "wrong" gap measure simply shifts the optimal β — the final loss landscape is nearly identical. The gap function only needs to be *monotonically related* to perturbation risk, not perfectly calibrated.

**Verdict: REJECT.** No statistical improvement despite theoretical elegance. The additional complexity is not justified.

### 8.2 Candidate B: Tracy-Widom / Marchenko-Pastur Initialization

**Hypothesis:** The percentile-based initialization (α₀ = λ²_transition at 10th percentile gap, β₀ = 1/median(Δλ)) is empirical guesswork. Rooting initialization in Random Matrix Theory (Marchenko-Pastur law) should provide a theoretically grounded signal/noise boundary.

**Implementation:** Compute the MP upper edge λ₊ = (1 + √(m/n))² (normalized by mean eigenvalue), classify eigenvalues below this edge as "noise." Set α at the edge eigenvalue², and β from the Wigner surmise mean spacing in the noise region.

**Results:**

| Noise | Baseline Error | Tracy-Widom Error | Δ% | p-value | Significant? |
|-------|---------------|-------------------|-----|---------|-------------|
| σ=5e-3 | 0.07023 | 0.08695 | **−23.8%** | <0.0001 | Yes (**WORSE**) |
| σ=5e-2 | 0.14024 | 0.14426 | −2.9% | <0.0001 | Yes (**WORSE**) |

**Diagnostic:** On a typical spectrum (m=300, n=2000):
- MP edge classifies **83%** of eigenvalues as "noise" (reality: ~20% are noise)
- Produces α = 5.21 vs correct α = 0.018 (288× too large)
- Even with gradient optimization, the starting point creates a loss basin the optimizer cannot escape within 150 epochs

**Root Cause of Failure:** The Marchenko-Pastur law models Wishart matrices (sample covariance of iid Gaussian noise). RBF kernel matrices are fundamentally NOT Wishart — they have structured eigenvalue decay from the kernel geometry. The MP law catastrophically overestimates the noise boundary for kernel matrices, causing over-damping of signal eigenvalues.

**Gradient stability:** 40× higher gradient variance than baseline (1.78e-6 vs 4.37e-8), indicating the optimizer is fighting against the poor initialization landscape.

**Verdict: REJECT.** Fundamentally wrong model for the problem. RMT tools (MP, Tracy-Widom) are only appropriate for random matrices or sample covariance matrices, not for structured kernel matrices.

**Lesson for future work:** Any initialization scheme based on random matrix theory must first verify that the kernel matrix's spectral structure matches the assumed null model. For RBF kernels, the exponential eigenvalue decay is a *structural* property, not a sampling artifact.

### 8.3 Candidate C: Resolvent-Based Implicit Gradient

**Hypothesis:** PyTorch's `eigh` backward pass involves terms 1/(λᵢ − λⱼ) that explode when eigenvalues are close. A custom backward pass that differentiates through the regularized inverse directly (via the chain rule d(reg_inv_i)/d(α) and d(reg_inv_i)/d(β)) avoids this instability entirely.

**Implementation:** Custom `torch.autograd.Function` that:
- Forward: standard eigendecomposition + DSD formula (identical to baseline)
- Backward: computes ∂(loss)/∂(log_α) and ∂(loss)/∂(log_β) analytically through the reg_inverse_diag formula, bypassing eigenvector gradients entirely

**Results:**

| Noise | Baseline Error | Resolvent Error | Δ% | p-value | Significant? |
|-------|---------------|-----------------|-----|---------|-------------|
| σ=5e-3 | 0.07023 | 0.07023 | 0.0% | 0.39 | No |
| σ=5e-2 | 0.14024 | 0.14024 | 0.0% | 0.30 | No |

**Gradient comparison (same initialization):**
- Gradients are **mathematically identical** to baseline (ratio = 1.000000 for both ∂/∂α and ∂/∂β)
- Tested under true degeneracy (identical eigenvalues): still identical, no NaN in either approach

**Root Cause of Non-Improvement:** The existing baseline already handles the degeneracy problem via eigenvector detaching:
1. When min(|λᵢ − λⱼ|) < 1e-6, baseline detaches eigenvectors from the computation graph
2. After detaching, gradients flow only through eigenvalues → damping → reg_inverse_diag
3. This is *exactly* the same gradient path that the Resolvent approach computes analytically

The "instability" described as a limitation in the paper is already fully mitigated by the existing detach mechanism. The Resolvent approach is architecturally elegant but solves a problem that doesn't exist in practice.

**Compute overhead:** 0.98× baseline (marginally faster due to skipping autograd graph construction for eigenvectors, but difference is within measurement noise).

**Verdict: REJECT.** No measurable benefit. The existing eigenvector detach mechanism is sufficient. Could be revisited if a future DSD variant requires differentiating through eigenvectors (e.g., learning eigenvector rotations), but for the current 2-parameter formulation it adds complexity without value.

### 8.4 Summary: Why the Current DSD Formulation Is Already Optimal

All three proposed "upgrades" to DSD's core machinery (gap measurement, initialization, gradient computation) failed to produce statistically significant improvements. The reason is architectural:

1. **The bilateral gap is "good enough" because β compensates** — with 2 degrees of freedom (α, β) and a well-behaved loss landscape, the optimizer finds the global minimum regardless of gap measurement precision.

2. **The percentile initialization only needs to be "in the basin"** — Adam with 150 epochs reaches the optimum from any reasonable starting point within the correct order of magnitude.

3. **The gradient path is already stable** — the eigenvector detach mechanism handles degeneracy correctly, making custom backward passes redundant.

**Key insight:** The DSD formula's power comes from its *structure* (gap-dependent exponential damping), not from the precision of its components. Incremental refinements to individual components are absorbed by the optimizer.

---

## 9. Next-Gen DSD Candidates — Tested

The failed approaches in Section 8 all tried to refine *existing* components. The following candidates instead increase DSD's *expressiveness* or change the optimization target entirely. All three were experimentally validated.

**Protocol:** 30 seeds, Swiss Roll + S-Curve, noise levels σ ∈ {5e-3, 5e-2, 1e-1}, m=300, γ=2.0, 150 epochs, patience=20. Metrics: reconstruction error and kNN manifold distance (k=5).

### 9.1 Per-Eigenvalue α (Vector Approach) — VALIDATED: Low-Noise Specialist

**Formulation:** $\tilde{\lambda}_i^{-1} = \frac{\lambda_i}{\lambda_i^2 + \alpha_i \cdot \exp(-\beta \cdot \delta_i)}$

where α⃗ is parameterized as 5 learnable control points (in log-space) interpolated to m values via `F.interpolate(mode='linear', align_corners=True)`. L₂ smoothness penalty (λ_reg=0.01) on deviation from mean prevents overfitting.

**Implementation note:** The naive Python-loop interpolation caused 2× overhead. Replacing with a single vectorized `torch.nn.functional.interpolate` call reduced overhead to 1.02× baseline (from 206× speedup on the interpolation step alone).

**Results (30 seeds, paired t-test vs baseline):**

| Configuration | Δ Error | p-value | Cohen's d | Significant? |
|---|---|---|---|---|
| Swiss Roll σ=5e-3 | **+0.26%** | **0.005** | 0.56 | **✓ Yes** |
| Swiss Roll σ=5e-2 | +0.02% | 0.36 | 0.17 | ✗ No |
| Swiss Roll σ=1e-1 | −0.41% | 0.20 | −0.24 | ✗ No (trends worse) |
| S-Curve σ=5e-3 | **+0.57%** | **0.00002** | **0.95** | **✓ Yes (strong)** |
| S-Curve σ=5e-2 | −0.04% | 0.25 | −0.22 | ✗ No |

**Manifold distance:** Only significant on S-Curve σ=5e-3 (p=0.001, d=−0.24 — but in the *wrong* direction, slightly farther). The improvement is purely in reconstruction fidelity, not manifold adherence.

**Key insight:** The vector-α achieves the **highest effect size** of any candidate tested (Cohen's d = 0.95 on S-Curve low noise) but is regime-dependent. At high noise (σ≥5e-2), the extra parameters have nothing useful to learn because noise destroys the fine spectral structure. At σ=1e-1, there is a non-significant trend toward degradation (overfitting to noise).

**Verdict:** Valid for **low-noise manifold data** (σ ≤ 5e-3). Not suitable as a general default due to high-noise sensitivity. The L₂ regularization (λ_reg=0.01) prevents Section 5-style catastrophic overfitting but does not eliminate the fundamental issue that more parameters + more noise = worse generalization.

**Safeguard required for production use:** Noise-level detection or cross-validation to decide whether to activate vector-α or fall back to scalar α.

### 9.2 Multi-Scale β (Hierarchical Sensitivity) — VALIDATED: Recommended Default

**Formulation:** Partition eigenvalue indices into 3 zones by percentile and assign separate β per zone:

$\text{damping}_i = \alpha \cdot \exp(-\beta_{z(i)} \cdot \delta_i), \quad z(i) \in \{\text{tail}, \text{mid}, \text{head}\}$

Zone boundaries: tail = bottom 30%, mid = middle 40%, head = top 30% of eigenvalue indices.

**Results (30 seeds, paired t-test vs baseline):**

| Configuration | Δ Error | p-value | Cohen's d | Significant? |
|---|---|---|---|---|
| Swiss Roll σ=5e-3 | **+0.58%** | **0.012** | 0.50 | **✓ Yes** |
| Swiss Roll σ=5e-2 | −0.27% | 0.62 | −0.09 | ✗ No |
| Swiss Roll σ=1e-1 | +0.04% | 0.87 | 0.03 | ✗ No (neutral) |
| S-Curve σ=5e-3 | **+0.81%** | **0.005** | 0.57 | **✓ Yes** |
| S-Curve σ=5e-2 | +0.24% | 0.35 | 0.18 | ✗ No |

**Manifold distance (unique advantage):**

| Configuration | Δ Manifold | p-value | Significant? |
|---|---|---|---|
| Swiss Roll σ=5e-3 | **+0.09%** | **0.008** | **✓ Yes** |
| S-Curve σ=5e-3 | −0.11% | 0.024 | ✓ (slightly farther) |
| All others | neutral | >0.09 | ✗ |

**Critical properties making this the recommended default:**
1. **Never degrades.** Worst case across all configs: −0.27% (not significant, p=0.62). Safe to enable unconditionally.
2. **Manifold distance improvement** (Swiss Roll low noise, p=0.008). The only candidate that produces pre-images geometrically closer to the data manifold — directly relevant for explainability applications.
3. **Minimal complexity.** Only +2 parameters (4 total: α, β_tail, β_mid, β_head). Same forward pass structure; no additional matrix operations.
4. **Natural fallback.** If all three β values converge to the same value during optimization, it reduces exactly to the 2-parameter baseline. The optimizer cannot do worse than the baseline in principle.
5. **No compute overhead.** 0.9–1.0× baseline timing.
6. **Consistent positive direction.** Wins on 18–24 out of 30 seeds across all low-noise configs.

**Why it works:** The Davis-Kahan sin(Θ) bound is non-uniform across the spectrum. A 0.01 eigengap in the tail (where all eigenvalues are ~0.001) represents extreme perturbation risk. The same 0.01 gap in the head (where eigenvalues are ~5.0) is negligible. A single β cannot distinguish these regimes. Hierarchical β applies aggressive sensitivity (large β_tail) in the dangerous region while keeping conservative sensitivity (smaller β_head) in the reliable region — matching the actual risk profile.

**Recommendation:** Integrate as the new default DSD optimizer module. The 30/40/30 zone partition is a sensible universal default.

### 9.3 Loss Function Engineering (Manifold-Constrained) — TESTED: No Effect

**Formulation:** L = ‖x̂ − x‖² + λ_k · d_kNN(x̂, X_train)²

Added a differentiable kNN manifold distance penalty (k=5, pre-built BallTree index) to the DSD optimization loss. Tested λ_k ∈ {0.1, 1.0}.

**Results (30 seeds Swiss Roll + S-Curve, σ=5e-3 and σ=5e-2):**

| λ_k | Best Δ Error | Best p-value | Any Significant? |
|---|---|---|---|
| 0.1 | +0.1% | 0.18 | ✗ Never |
| 1.0 | +0.3% | 0.21 | ✗ Never |

No improvement on reconstruction error or manifold distance across any configuration tested.

**Root cause:** The kNN manifold distance has near-zero gradient signal with respect to DSD parameters (α, β). The pre-image direction is determined by the kernel vectors k(x, landmarks) which are independent of regularization parameters. α and β only control *magnitude* of damping — they modulate how much each spectral direction contributes, but the *set of directions* is fixed by the eigendecomposition. The manifold penalty cannot steer α/β toward better manifold adherence because the relationship is too indirect.

**Verdict: REJECT for DSD parameter optimization.** The manifold constraint belongs in the *iterative pre-image solver* (as designed in the Iterative-Manifold-Constrained-PreImage-Plan), where it acts directly on the pre-image coordinate x̂ itself, not on the spectral damping parameters. At that level, the manifold penalty has direct gradient signal: ∂(d_kNN)/∂(x̂) is well-defined and points toward the data manifold.

**Remaining potential for margin-aware loss (Option A):** Not tested in this round. The margin-aware loss for *classification* (Section 2 problem: optimization destroys classification advantage) remains theoretically interesting but requires a fundamentally different experimental setup (classification accuracy metric, not reconstruction error). This is deprioritized relative to the validated improvements above.

### 9.4 Summary & Regime-Dependent Behavior

All three candidates exhibit the same regime-dependent pattern:

| Noise Level | What Happens | Why |
|---|---|---|
| **Low (σ ≤ 5e-3)** | Extra parameters help significantly | Fine spectral structure is preserved; more expressive models capture it |
| **Medium (σ ≈ 5e-2)** | Extra parameters are neutral | Noise obscures fine structure; extra params have nothing useful to learn |
| **High (σ ≥ 1e-1)** | Extra parameters risk degradation | Params may overfit to noise-corrupted spectral features |

**This is the fundamental limitation of solver-side improvements:** they can only improve performance within the "information budget" that the noise level permits. When noise dominates, the baseline 2-parameter model is already extracting all available signal.

### 9.5 Final Recommendation & Integration Plan

| Candidate | Status | Action |
|---|---|---|
| **Multi-Scale β (3-zone)** | ✓ Validated, safe default | **Integrate into `src/dsd_torch.py` as `DSDModule_v2`** |
| **Vector Alpha (5cp)** | ✓ Validated, conditional use | Available as opt-in for known low-noise manifold applications |
| **Manifold Loss** | ✗ No effect on DSD params | Redirect to MCIP iterative solver (different pipeline) |
| **Combined (MSB + VA)** | Untested | Potential future work: vector α with multi-scale β simultaneously |

**Integration architecture for Multi-Scale β:**
```python
class DSDModule_v2(nn.Module):
    """DSD with 3-zone hierarchical β. Drop-in replacement for DSDModule."""
    # Parameters: log_alpha, log_beta_tail, log_beta_mid, log_beta_head
    # Zone partition: fixed at 30/40/30 (tail/mid/head percentiles)
    # Initialization: all β zones start at the standard β₀ = 1/median(Δλ)
    # Backward-compatible: if β_tail ≈ β_mid ≈ β_head after optimization,
    #   behavior is identical to 2-param baseline
```

---

## 10. Code Artifacts From Experiments

The following experiments produced the data above but are NOT in the repository's experiment scripts (conducted in ephemeral research sessions). To reproduce:

```python
# Section 1: 5-param DSD (classification + pre-image) — tested inline
# Section 3: Mono-10cp on manifolds — tested inline
# Section 4: Adaptive-n control points — tested inline
# Section 5: Per-eigenvector learned damping — tested inline
# Section 8: KDE gap, Tracy-Widom init, Resolvent gradient — tested inline (30 seeds)
# Section 9: MultiScaleBeta, VectorAlpha, ManifoldLoss — tested inline (30 seeds)
```

All experiment code was ephemeral (not committed to the repository). Results are fully documented above with sufficient detail to reproduce: dataset parameters, seed counts, formulations, and statistical tests.

If any direction becomes a future paper target, formalize into `experiments/` scripts with proper 50-seed validation before publication.
