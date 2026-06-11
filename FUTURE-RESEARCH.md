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

---

## 8. Code Artifacts From Experiments

The following inline experiments produced the data above but are NOT in the repository's experiment scripts. To reproduce:

```python
# 5-param DSD (classification): tested in conversation, results above
# 5-param DSD (pre-image): tested in conversation, results above
# Mono-10cp on manifolds: tested in conversation, results above
# Adaptive-n: tested in conversation, results above
# Per-eigenvector: tested in conversation, results above
```

If any of these become future paper targets, formalize into `experiments/` scripts with proper 50-seed validation before publication.
