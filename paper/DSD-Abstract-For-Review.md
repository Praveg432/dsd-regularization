# Research Abstract for Faculty Review

## Differential Spectral Damping: Gap-Adaptive Regularization for Ill-Conditioned Kernel Methods

**Student:** Praveg Vashishtha  
**Program:** Executive M.Tech., Department of Computer Science and Engineering  
**Institution:** Indian Institute of Technology Patna  
**Contact:** praveg_pa2503mth259@iitp.ac.in  
**Code:** https://github.com/Praveg432/dsd-regularization

---

### Problem

Least-Squares Twin SVM (LSTSVM) — widely used for small-sample, high-dimensional classification in genomics, spectroscopy, and credit scoring — requires inverting cross-class product matrices (E₁ᵀE₁ + c⁻¹E₂ᵀE₂). At high dimensionality (d ≥ 100), these matrices exhibit extreme spectral tail-clustering: the bottom 50% of eigenvalues compress into only 7% of the total spectral range. The Davis-Kahan sin(Θ) theorem guarantees that eigenvectors corresponding to these clustered eigenvalues are directionally corrupted by numerical noise — regardless of subsequent scalar regularization. Standard Tikhonov regularization (adding γI) applies a single uniform penalty that cannot simultaneously avoid over-regularizing the reliable spectral head and under-regularizing the corrupted tail.

### Proposed Method

We propose **Differential Spectral Damping (DSD)**, a regularization formula that adapts its penalty to the localized eigengap structure:

$$\tilde{\lambda}_i^{-1} = \frac{\lambda_i}{\lambda_i^2 + \alpha \cdot \exp(-\beta \cdot \delta_i)}$$

where δᵢ = min(|λᵢ − λᵢ₋₁|, |λᵢ − λᵢ₊₁|) is the localized eigengap. The formula preserves exact inversion for well-separated eigenvalues (large δᵢ → zero penalty) while aggressively suppressing contributions from clustered eigenvalues (small δᵢ → maximum penalty). The two parameters (α, β) are initialized entirely from the spectral structure of the matrix itself — requiring zero cross-validation or hyperparameter tuning.

### Key Results (50-seed paired testing, fair baselines)

| Dataset | d | DSD (auto-init) | Tikhonov (grid-searched) | Advantage | p-value | Cohen's d |
|---------|---|-----------------|--------------------------|-----------|---------|-----------|
| **GINA (real-world)** | 970 | **85.9%** | 81.1% | **+4.8pp** | < 0.0001 | 4.49 |
| **Madelon (real-world)** | 500 | **57.7%** | 55.0% | **+2.6pp** | < 0.0001 | 1.76 |
| Synthetic | 200 | **71.4%** | 61.1% | **+10.4pp** | < 0.0001 | 1.57 |
| Synthetic | 100 | **79.1%** | 75.8% | **+3.3pp** | 0.0001 | 0.39 |
| Synthetic | 50 | 90.1% | 90.1% | 0 | 0.91 | — |

**Critical finding:** DSD uses *no optimization* (principled initialization only) while Tikhonov receives a 15-point grid search — yet DSD still dominates. Furthermore, gradient-optimizing DSD's parameters on reconstruction loss *degrades* classification accuracy to exactly Tikhonov's level (85.9% → 81.1%), proving the advantage is a structural prior, not a parametric one.

### Scientific Contributions

1. A novel eigengap-adaptive regularization formula motivated by Davis-Kahan eigenvector reliability theory
2. Empirical demonstration that structural spectral priors outperform parametric optimization for generalization
3. Clear operating regime characterization (d ≥ 100, spectral tail-clustering < 12%)
4. Connection to algorithmic stability theory (soft spectral truncation reduces effective hypothesis dimensionality)
5. Differentiable PyTorch implementation enabling end-to-end integration

### Significance

- **Effect size:** Cohen's d = 4.49 on GINA — exceptionally large for a regularization technique
- **Practical zero-tuning:** No cross-validation needed; parameters derived from spectrum
- **Computationally free:** O(m) overhead on existing O(m³) eigendecomposition
- **Validated on real data:** Two OpenML benchmarks (GINA d=970, Madelon d=500) confirm synthetic results transfer

### Limitations (Honestly Documented)

- DSD does not help at d ≤ 50 (insufficient tail-clustering)
- DSD does not beat optimally-tuned Tikhonov on pre-image reconstruction (ties)
- No formal stability proof (conjectured, empirically validated)
- Not evaluated on non-RBF kernels

### Target Venue

IEEE conference on machine learning or pattern recognition (ICASSP, ICPR, IJCNN, or IEEE TNNLS).

### Implementation

Complete open-source Python package: 10 source modules, 10 experiment scripts, 28 unit tests. All results reproducible via `python experiments/10_extended_validation.py`.

---

*Paper manuscript (6 pages, IEEE format) available upon request.*
