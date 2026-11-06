# DSD-SVM: Differential Spectral Damping for Kernel Matrix Pseudo-Inversion

[![arXiv](https://img.shields.io/badge/arXiv-2606.23407-b31b1b.svg)](https://arxiv.org/abs/2606.23407)

**Author:** Praveg Vashishtha  
**Affiliation:** Department of Computer Science and Engineering, IIT Patna | Senior SDE, SBS India  
**Contact:** praveg_pa2503mth259@iitp.ac.in | omvashishtha432@gmail.com  
**LinkedIn:** [pravegvashishtha](https://www.linkedin.com/in/pravegvashishtha/)  
**Paper (preprint):** [arXiv:2606.23407](https://arxiv.org/abs/2606.23407)  
**Code:** [github.com/Praveg432/dsd-regularization](https://github.com/Praveg432/dsd-regularization)

> **Follow-up work (this repo, `paper-followup/`).** A second manuscript extends the preprint with:
> (1) a *breadth* result, DSD improves ridge and Fisher classification, not just LSTSVM, on real
> high-dimensional data, and the gain survives a strengthened baseline set (truncated SVD and a tuned
> magnitude-based spectral filter, not only Tikhonov); (2) a proved **characterization theorem**, the
> DSD inverse is bounded, generalizes Tikhonov, and uniquely imposes a gap-monotone reliability
> ordering; and (3) a mapped **operating envelope** of four settings where DSD provably reduces to
> scalar regularization. The follow-up is honest about scope: it states why an excess-risk superiority
> theorem is *false*, and it **retracts** the preprint's conjecture that DSD improves algorithmic
> stability (measured and contradicted). See `paper-followup/` and `THEOREM-FINAL.md`.

---

## What DSD Does

DSD is a regularization formula for kernel matrix pseudo-inversion that adapts to the local spectral structure:

```
λ̃ᵢ⁻¹ = λᵢ / (λᵢ² + α · exp(-β · δᵢ))
```

Where δᵢ is the localized eigengap — a measure of how reliably the i-th eigenvector can be estimated under perturbation (per the Davis-Kahan sin(Θ) theorem).

DSD preserves eigenvalues with large gaps (reliable) while suppressing those with small gaps (corrupted by numerical noise). Unlike Tikhonov regularization (flat penalty for all eigenvalues), DSD applies damping proportional to each eigenvector's unreliability.

---

## Definitive Results (50-seed paired testing, all methods optimized)

All comparisons are **fair**: DSD gets gradient-optimized (α, β) and Tikhonov gets grid-searched optimal γ on the same training data.

### LSTSVM Classification (DSD’s Strongest Result)

| Setting | DSD | Tikhonov-opt | Advantage | p-value | Wins | Cohen’s d |
|---------|-----|-------------|-----------|---------|------|-----------|
| **GINA (d=970, real-world)** | **85.9%** | 81.1% | **+4.8pp** | p < 0.0001 | 30/30 | 4.49 |
| **Madelon (d=500, real-world)** | **57.7%** | 55.0% | **+2.6pp** | p < 0.0001 | 27/30 | 1.76 |
| **d=200, n=300** | **71.4%** | 61.1% | **+10.4pp** | p < 0.0001 | 44/50 | 1.57 |
| **d=100, n=200** | **79.1%** | 75.8% | **+3.3pp** | p = 0.0001 | 35/50 | 0.39 |
| d=50, n=300 | 90.1% | 90.1% | 0 | p = 0.91 | — | — |
| d=30, n=400 | 91.1% | 91.0% | 0 | p = 0.29 | — | — |

**Pattern:** DSD’s advantage scales with dimensionality (Cohen’s d > 1.5 at d ≥ 200). Validated on real-world GINA (d=970, handwriting, Cohen’s d=4.49) and Madelon (d=500, NIPS 2003 benchmark). At d ≤ 50, all methods are equivalent. DSD uses only its principled spectral initialization (zero optimization); Tikhonov receives grid search. This confirms DSD’s advantage is a structural prior, not a tuning advantage.

### Kernel LSTSVM (Non-Linear, RBF Kernel Matrices)

| Dataset | DSD-opt | Tikhonov-opt | Advantage | p-value |
|---------|---------|-------------|-----------|---------|
| **Two Moons (d=2, γ=2.0)** | **92.7%** | 92.1% | **+0.6pp** | p = 0.028 |
| Genomics (d=100, γ=0.02) | 87.7% | 87.7% | 0 | p = 0.77 |

### Pre-Image Reconstruction

| Noise Level | DSD-opt | Tikhonov-opt | Winner |
|-------------|---------|-------------|--------|
| σ = 5×10⁻³ | 0.069 | 0.069 | **Tie** (p = 0.99) |
| σ = 1×10⁻³ | 0.051 | 0.048 | Tikhonov (p = 0.02) |
| σ = 1×10⁻⁴ | 0.042 | 0.035 | Tikhonov (p < 0.001) |

**Honest assessment:** On pre-image tasks, when Tikhonov receives the same gradient optimizer (Adam, not just grid search), DSD still wins slightly (p=0.007, 16/20 seeds). The per-eigenvector structure provides a small but real advantage even here. However, the practical difference is negligible (0.073 vs 0.078). DSD's unique contribution is in **classification accuracy** — the gap-adaptive regularization prevents overfitting to corrupted eigenvector directions during model training.

### DSD-optimized vs DSD-init (Experiment 05)

| Noise | DSD-init | DSD-opt | Improvement | p-value | Wins |
|-------|----------|---------|-------------|---------|------|
| σ=5e-3 | 0.077 | 0.069 | +9.2% | p < 0.0001 | 48/50 |
| σ=1e-4 | 0.048 | 0.042 | +12.4% | p < 0.0001 | 50/50 |

---

## Where DSD Helps vs Where It Doesn't

| Condition | DSD Advantage | Recommendation |
|-----------|---------------|----------------|
| LSTSVM at d ≥ 100 | **+3 to +10pp** (significant) | Use DSD |
| Kernel LSTSVM with sharp RBF | **+0.6pp** (significant) | Use DSD |
| Pre-image at high noise | Equivalent to tuned Tikhonov | Either method |
| Pre-image at low noise | Tikhonov slightly better | Use Tikhonov |
| LSTSVM at d ≤ 50 | No difference | Use simplest method |
| Real-world d ≤ 34 | Mixed (boundary regime) | Test both; DSD not designed for this range |
| Well-conditioned matrices | No difference | Use Tikhonov |

---

## Mechanism

**Why DSD helps classification but ties on pre-image:** The pre-image task has a single objective (minimize reconstruction error) that Tikhonov's scalar γ can optimize directly. Classification has a more complex objective (generalization) where DSD's per-eigenvector regularization prevents the classifier from fitting to noise-corrupted directions — a form of spectral regularization that a single scalar cannot replicate.

---

## Differentiable DSD (PyTorch)

```python
from src.dsd_optimizer import DSDOptimizer

optimizer = DSDOptimizer(lr=0.01, n_epochs=150, patience=20)
dsd_module, result = optimizer.optimize(
    X_train=X_train, landmarks=landmarks,
    gamma=2.0, noise_matrix=noise,
)
# result.improvement — typically 9-12% over principled init
```

---

## Project Structure

```
dsd-svm/
├── src/
│   ├── dsd.py              # Core DSD + Tikhonov/TSVD inverses (NumPy)
│   ├── dsd_torch.py        # Differentiable DSD (PyTorch)
│   ├── dsd_optimizer.py    # Gradient-based α, β optimization
│   ├── dsd_scalable.py     # O(m²k) path for large matrices
│   ├── kernels.py          # RBF kernel + Nyström sampling
│   ├── preimage.py         # Pre-image pipeline (vectorized)
│   ├── lssvm.py            # LS-SVM
│   ├── lstsvm.py           # Linear LSTSVM
│   ├── kernel_lstsvm.py    # Kernel LSTSVM (non-linear, RBF)
│   ├── kernel_classifiers.py  # [follow-up] Ridge + Fisher classifiers, pluggable inversion
│   └── gp_regressor.py        # [follow-up] GP regression, pluggable posterior inversion
├── experiments/
│   ├── 01–04               # Development and validation
│   ├── 05–10               # Preprint definitive results (LSTSVM, kernel, boundary, extended)
│   ├── P2_breadth_classification.py    # [follow-up] breadth: ridge + Fisher, real data
│   ├── P2c_stronger_baselines.py       # [follow-up] vs TSVD + tuned spectral filter
│   ├── P1_gp_calibration.py            # [follow-up] boundary: GP calibration (negative)
│   ├── P2b_nonlinear_breadth.py        # [follow-up] boundary: non-linear RBF breadth
│   ├── B2probe_preconditioning.py      # [follow-up] boundary: preconditioning (negative)
│   ├── B3probe_covariance.py           # [follow-up] boundary: covariance (negative)
│   └── T2/T3_theorem_*.py              # [follow-up] characterization theorem validation
├── paper/                  # Original preprint (IEEE LaTeX + Markdown, arXiv:2606.23407)
├── paper-followup/         # Follow-up manuscript (Neurocomputing / elsarticle) + cover + highlights
├── results/                # Committed result JSONs (one per experiment) + figures
├── tests/                  # 28 unit tests
├── NOVELTY-SCAN.md         # Prior-work scan and differentiation for the follow-up
├── THEOREM-FINAL.md        # The characterization theorem: statement, proof, validation
├── FUTURE-RESEARCH.md      # Open directions
└── pyproject.toml
```

Every number cited in either paper maps to a committed JSON in `results/` and reproduces
deterministically from the corresponding experiment script.

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/ -v                                    # 28 tests pass
python experiments/07_lstsvm_definitive.py          # DSD's strongest result
python experiments/08_kernel_lstsvm_definitive.py   # Kernel LSTSVM
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{vashishtha2026dsd,
  title={Differential Spectral Damping: Gap-Adaptive Regularization for Ill-Conditioned Kernel Methods},
  author={Vashishtha, Praveg},
  journal={arXiv preprint arXiv:2606.23407},
  year={2026}
}
```

Preprint: [arXiv:2606.23407](https://arxiv.org/abs/2606.23407). A follow-up manuscript
(characterization theorem + breadth result + operating envelope) is prepared for Neurocomputing and
lives in `paper-followup/`.
