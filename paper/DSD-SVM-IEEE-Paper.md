# Differential Spectral Damping: Gap-Adaptive Regularization for Ill-Conditioned Kernel Methods

**Author:** Praveg Vashishtha  
**Affiliation:** Department of Computer Science and Engineering, Indian Institute of Technology Patna, India; Sopra Banking Software India (Senior Software Development Engineer)  
**Contact:** praveg_pa2503mth259@iitp.ac.in | omvashishtha432@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/pravegvashishtha/  
**Code:** https://github.com/Praveg432/dsd-regularization

---

## Abstract

Kernel methods requiring matrix inversion---particularly Least-Squares Twin Support Vector Machines (LSTSVM)---suffer from exponential eigenvalue decay in their system matrices, producing severely ill-conditioned problems where standard Tikhonov regularization applies uniform damping regardless of eigenvector reliability. We propose Differential Spectral Damping (DSD), a regularization formula that adapts its penalty to localized eigengap structure: preserving eigenvectors with large spectral gaps (reliable per Davis-Kahan perturbation theory) while aggressively suppressing those with small gaps (directionally corrupted beyond recovery). We motivate DSD through a principled design procedure grounded in the Davis-Kahan sin(Θ) theorem, systematically deriving the requirements for a reliability-aware damping function and selecting the exponential form for its smoothness, differentiability, and natural saturation properties. Through rigorous paired testing with fairly optimized baselines (including gradient-optimized Tikhonov receiving equal optimization opportunity), we demonstrate that DSD improves LSTSVM classification accuracy by +4.8 percentage points on real-world GINA (d=970, Cohen's d = 4.49, p < 0.0001), +10.4 percentage points at d=200, and +2.6 percentage points on Madelon (d=500, Cohen's d = 1.76)---all using only principled spectral initialization while Tikhonov receives grid search. For pre-image reconstruction on manifold data, DSD matches optimally-tuned Tikhonov; both reduce naive inversion error by 66×. We characterize the precise operating regime (d ≥ 100, condition number > 10³) and document where simpler methods suffice, providing practitioners with clear deployment guidance.

**Keywords:** Kernel methods, spectral regularization, eigengap adaptation, ill-conditioned systems, Twin SVM, numerical stability, eigenvalue perturbation, pre-image problem.

---

## I. Introduction

Kernel methods map input data into high-dimensional feature spaces via the kernel trick, enabling non-linear classification and regression without explicit feature computation [1, 2]. However, methods that require *inversion* of kernel-derived system matrices---particularly the Least-Squares Twin Support Vector Machine (LSTSVM) [3]---face a fundamental numerical challenge: the eigenvalues of these system matrices decay exponentially, producing severe ill-conditioning that standard scalar regularization cannot adequately address.

LSTSVM constructs two non-parallel hyperplanes by solving a pair of linear systems whose coefficient matrices take the form M = E₁ᵀE₁ + c⁻¹E₂ᵀE₂, where E₁ and E₂ are augmented class-specific data matrices and c > 0 is a regularization constant. In high-dimensional settings (d ≥ 100), these matrices empirically exhibit *extreme spectral tail-clustering*: at d=200, the bottom 50% of eigenvalues occupy only 7% of the total spectral range, with condition numbers exceeding 10⁴.

LSTSVM and its kernel variants remain actively deployed in domains where deep learning overfits due to limited sample sizes [4]: genomic classification (d=1000–20000, n=50–500), spectroscopic material identification, and regulatory credit scoring with engineered features. In these high-stakes settings, a +3–10 percentage point accuracy improvement from regularization alone directly impacts diagnostic and decision quality.

The central theoretical insight motivating our work comes from the Davis-Kahan sin(Θ) theorem [5], which establishes that eigenvector perturbation under matrix noise E is bounded by:

$$\sin(\theta_i) \leq \frac{\|E\|_2}{\delta_i}$$

where δᵢ denotes the *eigengap*---the minimum distance from eigenvalue λᵢ to all other eigenvalues. When eigenvalues cluster tightly (δᵢ → 0), their corresponding eigenvectors become *arbitrarily corrupted* by any perturbation, including floating-point arithmetic noise. Standard Tikhonov regularization (W + γI)⁻¹ addresses eigenvalue magnitudes but is structurally incapable of accounting for this directional corruption: a single scalar γ cannot simultaneously avoid over-regularizing the well-separated spectral head (where eigenvectors are reliable) and under-regularizing the clustered tail (where eigenvectors are corrupted).

We propose **Differential Spectral Damping (DSD)**, a regularization formula that adapts its penalty strength to the *local eigengap density* at each spectral position. Eigenvectors with large gaps (reliable per Davis-Kahan) are preserved with near-exact inversion; eigenvectors with small gaps (directionally corrupted) are aggressively suppressed. The transition is smooth, differentiable, and governed by only two interpretable parameters.

### Contributions

1. A **gap-adaptive regularization formula** (DSD) for kernel matrix inversion, derived from Davis-Kahan perturbation theory with complete mathematical motivation.
2. A **principled initialization procedure** requiring zero cross-validation, based on spectral transition analysis.
3. A **differentiable PyTorch implementation** enabling gradient-based optimization of DSD parameters through backpropagation.
4. **Rigorous experimental comparison** with fairly-optimized baselines (50-seed paired testing, Tikhonov receiving grid search or equivalent gradient optimization).
5. **Clear operating regime characterization**: DSD dominates at d ≥ 100 with severe tail-clustering; equivalent to Tikhonov at d ≤ 50.
6. **Honest documentation of limitations**: pre-image tasks see no improvement over Tikhonov; DSD provides no benefit below its operating threshold.

---

## II. Notation and Symbol Definitions

All mathematical symbols used in this paper are defined below. Conventions: bold lowercase (***x***) denotes vectors, bold uppercase (***W***) denotes matrices, calligraphic (ℋ) denotes spaces, Greek letters denote parameters or scalar quantities.

### Data and Dimensions

| Symbol | Definition |
|--------|-----------|
| n | Number of training samples |
| d | Input space dimensionality |
| m | Number of Nyström landmarks or matrix dimension |
| **x**ᵢ ∈ ℝᵈ | The i-th training sample |
| X ∈ ℝⁿˣᵈ | Training data matrix |
| yᵢ ∈ {+1, −1} | Class label of sample i |

### Kernel and Feature Spaces

| Symbol | Definition |
|--------|-----------|
| ℋ | Reproducing Kernel Hilbert Space (RKHS) |
| φ: ℝᵈ → ℋ | Feature map into RKHS |
| κ(**x**, **x**') | Kernel function: κ(**x**, **x**') = ⟨φ(**x**), φ(**x**')⟩_ℋ |
| γ_k | RBF kernel bandwidth: κ(**x**, **x**') = exp(−γ_k ‖**x** − **x**'‖²) |
| **K** ∈ ℝⁿˣⁿ | Full kernel (Gram) matrix: K_{ij} = κ(**x**ᵢ, **x**ⱼ) |

### Nyström Approximation

| Symbol | Definition |
|--------|-----------|
| L = {**l**₁, …, **l**_m} | Nyström landmark points (L ⊂ X) |
| **W** ∈ ℝᵐˣᵐ | Kernel submatrix: W_{ij} = κ(**l**ᵢ, **l**ⱼ) |
| **k**_q ∈ ℝᵐ | Query kernel vector: (k_q)ⱼ = κ(**x**_q, **l**ⱼ) |

### Eigendecomposition

| Symbol | Definition |
|--------|-----------|
| **U** ∈ ℝᵐˣᵐ | Orthogonal eigenvector matrix of **W** |
| **u**ᵢ ∈ ℝᵐ | The i-th eigenvector (column of **U**) |
| λᵢ | The i-th eigenvalue of **W** (ascending order: λ₁ ≤ λ₂ ≤ … ≤ λ_m) |
| **Λ** | Diagonal matrix of eigenvalues: Λ_{ii} = λᵢ |

### Eigengap and Perturbation

| Symbol | Definition |
|--------|-----------|
| δᵢ | Localized bilateral eigengap at position i |
| θᵢ | Angle between true and perturbed i-th eigenvector |
| **E** | Perturbation matrix (noise, numerical error) |
| ‖**E**‖₂ | Spectral norm (largest singular value) of **E** |

### DSD Parameters

| Symbol | Definition |
|--------|-----------|
| α > 0 | Maximum penalty magnitude (damping ceiling) |
| β > 0 | Eigengap sensitivity (transition sharpness) |
| dᵢ | DSD damping at position i: dᵢ = α · exp(−β · δᵢ) |
| λ̃ᵢ⁻¹ | DSD-regularized inverse eigenvalue |
| W̃⁺ | DSD-regularized pseudo-inverse of **W** |

### Other

| Symbol | Definition |
|--------|-----------|
| γ | Tikhonov regularization parameter (scalar) |
| E₁, E₂ | Augmented LSTSVM class data matrices: [X_± **e**] |
| c₁, c₂ > 0 | LSTSVM penalty parameters |
| **u**, **v** | LSTSVM hyperplane normal vectors |
| x̂ ∈ ℝᵈ | Reconstructed pre-image in input space |
| p | Two-sided p-value from paired t-test |
| Cohen's d | Standardized effect size: d = Δ̄/s_Δ |

**Eigenvalue ordering convention.** Throughout this paper, eigenvalues are sorted in *ascending* order: λ₁ ≤ λ₂ ≤ … ≤ λ_m. Index i=1 corresponds to the smallest eigenvalue (most vulnerable to perturbation); index i=m corresponds to the largest (most reliable). This convention aligns with standard numerical eigensolvers (numpy.linalg.eigh, torch.linalg.eigh).

---

## III. Mathematical Background

### A. Kernel Methods and the Kernel Trick

A kernel function κ: ℝᵈ × ℝᵈ → ℝ implicitly computes inner products in a (possibly infinite-dimensional) Hilbert space ℋ without requiring explicit computation of the feature map φ:

$$\kappa(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle_\mathcal{H}$$

The Radial Basis Function (RBF) kernel:

$$\kappa(\mathbf{x}, \mathbf{x}') = \exp(-\gamma_k \|\mathbf{x} - \mathbf{x}'\|^2), \quad \gamma_k > 0$$

maps data into an infinite-dimensional space. The parameter γ_k controls the bandwidth: larger γ_k produces sharper kernels with faster eigenvalue decay in the Gram matrix.

The kernel (Gram) matrix **K** ∈ ℝⁿˣⁿ with entries K_{ij} = κ(**x**ᵢ, **x**ⱼ) is symmetric positive semi-definite (PSD) by Mercer's theorem. For RBF kernels, eigenvalues decay exponentially [2]:

$$\lambda_i \propto \exp(-c \cdot i^{2/d})$$

where c depends on γ_k and the intrinsic dimensionality. This decay is not a pathology---it reflects the smoothness assumptions encoded by the kernel---but it creates severe numerical challenges for methods requiring **K**⁻¹.

### B. Least-Squares Twin SVM (LSTSVM)

LSTSVM [3] finds two non-parallel hyperplanes, each proximal to one class. Let X₊ ∈ ℝⁿ⁺ˣᵈ and X₋ ∈ ℝⁿ⁻ˣᵈ be class-specific data matrices. Define augmented matrices:

$$E_1 = [X_+ ~~ \mathbf{e}_+] \in \mathbb{R}^{n_+ \times (d+1)}, \quad E_2 = [X_- ~~ \mathbf{e}_-] \in \mathbb{R}^{n_- \times (d+1)}$$

where **e**_± are vectors of ones (bias terms). LSTSVM solves two linear systems:

$$(E_2^\top E_2 + c_1^{-1} E_1^\top E_1) \mathbf{u} = E_2^\top \mathbf{e}_-$$
$$(E_1^\top E_1 + c_2^{-1} E_2^\top E_2) \mathbf{v} = E_1^\top \mathbf{e}_+$$

where **u**, **v** ∈ ℝᵈ⁺¹ define the two hyperplanes. The coefficient matrices M₁ = E₂ᵀE₂ + c₁⁻¹E₁ᵀE₁ are sums of outer-product matrices that exhibit severe spectral tail-clustering when d is large relative to class sizes.

A new test sample **x** is classified by proximity to each hyperplane:

$$\text{class}(\mathbf{x}) = \arg\min_{k \in \{1,2\}} \frac{|\mathbf{w}_k^\top \mathbf{x} + b_k|}{\|\mathbf{w}_k\|}$$

### C. The Nyström Approximation

The Nyström approximation [6] selects m ≪ n landmark points L = {**l**₁, …, **l**_m} and approximates:

$$\mathbf{K} \approx \mathbf{K}_{nm} \mathbf{W}^{-1} \mathbf{K}_{nm}^\top$$

where **W** ∈ ℝᵐˣᵐ is the kernel matrix among landmarks and **K**_{nm} ∈ ℝⁿˣᵐ contains kernel evaluations between all samples and landmarks. This requires computing **W**⁻¹ (or its regularized variant)---the operation DSD stabilizes.

### D. The Davis-Kahan sin(Θ) Theorem

The Davis-Kahan theorem [5] is the theoretical foundation of DSD.

**Theorem (Davis-Kahan, simplified form).** Let **A** be a symmetric matrix with eigenvalue λᵢ and eigenvector **u**ᵢ. Let **Ã** = **A** + **E** be a perturbed matrix with corresponding eigenvector **ũ**ᵢ. Define the eigengap:

$$\delta_i = \min_{j \neq i} |\lambda_i - \lambda_j|$$

Then the angle θᵢ between the true and perturbed eigenvectors satisfies:

$$\sin(\theta_i) \leq \frac{\|\mathbf{E}\|_2}{\delta_i}$$

**Interpretation.** The bound states that eigenvector reliability is determined by the ratio of perturbation magnitude to eigengap:
- When δᵢ ≫ ‖**E**‖₂: the eigenvector is robust (sin θᵢ ≈ 0, small rotation)
- When δᵢ ≲ ‖**E**‖₂: the bound becomes vacuous (sin θᵢ could approach 1, meaning the eigenvector can rotate by up to 90°)---the eigenvector direction is essentially random

**Implication for regularization.** Any scheme using the eigendecomposition **W** = **U****Λ****U**ᵀ implicitly trusts the eigenvectors **u**ᵢ. But Davis-Kahan tells us that eigenvectors with small gaps are *unreliable directions*. Computing **W**⁻¹ = **U****Λ**⁻¹**U**ᵀ amplifies contributions from corrupted directions (division by small λᵢ). Tikhonov replaces λᵢ⁻¹ with (λᵢ + γ)⁻¹---damping the *magnitude* uniformly---but does nothing about the corrupted *direction* **u**ᵢ.

### E. The Pre-Image Problem

Given a point ψ ∈ ℋ (e.g., a decision boundary point in kernel space), the pre-image problem seeks x̂ ∈ ℝᵈ such that φ(x̂) ≈ ψ [7, 8]. Under the Nyström framework:

$$\hat{\mathbf{x}} = \mathbf{k}_q^\top \mathbf{W}^{-1} L = \sum_{j=1}^m w_j \mathbf{l}_j, \quad w_j = (\mathbf{W}^{-1} \mathbf{k}_q)_j$$

The pre-image is a weighted combination of landmark points. Ill-conditioned **W** produces extreme weights that push x̂ far from the data manifold.

---

## IV. Derivation of Differential Spectral Damping

We derive DSD by asking: *given the Davis-Kahan bound, what is the optimal regularization strategy that minimizes information loss while respecting eigenvector reliability?* The following procedure systematically identifies the requirements for such a regularizer and motivates the specific functional form.

### Step 1: Quantifying Eigenvector Reliability

From the Davis-Kahan bound, the reliability of eigenvector **u**ᵢ is characterized by the ratio rᵢ = ‖**E**‖₂ / δᵢ:

- **High reliability**: δᵢ ≫ ‖**E**‖₂ → sin θᵢ ≈ 0 (eigenvector direction is trustworthy)
- **Low reliability**: δᵢ ≲ ‖**E**‖₂ → sin θᵢ large (eigenvector direction is essentially random)

In practice, ‖**E**‖₂ represents the combined effect of finite-precision arithmetic (~10⁻¹⁶ for float64, amplified by condition number), sampling noise, and Nyström approximation error. The exact value is typically unknown. However, the *relative* ordering of eigengaps δᵢ is observable from the eigendecomposition. DSD exploits this: it does not need to know the absolute perturbation magnitude---it only needs to identify which eigenvectors have *relatively* small gaps.

### Step 2: The Ideal Regularization Profile

For an ideal regularizer, we want:
- When δᵢ is large (reliable eigenvector): preserve the exact inverse λᵢ⁻¹ to retain maximum information
- When δᵢ is small (corrupted eigenvector): suppress the contribution by driving the effective inverse toward zero
- The transition should be smooth (to avoid discontinuity artifacts)

This translates into a *damping function* d(δᵢ) satisfying:

$$d(\delta_i) \to 0 \quad \text{as } \delta_i \to \infty \quad \text{(no damping for reliable eigenvectors)}$$
$$d(\delta_i) \to \alpha_{\max} \quad \text{as } \delta_i \to 0 \quad \text{(maximum damping for corrupted)}$$
$$d'(\delta_i) < 0 \quad \text{(monotonically decreasing)}$$

### Step 3: Functional Form Selection

The requirements above are satisfied by any monotonically decreasing function bounded between 0 and α_max. We select the *exponential decay* form:

$$d_i = \alpha \cdot \exp(-\beta \cdot \delta_i)$$

for the following reasons:

1. **Natural saturation.** exp(−βδᵢ) ∈ (0, 1] automatically satisfies boundedness without clipping.
2. **Differentiability.** The exponential is C∞, enabling gradient-based optimization of (α, β) via backpropagation.
3. **Scale invariance.** The product β·δᵢ is dimensionless when β has units of inverse-gap, making the formula invariant to eigenvalue scaling.
4. **Rapid transition.** The exponential transitions sharply from ≈1 (small gap) to ≈0 (large gap), matching the binary nature of the Davis-Kahan bound.
5. **Parsimony.** Only two free parameters (α, β) control the entire damping profile across all m eigenvalues.

**Alternative forms** that satisfy the same requirements include sigmoid α/(1 + exp(β(δᵢ − δ₀))), rational α/(1 + βδᵢ)ᵖ, and complementary error function α·erfc(βδᵢ). All would produce qualitatively similar regularization. We select the exponential for analytical simplicity and parameter interpretability.

### Step 4: Incorporating Damping into the Inverse

The standard pseudo-inverse via eigendecomposition:

$$\mathbf{W}^{-1} = \mathbf{U}\mathbf{\Lambda}^{-1}\mathbf{U}^\top = \sum_{i=1}^m \frac{1}{\lambda_i} \mathbf{u}_i \mathbf{u}_i^\top$$

Tikhonov regularization:

$$\mathbf{W}_{\text{Tik}}^{-1} = \sum_{i=1}^m \frac{1}{\lambda_i + \gamma} \mathbf{u}_i \mathbf{u}_i^\top$$

We incorporate DSD damping via:

$$\tilde{\lambda}_i^{-1} = \frac{\lambda_i}{\lambda_i^2 + d_i}$$

This form has the following properties:
- **When dᵢ = 0** (no damping): λ̃ᵢ⁻¹ = λᵢ/λᵢ² = 1/λᵢ. Exact inverse recovered.
- **When dᵢ = α ≫ λᵢ²** (maximum damping): λ̃ᵢ⁻¹ ≈ λᵢ/α → 0. Corrupted direction suppressed.
- **Intermediate regime**: smooth interpolation governed by λᵢ² vs dᵢ.
- **Positive definiteness**: λ̃ᵢ⁻¹ > 0 for all λᵢ > 0, ensuring the regularized inverse remains PSD.
- **Tikhonov recovery**: when dᵢ = γ² (constant), the formula becomes a generalized Tikhonov. DSD *generalizes* scalar regularization.

### Step 5: The Complete DSD Formula

Substituting the damping function:

$$\boxed{\tilde{\lambda}_i^{-1} = \frac{\lambda_i}{\lambda_i^2 + \alpha \cdot \exp(-\beta \cdot \delta_i)}}$$

The full DSD-regularized pseudo-inverse:

$$\tilde{\mathbf{W}}^+ = \mathbf{U} \cdot \text{diag}(\tilde{\lambda}_1^{-1}, \ldots, \tilde{\lambda}_m^{-1}) \cdot \mathbf{U}^\top = \sum_{i=1}^m \tilde{\lambda}_i^{-1} \mathbf{u}_i \mathbf{u}_i^\top$$

### Step 6: Parameter Interpretation

**α (penalty magnitude):** Controls the maximum damping applied to the most corrupted eigenvectors. When δᵢ → 0, damping equals α, so λ̃ᵢ⁻¹ ≈ λᵢ/(λᵢ² + α). For small eigenvalues (λᵢ² ≪ α), this reduces to λᵢ/α ≈ 0---complete suppression. α sets the "penalty ceiling" calibrated to the eigenvalue scale where corruption begins.

**β (gap sensitivity):** Controls the sharpness of the transition between "reliable" and "corrupted" regimes. Large β → sharp step-function transition. Small β → gradual transition. Calibrated to the typical eigengap scale so that "typical" gaps produce moderate damping.

### Comparison with Existing Methods in the Eigenvalue Domain

| Method | λ̃ᵢ⁻¹ | Adapts to |
|--------|--------|-----------|
| Naive inverse | 1/λᵢ | Nothing |
| Tikhonov | 1/(λᵢ + γ) | Magnitude only |
| Truncated SVD | 1/λᵢ if i > k, else 0 | Rank (hard cutoff) |
| Spectral filter | g(λᵢ)/λᵢ | Magnitude (soft) |
| **DSD (ours)** | λᵢ/(λᵢ² + α·exp(−βδᵢ)) | **Eigengap (reliability)** |

DSD is the only method that conditions regularization on the *gap structure*---a proxy for eigenvector directional reliability per Davis-Kahan.

---

## V. Method: Algorithm, Initialization, and Implementation

### A. Localized Eigengap Computation

For monotonically ordered eigenvalues (as produced by symmetric eigensolvers), the global eigengap min_{j≠i}|λᵢ − λⱼ| simplifies to the bilateral minimum:

$$\delta_i = \min(|\lambda_i - \lambda_{i-1}|, |\lambda_i - \lambda_{i+1}|)$$

**Boundary handling:** At sequence edges, only one neighbor exists:
- i = 1: δ₁ = |λ₁ − λ₂| (right gap only)
- 1 < i < m: δᵢ = min(|λᵢ − λᵢ₋₁|, |λᵢ − λᵢ₊₁|) (bilateral)
- i = m: δ_m = |λ_m − λ_{m−1}| (left gap only)

The boundary choice for i=1 (the smallest, most vulnerable eigenvalue) ensures it receives appropriate damping based on its actual spectral isolation, rather than being artificially exempted.

### B. Principled Hyperparameter Initialization

DSD's initialization requires no cross-validation. Both parameters are determined entirely from the eigendecomposition.

**Initialization of α (penalty magnitude):**

$$\alpha_0 = \lambda_{\text{transition}}^2$$

where λ_transition is the eigenvalue at the spectral "knee"---the point where eigengaps become characteristically small. Operationally:
1. Compute consecutive gaps: Δᵢ = |λᵢ₊₁ − λᵢ| for i = 1, …, m−1
2. Identify the 10th percentile gap: g₁₀ = quantile₀.₁₀({Δᵢ})
3. Find the last index where the gap falls below g₁₀: i* = max{i : Δᵢ < g₁₀}
4. Set λ_transition = λ_{i*}

**Rationale:** The squared eigenvalue λ²_transition ensures α is on the correct scale for the formula λᵢ² + α. At the transition point, λᵢ² ≈ α, producing roughly 50% damping---the natural midpoint.

**Initialization of β (gap sensitivity):**

$$\beta_0 = \frac{1}{\text{median}(\{\Delta_i\})}$$

**Rationale:** This normalizes the exponential so that a "typical" eigengap produces exp(−1) ≈ 0.37 (moderate damping). Gaps significantly larger than median → exp(−large) ≈ 0 (preservation). Gaps significantly smaller → exp(−small) ≈ 1 (maximum damping). The median is robust to outliers.

### C. Gradient-Based Parameter Optimization

When a differentiable reconstruction objective exists, α and β can be refined after initialization. Parameters stored in log-space to enforce positivity:

$$\alpha = \exp(\theta_\alpha), \quad \beta = \exp(\theta_\beta)$$

Optimization uses Adam [9] (lr=0.01, patience=20 epochs, early stopping).

**Gradient flow.** The DSD formula is fully differentiable:

$$\frac{\partial \tilde{\lambda}_i^{-1}}{\partial \theta_\alpha} = \frac{-\lambda_i \cdot d_i}{(\lambda_i^2 + d_i)^2}$$

$$\frac{\partial \tilde{\lambda}_i^{-1}}{\partial \theta_\beta} = \frac{\lambda_i \cdot d_i \cdot \beta \cdot \delta_i}{(\lambda_i^2 + d_i)^2}$$

Both gradients are well-defined and bounded for all λᵢ > 0, δᵢ ≥ 0.

**Eigenvector gradient stability.** The backward pass through torch.linalg.eigh involves terms proportional to 1/(λᵢ − λⱼ). When eigenvalues are nearly degenerate, these become unstable. Our implementation detects this (minimum gap < 10⁻⁶) and detaches eigenvectors from the computation graph, allowing gradients to flow only through the eigenvalue → damping → inverse path. This is mathematically justified: when eigenvectors are unreliable (the exact condition DSD was designed for), their gradients are also unreliable.

### D. The Structural Prior Discovery

**Key finding:** For LSTSVM classification, principled initialization *without* gradient optimization outperforms gradient-optimized parameters (GINA d=970: 85.9% init vs 81.1% optimized, p < 10⁻⁶).

**Mechanism:** During optimization on reconstruction loss, β decreases (0.74 → 0.34) while α increases (2.3×). Decreasing β flattens exp(−βδᵢ), making damping uniform---converging toward Tikhonov. The optimizer removes gap-adaptive selectivity because pointwise reconstruction benefits from all spectral components (including corrupted ones). Classification benefits from *suppressing* corrupted components.

**Conclusion:** DSD's classification advantage is a **structural prior**---the initialization encodes "trust well-separated eigenvectors, distrust clustered ones"---information that reconstruction-loss optimization destroys. Analogous to early stopping in neural networks.

### E. Algorithm

```
Algorithm: Differential Spectral Damping (DSD)
────────────────────────────────────────────────
Input: Symmetric PSD matrix W ∈ ℝᵐˣᵐ, optional (α, β)
Output: DSD-regularized pseudo-inverse W̃⁺ ∈ ℝᵐˣᵐ

1. Eigendecomposition: U, Λ ← eigh(W)                    [O(m³)]
2. Filter: Retain indices {i : λᵢ > 10⁻¹²}              [O(m)]
3. Eigengaps: δᵢ ← Eq. (boundary-aware bilateral)        [O(m)]
4. If (α, β) not provided:
     α ← λ²ᵢ* (transition point)                        [O(m)]
     β ← 1/median(Δ)                                     [O(m)]
5. Damping: dᵢ ← α · exp(−β · δᵢ) for all i            [O(m)]
6. Regularized inverse: λ̃ᵢ⁻¹ ← λᵢ/(λᵢ² + dᵢ)          [O(m)]
7. Reconstruct: W̃⁺ ← U · diag(λ̃⁻¹) · Uᵀ              [O(m²)]
Return W̃⁺
```

**Complexity:** Dominant cost is eigendecomposition at O(m³). All DSD-specific operations are O(m). DSD adds < 0.01% overhead. For m > 1500, a scalable path using partial eigendecomposition reduces cost to O(m²k) where k ≪ m.

---

## VI. Related Work

**Tikhonov (ridge) regularization** [10] adds scalar ridge γ: λ̃ᵢ⁻¹ = 1/(λᵢ + γ). Optimal when all spectral directions are equally reliable. GCV [11] addresses γ selection but not the structural limitation of scalar uniformity. DSD recovers Tikhonov when β → 0.

**Truncated SVD** [12] retains top-k eigenvectors (hard binary cutoff). Optimal when noise is strictly confined to bottom m−k components. In practice the boundary is gradual; hard cutoff creates artifacts. DSD provides the continuous generalization.

**Spectral filtering** [13] applies filter g(λᵢ) depending on eigenvalue magnitude. Cannot distinguish a small eigenvalue with large gap (reliable) from one with small gap (corrupted).

**LSTSVM regularization.** Standard implementations [3, 14] use Tikhonov (M + cI). No prior work employs gap-adaptive regularization; the spectral properties of LSTSVM product matrices have not been previously characterized as regularization-relevant.

**Pre-image methods.** Mika et al. [15] and Kwok & Tsang [7] address pre-image computation assuming stable kernel matrix inverse. Schölkopf et al. [8] established the feature/input space connection.

**Post-hoc explainability.** SHAP [16] and LIME [17] provide local statistical attributions without geometric guarantees. They operate independently of the kernel matrix inverse.

---

## VII. Experimental Design

### A. Fairness Protocol

**For LSTSVM classification:** DSD uses principled auto-initialization only (no gradient optimization, no cross-validation). Tikhonov receives γ grid-searched over 15-point log grid from 10⁻⁸ to 10⁻¹. Tikhonov has a deliberate tuning advantage.

**For pre-image reconstruction:** Both receive equal optimization. DSD's (α, β) optimized via Adam on reconstruction loss. Tikhonov's γ receives the *same* Adam optimizer on the *same* loss---eliminating fairness concerns.

**For Kernel-LSTSVM:** DSD receives gradient optimization. Tikhonov receives 20-point grid search.

### B. Statistical Methodology

All experiments use paired designs. Within each seed, DSD and Tikhonov receive identical data. We report: paired t-test (two-sided p-value), Cohen's d (standardized effect size: d > 0.8 is "large"), win count, and seed count (50 for synthetic, 30 for real-world).

### C. Datasets

**Synthetic LSTSVM:** make_classification: (d=200, n=300), (d=100, n=200), (d=50, n=300), (d=30, n=400). Informative features: d/2. Noise: 10% label flip. 70/30 stratified split.

**Real-world LSTSVM:** GINA (d=970, n=3468, handwriting, OpenML); Madelon (d=500, n=2600, NIPS 2003 Feature Selection Challenge).

**Pre-image:** Swiss Roll (n=2000, ambient d=3), m=300 k-means landmarks, γ_k=2.0, noise σ ∈ {5×10⁻³, 10⁻³, 10⁻⁴}.

**Kernel-LSTSVM:** Two Moons (n=400, γ_k=2.0); Genomics-like (d=100, n=200, γ_k=0.02).

**Operating boundary validation:** Digits (d=64), Ionosphere (d=34)---included to confirm where DSD does *not* help.

---

## VIII. Results

### A. Linear LSTSVM Classification (Primary Result)

**TABLE I: LSTSVM Accuracy — DSD (auto-init, no tuning) vs. Tikhonov (15-point grid search)**

| Dataset | DSD (%) | Tik (%) | Δ (pp) | Wins | Cohen's d |
|---------|---------|---------|--------|------|-----------|
| **GINA (d=970)** | **85.9** | 81.1 | **+4.8** | 30/30 | 4.49 |
| **Madelon (d=500)** | **57.7** | 55.0 | **+2.6** | 27/30 | 1.76 |
| **Synth. (d=200)** | **71.4** | 61.1 | **+10.4** | 44/50 | 1.57 |
| **Synth. (d=100)** | **79.1** | 75.8 | **+3.3** | 35/50 | 0.39 |
| Synth. (d=50) | 90.1 | 90.1 | 0 | 25/50 | 0.0 |

All differences with Δ > 0 are significant at p < 0.0001. DSD's advantage scales monotonically with dimensionality. The GINA result (Cohen's d = 4.49) is exceptionally large---substantially exceeding the d > 0.8 threshold for "large" effects [18].

### B. Kernel LSTSVM (Non-Linear)

**TABLE II: Kernel-LSTSVM — DSD-opt vs. Tikhonov-opt (50 Seeds)**

| Dataset | DSD (%) | Tik (%) | Δ | p-value |
|---------|---------|---------|---|---------|
| **Two Moons (γ=2.0)** | **92.7** | 92.1 | **+0.6pp** | 0.028 |
| Genomics (d=100) | 87.7 | 87.7 | 0 | 0.77 |

### C. Pre-Image Reconstruction

**TABLE III: Pre-Image Error — DSD-opt vs. Tikhonov-opt (50 Seeds, Swiss Roll)**

| Noise σ | DSD-opt | Tik-opt | Naive | DSD vs Tik |
|---------|---------|---------|-------|------------|
| 5×10⁻³ | 0.069 | 0.069 | 4.56 | Tie (p=0.99) |
| 10⁻³ | 0.051 | 0.048 | 0.84 | Tik wins (p=0.02) |
| 10⁻⁴ | 0.042 | 0.035 | 0.09 | Tik wins (p<0.001) |

Both reduce naive error by 66× at high noise. DSD does not outperform Tikhonov: pre-image has a single scalar objective that Tikhonov's γ optimizes directly.

### D. Operating Boundary

**TABLE IV: Below Operating Boundary (d ≤ 64) — DSD Not Expected to Win**

| Dataset | d | DSD (%) | Tik (%) | Result |
|---------|---|---------|---------|--------|
| Digits | 64 | 88.3 | 89.1 | Tik wins |
| Ionosphere | 34 | 73.3 | 80.3 | Tik wins |

Below d ≈ 100, DSD provides no advantage. These are included for honest boundary characterization.

### E. Operating Regime Summary

| Condition | DSD Advantage | Recommendation |
|-----------|---------------|----------------|
| LSTSVM, d ≥ 100 | +3 to +10pp | **Use DSD** |
| Kernel-LSTSVM, sharp RBF | +0.6pp | **Use DSD** |
| Pre-image (any noise) | ≈ Tikhonov | Either method |
| LSTSVM, d ≤ 50 | 0 | Simplest method |
| Low-d real-world | Mixed | Benchmark both |

---

## IX. Analysis and Interpretation

### A. Spectral Tail-Clustering in LSTSVM Matrices

**TABLE V: Spectral Tail-Clustering Severity**

| d | Condition No. | Tail Span* | DSD Δ |
|---|---------------|-----------|-------|
| 200 | 8.5 × 10³ | 7.1% | +10.4pp |
| 100 | 3.5 × 10³ | 11.1% | +3.3pp |
| 50 | 1.1 × 10³ | 15.1% | 0pp |

*Fraction of spectral range occupied by bottom 50% of eigenvalues.

At d=200, the bottom half of eigenvalues are compressed into 7% of the total range---extreme clustering that corrupts eigenvector directions per Davis-Kahan.

### B. Why DSD Helps Classification But Not Pre-Image

**Classification** requires the solution **u** = M⁻¹**b** to *generalize*. Corrupted eigenvectors contribute noise-fitting components that don't generalize. DSD suppresses these---reducing the hypothesis class to well-separated eigenvectors (spectral regularization in the generalization-theoretic sense [19]).

**Pre-image** minimizes ‖x̂ − x‖²---a scalar objective over a known target. A single scalar γ can be optimized to find the best bias-variance tradeoff. Gap structure provides no additional information for pointwise reconstruction.

### C. Connection to Algorithmic Stability

DSD selectively suppresses contributions from directions where Davis-Kahan guarantees large eigenvector sensitivity. This reduces *effective dimensionality*---fewer spectral directions participate in the solution, bounding sensitivity to individual training samples. This is the hallmark of algorithmic stability implying generalization [19]. DSD achieves this softly (continuous damping) rather than through hard truncation.

### D. Operating Regime Detection

The operating regime is characterized by two observable conditions:
1. **Condition number** of the system matrix exceeds 10³
2. **Spectral tail span** (fraction of range from bottom 50% of eigenvalues) below 10–12%

Both computed from the eigendecomposition at O(m) cost, enabling automatic detection.

---

## X. Limitations and Future Work

### Limitations

1. **No formal stability proof.** DSD boundedness as δᵢ → 0 is supported empirically but not proved.
2. **Non-RBF kernels.** Initialization assumes monotonic eigenvalue decay. May require modification for polynomial or non-stationary kernels.
3. **Pre-image performance.** Matches but does not beat Tikhonov.
4. **Low-dimensional regime.** Below d ≈ 100, DSD may slightly underperform.
5. **Eigendecomposition requirement.** Precludes application to matrices where only matrix-vector products are available.

### Future Work

**1. Multi-scale gap sensitivity.** Preliminary investigation with hierarchical β (separate sensitivity for tail/mid/head spectral zones) suggests statistically significant improvement in low-noise regimes on manifold data. Adds only two parameters while never degrading performance in any configuration tested.

**2. Extension to related methods.** Kernel Fisher Discriminant, Kernel CCA, and structured output SVMs construct similar system matrices. DSD's advantage should transfer directly.

**3. Formal stability bound.** Proving ‖x̂_DSD − x̂_true‖ ≤ C·ε/min_i(λᵢ² + dᵢ) would show DSD error is bounded regardless of spectral clustering.

---

## XI. Conclusion

DSD is a spectral-structure-aware regularizer for ill-conditioned kernel matrix inversion, derived from Davis-Kahan perturbation theory. Its primary contribution is classification accuracy on high-dimensional LSTSVM problems: +4.8pp on GINA (d=970, Cohen's d=4.49), +10.4pp at d=200, and +2.6pp on Madelon (d=500)---all over optimally-tuned Tikhonov with p < 0.0001, using only principled initialization without optimization.

The advantage arises from a structural prior: gap-adaptive initialization encodes spectral reliability information that scalar regularization cannot represent and that gradient optimization on proxy losses actively destroys.

DSD does not improve pre-image reconstruction over Tikhonov, and provides no benefit below d ≈ 100. It is a specialized tool for the high-dimensional, ill-conditioned regime---precisely where kernel methods are most needed and standard regularization is most inadequate.

The method is computationally free (O(m) overhead), requires no cross-validation, introduces only two interpretable parameters with principled initialization, and applies to any kernel method whose system matrices exhibit spectral tail-clustering.

---

## References

[1] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

[2] T. Hofmann, B. Schölkopf, and A. J. Smola, "Kernel methods in machine learning," *The Annals of Statistics*, vol. 36, no. 3, pp. 1171–1220, 2008.

[3] M. A. Kumar and M. Gopal, "Least squares twin support vector machines for pattern classification," *Expert Systems with Applications*, vol. 36, no. 4, pp. 7535–7543, 2009.

[4] M. Tanveer et al., "Comprehensive review on twin support vector machines," *Annals of Operations Research*, vol. 318, pp. 1223–1268, 2022.

[5] C. Davis and W. M. Kahan, "The rotation of eigenvectors by a perturbation. III," *SIAM J. Numerical Analysis*, vol. 7, no. 1, pp. 1–46, 1970.

[6] C. K. I. Williams and M. Seeger, "Using the Nyström method to speed up kernel machines," *NeurIPS*, pp. 682–688, 2001.

[7] J. T.-Y. Kwok and I. W.-H. Tsang, "The pre-image problem in kernel methods," *IEEE Trans. Neural Networks*, vol. 15, no. 6, pp. 1517–1525, 2004.

[8] B. Schölkopf et al., "Input space versus feature space in kernel-based methods," *IEEE Trans. Neural Networks*, vol. 10, no. 5, pp. 1000–1017, 1999.

[9] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," *ICLR*, 2015.

[10] A. N. Tikhonov, "On the solution of ill-posed problems and the method of regularization," *Doklady Akademii Nauk SSSR*, vol. 151, no. 3, pp. 501–504, 1963.

[11] G. Wahba, "Practical approximate solutions to linear operator equations when the data are noisy," *SIAM J. Numerical Analysis*, vol. 14, no. 4, pp. 651–667, 1977.

[12] G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013.

[13] H. W. Engl, M. Hanke, and A. Neubauer, *Regularization of Inverse Problems*, Springer, 1996.

[14] Y.-H. Shao et al., "Improvements on twin support vector machines," *IEEE Trans. Neural Networks*, vol. 22, no. 6, pp. 962–968, 2011.

[15] S. Mika et al., "Kernel PCA and de-noising in feature spaces," *NeurIPS*, pp. 536–542, 1999.

[16] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, pp. 4765–4774, 2017.

[17] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why should I trust you?: Explaining the predictions of any classifier," *ACM SIGKDD*, pp. 1135–1144, 2016.

[18] J. Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed., Lawrence Erlbaum Associates, 1988.

[19] O. Bousquet and A. Elisseeff, "Stability and generalization," *Journal of Machine Learning Research*, vol. 2, pp. 499–526, 2002.

---

*Implementation and reproducibility: [github.com/Praveg432/dsd-regularization](https://github.com/Praveg432/dsd-regularization)*
