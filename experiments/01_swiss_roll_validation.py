"""
Experiment 01: Swiss Roll Topological Verification

Demonstrates that DSD produces manifold-consistent pre-images while
Tikhonov/naive methods scatter off-manifold.

Run: python experiments/01_swiss_roll_validation.py
Output: results/figures/swiss_roll_comparison.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from src.dsd import dsd_regularized_inverse, DSDResult
from src.kernels import rbf_kernel_matrix, rbf_kernel_vector, nystrom_sample
from src.preimage import preimage_pipeline


def generate_boundary_points(model, X, n_points=100):
    """Find points near the SVM decision boundary."""
    decision_values = model.decision_function(X)
    # Points closest to the boundary (decision_function ≈ 0)
    boundary_indices = np.argsort(np.abs(decision_values))[:n_points]
    return X[boundary_indices]


def run_experiment():
    print("=" * 60)
    print("DSD-SVM Experiment 01: Swiss Roll Topological Verification")
    print("=" * 60)
    
    # Generate Swiss Roll data
    np.random.seed(42)
    n_samples = 3000
    X, color = make_swiss_roll(n_samples=n_samples, noise=0.3, random_state=42)
    
    # Binary classification: split by color (manifold parameter)
    median_color = np.median(color)
    y = (color > median_color).astype(int)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train RBF-SVM
    gamma = 0.5
    svm = SVC(kernel='rbf', gamma=gamma, C=10.0)
    svm.fit(X_scaled, y)
    print(f"SVM accuracy: {svm.score(X_scaled, y):.3f}")
    
    # Find boundary points
    boundary_points = generate_boundary_points(svm, X_scaled, n_points=150)
    print(f"Boundary points selected: {len(boundary_points)}")
    
    # Compute pre-images with each method
    m = 300  # Nyström landmarks
    methods = ["dsd", "tikhonov", "tsvd", "naive"]
    preimages = {}
    
    for method in methods:
        print(f"\nComputing pre-images: {method}...")
        try:
            preimages[method] = preimage_pipeline(
                X_train=X_scaled,
                X_boundary=boundary_points,
                gamma=gamma,
                m=m,
                method=method,
                tikhonov_gamma=1e-2,
                tsvd_rank=50,
            )
            
            # Compute reconstruction error
            mse = np.mean(np.sum((preimages[method] - boundary_points) ** 2, axis=1))
            print(f"  {method}: MSE = {mse:.4f}")
        except Exception as e:
            print(f"  {method}: FAILED — {e}")
            preimages[method] = None
    
    # Visualization
    fig = plt.figure(figsize=(16, 12))
    
    titles = {
        "dsd": "DSD (Proposed)",
        "tikhonov": "Tikhonov Regularization",
        "tsvd": "Truncated SVD",
        "naive": "Naive Pseudo-Inverse",
    }
    
    for idx, (method, title) in enumerate(titles.items()):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        
        # Plot original data (light)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
                   c=color, cmap='coolwarm', alpha=0.05, s=1)
        
        # Plot boundary points (green)
        ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
                   c='green', s=15, alpha=0.5, label='True boundary')
        
        # Plot pre-images (red)
        if preimages[method] is not None:
            ax.scatter(preimages[method][:, 0], preimages[method][:, 1], preimages[method][:, 2],
                       c='red', s=20, alpha=0.8, label=f'{method} pre-image')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'figures'), exist_ok=True)
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'figures')

    plt.suptitle('Pre-Image Reconstruction on Swiss Roll Manifold\n'
                 'DSD preserves manifold structure; baselines scatter off-manifold',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'swiss_roll_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: results/figures/swiss_roll_comparison.png")
    plt.close()
    
    # Print spectral analysis
    print("\n" + "=" * 60)
    print("SPECTRAL ANALYSIS (DSD internals)")
    print("=" * 60)
    
    landmarks, _ = nystrom_sample(X_scaled, m=m, method="kmeans")
    W = rbf_kernel_matrix(landmarks, gamma=gamma)
    result = dsd_regularized_inverse(W)
    
    print(f"  Eigenvalue range: [{result.eigenvalues[0]:.2e}, {result.eigenvalues[-1]:.2e}]")
    print(f"  Condition number (original): {result.condition_number_original:.2e}")
    print(f"  Condition number (DSD-regularized): {result.condition_number_regularized:.2e}")
    print(f"  Condition improvement: {result.condition_number_original / result.condition_number_regularized:.1f}x")
    print(f"  α (auto): {result.alpha:.4f}")
    print(f"  β (auto): {result.beta:.4f}")
    print(f"  Eigengap range: [{result.eigengaps.min():.2e}, {result.eigengaps.max():.2e}]")
    print(f"  Damping range: [{result.damping.min():.2e}, {result.damping.max():.2e}]")
    
    # Plot eigenvalue spectrum + damping
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].semilogy(result.eigenvalues, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Eigenvalue (log scale)')
    axes[0].set_title('Eigenvalue Spectrum (Exponential Decay)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(result.eigengaps, 'g-', linewidth=1.5)
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigengap δᵢ')
    axes[1].set_title('Localized Eigengaps')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].semilogy(result.damping, 'r-', linewidth=1.5, label='DSD damping')
    axes[2].axhline(y=result.alpha, color='k', linestyle='--', alpha=0.5, label=f'α={result.alpha:.3f}')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('Damping magnitude (log scale)')
    axes[2].set_title('DSD Adaptive Damping')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'spectral_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"Figure saved: results/figures/spectral_analysis.png")
    plt.close()
    
    print("\n✓ Experiment 01 complete.")


if __name__ == "__main__":
    run_experiment()
