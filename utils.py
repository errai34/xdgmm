from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import torch
from matplotlib.patches import Ellipse


def generate_synthetic_data(
    n_samples: int,
    n_components: int,
    n_dimensions: int = 2,
    random_state: int = None,
    noise_scale: float = 0.1,
    separation: float = 5.0,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data from a GMM with measurement uncertainties

    Args:
        n_samples: Number of samples
        n_components: Number of components
        n_dimensions: Number of dimensions
        random_state: Random seed
        noise_scale: Scale of measurement noise
        separation: Average distance between components
        device: Device to use

    Returns:
        Tuple of (X, true_labels, noise_covars, component_params)
            X: Observed data with noise
            true_labels: True component labels
            noise_covars: Measurement uncertainty covariance matrices
            component_params: Dict with 'means' and 'covariances' of true components
    """
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate component parameters
    means = torch.randn(n_components, n_dimensions, device=device) * separation

    # Generate random covariance matrices
    # First create random rotation matrices
    covariances = []
    for _ in range(n_components):
        # Create a random matrix
        A = torch.randn(n_dimensions, n_dimensions, device=device)
        # Create a positive definite matrix
        cov = torch.matmul(A, A.T)
        # Scale to reasonable size
        cov = cov / cov.norm() * torch.rand(1, device=device) * 2.0
        covariances.append(cov)

    covariances = torch.stack(covariances)

    # Generate samples from each component
    samples_per_component = torch.multinomial(
        torch.ones(n_components, device=device), n_samples, replacement=True
    )

    X_clean = torch.zeros(n_samples, n_dimensions, device=device)
    labels = torch.zeros(n_samples, dtype=torch.long, device=device)

    idx = 0
    for k in range(n_components):
        n = samples_per_component[k].item()
        if n > 0:
            # Sample from multivariate normal
            mvn = torch.distributions.MultivariateNormal(
                loc=means[k], covariance_matrix=covariances[k]
            )
            X_clean[idx : idx + n] = mvn.sample((n,))
            labels[idx : idx + n] = k
            idx += n

    # Generate random noise covariances for each sample
    noise_covars = torch.zeros(n_samples, n_dimensions, n_dimensions, device=device)

    for i in range(n_samples):
        # Create a random noise covariance matrix
        A = torch.randn(n_dimensions, n_dimensions, device=device) * noise_scale
        noise_cov = torch.matmul(A, A.T)
        noise_covars[i] = noise_cov

    # Add noise to samples
    X_noisy = torch.zeros_like(X_clean)

    for i in range(n_samples):
        noise_dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(n_dimensions, device=device),
            covariance_matrix=noise_covars[i],
        )
        X_noisy[i] = X_clean[i] + noise_dist.sample()

    component_params = {"means": means, "covariances": covariances}

    return X_noisy, labels, noise_covars, component_params


def plot_confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Plot an ellipse representing a covariance matrix

    Args:
        mean: The center of the ellipse
        cov: The covariance matrix
        ax: The matplotlib axis
        n_std: Number of standard deviations
        facecolor: Fill color
        **kwargs: Additional arguments for Ellipse
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Create the ellipse with unit radius
    ellipse = Ellipse(
        (0, 0),
        width=2.0,  # Unit circle initially
        height=2.0,
        facecolor=facecolor,
        **kwargs,
    )

    # Compute angle
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Scaling factor for standard deviations
    scale_x = np.sqrt(eigenvalues[0]) * n_std
    scale_y = np.sqrt(eigenvalues[1]) * n_std

    # Create transformation
    transf = (
        transforms.Affine2D()
        .rotate_deg(angle)
        .scale(scale_x, scale_y)
        .translate(mean[0], mean[1])
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_gmm_results(
    X, noise_covars, true_params, model, n_samples=0, figsize=(14, 6)
):
    """
    Simple, ultra-clear visualization comparing true and fitted models
    
    Args:
        X: Observed data
        noise_covars: Measurement uncertainties 
        true_params: True component parameters
        model: Fitted DeconvGMM model
        n_samples: Not used
        figsize: Figure size
    """
    # Convert tensors to numpy
    X_np = X.detach().cpu().numpy()
    
    # Create figure with two subplots - side by side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Define colors for clusters
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    # --- PLOT 1: True Distribution ---
    true_means_np = true_params["means"].detach().cpu().numpy()
    true_covs_np = true_params["covariances"].detach().cpu().numpy()
    
    # True cluster assignments (by proximity to true centers)
    true_assignments = []
    for i in range(len(X_np)):
        dists = np.sum((X_np[i] - true_means_np)**2, axis=1)
        true_assignments.append(np.argmin(dists))
    true_assignments = np.array(true_assignments)
    
    # Plot data colored by true assignments
    for k in range(model.n_components):
        mask = true_assignments == k
        ax1.scatter(
            X_np[mask, 0], 
            X_np[mask, 1], 
            s=15, 
            alpha=0.5,
            color=colors[k],
            label=f"Cluster {k+1}"
        )
    
    # Plot true component ellipses - bold and clear
    for k in range(len(true_means_np)):
        # Plot center
        ax1.scatter(
            true_means_np[k, 0],
            true_means_np[k, 1],
            s=200,
            marker='X',
            color=colors[k],
            edgecolor='black',
            linewidth=1.5,
            zorder=100
        )
        
        # Plot ellipse - thick line
        plot_confidence_ellipse(
            true_means_np[k],
            true_covs_np[k],
            ax1,
            n_std=2.0,
            edgecolor=colors[k],
            linewidth=3,
            alpha=0.8
        )
    
    ax1.set_xlabel("X", fontsize=12)
    ax1.set_ylabel("Y", fontsize=12)
    ax1.set_title("True GMM Components", fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # --- PLOT 2: Fitted Distribution ---
    # Get fitted parameters
    fitted_means_np = model.means.detach().cpu().numpy()
    fitted_covs_np = model.covariances.detach().cpu().numpy()
    
    # Get predicted cluster assignments
    pred_probs = model.predict_proba((X, noise_covars)).detach().cpu().numpy()
    pred_assignments = np.argmax(pred_probs, axis=1)
    
    # Plot data colored by predicted assignments
    for k in range(model.n_components):
        mask = pred_assignments == k
        ax2.scatter(
            X_np[mask, 0], 
            X_np[mask, 1], 
            s=15, 
            alpha=0.5,
            color=colors[k],
            label=f"Cluster {k+1}"
        )
    
    # Plot fitted component ellipses - bold and clear
    for k in range(len(fitted_means_np)):
        # Plot center
        ax2.scatter(
            fitted_means_np[k, 0],
            fitted_means_np[k, 1],
            s=200,
            marker='o',
            color=colors[k],
            edgecolor='black',
            linewidth=1.5,
            zorder=100
        )
        
        # Plot ellipse
        plot_confidence_ellipse(
            fitted_means_np[k],
            fitted_covs_np[k],
            ax2,
            n_std=2.0,
            edgecolor=colors[k],
            linewidth=3,
            alpha=0.8
        )
    
    ax2.set_xlabel("X", fontsize=12)
    ax2.set_ylabel("Y", fontsize=12)
    ax2.set_title("Fitted XDGMM Components", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Ensure both plots have the same limits for fair comparison
    xlim = [min(ax1.get_xlim()[0], ax2.get_xlim()[0]), max(ax1.get_xlim()[1], ax2.get_xlim()[1])]
    ylim = [min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1])]
    
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    
    # Add parameter comparison
    parameter_text = "Model Parameters:\n"
    mse_means = 0
    mse_covs = 0
    
    # Calculate mean errors between true and fitted
    for k in range(model.n_components):
        mean_diff = np.linalg.norm(true_means_np[k] - fitted_means_np[k])
        cov_diff = np.linalg.norm(true_covs_np[k] - fitted_covs_np[k])
        mse_means += mean_diff**2
        mse_covs += cov_diff**2
    
    mse_means /= model.n_components
    mse_covs /= model.n_components
    
    parameter_text += f"Mean MSE: {mse_means:.4f}\n"
    parameter_text += f"Cov MSE: {mse_covs:.4f}"
    
    # Add text to figure
    plt.figtext(0.5, 0.01, parameter_text, ha="center", fontsize=10, 
                bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the parameter text
    
    return fig
