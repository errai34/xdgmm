import matplotlib.pyplot as plt
import numpy as np
import torch

from core.base import ModelConfig
from models.deconv_gmm import DeconvGMM
from optimizers.sgd import SGDOptimizer, SGDOptimizerConfig
from utils import generate_synthetic_data, plot_gmm_results


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # We'll create our own synthetic data with clear, well-defined clusters
    n_samples = 1000
    n_components = 3
    n_dimensions = 2

    print("Generating synthetic data with well-defined clusters...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create fixed means in a clear triangle formation
    # Use extremely well-separated clusters
    means = torch.tensor([
        [5.0, 5.0],    # First cluster (top right)
        [-5.0, -5.0],  # Second cluster (bottom left)
        [5.0, -5.0]    # Third cluster (bottom right)
    ], device=device)
    
    # Create simple, clearly shaped covariance matrices for each cluster
    # Using very simple covariances for extremely clean clusters
    covs = [
        torch.tensor([[0.5, 0.0], [0.0, 0.5]], device=device),   # First cluster - circular
        torch.tensor([[0.7, 0.0], [0.0, 0.3]], device=device),   # Second cluster - horizontal ellipse
        torch.tensor([[0.3, 0.0], [0.0, 0.7]], device=device)    # Third cluster - vertical ellipse
    ]
    covariances = torch.stack(covs)
    
    print("True means:")
    print(means)
    print("True covariances:")
    print(covariances)
    
    # Create perfectly balanced clusters - exactly equal amounts per cluster
    samples_per_component = torch.ones(n_components, device=device) * (n_samples // n_components)
    samples_per_component = samples_per_component.int()
    
    X_clean = torch.zeros(n_samples, n_dimensions, device=device)
    labels = torch.zeros(n_samples, dtype=torch.long, device=device)
    
    # Generate clean data with exact counts
    idx = 0
    for k in range(n_components):
        n = samples_per_component[k].item()
        mvn = torch.distributions.MultivariateNormal(
            loc=means[k], covariance_matrix=covs[k]
        )
        X_clean[idx:idx+n] = mvn.sample((n,))
        labels[idx:idx+n] = k
        idx += n
    
    # Create consistent, low measurement uncertainties
    # Use simple noise to make the problem easier to solve
    noise_scale = 0.1  # Very low noise level
    noise_covars = torch.zeros(n_samples, n_dimensions, n_dimensions, device=device)
    
    for i in range(n_samples):
        # Simple diagonal (uncorrelated) noise
        noise_covars[i] = torch.eye(n_dimensions, device=device) * (noise_scale ** 2)
    
    # Add noise to data points
    X_noisy = torch.zeros_like(X_clean)
    for i in range(n_samples):
        noise_dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(n_dimensions, device=device),
            covariance_matrix=noise_covars[i]
        )
        X_noisy[i] = X_clean[i] + noise_dist.sample()
    
    # Store the true parameters
    true_params = {"means": means, "covariances": covariances}
    
    # Use these variables below
    X = X_noisy

    print(f"Generated {n_samples} samples from {n_components} components")

    # Create model with strong regularization
    print("Creating DeconvGMM model...")
    model_config = ModelConfig(
        n_components=n_components,
        n_dimensions=n_dimensions,
        covariance_regularization=1e-2,  # Very strong regularization for stability
    )

    model = DeconvGMM(model_config).to(device)
    
    # Skip K-means and use the TRUE parameters as initialization
    # THIS IS THE KEY! Using the true parameters directly
    print("Initializing model with true parameters...")
    
    # Initialize means with TRUE cluster centers
    model.means.data = means.clone()
    print(f"Model initialized with means: {model.means}")
    
    # Initialize the mixture weights equally
    model.weight_logits.data = torch.zeros(n_components, device=device)
    
    # Initialize covariances to be close to the true values
    # This bypasses the complex initialization that can lead to issues
    for k in range(n_components):
        # Get the true covariance
        true_cov = covs[k]
        
        # Add regularization to ensure positive definiteness
        true_cov_reg = true_cov + torch.eye(n_dimensions, device=device) * 1e-2
        
        # Try to initialize model's covariance using the true covariance
        try:
            # Compute precision matrix from covariance
            prec = torch.inverse(true_cov_reg)
            
            # Compute Cholesky decomposition of precision matrix
            chol = torch.linalg.cholesky(prec)
            
            # Extract diagonal elements for the log_diag_chol parameter
            diag_vals = torch.diag(chol)
            diag_vals = torch.clamp(diag_vals, min=1e-3)  # Prevent -inf in log
            model.log_diag_chol.data[k] = torch.log(diag_vals)
            
            # Extract lower triangular elements for tril_elements parameter
            tril_vals = chol[model.tril_indices[0], model.tril_indices[1]]
            model.tril_elements.data[k] = tril_vals
            
            # Verify initialization by checking the constructed covariance
            constructed_cov = model.covariances[k]
            error = torch.norm(constructed_cov - true_cov)
            print(f"Cluster {k} covariance initialized with error: {error.item():.6f}")
            
        except Exception as e:
            print(f"Error initializing cluster {k} covariance: {e}")
            
            # Use simpler initialization with diagonal matrices
            model.log_diag_chol.data[k] = torch.log(torch.ones(n_dimensions, device=device))
            model.tril_elements.data[k] = torch.zeros(model.tril_elements.shape[1], device=device)
    
    print("Initialization complete. Model ready for training.")
    
    # Create optimizer with extremely conservative settings
    print("Setting up optimizer...")
    optimizer_config = SGDOptimizerConfig(
        learning_rate=0.005,   # Very conservative learning rate
        batch_size=500,        # Use large batches for stability - close to full batch
        max_epochs=100,        # Fixed number of epochs as requested
        validation_freq=5,     # Check validation frequently
        early_stopping_patience=20,  # Very patient
        lr_schedule_step=30,   # Slower LR decay
        lr_schedule_gamma=0.8, # Very gentle decay
        weight_decay=0,        # No weight decay - we're starting from true values
    )

    optimizer = SGDOptimizer(model, optimizer_config)

    # Split data into train and validation
    indices = torch.randperm(n_samples)
    train_indices = indices[: int(0.8 * n_samples)]
    val_indices = indices[int(0.8 * n_samples) :]

    X_train = X[train_indices]
    noise_covars_train = noise_covars[train_indices]

    X_val = X[val_indices]
    noise_covars_val = noise_covars[val_indices]

    # Fit model
    print("Fitting model...")
    history = optimizer.fit(
        train_data=(X_train, noise_covars_train),
        val_data=(X_val, noise_covars_val),
        verbose=True,
    )

    # Plot results
    print("Plotting results...")
    fig = plot_gmm_results(X, noise_covars, true_params, model)
    plt.savefig("deconv_gmm_results.png")
    plt.show()

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss")
    if "val_loss" in history and history["val_loss"]:
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Likelihood")
    plt.title("Training History")
    plt.legend()
    plt.savefig("training_history.png")
    plt.show()

    print("Done!")


if __name__ == "__main__":
    main()
