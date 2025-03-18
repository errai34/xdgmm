from typing import Tuple

import torch
import torch.distributions as dist

from .gmm import GMM


class DeconvGMM(GMM):
    """
    Extreme Deconvolution Gaussian Mixture Model

    Extends GMM to handle measurement uncertainties.
    """

    def log_likelihood(self, data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculate log likelihood accounting for measurement uncertainty

        Args:
            data: Tuple of (X, uncertainty_covars)
                X: Data tensor of shape (n_samples, n_dimensions)
                uncertainty_covars: Uncertainty covariance matrices of shape
                                   (n_samples, n_dimensions, n_dimensions)

        Returns:
            Log likelihood of shape (n_samples,)
        """
        X, noise_covars = data
        return torch.logsumexp(self._log_component_likelihood(X, noise_covars), dim=1)

    def _log_component_likelihood(
        self, X: torch.Tensor, noise_covars: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the log likelihood of each component with uncertainty

        Args:
            X: Data tensor of shape (n_samples, n_dimensions)
            noise_covars: Uncertainty covariance matrices of shape
                         (n_samples, n_dimensions, n_dimensions)

        Returns:
            Component log likelihoods of shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        n_dims = X.shape[1]
        device = X.device
        
        # Pre-allocate output tensor
        log_likelihoods = torch.full((n_samples, self.n_components), -1e10, device=device)
        
        # Precompute log weights once - with numerical safeguards
        log_weights = torch.log(torch.clamp(self.weights, min=1e-10))
        
        # Constants for multivariate normal distribution
        log_2pi = torch.log(torch.tensor(2 * torch.pi, device=device))
        
        # Process in small batches to avoid memory issues
        batch_size = 64  
        
        for k in range(self.n_components):
            # Get this component's mean and covariance matrix
            mean_k = self.means[k]
            cov_k = self.covariances[k]
            
            # Process data in batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_size_actual = end_idx - start_idx
                
                # Get data and noise for this batch
                batch_X = X[start_idx:end_idx]
                batch_noise = noise_covars[start_idx:end_idx]
                
                # Compute results for this batch
                batch_log_probs = torch.empty(batch_size_actual, device=device)
                
                for i in range(batch_size_actual):
                    try:
                        # Total covariance = component covariance + noise covariance
                        total_cov = cov_k + batch_noise[i]
                        
                        # Add regularization to ensure positive definiteness
                        total_cov = total_cov + torch.eye(n_dims, device=device) * 1e-6
                        
                        # Manual log_prob calculation (more reliable than torch.distributions)
                        # Steps: 1) Compute Cholesky, 2) Solve linear system, 3) Compute quadratic form
                        
                        try:
                            # Try Cholesky decomposition
                            L = torch.linalg.cholesky(total_cov)
                            
                            # Compute centered points
                            diff = batch_X[i] - mean_k
                            
                            # Solve triangular system Lx = diff
                            alpha = torch.triangular_solve(diff.unsqueeze(1), L, upper=False)[0].squeeze()
                            
                            # Compute quadratic term (diff)ᵀ Σ⁻¹ (diff)
                            quad = torch.sum(alpha**2)
                            
                            # Log determinant via sum of log of diagonal elements of L
                            logdet = 2 * torch.sum(torch.log(torch.diagonal(L)))
                            
                            # Final log probability
                            log_prob = -0.5 * (n_dims * log_2pi + logdet + quad)
                            
                            batch_log_probs[i] = log_prob
                            
                        except Exception as e:
                            # Fallback using torch.distributions if Cholesky fails
                            try:
                                mvn = dist.MultivariateNormal(
                                    loc=mean_k,
                                    covariance_matrix=total_cov
                                )
                                batch_log_probs[i] = mvn.log_prob(batch_X[i])
                            except:
                                # If all fails, use very low likelihood
                                batch_log_probs[i] = torch.tensor(-1e10, device=device)
                                
                    except Exception as e:
                        # Any unexpected errors lead to low likelihood
                        batch_log_probs[i] = torch.tensor(-1e10, device=device)
                
                # Store results with log weights
                log_likelihoods[start_idx:end_idx, k] = batch_log_probs + log_weights[k]
        
        # Apply additional safeguards
        # 1. Replace NaN and inf values
        log_likelihoods = torch.nan_to_num(log_likelihoods, nan=-1e10, posinf=-1e10, neginf=-1e10)
        
        # 2. Apply max clamping to prevent overflow
        log_likelihoods = torch.clamp(log_likelihoods, max=1e6)
        
        return log_likelihoods

    def predict_proba(self, data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculate posterior probabilities with uncertainty

        Args:
            data: Tuple of (X, uncertainty_covars)
                X: Data tensor of shape (n_samples, n_dimensions)
                uncertainty_covars: Uncertainty covariance matrices

        Returns:
            Posterior probabilities of shape (n_samples, n_components)
        """
        X, noise_covars = data
        log_component_likelihood = self._log_component_likelihood(X, noise_covars)
        log_prob = torch.logsumexp(log_component_likelihood, dim=1, keepdim=True)
        return torch.exp(log_component_likelihood - log_prob)

    def sample(self, n_samples: int, with_noise: bool = False) -> torch.Tensor:
        """
        Generate samples from the model

        Args:
            n_samples: Number of samples to generate
            with_noise: Whether to add measurement noise

        Returns:
            Samples of shape (n_samples, n_dimensions)
        """
        # Get clean samples from the base implementation
        samples = super().sample(n_samples)

        if with_noise:
            # This would require noise covariances which aren't available in this context
            # In a real application, you might generate random noise covariances
            pass

        return samples
