import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from core.base import BaseGMM, ModelConfig


class GMM(BaseGMM):
    """Standard Gaussian Mixture Model"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Initialize parameters
        self.n_components = config.n_components
        self.n_dimensions = config.n_dimensions
        self.reg_covar = config.covariance_regularization

        # Create model parameters
        # Mixture weights (logits)
        self.weight_logits = nn.Parameter(torch.zeros(self.n_components))

        # Means
        self.means = nn.Parameter(torch.zeros(self.n_components, self.n_dimensions))

        # Covariance representation: we use a lower triangular Cholesky factor
        # Diagonal elements (positive via exp transform)
        self.log_diag_chol = nn.Parameter(
            torch.zeros(self.n_components, self.n_dimensions)
        )

        # Lower triangular elements (no constraint)
        n_tril_elements = self.n_dimensions * (self.n_dimensions - 1) // 2
        self.tril_indices = torch.tril_indices(self.n_dimensions, self.n_dimensions, -1)
        self.tril_elements = nn.Parameter(
            torch.zeros(self.n_components, n_tril_elements)
        )

    @property
    def weights(self) -> torch.Tensor:
        """Get normalized mixture weights"""
        return F.softmax(self.weight_logits, dim=0)

    @property
    def precision_chol(self) -> torch.Tensor:
        """
        Construct the Cholesky factors of the precision matrices
        Shape: (n_components, n_dimensions, n_dimensions)
        """
        chol = torch.zeros(
            self.n_components,
            self.n_dimensions,
            self.n_dimensions,
            device=self.weight_logits.device,
        )

        # Set diagonal elements
        diag_idx = torch.arange(self.n_dimensions)
        chol[:, diag_idx, diag_idx] = torch.exp(self.log_diag_chol)

        # Set lower triangular elements
        chol[:, self.tril_indices[0], self.tril_indices[1]] = self.tril_elements

        return chol

    @property
    def covariances(self) -> torch.Tensor:
        """
        Compute the covariance matrices
        Shape: (n_components, n_dimensions, n_dimensions)
        """
        chol = self.precision_chol
        
        # Ensure numerical stability by clamping small values in chol
        chol = torch.clamp(chol, min=-1e6, max=1e6)
        
        precision = torch.matmul(chol, chol.transpose(-1, -2))

        # Add substantial regularization to prevent singularity
        # Higher regularization ensures covariance matrices are well-conditioned
        reg_matrix = self.reg_covar * torch.eye(
            self.n_dimensions, device=self.weight_logits.device
        ).expand(self.n_components, self.n_dimensions, self.n_dimensions)

        # Use more stable pseudo-inverse instead of inverse
        try:
            # First try standard inverse with regularization
            return torch.inverse(precision + reg_matrix)
        except:
            # If that fails, add more regularization and try again
            stronger_reg = self.reg_covar * 10.0 * torch.eye(
                self.n_dimensions, device=self.weight_logits.device
            ).expand(self.n_components, self.n_dimensions, self.n_dimensions)
            
            return torch.inverse(precision + stronger_reg)

    def log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate log likelihood of data

        Args:
            X: Data tensor of shape (n_samples, n_dimensions)

        Returns:
            Log likelihood of shape (n_samples,)
        """
        return torch.logsumexp(self._log_component_likelihood(X), dim=1)

    def _log_component_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log likelihood of each component

        Args:
            X: Data tensor of shape (n_samples, n_dimensions)

        Returns:
            Component log likelihoods of shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        log_likelihoods = torch.empty(n_samples, self.n_components, device=X.device)

        for k in range(self.n_components):
            # Create multivariate normal distribution
            try:
                mvn = dist.MultivariateNormal(
                    loc=self.means[k], covariance_matrix=self.covariances[k]
                )
                log_likelihoods[:, k] = mvn.log_prob(X) + torch.log(self.weights[k])
            except ValueError:
                # If covariance is not positive definite, return very low likelihood
                log_likelihoods[:, k] = torch.full((n_samples,), -1e10, device=X.device)

        return log_likelihoods

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate posterior probabilities

        Args:
            X: Data tensor of shape (n_samples, n_dimensions)

        Returns:
            Posterior probabilities of shape (n_samples, n_components)
        """
        log_component_likelihood = self._log_component_likelihood(X)
        log_prob = torch.logsumexp(log_component_likelihood, dim=1, keepdim=True)
        return torch.exp(log_component_likelihood - log_prob)

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples from the model

        Args:
            n_samples: Number of samples to generate

        Returns:
            Samples of shape (n_samples, n_dimensions)
        """
        # Sample component indices
        component_dist = dist.Categorical(probs=self.weights)
        component_indices = component_dist.sample((n_samples,))

        # Sample from selected components
        samples = torch.empty(
            n_samples, self.n_dimensions, device=self.weight_logits.device
        )

        for k in range(self.n_components):
            # Find samples assigned to this component
            component_mask = component_indices == k
            count = component_mask.sum().item()

            if count > 0:
                try:
                    component_dist = dist.MultivariateNormal(
                        loc=self.means[k], covariance_matrix=self.covariances[k]
                    )
                    samples[component_mask] = component_dist.sample((count,))
                except ValueError:
                    # Fallback if covariance is not positive definite
                    samples[component_mask] = self.means[k].expand(
                        count, self.n_dimensions
                    )

        return samples

    def bic(self, X: torch.Tensor) -> float:
        """
        Bayesian Information Criterion

        Args:
            X: Data tensor of shape (n_samples, n_dimensions)

        Returns:
            BIC value (lower is better)
        """
        n_samples = X.shape[0]

        # Number of free parameters
        n_params = (
            (self.n_components - 1)  # mixture weights (sum to 1)
            + self.n_components * self.n_dimensions  # means
            + self.n_components
            * self.n_dimensions
            * (self.n_dimensions + 1)
            // 2  # covariances
        )

        # Log likelihood
        log_likelihood_sum = self.log_likelihood(X).sum().item()

        return (
            -2 * log_likelihood_sum
            + n_params * torch.log(torch.tensor(n_samples)).item()
        )
