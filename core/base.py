from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration for GMM models"""

    n_components: int
    n_dimensions: int
    covariance_regularization: float = 1e-6


class BaseGMM(nn.Module, ABC):
    """Base class for all GMM variants"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """Calculate log likelihood of data"""
        pass

    @abstractmethod
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Calculate posterior probabilities"""
        pass

    @abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate samples from the model"""
        pass
