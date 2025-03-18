from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.data as data_utils
from tqdm import tqdm

from core.base import BaseGMM


@dataclass
class SGDOptimizerConfig:
    """SGD optimizer configuration"""

    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 100
    validation_freq: int = 1
    early_stopping_patience: int = 10
    lr_schedule_step: int = 20
    lr_schedule_gamma: float = 0.5
    weight_decay: float = 1e-5


class SGDOptimizer:
    """SGD optimization for GMM models"""

    def __init__(self, model: BaseGMM, config: SGDOptimizerConfig):
        """
        Initialize the optimizer

        Args:
            model: The GMM model to optimize
            config: Optimizer configuration
        """
        self.model = model
        self.config = config

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_schedule_step,
            gamma=config.lr_schedule_gamma,
        )

        # Training history
        self.history = {"train_loss": [], "val_loss": []}

    def fit(
        self,
        train_data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        val_data: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Fit the model using SGD

        Args:
            train_data: Training data, either a tensor X or a tuple (X, noise_covars)
            val_data: Optional validation data
            verbose: Whether to display progress

        Returns:
            Training history
        """
        # Prepare dataset and dataloader
        if isinstance(train_data, tuple):
            X_train, noise_covars_train = train_data
            dataset = DeconvDataset(X_train, noise_covars_train)
        else:
            dataset = SimpleDataset(train_data)

        dataloader = data_utils.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=True
        )

        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        for epoch in range(self.config.max_epochs):
            # Train for one epoch
            train_loss = self._train_epoch(dataloader, verbose)
            self.history["train_loss"].append(train_loss)

            # Validate if requested
            if val_data is not None and epoch % self.config.validation_freq == 0:
                val_loss = self._validate(val_data)
                self.history["val_loss"].append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    # Restore best model
                    self.model.load_state_dict(best_model_state)
                    break

                if verbose:
                    print(
                        f"Epoch {epoch}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}"
                    )
            elif verbose:
                print(f"Epoch {epoch}: train_loss = {train_loss:.4f}")

            # Update learning rate
            self.scheduler.step()

        return self.history

    def _train_epoch(self, dataloader: data_utils.DataLoader, verbose: bool) -> float:
        """Train for one epoch and return average loss"""
        self.model.train()
        total_loss = 0.0
        n_batches = len(dataloader)

        iterator = tqdm(dataloader) if verbose else dataloader

        for batch in iterator:
            self.optimizer.zero_grad()

            # Compute negative log likelihood
            if isinstance(batch, tuple):
                X, noise_covars = batch
                loss = -self.model.log_likelihood((X, noise_covars)).mean()
            else:
                loss = -self.model.log_likelihood(batch).mean()

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / n_batches

    def _validate(
        self, val_data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """Compute validation loss"""
        self.model.eval()

        with torch.no_grad():
            if isinstance(val_data, tuple):
                X_val, noise_covars_val = val_data
                val_loss = (
                    -self.model.log_likelihood((X_val, noise_covars_val)).mean().item()
                )
            else:
                val_loss = -self.model.log_likelihood(val_data).mean().item()

        return val_loss


class SimpleDataset(data_utils.Dataset):
    """Dataset for standard GMM"""

    def __init__(self, X: torch.Tensor):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class DeconvDataset(data_utils.Dataset):
    """Dataset for deconvolution GMM"""

    def __init__(self, X: torch.Tensor, noise_covars: torch.Tensor):
        self.X = X
        self.noise_covars = noise_covars

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.noise_covars[idx]
