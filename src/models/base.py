"""Base model class for EEG neural networks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all EEG classification models.

    Provides common functionality for parameter counting and
    SNN state management. All models should inherit from this class.

    Attributes:
        n_channels: Number of EEG input channels.
        n_samples: Number of time samples per trial.
        n_classes: Number of output classes.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
    ) -> None:
        """Initialize base model.

        Args:
            n_channels: Number of EEG input channels.
            n_samples: Number of time samples per trial.
            n_classes: Number of output classes.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, channels, samples).

        Returns:
            Output logits of shape (batch, n_classes).
        """
        pass

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_snn_states(self) -> None:
        """Reset SNN neuron membrane potentials.

        Override this method in SNN models to reset stateful neurons
        before each forward pass. Default implementation does nothing
        for non-SNN models.
        """
        pass
