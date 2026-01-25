"""PyTorch Dataset for EEG trials."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from numpy.typing import NDArray


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG motor imagery trials.

    Loads preprocessed EEG data and provides samples for training.

    Attributes:
        data: EEG trial data of shape (n_trials, n_channels, n_samples).
        labels: Class labels for each trial.
        transform: Optional transform to apply to each sample.
    """

    def __init__(
        self,
        data: NDArray[np.float32],
        labels: NDArray[np.int64],
        transform: callable | None = None,
    ) -> None:
        """Initialize EEG dataset.

        Args:
            data: EEG data array of shape (n_trials, n_channels, n_samples).
            labels: Label array of shape (n_trials,).
            transform: Optional callable to transform samples.

        Raises:
            ValueError: If data and labels have mismatched lengths.
        """
        if len(data) != len(labels):
            msg = f"Data and labels length mismatch: {len(data)} vs {len(labels)}"
            raise ValueError(msg)

        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (data, label) tensors.
        """
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    @classmethod
    def from_file(
        cls,
        data_path: str | Path,
        transform: callable | None = None,
    ) -> "EEGDataset":
        """Load dataset from preprocessed numpy files.

        Args:
            data_path: Path to directory containing data.npy and labels.npy.
            transform: Optional transform to apply.

        Returns:
            Initialized EEGDataset instance.

        Raises:
            FileNotFoundError: If data files don't exist.
        """
        data_path = Path(data_path)
        data_file = data_path / "data.npy"
        labels_file = data_path / "labels.npy"

        if not data_file.exists():
            msg = f"Data file not found: {data_file}"
            raise FileNotFoundError(msg)

        if not labels_file.exists():
            msg = f"Labels file not found: {labels_file}"
            raise FileNotFoundError(msg)

        data = np.load(data_file)
        labels = np.load(labels_file)

        return cls(data, labels, transform)

    @property
    def n_channels(self) -> int:
        """Return number of EEG channels."""
        return self.data.shape[1]

    @property
    def n_samples(self) -> int:
        """Return number of time samples per trial."""
        return self.data.shape[2]

    @property
    def n_classes(self) -> int:
        """Return number of unique classes."""
        return len(torch.unique(self.labels))
