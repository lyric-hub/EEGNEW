"""MOABB data loader for subject-independent EEG classification.

Uses MOABB's MotorImagery paradigm to load and preprocess BCI data
with proper subject-based splits for subject-independent evaluation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def load_moabb_data(
    config: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load EEG data using MOABB paradigm with subject-based splits.

    Args:
        config: Hydra configuration containing data parameters.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Initialize dataset
    dataset = BNCI2014_001()

    # Configure paradigm from config
    paradigm_config = config.data.paradigm
    paradigm = MotorImagery(
        n_classes=paradigm_config.n_classes,
        fmin=paradigm_config.fmin,
        fmax=paradigm_config.fmax,
        tmin=paradigm_config.tmin,
        tmax=paradigm_config.tmax,
        channels=paradigm_config.channels,
        resample=paradigm_config.resample,
    )

    # Load training data (subjects 1-8)
    train_subjects = list(config.data.train_subjects)
    logger.info("Loading training data for subjects: %s", train_subjects)
    X_train, y_train, _ = paradigm.get_data(dataset, subjects=train_subjects)

    # Load test data (subject 9)
    test_subjects = list(config.data.test_subjects)
    logger.info("Loading test data for subjects: %s", test_subjects)
    X_test, y_test, _ = paradigm.get_data(dataset, subjects=test_subjects)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train, y_test]))
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    logger.info("Classes: %s", label_encoder.classes_)
    logger.info("Train data shape: %s, labels: %s", X_train.shape, y_train_encoded.shape)
    logger.info("Test data shape: %s, labels: %s", X_test.shape, y_test_encoded.shape)

    # Split training data into train/val
    val_ratio = config.data.val_ratio
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train,
        y_train_encoded,
        test_size=val_ratio,
        stratify=y_train_encoded,
        random_state=config.seed,
    )

    logger.info(
        "After split - Train: %d, Val: %d, Test: %d",
        len(X_train_split),
        len(X_val),
        len(X_test),
    )

    # Create PyTorch tensors
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_split.astype(np.float32)),
        torch.from_numpy(y_train_split.astype(np.int64)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.int64)),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test.astype(np.float32)),
        torch.from_numpy(y_test_encoded.astype(np.int64)),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_data_info(train_loader: DataLoader) -> tuple[int, int, int]:
    """Extract data dimensions from a DataLoader.

    Args:
        train_loader: Training DataLoader.

    Returns:
        Tuple of (n_channels, n_samples, n_classes).
    """
    sample_batch, labels = next(iter(train_loader))
    n_channels = sample_batch.shape[1]
    n_samples = sample_batch.shape[2]
    n_classes = len(torch.unique(labels))

    return n_channels, n_samples, n_classes
