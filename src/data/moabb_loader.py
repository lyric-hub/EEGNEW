"""MOABB data loader for subject-independent EEG classification.

Uses MOABB's MotorImagery paradigm to load and preprocess BCI data
with proper subject-based splits for subject-independent evaluation.

v3.5: Added subject ID propagation for domain adaptation (GRL + MMD).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from src.data.augmentation import AugmentedDataset, EEGAugmentation

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _extract_subject_ids(metadata: "pd.DataFrame") -> np.ndarray:
    """Extract and encode subject IDs from MOABB metadata.

    Args:
        metadata: MOABB metadata DataFrame with 'subject' column.

    Returns:
        Integer-encoded subject IDs array.
    """
    subject_encoder = LabelEncoder()
    return subject_encoder.fit_transform(metadata["subject"].values)


def load_moabb_data(
    config: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load EEG data using MOABB paradigm with subject-based splits.

    Returns DataLoaders yielding (data, label, subject_id) tuples.
    Subject IDs are integer-encoded per split (0-indexed within each split).

    Args:
        config: Hydra configuration containing data parameters.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
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

    # Load data with metadata (subject IDs needed for domain adaptation)
    train_subjects = list(config.data.train_subjects)
    logger.info("Loading training data for subjects: %s", train_subjects)
    X_train, y_train, meta_train = paradigm.get_data(dataset, subjects=train_subjects)

    val_subjects = list(config.data.val_subjects)
    logger.info("Loading validation data for subjects: %s", val_subjects)
    X_val, y_val, meta_val = paradigm.get_data(dataset, subjects=val_subjects)

    test_subjects = list(config.data.test_subjects)
    logger.info("Loading test data for subjects: %s", test_subjects)
    X_test, y_test, meta_test = paradigm.get_data(dataset, subjects=test_subjects)

    # Encode class labels
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train, y_val, y_test]))
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    # Encode subject IDs (0-indexed within each split)
    subj_train = _extract_subject_ids(meta_train)
    subj_val = _extract_subject_ids(meta_val)
    subj_test = _extract_subject_ids(meta_test)

    n_train_subjects = len(np.unique(subj_train))
    logger.info("Classes: %s", label_encoder.classes_)
    logger.info(
        "Train: %d samples (%d subjects), Val: %d, Test: %d",
        len(X_train),
        n_train_subjects,
        len(X_val),
        len(X_test),
    )

    # Create 3-tuple TensorDatasets: (data, label, subject_id)
    train_dataset = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train_encoded.astype(np.int64)),
        torch.from_numpy(subj_train.astype(np.int64)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val_encoded.astype(np.int64)),
        torch.from_numpy(subj_val.astype(np.int64)),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test.astype(np.float32)),
        torch.from_numpy(y_test_encoded.astype(np.int64)),
        torch.from_numpy(subj_test.astype(np.int64)),
    )

    # Apply augmentation to training data only
    use_augmentation = getattr(config.data, 'use_augmentation', True)
    if use_augmentation:
        augmentation = EEGAugmentation(
            channel_dropout_prob=getattr(config.data, 'aug_channel_dropout', 0.2),
            channel_shuffle_prob=0.0,  # Disabled: prevents spatial corruption
            time_shift_max=getattr(config.data, 'aug_time_shift', 50),
            time_warp_prob=getattr(config.data, 'aug_time_warp', 0.2),
            gaussian_noise_prob=getattr(config.data, 'aug_noise_prob', 0.3),
            noise_snr_db=getattr(config.data, 'aug_noise_snr', 20.0),
        )
        train_dataset = AugmentedDataset(train_dataset, augmentation, apply_augmentation=True)
        logger.info("Training data augmentation enabled")

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
