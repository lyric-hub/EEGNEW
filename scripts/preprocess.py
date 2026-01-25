#!/usr/bin/env python
"""Preprocess EEG data for training.

This script loads raw EEG data from MOABB, applies preprocessing,
and saves the processed data for training.

Usage:
    python scripts/preprocess.py

Output:
    data/processed/
        ├── train/
        │   ├── data.npy
        │   └── labels.npy
        ├── val/
        │   └── ...
        └── test/
            └── ...
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import seed_everything, setup_logging

logger = logging.getLogger(__name__)


def load_moabb_data(
    subjects: list[int],
    fmin: float = 4.0,
    fmax: float = 38.0,
    resample: float = 128.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess data from MOABB.

    Args:
        subjects: List of subject IDs to load.
        fmin: Minimum frequency for bandpass filter.
        fmax: Maximum frequency for bandpass filter.
        resample: Resampling frequency.

    Returns:
        Tuple of (data, labels) arrays.
    """
    dataset = BNCI2014_001()

    paradigm = MotorImagery(
        n_classes=4,
        fmin=fmin,
        fmax=fmax,
        resample=resample,
    )

    all_data = []
    all_labels = []

    for subject in subjects:
        logger.info("Loading subject %d...", subject)
        x, labels, meta = paradigm.get_data(dataset, subjects=[subject])
        all_data.append(x)
        all_labels.append(labels)

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Encode labels as integers
    unique_labels = sorted(set(labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    labels = np.array([label_map[label] for label in labels], dtype=np.int64)

    logger.info(
        "Loaded %d trials, %d channels, %d samples",
        data.shape[0],
        data.shape[1],
        data.shape[2],
    )
    logger.info("Labels: %s", unique_labels)

    return data.astype(np.float32), labels


def split_data(
    data: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split data into train, validation, and test sets.

    Args:
        data: EEG data array.
        labels: Label array.
        test_size: Proportion for test set.
        val_size: Proportion for validation set (from remaining after test).
        random_state: Random seed.

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing (data, labels) tuples.
    """
    # Split off test set
    x_temp, x_test, y_temp, y_test = train_test_split(
        data,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Split remaining into train and val
    val_ratio = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(y_train),
        len(y_val),
        len(y_test),
    )

    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }


def save_splits(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    """Save data splits to disk.

    Args:
        splits: Dictionary of data splits.
        output_dir: Base output directory.
    """
    for split_name, (data, labels) in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        np.save(split_dir / "data.npy", data)
        np.save(split_dir / "labels.npy", labels)

        logger.info("Saved %s split to %s", split_name, split_dir)


def main() -> None:
    """Main entry point."""
    setup_logging(level=logging.INFO)
    seed_everything(42)

    # Load config
    config_path = PROJECT_ROOT / "config" / "data" / "default.yaml"
    config = OmegaConf.load(config_path)

    output_dir = PROJECT_ROOT / "data" / "processed"

    # Load data
    data, labels = load_moabb_data(
        subjects=list(config.subjects),
        fmin=config.preprocessing.low_freq,
        fmax=config.preprocessing.high_freq,
        resample=config.preprocessing.resample_freq,
    )

    # Split data
    splits = split_data(
        data,
        labels,
        test_size=config.split.test_size,
        val_size=config.split.val_size,
    )

    # Save
    save_splits(splits, output_dir)

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
