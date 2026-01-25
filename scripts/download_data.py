#!/usr/bin/env python
"""Download EEG datasets from MOABB.

This script downloads and caches motor imagery datasets from MOABB
for training EEG classification models.

Usage:
    python scripts/download_data.py

Output:
    data/raw/<dataset_name>/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import moabb
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def download_bnci2014_001(output_dir: Path) -> None:
    """Download BNCI2014_001 dataset.

    This is a 4-class motor imagery dataset with 9 subjects.
    Classes: left hand, right hand, both feet, tongue.

    Args:
        output_dir: Directory to save dataset info.
    """
    logger.info("Downloading BNCI2014_001 dataset...")

    # Initialize dataset (MOABB handles caching internally)
    dataset = BNCI2014_001()

    # Initialize paradigm to get data
    paradigm = MotorImagery(
        n_classes=4,
        fmin=4.0,
        fmax=38.0,
        resample=128.0,
    )

    # Get all subjects
    subjects = dataset.subject_list
    logger.info("Dataset has %d subjects: %s", len(subjects), subjects)

    # Download data for each subject (MOABB caches to ~/mne_data)
    for subject in subjects:
        logger.info("Downloading data for subject %d...", subject)
        try:
            x, labels, meta = paradigm.get_data(dataset, subjects=[subject])
            logger.info(
                "Subject %d: %d trials, %d classes",
                subject,
                len(labels),
                len(set(labels)),
            )
        except Exception as e:
            logger.error("Failed to download subject %d: %s", subject, e)
            continue

    # Save dataset info
    output_dir.mkdir(parents=True, exist_ok=True)
    info_file = output_dir / "dataset_info.txt"
    with open(info_file, "w") as f:
        f.write(f"Dataset: BNCI2014_001\n")
        f.write(f"Subjects: {subjects}\n")
        f.write(f"Classes: left_hand, right_hand, feet, tongue\n")
        f.write(f"MOABB cache: ~/mne_data/\n")

    logger.info("Download complete. Dataset info saved to %s", info_file)


def main() -> None:
    """Main entry point."""
    setup_logging(level=logging.INFO)

    output_dir = PROJECT_ROOT / "data" / "raw"

    # Set MOABB log level
    moabb.set_log_level("WARNING")

    download_bnci2014_001(output_dir)


if __name__ == "__main__":
    main()
