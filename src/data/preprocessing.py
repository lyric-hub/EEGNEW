"""EEG preprocessing pipeline using MNE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mne
import numpy as np
from omegaconf import DictConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """Preprocessor for EEG data using MNE.

    Handles bandpass filtering, epoching, artifact rejection,
    and resampling of raw EEG data.

    Attributes:
        config: Preprocessing configuration from Hydra.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize preprocessor with configuration.

        Args:
            config: Hydra configuration containing preprocessing params.
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate preprocessing configuration.

        Raises:
            ValueError: If configuration values are invalid.
        """
        prep = self.config.preprocessing
        if prep.low_freq >= prep.high_freq:
            msg = f"low_freq ({prep.low_freq}) must be less than high_freq ({prep.high_freq})"
            raise ValueError(msg)

        if prep.tmin >= prep.tmax:
            msg = f"tmin ({prep.tmin}) must be less than tmax ({prep.tmax})"
            raise ValueError(msg)

    def preprocess_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply preprocessing to raw EEG data.

        Args:
            raw: MNE Raw object with EEG data.

        Returns:
            Preprocessed Raw object.
        """
        prep = self.config.preprocessing

        # Pick EEG channels only
        raw = raw.copy().pick_types(eeg=True)
        logger.info("Picked %d EEG channels", len(raw.ch_names))

        # Apply bandpass filter
        raw.filter(
            l_freq=prep.low_freq,
            h_freq=prep.high_freq,
            method="iir",
            verbose=False,
        )
        logger.info(
            "Applied bandpass filter: %.1f-%.1f Hz",
            prep.low_freq,
            prep.high_freq,
        )

        # Resample if needed
        if raw.info["sfreq"] != prep.resample_freq:
            raw.resample(prep.resample_freq, verbose=False)
            logger.info("Resampled to %d Hz", prep.resample_freq)

        return raw

    def create_epochs(
        self,
        raw: mne.io.Raw,
        events: NDArray[np.int64],
        event_id: dict[str, int],
    ) -> mne.Epochs:
        """Create epochs from raw data and events.

        Args:
            raw: Preprocessed Raw object.
            events: MNE events array of shape (n_events, 3).
            event_id: Dictionary mapping event names to IDs.

        Returns:
            MNE Epochs object.
        """
        prep = self.config.preprocessing

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=prep.tmin,
            tmax=prep.tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )
        logger.info("Created %d epochs", len(epochs))

        # Apply artifact rejection
        if prep.reject_threshold is not None:
            reject = {"eeg": prep.reject_threshold}
            epochs.drop_bad(reject=reject, verbose=False)
            logger.info("After artifact rejection: %d epochs", len(epochs))

        return epochs

    def epochs_to_arrays(
        self,
        epochs: mne.Epochs,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Convert epochs to numpy arrays.

        Args:
            epochs: MNE Epochs object.

        Returns:
            Tuple of (data, labels) arrays.
        """
        data = epochs.get_data().astype(np.float32)
        labels = epochs.events[:, -1].astype(np.int64)

        # Normalize labels to start from 0
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[label] for label in labels], dtype=np.int64)

        logger.info("Data shape: %s, Labels shape: %s", data.shape, labels.shape)
        return data, labels

    def save_processed(
        self,
        data: NDArray[np.float32],
        labels: NDArray[np.int64],
        output_dir: str | Path,
    ) -> None:
        """Save processed data to disk.

        Args:
            data: Processed EEG data.
            labels: Class labels.
            output_dir: Directory to save files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "data.npy", data)
        np.save(output_dir / "labels.npy", labels)

        logger.info("Saved processed data to %s", output_dir)
