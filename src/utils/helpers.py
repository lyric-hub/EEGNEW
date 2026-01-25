"""Helper utilities for EEG classification."""

from __future__ import annotations

import logging
import random
import sys
from typing import Literal

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: Literal["auto", "cpu", "cuda", "mps"] = "auto") -> torch.device:
    """Get the best available compute device.

    Args:
        preference: Device preference.
            - "auto": Automatically select best available device.
            - "cpu": Force CPU.
            - "cuda": Force CUDA (fails if unavailable).
            - "mps": Force Apple Metal (fails if unavailable).

    Returns:
        PyTorch device object.

    Raises:
        RuntimeError: If requested device is not available.
    """
    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA requested but not available"
            raise RuntimeError(msg)
        return torch.device("cuda")

    if preference == "mps":
        if not torch.backends.mps.is_available():
            msg = "MPS requested but not available"
            raise RuntimeError(msg)
        return torch.device("mps")

    # Auto-detect best device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logging.info("Using CUDA device: %s", gpu_name)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple Metal (MPS) device")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")

    return device


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to log file.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Reduce verbosity of some libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("mne").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
