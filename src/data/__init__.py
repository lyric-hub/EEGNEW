"""Data loading utilities for EEG classification."""

from src.data.dataset import EEGDataset
from src.data.moabb_loader import get_data_info, load_moabb_data

__all__ = ["EEGDataset", "load_moabb_data", "get_data_info"]

