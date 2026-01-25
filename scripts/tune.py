#!/usr/bin/env python
"""Hyperparameter tuning with Optuna.

This script uses Optuna to search for optimal hyperparameters
for the EEG classification model.

Usage:
    python scripts/tune.py --n-trials 50
    python scripts/tune.py --n-trials 100 --timeout 3600

Output:
    outputs/tuning/best_params.yaml
    outputs/tuning/study.db (SQLite database)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import optuna
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import EEGDataset
from src.models.custom import CustomModel
from src.training.trainer import Trainer
from src.utils.helpers import get_device, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def create_model(trial: optuna.Trial, n_channels: int, n_samples: int, n_classes: int) -> CustomModel:
    """Create model with hyperparameters suggested by Optuna.

    Args:
        trial: Optuna trial object.
        n_channels: Number of EEG channels.
        n_samples: Number of time samples.
        n_classes: Number of classes.

    Returns:
        Model with suggested hyperparameters.
    """
    # ============================================
    # Define your hyperparameter search space here
    # ============================================

    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7)

    model = CustomModel(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
    )

    return model


def objective(trial: optuna.Trial, config: dict, device: torch.device) -> float:
    """Optuna objective function.

    Args:
        trial: Optuna trial object.
        config: Base configuration.
        device: Compute device.

    Returns:
        Validation accuracy (to maximize).
    """
    # Suggest training hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Load data
    data_dir = PROJECT_ROOT / "data" / "processed"
    train_dataset = EEGDataset.from_file(data_dir / "train")
    val_dataset = EEGDataset.from_file(data_dir / "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    model = create_model(
        trial,
        n_channels=train_dataset.n_channels,
        n_samples=train_dataset.n_samples,
        n_classes=train_dataset.n_classes,
    )

    logger.info(
        "Trial %d: lr=%.6f, weight_decay=%.6f, batch_size=%d, params=%d",
        trial.number,
        lr,
        weight_decay,
        batch_size,
        model.count_parameters(),
    )

    # Update config with suggested values
    config_copy = OmegaConf.create(config)
    config_copy.training.optimizer.lr = lr
    config_copy.training.optimizer.weight_decay = weight_decay
    config_copy.training.epochs = 30  # Fewer epochs for tuning
    config_copy.training.early_stopping.patience = 10
    config_copy.training.logging.use_mlflow = False

    # Train
    output_dir = PROJECT_ROOT / "outputs" / "tuning" / f"trial_{trial.number}"
    trainer = Trainer(
        model=model,
        config=config_copy,
        device=device,
        output_dir=output_dir,
    )

    try:
        trainer.fit(train_loader, val_loader)
    except Exception as e:
        logger.error("Trial %d failed: %s", trial.number, e)
        raise optuna.TrialPruned()

    # Report intermediate values for pruning
    trial.report(trainer.best_val_accuracy, step=trainer.best_epoch)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return trainer.best_val_accuracy


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    seed_everything(args.seed)

    # Load base config
    config_path = PROJECT_ROOT / "config"
    config = OmegaConf.merge(
        OmegaConf.load(config_path / "config.yaml"),
        OmegaConf.load(config_path / "training" / "default.yaml"),
    )

    device = get_device("auto")

    # Create output directory
    output_dir = PROJECT_ROOT / "outputs" / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Optuna study
    study = optuna.create_study(
        study_name="eeg-mi-tuning",
        direction="maximize",
        storage=f"sqlite:///{output_dir}/study.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, OmegaConf.to_container(config), device),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("Best trial:")
    logger.info("  Value (accuracy): %.4f", study.best_trial.value)
    logger.info("  Params:")
    for key, value in study.best_trial.params.items():
        logger.info("    %s: %s", key, value)

    # Save best parameters
    best_params_file = output_dir / "best_params.yaml"
    with open(best_params_file, "w") as f:
        yaml.dump(study.best_trial.params, f, default_flow_style=False)
    logger.info("Best parameters saved to %s", best_params_file)

    # Print optimization history
    logger.info("=" * 60)
    logger.info("Optimization summary:")
    logger.info("  Total trials: %d", len(study.trials))
    logger.info("  Completed: %d", len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
    logger.info("  Pruned: %d", len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))


if __name__ == "__main__":
    main()
