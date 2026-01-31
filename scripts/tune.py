#!/usr/bin/env python
"""Hyperparameter tuning with Optuna for S-MSTT v3.1.

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
import json
import logging
import sys
from pathlib import Path

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.moabb_loader import get_data_info, load_moabb_data
from src.models.custom import CustomModel
from src.training.trainer import Trainer
from src.utils.helpers import get_device, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def create_model(
    trial: optuna.Trial,
    n_channels: int,
    n_samples: int,
    n_classes: int,
) -> CustomModel:
    """Create model with hyperparameters suggested by Optuna.

    Args:
        trial: Optuna trial object.
        n_channels: Number of EEG channels.
        n_samples: Number of time samples.
        n_classes: Number of classes.

    Returns:
        Model with suggested hyperparameters.
    """
    # S-MSTT v3.1 hyperparameter search space
    embed_dim = trial.suggest_categorical("embed_dim", [32, 64])
    depth = trial.suggest_int("depth", 2, 4)
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    t_steps = trial.suggest_categorical("t_steps", [2, 4])
    tau = trial.suggest_float("tau", 1.0, 3.0)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.3)
    attn_residual_ratio = trial.suggest_float("attn_residual_ratio", 0.0, 0.2)

    model = CustomModel(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        t_steps=t_steps,
        tau=tau,
        dropout_rate=dropout_rate,
        attn_residual_ratio=attn_residual_ratio,
        motor_indices=[6, 7, 8, 9, 10, 11, 12],
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
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    spike_loss_weight = trial.suggest_float("spike_loss_weight", 0.01, 0.3)

    # Load data using MOABB
    config_obj = OmegaConf.create(config)
    train_loader, val_loader, _ = load_moabb_data(config_obj)
    n_channels, n_samples, n_classes = get_data_info(train_loader)

    # Create model
    model = create_model(trial, n_channels, n_samples, n_classes)

    logger.info(
        "Trial %d: embed_dim=%d, depth=%d, lr=%.6f, params=%d",
        trial.number,
        trial.params["embed_dim"],
        trial.params["depth"],
        lr,
        model.count_parameters(),
    )

    # Update config with suggested values
    config_copy = OmegaConf.create(config)
    config_copy.training.optimizer.lr = lr
    config_copy.training.optimizer.weight_decay = weight_decay
    config_copy.training.spike_loss_weight = spike_loss_weight
    config_copy.training.epochs = 20  # Fewer epochs for tuning
    config_copy.training.early_stopping.patience = 8
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
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("Trial %d: OOM, pruning", trial.number)
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
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
    parser.add_argument("--n-trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    seed_everything(args.seed)

    # Load base config
    config_path = PROJECT_ROOT / "config"
    config = OmegaConf.merge(
        OmegaConf.load(config_path / "config.yaml"),
        {"data": OmegaConf.load(config_path / "data" / "default.yaml")},
        {"model": OmegaConf.load(config_path / "model" / "custom.yaml")},
        {"training": OmegaConf.load(config_path / "training" / "default.yaml")},
    )

    device = get_device("auto")

    # Create output directory
    output_dir = PROJECT_ROOT / "outputs" / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Optuna study
    study = optuna.create_study(
        study_name="eeg-smstt-tuning",
        direction="maximize",
        storage=f"sqlite:///{output_dir}/study.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    )

    # MLflow callback for tracking each trial
    mlflow.set_experiment("optuna-tuning")
    mlflow_callback = MLflowCallback(
        tracking_uri="mlruns",
        metric_name="val_accuracy",
        create_experiment=False,
    )

    logger.info("=" * 60)
    logger.info("S-MSTT v3.1 Hyperparameter Tuning")
    logger.info("=" * 60)

    # Run optimization with MLflow tracking
    study.optimize(
        lambda trial: objective(trial, OmegaConf.to_container(config), device),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
        callbacks=[mlflow_callback],
    )

    # Print results
    logger.info("=" * 60)
    logger.info("Best trial:")
    logger.info("  Value (accuracy): %.4f", study.best_trial.value)
    logger.info("  Params:")
    for key, value in study.best_trial.params.items():
        logger.info("    %s: %s", key, value)

    # Save best parameters as YAML
    best_params_file = output_dir / "best_params.yaml"
    with open(best_params_file, "w") as f:
        yaml.dump(study.best_trial.params, f, default_flow_style=False)
    logger.info("Best parameters saved to %s", best_params_file)

    # Save best parameters as JSON (for DVC metrics)
    best_json = output_dir / "best_params.json"
    with open(best_json, "w") as f:
        json.dump({
            "best_val_accuracy": study.best_trial.value,
            "params": study.best_trial.params,
        }, f, indent=2)

    # Save trials to CSV
    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)

    # Print optimization history
    logger.info("=" * 60)
    logger.info("Optimization summary:")
    logger.info("  Total trials: %d", len(study.trials))
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    logger.info("  Completed: %d", completed)
    logger.info("  Pruned: %d", pruned)


if __name__ == "__main__":
    main()
