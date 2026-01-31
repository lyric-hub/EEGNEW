#!/usr/bin/env python
"""Train S-MSTT SNN for subject-independent EEG classification.

Uses MOABB's MotorImagery paradigm with subject-based splits:
- Train on subjects 1-8
- Test on subject 9

Usage:
    python scripts/train.py
    python scripts/train.py model.t_steps=16 training.lr=0.0005
"""

from __future__ import annotations

import logging
import json
import sys
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.moabb_loader import get_data_info, load_moabb_data
from src.models.custom import CustomModel
from src.training.trainer import Trainer
from src.utils.helpers import get_device, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def build_model(
    config: DictConfig,
    n_channels: int,
    n_samples: int,
    n_classes: int,
) -> torch.nn.Module:
    """Build S-MSTT v3.1 model from configuration.

    Args:
        config: Model configuration.
        n_channels: Number of EEG channels from data.
        n_samples: Number of time samples from data.
        n_classes: Number of classes from data.

    Returns:
        Initialized S-MSTT v3.1 model.
    """
    model = CustomModel(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        # S-MSTT v3.1 parameters
        embed_dim=config.model.get("embed_dim", 64),
        depth=config.model.get("depth", 4),
        num_heads=config.model.get("num_heads", 4),
        mlp_ratio=config.model.get("mlp_ratio", 4.0),
        t_steps=config.model.get("t_steps", 4),
        tau=config.model.get("tau", 2.0),
        dropout_rate=config.model.get("dropout_rate", 0.1),
        use_alignment=config.model.get("use_alignment", True),
        use_channel_attention=config.model.get("use_channel_attention", True),
        use_multiscale=config.model.get("use_multiscale", True),
        use_learned_aggregation=config.model.get("use_learned_aggregation", True),
        alignment_momentum=config.model.get("alignment_momentum", 0.1),
        spike_target_rate=config.model.get("spike_target_rate", 0.25),
        attn_residual_ratio=config.model.get("attn_residual_ratio", 0.1),
        motor_indices=config.model.get("motor_indices", [6, 7, 8, 9, 10, 11, 12]),
    )

    logger.info(
        "Built S-MSTT v3.1 with %d parameters (channels=%d, samples=%d, classes=%d)",
        model.count_parameters(),
        n_channels,
        n_samples,
        n_classes,
    )

    return model


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(config: DictConfig) -> float:
    """Main training function.

    Args:
        config: Hydra configuration.

    Returns:
        Best validation accuracy.
    """
    # Setup
    setup_logging(level=logging.INFO)
    seed_everything(config.seed)

    logger.info("=" * 60)
    logger.info("S-MSTT Training: Subject-Independent EEG Classification")
    logger.info("=" * 60)
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    # Device
    device = get_device(config.device)
    logger.info("Using device: %s", device)

    # Load data using MOABB
    logger.info("Loading data from MOABB...")
    train_loader, val_loader, test_loader = load_moabb_data(config)

    # Get data dimensions
    n_channels, n_samples, n_classes = get_data_info(train_loader)
    logger.info(
        "Data: %d channels, %d samples, %d classes",
        n_channels,
        n_samples,
        n_classes,
    )

    # Build model
    model = build_model(config, n_channels, n_samples, n_classes)

    # Training setup
    output_dir = Path(config.paths.models)

    # MLflow setup
    if config.training.logging.use_mlflow:
        mlflow.set_experiment(config.experiment_name)
        mlflow.start_run()
        mlflow.log_params({
            "model_name": "S-MSTT",
            "train_subjects": str(list(config.data.train_subjects)),
            "test_subjects": str(list(config.data.test_subjects)),
            "n_channels": n_channels,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "t_steps": config.model.get("t_steps", 8),
        })

    try:
        trainer = Trainer(
            model=model,
            config=config,
            device=device,
            output_dir=output_dir,
        )

        # Train
        logger.info("Starting training...")
        history = trainer.fit(train_loader, val_loader)

        # Final evaluation on test subject (subject 9)
        logger.info("=" * 60)
        logger.info("Evaluating on test subject (subject 9)...")
        test_loss, test_acc, test_kappa = trainer.validate(test_loader)
        logger.info(
            "TEST RESULTS - Loss: %.4f, Accuracy: %.4f, Kappa: %.4f",
            test_loss,
            test_acc,
            test_kappa,
        )
        logger.info("=" * 60)

        if config.training.logging.use_mlflow:
            mlflow.log_metrics({
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_kappa": test_kappa,
            })
            mlflow.log_artifact(str(output_dir / "best_model.pt"))

        # Save metrics to JSON for DVC
        metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_kappa": float(test_kappa),
            "best_val_accuracy": float(trainer.best_val_accuracy),
        }
        metrics_path = Path("outputs/metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Save history to CSV for DVC plots
        history_df = pd.DataFrame(history)
        history_df.index.name = "epoch"
        history_df.to_csv("outputs/history.csv")
        logger.info("Saved metrics to %s and history to outputs/history.csv", metrics_path)

    finally:
        if config.training.logging.use_mlflow:
            mlflow.end_run()

    return trainer.best_val_accuracy


if __name__ == "__main__":
    main()
