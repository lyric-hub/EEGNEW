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
import sys
from pathlib import Path

import hydra
import mlflow
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
    """Build S-MSTT model from configuration.

    Args:
        config: Model configuration.
        n_channels: Number of EEG channels from data.
        n_samples: Number of time samples from data.
        n_classes: Number of classes from data.

    Returns:
        Initialized S-MSTT model.
    """
    model = CustomModel(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        t_steps=config.model.get("t_steps", 8),
        f1=config.model.get("f1", 32),
        depth_multiplier=config.model.get("depth_multiplier", 2),
        num_heads=config.model.get("num_heads", 4),
        dropout_rate=config.model.get("dropout_rate", 0.2),
        attention_pool=config.model.get("attention_pool", 8),
        tau=config.model.get("tau", 4.0),
        v_threshold=config.model.get("v_threshold", 1.0),
        use_preprocessing=config.model.get("use_preprocessing", True),
        sample_rate=config.model.get("sample_rate", 250),
    )

    logger.info(
        "Built S-MSTT with %d parameters (channels=%d, samples=%d, classes=%d)",
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

    finally:
        if config.training.logging.use_mlflow:
            mlflow.end_run()

    return trainer.best_val_accuracy


if __name__ == "__main__":
    main()
