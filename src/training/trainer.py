"""Training loop with MLflow logging and early stopping."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.training.metrics import compute_all_metrics

if TYPE_CHECKING:
    from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Monitors a metric and stops training if no improvement is seen
    for a specified number of epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement before stopping.
            min_delta: Minimum change to qualify as an improvement.
            mode: 'min' for loss, 'max' for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """Training loop manager with logging and checkpointing.

    Handles the training loop, validation, early stopping,
    MLflow logging, and model checkpointing.
    """

    def __init__(
        self,
        model: BaseModel,
        config: DictConfig,
        device: torch.device,
        output_dir: str | Path,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Model to train.
            config: Training configuration.
            device: Device to train on.
            output_dir: Directory for checkpoints and logs.
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.weight_decay,
            betas=tuple(config.training.optimizer.betas),
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.scheduler.T_max,
            eta_min=config.training.scheduler.eta_min,
        )

        # Early stopping
        if config.training.early_stopping.enabled:
            mode = "min" if "loss" in config.training.early_stopping.monitor else "max"
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping.patience,
                min_delta=config.training.early_stopping.min_delta,
                mode=mode,
            )
        else:
            self.early_stopping = None

        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Tuple of (average loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Reset SNN states before each forward pass
            if hasattr(self.model, 'reset_snn_states'):
                self.model.reset_snn_states()

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            ce_loss = self.criterion(outputs, batch_labels)
            
            # Add spike rate regularization if available
            if hasattr(self.model, 'get_spike_loss'):
                spike_loss = self.model.get_spike_loss()
                loss = ce_loss + spike_loss
            else:
                loss = ce_loss

            # Backward pass
            loss.backward()
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * batch_data.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader.dataset)
        metrics = compute_all_metrics(np.array(all_labels), np.array(all_preds))

        return avg_loss, metrics["accuracy"]

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> tuple[float, float, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader.

        Returns:
            Tuple of (average loss, accuracy, kappa).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Reset SNN states before each forward pass
            if hasattr(self.model, 'reset_snn_states'):
                self.model.reset_snn_states()

            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)

            total_loss += loss.item() * batch_data.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader.dataset)
        metrics = compute_all_metrics(np.array(all_labels), np.array(all_preds))

        return avg_loss, metrics["accuracy"], metrics["kappa"]

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[float]]:
        """Train the model.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Dictionary of training history.
        """
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_kappa": [],
        }

        # Start MLflow run if enabled
        use_mlflow = self.config.training.logging.use_mlflow
        # Note: params are logged in train.py, not here, to avoid conflicts

        for epoch in range(self.config.training.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            # Validate
            val_loss, val_acc, val_kappa = self.validate(val_loader)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["val_kappa"].append(val_kappa)

            # Update scheduler
            self.scheduler.step()

            # Log metrics
            logger.info(
                "Epoch %d/%d - Train Loss: %.4f, Train Acc: %.4f, "
                "Val Loss: %.4f, Val Acc: %.4f, Val Kappa: %.4f",
                epoch + 1,
                self.config.training.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_kappa,
            )

            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_kappa": val_kappa,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                    },
                    step=epoch,
                )

            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch
                self._save_checkpoint("best_model.pt")

            # Early stopping
            if self.early_stopping is not None:
                monitor_value = (
                    val_loss
                    if "loss" in self.config.training.early_stopping.monitor
                    else val_acc
                )
                if self.early_stopping(monitor_value):
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

        # Save last model
        self._save_checkpoint("last_model.pt")

        logger.info(
            "Training complete. Best accuracy: %.4f at epoch %d",
            self.best_val_accuracy,
            self.best_epoch + 1,
        )

        return history

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
        }
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.debug("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_accuracy = checkpoint["best_val_accuracy"]
        self.best_epoch = checkpoint["best_epoch"]
        logger.info("Loaded checkpoint from %s", path)
