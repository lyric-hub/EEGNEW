"""Training loop with MLflow logging, early stopping, and domain adaptation.

v3.5: Added GRL adversarial + MMD losses for subject-invariant training.
"""

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

from src.models.custom import MaximumMeanDiscrepancy
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
        min_delta: float = 0.001,
        monitor: str = "val_loss",
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement before stopping.
            min_delta: Minimum change to qualify as an improvement.
            monitor: Metric to monitor (e.g., 'val_loss', 'val_accuracy').
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.monitor in ("val_loss", "train_loss"):
            score = -value  # For loss, we want to minimize, so negate for 'max' logic
        else:
            score = value   # For accuracy, we want to maximize

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.should_stop


class Trainer:
    """Trainer with domain adaptation support (GRL + MMD).

    Handles training, validation, checkpointing, and MLflow logging.
    Supports 3-tuple DataLoaders: (data, label, subject_id).

    Args:
        model: S-MSTT model instance.
        config: Training configuration.
        device: Target device.
    """

    def __init__(
        self,
        model: BaseModel,
        config: DictConfig,
        device: torch.device,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Model to train.
            config: Training configuration.
            device: Device to train on.
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

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

        # Loss
        self.criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.mmd_loss_fn = MaximumMeanDiscrepancy()

        # Loss weights
        self.domain_loss_weight = config.training.get("domain_loss_weight", 0.1)
        self.mmd_loss_weight = config.training.get("mmd_loss_weight", 0.1)

        # Early stopping
        es_config = config.training.early_stopping
        self.early_stopping = EarlyStopping(
            patience=es_config.patience,
            min_delta=es_config.min_delta,
            monitor=es_config.monitor,
        ) if es_config.enabled else None

        # Checkpointing
        self.checkpoint_dir = Path(config.training.get("checkpoint_dir", "models"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

        # Store epoch-level domain metrics for logging
        self._last_domain_loss = 0.0
        self._last_mmd_loss = 0.0

        logger.info("Trainer initialized (device=%s, domain_weight=%.2f, mmd_weight=%.2f)",
                     device, self.domain_loss_weight, self.mmd_loss_weight)

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train for one epoch with domain adaptation losses.

        Supports both 2-tuple (data, label) and 3-tuple (data, label, subject_id)
        DataLoaders for backward compatibility.

        Args:
            train_loader: Training data loader.

        Returns:
            Tuple of (average loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        total_domain_loss = 0.0
        total_mmd_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            # Unpack: support both 2-tuple and 3-tuple
            if len(batch) == 3:
                batch_data, batch_labels, batch_subjects = batch
                batch_subjects = batch_subjects.to(self.device)
            else:
                batch_data, batch_labels = batch
                batch_subjects = None

            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Reset SNN states before each forward pass
            if hasattr(self.model, 'reset_snn_states'):
                self.model.reset_snn_states()

            # Forward pass
            self.optimizer.zero_grad()

            if batch_subjects is not None and hasattr(self.model, 'forward_with_domain'):
                outputs, domain_logits = self.model.forward_with_domain(batch_data)
            else:
                outputs = self.model(batch_data)
                domain_logits = None

            # Classification loss
            ce_loss = self.criterion(outputs, batch_labels)

            # Spike rate regularization
            spike_loss = torch.tensor(0.0, device=self.device)
            spike_weight = 0.0
            if hasattr(self.model, 'get_spike_loss'):
                spike_loss = self.model.get_spike_loss()
                spike_weight = self.config.training.get("spike_loss_weight", 0.1)

            # Domain adversarial loss (GRL handles gradient reversal)
            domain_loss = torch.tensor(0.0, device=self.device)
            if domain_logits is not None and batch_subjects is not None:
                domain_loss = self.domain_criterion(domain_logits, batch_subjects)
                total_domain_loss += domain_loss.item() * batch_data.size(0)

            # MMD loss between random subject pairs
            mmd_loss = torch.tensor(0.0, device=self.device)
            if batch_subjects is not None and hasattr(self.model, 'get_features'):
                features = self.model.get_features()
                if features is not None:
                    unique_subjects = batch_subjects.unique()
                    if len(unique_subjects) >= 2:
                        # Pick two random subjects from batch
                        idx = torch.randperm(len(unique_subjects))[:2]
                        s1, s2 = unique_subjects[idx[0]], unique_subjects[idx[1]]
                        feat_s1 = features[batch_subjects == s1]
                        feat_s2 = features[batch_subjects == s2]
                        if len(feat_s1) >= 2 and len(feat_s2) >= 2:
                            mmd_loss = self.mmd_loss_fn(feat_s1, feat_s2)
                            total_mmd_loss += mmd_loss.item() * batch_data.size(0)

            # Total loss
            loss = (
                ce_loss
                + spike_weight * spike_loss
                + self.domain_loss_weight * domain_loss
                + self.mmd_loss_weight * mmd_loss
            )

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

        n_samples = len(train_loader.dataset)
        avg_loss = total_loss / n_samples
        metrics = compute_all_metrics(np.array(all_labels), np.array(all_preds))

        # Store epoch-level domain metrics for logging
        self._last_domain_loss = total_domain_loss / n_samples
        self._last_mmd_loss = total_mmd_loss / n_samples

        return avg_loss, metrics["accuracy"]

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> tuple[float, float, float]:
        """Validate the model.

        Handles both 2-tuple and 3-tuple DataLoaders (subject IDs ignored).

        Args:
            val_loader: Validation data loader.

        Returns:
            Tuple of (average loss, accuracy, kappa).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in val_loader:
            # Unpack: take only data and labels, ignore subject_id if present
            batch_data = batch[0].to(self.device)
            batch_labels = batch[1].to(self.device)

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
            "train_domain_loss": [],
            "train_mmd_loss": [],
        }

        # Start MLflow run if enabled
        use_mlflow = self.config.training.logging.use_mlflow
        # Note: params are logged in train.py, not here, to avoid conflicts
        total_epochs = self.config.training.epochs

        for epoch in range(total_epochs):
            # Update GRL progress schedule
            if hasattr(self.model, 'grl'):
                self.model.grl.set_progress(epoch / max(total_epochs - 1, 1))

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["train_domain_loss"].append(self._last_domain_loss)
            history["train_mmd_loss"].append(self._last_mmd_loss)


            # Validate
            val_loss, val_acc, val_kappa = self.validate(val_loader)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["val_kappa"].append(val_kappa)

            # Update scheduler
            self.scheduler.step()

            # Log metrics
            spike_rate = self.model.get_spike_rate() if hasattr(self.model, 'get_spike_rate') else 0.0
            logger.info(
                "Epoch %d/%d - Train Loss: %.4f, Train Acc: %.4f, "
                "Val Loss: %.4f, Val Acc: %.4f, Val Kappa: %.4f, Spike: %.2f%%",
                epoch + 1,
                self.config.training.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_kappa,
                spike_rate * 100,
            )

            if use_mlflow:
                metrics_to_log = {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_kappa": val_kappa,
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "domain_loss": self._last_domain_loss,
                    "mmd_loss": self._last_mmd_loss,
                }
                # Log spike rate if available
                if hasattr(self.model, 'get_spike_rate'):
                    metrics_to_log["spike_rate"] = self.model.get_spike_rate()
                # Log per-layer spike rates for dead neuron diagnosis
                if hasattr(self.model, 'get_layer_rates'):
                    for layer_name, rate in self.model.get_layer_rates().items():
                        metrics_to_log[f"spike/{layer_name}"] = rate
                # Log GRL lambda schedule
                if hasattr(self.model, 'grl'):
                    metrics_to_log["grl_lambda"] = self.model.grl.current_lambda
                mlflow.log_metrics(metrics_to_log, step=epoch)

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
        path = self.checkpoint_dir / filename
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
