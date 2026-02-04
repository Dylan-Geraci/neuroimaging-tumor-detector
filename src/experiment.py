"""
MLflow experiment tracking for training runs.

Logs hyperparameters, metrics per epoch, and model artifacts.
Usage:
    from src.experiment import ExperimentTracker
    tracker = ExperimentTracker("brain-tumor-resnet18")
    tracker.log_params({"lr": 0.001, "batch_size": 32, ...})
    for epoch in range(num_epochs):
        tracker.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
    tracker.log_model(model, "best_model")
    tracker.end()
"""

from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch


class ExperimentTracker:
    """Wraps MLflow for brain tumor classification experiments."""

    def __init__(self, experiment_name: str = "brain-tumor-classification", tracking_uri: Optional[str] = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()

    def log_params(self, params: dict):
        """Log hyperparameters (lr, batch_size, epochs, etc.)."""
        mlflow.log_params(params)

    def log_epoch(self, epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float):
        """Log metrics for a single training epoch."""
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            step=epoch,
        )

    def log_test_metrics(self, accuracy: float, precision: float, recall: float, f1: float):
        """Log final test-set evaluation metrics."""
        mlflow.log_metrics(
            {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
            }
        )

    def log_model(self, model, artifact_name: str = "model"):
        """Log a PyTorch model as an MLflow artifact."""
        mlflow.pytorch.log_model(model, artifact_name)

    def log_artifact(self, path: str):
        """Log a local file (e.g. confusion matrix image) as an artifact."""
        if Path(path).exists():
            mlflow.log_artifact(path)

    def end(self):
        """End the active MLflow run."""
        mlflow.end_run()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end()
