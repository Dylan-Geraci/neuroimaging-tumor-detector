"""
Training script for brain tumor classification model.

Handles:
- Training loop with validation
- Metrics tracking (loss, accuracy)
- Model checkpointing (save best model)
- Learning rate scheduling
- Early stopping
"""

import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import get_data_loaders, CLASSES
from src.model import create_model, count_parameters


class Trainer:
    """
    Trainer class for brain tumor classification.

    Handles complete training pipeline including validation,
    checkpointing, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        checkpoint_dir: str = 'models'
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on ('cuda', 'mps', 'cpu')
            learning_rate: Initial learning rate
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Loss function - CrossEntropyLoss for multi-class classification
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer - Adam is good default for medical imaging
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4  # L2 regularization to prevent overfitting
        )

        # Learning rate scheduler - reduce LR when validation plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,  # Reduce by half
            patience=5,  # Wait 5 epochs before reducing
            verbose=True
        )

        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for training
        pbar = tqdm(self.train_loader, desc='Training', leave=False)

        for images, labels in pbar:
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.

        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)

            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Track metrics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, filename: str = 'best_model.pth'):
        """
        Save model checkpoint.

        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, checkpoint_path)

        print(f"✓ Checkpoint saved to {checkpoint_path}")

    def train(
        self,
        num_epochs: int = 30,
        early_stopping_patience: int = 10,
        save_best_only: bool = True
    ) -> Dict:
        """
        Complete training loop.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for this many epochs
            save_best_only: Only save model when validation improves

        Returns:
            Training history dictionary
        """
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Early stopping patience: {early_stopping_patience}")
        trainable, total = count_parameters(self.model)
        print(f"Trainable parameters: {trainable:,}")
        print("=" * 60)
        print()

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch + 1}/{num_epochs}] ({epoch_time:.1f}s)")
            print(
                f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
            print(
                f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            # Check for improvement
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                improved = True

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                improved = True

            # Save checkpoint if improved
            if improved:
                self.epochs_no_improve = 0
                if save_best_only:
                    self.save_checkpoint('best_model.pth')
                print(
                    f"  ✓ New best! Val Loss: {self.best_val_loss:.4f}, Val Acc: {self.best_val_acc * 100:.2f}%")
            else:
                self.epochs_no_improve += 1

            # Early stopping check
            if self.epochs_no_improve >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                print(
                    f"  No improvement for {early_stopping_patience} consecutive epochs")
                break

            print()

        # Training complete
        total_time = time.time() - start_time
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Acc: {self.best_val_acc * 100:.2f}%")
        print("=" * 60)

        return self.history


def train_model(
    data_dir: str = 'data',
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    checkpoint_dir: str = 'models'
):
    """
    Main training function.

    Args:
        data_dir: Directory containing Training/ and Testing/ folders
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        early_stopping_patience: Stop if no improvement for this many epochs
        checkpoint_dir: Directory to save checkpoints
    """
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size
    )

    # Create model
    print("Creating model...")
    model = create_model(num_classes=len(CLASSES), pretrained=True)

    # Determine device
    device = next(model.parameters()).device

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device),
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir
    )

    # Train
    history = trainer.train(
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience
    )

    return trainer, history


if __name__ == "__main__":
    # Train the model
    trainer, history = train_model(
        data_dir='data',
        num_epochs=30,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=10
    )
