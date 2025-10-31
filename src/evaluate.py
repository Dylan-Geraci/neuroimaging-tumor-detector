"""
Test set evaluation for brain tumor classification model.

This script:
- Loads the trained model from checkpoint
- Evaluates on the held-out test set
- Generates detailed metrics and visualizations
- Saves results and identifies misclassified examples
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data import get_data_loaders, CLASSES
from model import create_model


def load_trained_model(
    checkpoint_path: str,
    num_classes: int = 4,
    device: str = 'cpu'
) -> nn.Module:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to saved model checkpoint
        num_classes: Number of output classes
        device: Device to load model to

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from {checkpoint_path}...")

    # Create model architecture
    model = create_model(num_classes=num_classes, pretrained=False, device=device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode
    model.eval()

    print(f"Model loaded successfully (Epoch {checkpoint.get('epoch', 'N/A')})")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A'):.4f}")

    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int, int]]]:
    """
    Run inference on test set and collect predictions.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run inference on

    Returns:
        (all_labels, all_predictions, misclassified_info)
        misclassified_info is list of (image_path, true_label, pred_label)
    """
    all_labels = []
    all_predictions = []
    misclassified = []

    print("\nRunning inference on test set...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            # Collect results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

            # Track misclassified examples
            for i, (true_label, pred_label) in enumerate(zip(labels, predictions)):
                if true_label != pred_label:
                    # Get image path from dataset
                    dataset_idx = batch_idx * test_loader.batch_size + i
                    if dataset_idx < len(test_loader.dataset):
                        img_path = test_loader.dataset.base_dataset.samples[
                            test_loader.dataset.indices[dataset_idx]
                        ][0]
                        misclassified.append((img_path, true_label.item(), pred_label.item()))

    return np.array(all_labels), np.array(all_predictions), misclassified


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    save_path: str
) -> None:
    """
    Create and save confusion matrix visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        save_path: Path to save figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize by true label (rows sum to 1)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes,
                yticklabels=classes, ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

    # Plot normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax2,
                cbar_kws={'label': 'Percentage'})
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def save_evaluation_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    misclassified: List[Tuple[str, int, int]],
    save_path: str
) -> None:
    """
    Save detailed evaluation results to text file.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        misclassified: List of misclassified examples
        save_path: Path to save results
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BRAIN TUMOR CLASSIFICATION - TEST SET EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        f.write(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Total Test Samples: {len(y_true)}\n")
        f.write(f"Correctly Classified: {(y_true == y_pred).sum()}\n")
        f.write(f"Misclassified: {(y_true != y_pred).sum()}\n\n")

        # Per-class metrics
        f.write("=" * 80 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("=" * 80 + "\n\n")

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(classes))
        )

        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 80 + "\n")

        for i, class_name in enumerate(classes):
            f.write(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                   f"{f1[i]:<12.4f} {support[i]:<10}\n")

        # Macro and weighted averages
        f.write("-" * 80 + "\n")
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        f.write(f"{'Macro Avg':<15} {macro_p:<12.4f} {macro_r:<12.4f} {macro_f1:<12.4f}\n")
        f.write(f"{'Weighted Avg':<15} {weighted_p:<12.4f} {weighted_r:<12.4f} {weighted_f1:<12.4f}\n\n")

        # Classification report
        f.write("=" * 80 + "\n")
        f.write("SKLEARN CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(classification_report(y_true, y_pred, target_names=classes))
        f.write("\n")

        # Confusion matrix
        f.write("=" * 80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 80 + "\n\n")
        cm = confusion_matrix(y_true, y_pred)

        # Header
        f.write(f"{'True \\ Pred':<15}")
        for class_name in classes:
            f.write(f"{class_name:<12}")
        f.write("\n" + "-" * 80 + "\n")

        # Rows
        for i, class_name in enumerate(classes):
            f.write(f"{class_name:<15}")
            for j in range(len(classes)):
                f.write(f"{cm[i, j]:<12}")
            f.write("\n")
        f.write("\n")

        # Misclassified examples
        f.write("=" * 80 + "\n")
        f.write(f"MISCLASSIFIED EXAMPLES ({len(misclassified)} total)\n")
        f.write("=" * 80 + "\n\n")

        if misclassified:
            # Group by (true_label, pred_label) pairs
            error_types = {}
            for img_path, true_label, pred_label in misclassified:
                key = (true_label, pred_label)
                if key not in error_types:
                    error_types[key] = []
                error_types[key].append(img_path)

            # Print grouped by confusion type
            for (true_label, pred_label), paths in sorted(error_types.items()):
                true_name = classes[true_label]
                pred_name = classes[pred_label]
                f.write(f"\n{true_name} misclassified as {pred_name} ({len(paths)} cases):\n")
                f.write("-" * 80 + "\n")
                for path in paths[:10]:  # Show first 10 examples
                    f.write(f"  {path}\n")
                if len(paths) > 10:
                    f.write(f"  ... and {len(paths) - 10} more\n")
        else:
            f.write("Perfect classification! No misclassified examples.\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Detailed results saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate brain tumor classification model on test set'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/best_model.pth',
        help='Path to model checkpoint (default: models/best_model.pth)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Root data directory (default: data)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/mps/cpu, default: auto-detect)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    _, _, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )
    print(f"Test set: {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    # Load model
    model = load_trained_model(args.checkpoint, num_classes=len(CLASSES), device=device)

    # Run evaluation
    y_true, y_pred, misclassified = evaluate_model(model, test_loader, device)

    # Calculate and print overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total Samples: {len(y_true)}")
    print(f"Correctly Classified: {(y_true == y_pred).sum()}")
    print(f"Misclassified: {(y_true != y_pred).sum()}")

    # Print per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for i, class_name in enumerate(CLASSES):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == i).sum() / mask.sum()
            print(f"  {class_name:12s}: {class_acc:.4f} ({class_acc*100:.2f}%)")

    # Generate and save confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, CLASSES, cm_path)

    # Save detailed results
    results_path = os.path.join(args.output_dir, 'test_evaluation.txt')
    save_evaluation_results(y_true, y_pred, CLASSES, misclassified, results_path)

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
