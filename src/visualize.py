"""
Generate Grad-CAM visualizations for brain tumor classification model.

This script loads the trained model and creates heatmap visualizations
showing what regions of the MRI the model focuses on for each prediction.
Useful for validating that the model learned to identify actual tumor features.
"""

import os
import argparse
from pathlib import Path
import random

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.model import create_model
from src.data import get_transforms, CLASSES
from src.gradcam import create_gradcam


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Create model architecture
    model = create_model(num_classes=len(CLASSES), pretrained=False, device=device)

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Best validation accuracy: {checkpoint['best_val_acc'] * 100:.2f}%")

    return model


def get_sample_images(data_dir: str, num_per_class: int = 3) -> dict:
    """
    Get sample images from test set for each class.

    Args:
        data_dir: Root data directory
        num_per_class: Number of samples to get per class

    Returns:
        Dict mapping class name to list of image paths
    """
    test_path = Path(data_dir) / "Testing"
    samples = {}

    for class_name in CLASSES:
        class_path = test_path / class_name
        if not class_path.exists():
            print(f"Warning: {class_path} not found, skipping")
            continue

        # Get all images in class
        image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))

        # Sample randomly
        if len(image_files) > num_per_class:
            image_files = random.sample(image_files, num_per_class)

        samples[class_name] = [str(f) for f in image_files]

    return samples


def create_visualization_grid(
    gradcam,
    image_path: str,
    preprocess_fn,
    class_names: list
) -> plt.Figure:
    """
    Create a 3-panel visualization: Original | Heatmap | Overlay.

    Args:
        gradcam: BrainTumorGradCAM instance
        image_path: Path to image
        preprocess_fn: Preprocessing function
        class_names: List of class names

    Returns:
        Matplotlib figure
    """
    # Generate visualizations
    original, heatmap, overlay, pred_info = gradcam.visualize_prediction(
        image_path, preprocess_fn, class_names
    )

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original MRI', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Add prediction info as suptitle
    true_class = Path(image_path).parent.name
    pred_class = pred_info['class_name']
    confidence = pred_info['confidence'] * 100

    correct = "✓" if true_class == pred_class else "✗"
    color = 'green' if true_class == pred_class else 'red'

    fig.suptitle(
        f"{correct} True: {true_class} | Predicted: {pred_class} ({confidence:.1f}%)",
        fontsize=16,
        fontweight='bold',
        color=color
    )

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for brain tumor model"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Root data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations/gradcam',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=3,
        help='Number of samples per class'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/mps/cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GRAD-CAM VISUALIZATION GENERATION")
    print("=" * 70)
    print()

    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Device: {device}")
    print()

    # Load model
    model = load_model(args.checkpoint, device)
    print()

    # Create Grad-CAM
    gradcam = create_gradcam(model, device)
    print("✓ Grad-CAM initialized")
    print()

    # Get preprocessing function
    preprocess = get_transforms(augment=False)

    # Get sample images
    print(f"Collecting {args.num_samples} samples per class from {args.data_dir}/Testing...")
    samples = get_sample_images(args.data_dir, args.num_samples)

    total_samples = sum(len(paths) for paths in samples.values())
    print(f"✓ Found {total_samples} total samples across {len(samples)} classes")
    print()

    # Generate visualizations
    print("Generating Grad-CAM visualizations...")
    print("-" * 70)

    for class_name, image_paths in samples.items():
        print(f"\nClass: {class_name}")

        for idx, image_path in enumerate(image_paths, 1):
            image_name = Path(image_path).stem

            # Create visualization
            fig = create_visualization_grid(
                gradcam, image_path, preprocess, CLASSES
            )

            # Save figure
            output_path = output_dir / f"{class_name}_{idx}_{image_name}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"  [{idx}/{len(image_paths)}] Saved: {output_path.name}")

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Total visualizations: {total_samples}")
    print()
    print("Next steps:")
    print("  1. Review the visualizations to verify the model focuses on tumor regions")
    print("  2. Check if misclassifications (✗) have unusual heatmap patterns")
    print("  3. Use these insights to validate model training effectiveness")
    print()


if __name__ == "__main__":
    main()
