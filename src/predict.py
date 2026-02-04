"""
Inference script for brain tumor classification.

This script loads a trained model and makes predictions on individual MRI images.
Can be used standalone or imported as a module for the web application.
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

from src.model import create_model
from src.data import CLASSES


class TemperatureScaler:
    """
    Post-hoc temperature scaling for better calibrated probabilities.

    After training, fit the temperature on the validation set, then use
    scaled_probs = softmax(logits / T) for better confidence estimates.

    Usage:
        scaler = TemperatureScaler()
        scaler.fit(model, val_loader, device)
        calibrated_probs = scaler.scale(logits)
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, model: nn.Module, val_loader, device: str, lr: float = 0.01, max_iter: int = 100):
        """Learn optimal temperature on validation data using NLL loss."""
        temp = nn.Parameter(torch.ones(1, device=device) * self.temperature)
        optimizer = torch.optim.LBFGS([temp], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        all_logits = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))
                all_logits.append(logits)
                all_labels.append(labels.to(device))

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        def closure():
            optimizer.zero_grad()
            loss = criterion(all_logits / temp, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = temp.item()
        return self.temperature

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits and return probabilities."""
        return torch.softmax(logits / self.temperature, dim=1)


def get_inference_transforms() -> transforms.Compose:
    """
    Get transforms for inference (no augmentation).
    Same as validation transforms from training.

    Returns:
        Composed transforms for preprocessing
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def load_model(
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
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create model architecture
    model = create_model(num_classes=num_classes, pretrained=False, device=device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode
    model.eval()

    return model


def preprocess_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """
    Load and preprocess a single image.

    Args:
        image_path: Path to image file
        transform: Transforms to apply

    Returns:
        Preprocessed image tensor ready for model input
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    image = Image.open(image_path).convert('L')  # Ensure grayscale

    # Apply transforms
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: str = 'cpu'
) -> Tuple[int, np.ndarray]:
    """
    Make prediction on a single image.

    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on

    Returns:
        (predicted_class_index, probabilities_array)
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_class = outputs.argmax(1).item()
        probs_array = probabilities.cpu().numpy()[0]

    return predicted_class, probs_array


def predict_image(
    image_path: str,
    checkpoint_path: str = 'models/best_model.pth',
    device: str = None
) -> Dict[str, any]:
    """
    High-level function to predict tumor type from an image path.
    Useful for importing in other scripts or web application.

    Args:
        image_path: Path to MRI image
        checkpoint_path: Path to model checkpoint
        device: Device to use (None for auto-detect)

    Returns:
        Dictionary with:
            - predicted_class: class name (str)
            - predicted_index: class index (int)
            - confidence: confidence percentage (float)
            - probabilities: dict of {class_name: probability}
    """
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    # Load model
    model = load_model(checkpoint_path, num_classes=len(CLASSES), device=device)

    # Prepare transforms
    transform = get_inference_transforms()

    # Preprocess image
    image_tensor = preprocess_image(image_path, transform)

    # Make prediction
    pred_idx, probs = predict(model, image_tensor, device)

    # Format results
    predicted_class = CLASSES[pred_idx]
    confidence = probs[pred_idx] * 100

    probabilities = {
        class_name: float(prob) for class_name, prob in zip(CLASSES, probs)
    }

    return {
        'predicted_class': predicted_class,
        'predicted_index': pred_idx,
        'confidence': confidence,
        'probabilities': probabilities
    }


def main():
    parser = argparse.ArgumentParser(
        description='Predict brain tumor type from MRI image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/predict.py --image data/Testing/glioma/image_001.jpg
  python src/predict.py --image scan.jpg --checkpoint models/best_model.pth
  python src/predict.py --image scan.jpg --device cuda
        """
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to MRI image file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/best_model.pth',
        help='Path to model checkpoint (default: models/best_model.pth)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/mps/cpu, default: auto-detect)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed information'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return

    # Make prediction
    if args.verbose:
        print(f"Loading model from {args.checkpoint}...")
        print(f"Processing image: {args.image}")

    try:
        result = predict_image(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            device=args.device
        )

        # Display results
        print()
        print("=" * 60)
        print("BRAIN TUMOR CLASSIFICATION RESULT")
        print("=" * 60)
        print(f"\nImage: {args.image}")
        print(f"\nPrediction: {result['predicted_class'].upper()}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nAll Probabilities:")
        print("-" * 60)

        # Sort by probability (descending)
        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for class_name, prob in sorted_probs:
            bar_length = int(prob * 40)  # Scale to 40 characters
            bar = "█" * bar_length
            print(f"  {class_name:12s}: {prob*100:6.2f}% {bar}")

        print("=" * 60)
        print()

    except Exception as e:
        print(f"Error during prediction: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
