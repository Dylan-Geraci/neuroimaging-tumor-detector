"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability.

Generates heatmaps showing which regions of the brain MRI the model focuses on
when making predictions. This is crucial for validating that the model learns
to identify actual tumor features rather than spurious correlations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class BrainTumorGradCAM:
    """
    Wrapper for Grad-CAM visualization on brain tumor classification model.

    This class makes it easy to generate heatmaps showing what the model
    focuses on when making predictions.
    """

    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Args:
            model: Trained BrainTumorClassifier model
            device: Device to run on ('cuda', 'mps', 'cpu')
        """
        self.model = model
        self.device = device
        self.model.eval()

        # Target layer for Grad-CAM (last conv layer in ResNet18)
        # For ResNet18, this is model.model.layer4[-1]
        target_layers = [self.model.model.layer4[-1]]

        # Initialize Grad-CAM
        self.cam = GradCAM(model=self.model, target_layers=target_layers)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        use_rgb: bool = False
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an input image.

        Args:
            input_tensor: Preprocessed image tensor (1, 1, 224, 224)
            target_class: Class to generate CAM for (None = predicted class)
            use_rgb: If True, return RGB heatmap; if False, return grayscale

        Returns:
            Heatmap as numpy array (H, W) or (H, W, 3)
        """
        # Get prediction if target class not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        # Create target for Grad-CAM
        targets = [ClassifierOutputTarget(target_class)]

        # Generate CAM
        # Note: input_tensor should be (1, 1, 224, 224)
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)

        # grayscale_cam is (batch, H, W), take first batch
        grayscale_cam = grayscale_cam[0, :]

        if use_rgb:
            # Convert to RGB heatmap using jet colormap
            heatmap = cv2.applyColorMap(
                np.uint8(255 * grayscale_cam),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            return heatmap
        else:
            return grayscale_cam

    def visualize_prediction(
        self,
        image_path: str,
        preprocess_fn,
        class_names: list = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Complete visualization pipeline: load image, predict, generate CAM.

        Args:
            image_path: Path to MRI image
            preprocess_fn: Function to preprocess image for model
            class_names: List of class names (default: glioma, meningioma, notumor, pituitary)

        Returns:
            (original_image, heatmap, overlay, prediction_info)
            - original_image: Original image as numpy array (H, W)
            - heatmap: Grad-CAM heatmap (H, W, 3)
            - overlay: Original image with heatmap overlay (H, W, 3)
            - prediction_info: Dict with 'class', 'class_name', 'confidence', 'all_probs'
        """
        if class_names is None:
            class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

        # Load and preprocess image
        original_image = Image.open(image_path).convert('L')  # Grayscale

        # Resize to 224x224 to match model input (for visualization consistency)
        original_image_resized = original_image.resize((224, 224), Image.Resampling.LANCZOS)
        original_np = np.array(original_image_resized)

        # Preprocess for model (apply transforms to original, not resized)
        input_tensor = preprocess_fn(original_image).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[predicted_class].item()

        # Generate Grad-CAM heatmap
        grayscale_cam = self.generate_heatmap(input_tensor, predicted_class)

        # Normalize original image to [0, 1] for overlay
        original_normalized = original_np / 255.0

        # Convert grayscale to RGB for visualization
        original_rgb = np.stack([original_normalized] * 3, axis=-1)

        # Create overlay using show_cam_on_image
        overlay = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)

        # Create colored heatmap for separate display
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * grayscale_cam),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Prediction info
        prediction_info = {
            'class': predicted_class,
            'class_name': class_names[predicted_class],
            'confidence': confidence,
            'all_probs': {
                class_names[i]: probabilities[i].item()
                for i in range(len(class_names))
            }
        }

        return original_np, heatmap_colored, overlay, prediction_info


def create_gradcam(model: torch.nn.Module, device: str = 'cpu') -> BrainTumorGradCAM:
    """
    Factory function to create Grad-CAM visualizer.

    Args:
        model: Trained BrainTumorClassifier
        device: Device to run on

    Returns:
        BrainTumorGradCAM instance
    """
    return BrainTumorGradCAM(model, device)


if __name__ == "__main__":
    # Demo: Load model and generate a sample Grad-CAM
    print("Grad-CAM Module")
    print("=" * 50)
    print("This module provides Grad-CAM visualization for the")
    print("brain tumor classification model.")
    print()
    print("Use visualize.py to generate example visualizations.")
