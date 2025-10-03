"""
CNN model for brain tumor classification.

Uses transfer learning with ResNet18, modified for:
- Single-channel grayscale MRI input (instead of 3-channel RGB)
- 4-class classification (glioma, meningioma, notumor, pituitary)
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class BrainTumorClassifier(nn.Module):
    """
    CNN for classifying brain MRI scans into 4 tumor categories.

    Architecture:
    - Base: ResNet18 pretrained on ImageNet
    - Modified: First conv layer accepts 1-channel grayscale
    - Output: 4 classes (glioma, meningioma, notumor, pituitary)

    Why ResNet18:
    - Good accuracy/speed tradeoff for medical imaging
    - Residual connections help with gradient flow
    - Pretrained features (edges, textures) transfer well even from RGB to grayscale
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        """
        Args:
            num_classes: Number of output classes (default: 4)
            pretrained: Use ImageNet pretrained weights (default: True)
        """
        super(BrainTumorClassifier, self).__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=pretrained)

        # Modify first convolutional layer for grayscale input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # New: Conv2d(1, 64, kernel_size=7, stride=2, padding=3)

        original_conv = self.model.conv1

        # Create new conv layer with 1 input channel
        self.model.conv1 = nn.Conv2d(
            in_channels=1,  # Grayscale
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Initialize new conv layer weights
        if pretrained:
            # Average RGB weights across channels to initialize grayscale weights
            # This preserves learned edge detectors from ImageNet
            with torch.no_grad():
                self.model.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        # Modify final fully connected layer for num_classes
        # Original: Linear(512, 1000) for ImageNet
        # New: Linear(512, 4) for our 4 tumor classes

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, 224, 224)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before final classification layer.
        Useful for visualization and feature analysis.

        Args:
            x: Input tensor of shape (batch_size, 1, 224, 224)

        Returns:
            Features of shape (batch_size, 512)
        """
        # Forward through all layers except final FC
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def create_model(
    num_classes: int = 4,
    pretrained: bool = True,
    device: Optional[str] = None
) -> BrainTumorClassifier:
    """
    Factory function to create and initialize the model.

    Args:
        num_classes: Number of output classes (default: 4)
        pretrained: Use ImageNet pretrained weights (default: True)
        device: Device to move model to ('cuda', 'mps', 'cpu', or None for auto)

    Returns:
        Initialized model on specified device
    """
    model = BrainTumorClassifier(num_classes=num_classes, pretrained=pretrained)

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    model = model.to(device)
    print(f"Model created and moved to device: {device}")

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count trainable and total parameters in model.

    Args:
        model: PyTorch model

    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    # Demo: Create model and show architecture
    print("Creating BrainTumorClassifier...")
    model = create_model(num_classes=4, pretrained=True)

    # Count parameters
    trainable, total = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")

    # Test forward pass with dummy data
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 1, 224, 224)  # Batch of 4 grayscale images

    # Move to same device as model
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)

    with torch.no_grad():
        output = model(dummy_input)
        features = model.get_features(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be [4, 4]
    print(f"Feature shape: {features.shape}")  # Should be [4, 512]
    print(f"\nSample output (logits): {output[0]}")

    # Show probabilities using softmax
    probs = torch.softmax(output[0], dim=0)
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    print(f"\nSample probabilities:")
    for cls, prob in zip(classes, probs):
        print(f"  {cls:12s}: {prob.item():.4f}")
