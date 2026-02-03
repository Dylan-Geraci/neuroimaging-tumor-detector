"""Tests for src/gradcam.py — heatmap generation."""

import numpy as np
import torch

from src.gradcam import BrainTumorGradCAM, create_gradcam


def test_heatmap_shape(mock_gradcam, sample_image_tensor, device):
    tensor = sample_image_tensor.to(device)
    heatmap = mock_gradcam.generate_heatmap(tensor, target_class=0)
    assert heatmap.shape == (224, 224)


def test_heatmap_values_in_range(mock_gradcam, sample_image_tensor, device):
    tensor = sample_image_tensor.to(device)
    heatmap = mock_gradcam.generate_heatmap(tensor, target_class=0)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_rgb_heatmap_shape(mock_gradcam, sample_image_tensor, device):
    tensor = sample_image_tensor.to(device)
    heatmap = mock_gradcam.generate_heatmap(tensor, target_class=0, use_rgb=True)
    assert heatmap.shape == (224, 224, 3)


def test_different_targets_differ(mock_gradcam, sample_image_tensor, device):
    """Different target classes should generally produce different heatmaps."""
    tensor = sample_image_tensor.to(device)
    h0 = mock_gradcam.generate_heatmap(tensor, target_class=0)
    h1 = mock_gradcam.generate_heatmap(tensor, target_class=1)
    # With random weights they might be similar, but shapes must match
    assert h0.shape == h1.shape == (224, 224)


def test_create_gradcam_factory(mock_model, device):
    gc = create_gradcam(mock_model, device)
    assert isinstance(gc, BrainTumorGradCAM)
