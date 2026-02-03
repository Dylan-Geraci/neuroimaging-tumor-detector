"""Tests for src/model.py — architecture, shapes, factory."""

import torch

from src.model import BrainTumorClassifier, create_model, count_parameters
from src.data import CLASSES


def test_forward_output_shape(mock_model, device):
    x = torch.randn(2, 1, 224, 224, device=device)
    out = mock_model(x)
    assert out.shape == (2, len(CLASSES))


def test_get_features_shape(mock_model, device):
    x = torch.randn(1, 1, 224, 224, device=device)
    features = mock_model.get_features(x)
    assert features.shape == (1, 512)


def test_count_parameters(mock_model):
    trainable, total = count_parameters(mock_model)
    assert trainable > 0
    assert total >= trainable


def test_create_model_pretrained_false():
    model = create_model(num_classes=4, pretrained=False, device="cpu")
    assert isinstance(model, BrainTumorClassifier)
    x = torch.randn(1, 1, 224, 224)
    out = model(x)
    assert out.shape == (1, 4)


def test_create_model_custom_classes():
    model = create_model(num_classes=2, pretrained=False, device="cpu")
    x = torch.randn(1, 1, 224, 224)
    out = model(x)
    assert out.shape == (1, 2)


def test_single_channel_input(mock_model, device):
    """Model must accept 1-channel (grayscale) input."""
    x = torch.randn(1, 1, 224, 224, device=device)
    out = mock_model(x)
    assert out.shape[1] == len(CLASSES)
