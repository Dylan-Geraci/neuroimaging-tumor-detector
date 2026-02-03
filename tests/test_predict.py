"""Tests for src/predict.py — inference helpers."""

import numpy as np
import torch
from PIL import Image

from src.predict import get_inference_transforms, predict
from src.data import CLASSES


def test_inference_transform_shape():
    transform = get_inference_transforms()
    img = Image.fromarray(np.random.randint(0, 256, (300, 400), dtype=np.uint8), mode="L")
    tensor = transform(img)
    assert tensor.shape == (1, 224, 224)


def test_predict_returns_valid_class(mock_model, sample_image_tensor, device):
    pred_idx, probs = predict(mock_model, sample_image_tensor, device)
    assert 0 <= pred_idx < len(CLASSES)


def test_predict_probabilities_sum_to_one(mock_model, sample_image_tensor, device):
    _, probs = predict(mock_model, sample_image_tensor, device)
    assert abs(probs.sum() - 1.0) < 1e-4


def test_predict_probabilities_non_negative(mock_model, sample_image_tensor, device):
    _, probs = predict(mock_model, sample_image_tensor, device)
    assert (probs >= 0).all()


def test_predict_returns_numpy_array(mock_model, sample_image_tensor, device):
    _, probs = predict(mock_model, sample_image_tensor, device)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(CLASSES),)
