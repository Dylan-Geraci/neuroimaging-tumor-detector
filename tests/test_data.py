"""Tests for src/data.py — transforms, constants."""

import torch
import numpy as np
from PIL import Image

from src.data import CLASSES, NUM_CLASSES, get_transforms


def test_classes_has_four_entries():
    assert len(CLASSES) == 4
    assert NUM_CLASSES == 4


def test_classes_contents():
    assert "glioma" in CLASSES
    assert "meningioma" in CLASSES
    assert "notumor" in CLASSES
    assert "pituitary" in CLASSES


def test_eval_transform_output_shape():
    transform = get_transforms(augment=False)
    img = Image.fromarray(np.random.randint(0, 256, (300, 400), dtype=np.uint8), mode="L")
    tensor = transform(img)
    assert tensor.shape == (1, 224, 224)


def test_train_transform_output_shape():
    transform = get_transforms(augment=True)
    img = Image.fromarray(np.random.randint(0, 256, (300, 400), dtype=np.uint8), mode="L")
    tensor = transform(img)
    assert tensor.shape == (1, 224, 224)


def test_eval_transform_deterministic():
    """Inference transforms should be deterministic."""
    transform = get_transforms(augment=False)
    arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    t1 = transform(img)
    t2 = transform(img)
    assert torch.equal(t1, t2)


def test_transform_normalizes():
    """After normalization with mean=0.5 std=0.5, values should be in [-1, 1]."""
    transform = get_transforms(augment=False)
    img = Image.fromarray(np.full((224, 224), 128, dtype=np.uint8), mode="L")
    tensor = transform(img)
    # 128/255 ≈ 0.502; (0.502 - 0.5) / 0.5 ≈ 0.004
    assert tensor.min() >= -1.01
    assert tensor.max() <= 1.01
