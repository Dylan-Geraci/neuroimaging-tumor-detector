"""Shared test fixtures."""

import io

import numpy as np
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from src.model import create_model
from src.data import CLASSES
from src.predict import get_inference_transforms
from src.gradcam import create_gradcam


@pytest.fixture(scope="session")
def device():
    return "cpu"


@pytest.fixture(scope="session")
def mock_model(device):
    """Create model with random weights (no checkpoint needed)."""
    model = create_model(num_classes=len(CLASSES), pretrained=False, device=device)
    model.eval()
    return model


@pytest.fixture(scope="session")
def mock_gradcam(mock_model, device):
    return create_gradcam(mock_model, device)


@pytest.fixture(scope="session")
def mock_transform():
    return get_inference_transforms()


@pytest.fixture()
def sample_image_bytes():
    """Generate a synthetic 224x224 grayscale image as bytes."""
    arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture()
def sample_image_tensor(mock_transform):
    """Generate a preprocessed image tensor ready for the model."""
    arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    return mock_transform(img).unsqueeze(0)


@pytest.fixture()
def app_client(mock_model, mock_gradcam, mock_transform, device):
    """TestClient with mocked app.state (no checkpoint file needed)."""
    from main import app

    app.state.model = mock_model
    app.state.gradcam = mock_gradcam
    app.state.device = device
    app.state.transform = mock_transform

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
