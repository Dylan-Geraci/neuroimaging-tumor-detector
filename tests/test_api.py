"""Tests for API endpoints."""

import io

import numpy as np
from PIL import Image


def _make_image_bytes(fmt="PNG"):
    arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


def test_health_endpoint(app_client):
    resp = app_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert len(data["classes"]) == 4


def test_predict_single_valid(app_client):
    img_bytes = _make_image_bytes()
    resp = app_client.post("/predict", files={"file": ("test.png", img_bytes, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "prediction" in data
    assert "images" in data
    assert data["prediction"]["class"] in ["glioma", "meningioma", "notumor", "pituitary"]
    assert 0.0 <= data["prediction"]["confidence"] <= 1.0


def test_predict_invalid_file_type(app_client):
    resp = app_client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_predict_missing_file(app_client):
    resp = app_client.post("/predict")
    assert resp.status_code == 422


def test_predict_batch_multi_file(app_client):
    files = [("files", (f"scan_{i}.png", _make_image_bytes(), "image/png")) for i in range(3)]
    resp = app_client.post("/predict/batch", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["processed_count"] == 3
    assert len(data["individual_predictions"]) == 3
    assert "aggregated_prediction" in data


def test_predict_batch_empty(app_client):
    resp = app_client.post("/predict/batch", files=[("files", ("empty.txt", b"", "text/plain"))])
    assert resp.status_code == 400


def test_predict_returns_probabilities(app_client):
    img_bytes = _make_image_bytes()
    resp = app_client.post("/predict", files={"file": ("test.png", img_bytes, "image/png")})
    data = resp.json()
    probs = data["prediction"]["probabilities"]
    assert len(probs) == 4
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-4


def test_predict_returns_base64_images(app_client):
    img_bytes = _make_image_bytes()
    resp = app_client.post("/predict", files={"file": ("test.png", img_bytes, "image/png")})
    data = resp.json()
    for key in ["original", "heatmap", "overlay"]:
        assert data["images"][key].startswith("data:image/png;base64,")
