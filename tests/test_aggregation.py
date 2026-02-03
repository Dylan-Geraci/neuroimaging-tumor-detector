"""Tests for prediction aggregation logic."""

from main import aggregate_predictions


def _make_pred(cls, confidence, probs=None):
    if probs is None:
        probs = {"glioma": 0.0, "meningioma": 0.0, "notumor": 0.0, "pituitary": 0.0}
        probs[cls] = confidence
        remaining = 1.0 - confidence
        others = [k for k in probs if k != cls]
        for k in others:
            probs[k] = remaining / len(others)
    return {"class": cls, "confidence": confidence, "probabilities": probs}


def test_single_prediction():
    result = aggregate_predictions([_make_pred("glioma", 0.9)])
    assert result["class"] == "glioma"
    assert result["agreement_score"] == 1.0


def test_unanimous_agreement():
    preds = [_make_pred("meningioma", 0.8) for _ in range(5)]
    result = aggregate_predictions(preds)
    assert result["class"] == "meningioma"
    assert result["agreement_score"] == 1.0


def test_split_predictions():
    preds = [
        _make_pred("glioma", 0.9),
        _make_pred("glioma", 0.85),
        _make_pred("meningioma", 0.7),
    ]
    result = aggregate_predictions(preds)
    assert result["class"] in ["glioma", "meningioma"]
    assert 0.0 < result["agreement_score"] <= 1.0


def test_empty_list():
    result = aggregate_predictions([])
    assert result == {}


def test_confidence_averaging():
    preds = [
        _make_pred("notumor", 0.6, {"glioma": 0.1, "meningioma": 0.1, "notumor": 0.6, "pituitary": 0.2}),
        _make_pred("notumor", 0.8, {"glioma": 0.05, "meningioma": 0.05, "notumor": 0.8, "pituitary": 0.1}),
    ]
    result = aggregate_predictions(preds)
    assert result["class"] == "notumor"
    expected_avg = (0.6 + 0.8) / 2
    assert abs(result["probabilities"]["notumor"] - expected_avg) < 1e-6


def test_agreement_score_calculation():
    preds = [
        _make_pred("glioma", 0.9),
        _make_pred("glioma", 0.8),
        _make_pred("meningioma", 0.7),
        _make_pred("pituitary", 0.6),
    ]
    result = aggregate_predictions(preds)
    # Regardless of who wins, agreement score should reflect count/total
    total = 4
    winning_class = result["class"]
    expected_count = sum(1 for p in preds if p["class"] == winning_class)
    assert abs(result["agreement_score"] - expected_count / total) < 1e-6
