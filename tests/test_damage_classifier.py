"""
Unit tests for damage classifier.
"""

import pytest
from src.models.damage_classifier import DamageClassifier, DamageSeverity


def test_classifier_initialization():
    """Test classifier can be initialized."""
    classifier = DamageClassifier()
    assert classifier is not None
    assert classifier.minor_th == 0.02


def test_no_detections():
    """Test handling of images with no detections."""
    classifier = DamageClassifier()
    result = classifier.classify_damage([], (1920, 1080))

    assert result["severity"] is None
    assert result["damage_count"] == 0
    assert result["area_ratio"] == 0


def test_minor_damage_classification():
    """Test classification of minor damage."""
    classifier = DamageClassifier()
    detections = [
        {
            "bbox": [100, 100, 200, 200],
            "confidence": 0.9,
            "class": "scratch"
        }
    ]

    result = classifier.classify_damage(detections, (1920, 1080))

    assert result["severity"] == "minor"
    assert result["damage_count"] == 1
    assert result["area_ratio"] > 0


def test_multiple_damages():
    """Test handling multiple damage detections."""
    classifier = DamageClassifier()
    detections = [
        {"bbox": [100, 100, 300, 300], "confidence": 0.85, "class": "dent"},
        {"bbox": [400, 400, 600, 600], "confidence": 0.92, "class": "scratch"},
        {"bbox": [700, 700, 900, 900], "confidence": 0.78, "class": "crack"}
    ]

    result = classifier.classify_damage(detections, (1920, 1080))

    assert result["damage_count"] == 3
    assert "dent" in result["damage_types"]
    assert "scratch" in result["damage_types"]
    assert "crack" in result["damage_types"]


def test_batch_classify():
    """Test batch classification."""
    classifier = DamageClassifier()
    detections_list = [
        [{"bbox": [0, 0, 100, 100], "confidence": 0.9, "class": "scratch"}],
        [{"bbox": [0, 0, 200, 200], "confidence": 0.8, "class": "dent"}]
    ]
    shapes = [(1920, 1080), (1920, 1080)]

    results = classifier.batch_classify(detections_list, shapes)

    assert len(results) == 2
    assert results[0]["damage_count"] == 1
    assert results[1]["damage_count"] == 1
