"""
Unit tests for cost estimator.
"""

import pytest
from src.models.cost_estimator import CostEstimator, RepairCosts


def test_estimator_initialization():
    """Test cost estimator initialization."""
    estimator = CostEstimator()
    assert estimator is not None
    assert estimator.currency == "USD"


def test_no_damage_cost():
    """Test cost estimation with no damage."""
    estimator = CostEstimator()
    classification = {
        "severity": None,
        "damage_count": 0,
        "damage_types": {}
    }

    result = estimator.estimate_cost(classification)

    assert result["estimated_cost"] == 0
    assert result["min_cost"] == 0
    assert result["max_cost"] == 0


def test_minor_damage_cost():
    """Test cost estimation for minor damage."""
    estimator = CostEstimator()
    classification = {
        "severity": "minor",
        "damage_count": 1,
        "damage_types": {"scratch": 1}
    }

    result = estimator.estimate_cost(classification)

    assert result["estimated_cost"] > 0
    assert result["min_cost"] < result["estimated_cost"]
    assert result["max_cost"] > result["estimated_cost"]
    assert result["currency"] == "USD"


def test_severe_damage_multiplier():
    """Test that severe damage increases cost."""
    estimator = CostEstimator()

    minor = {
        "severity": "minor",
        "damage_count": 1,
        "damage_types": {"scratch": 1}
    }

    severe = {
        "severity": "severe",
        "damage_count": 1,
        "damage_types": {"scratch": 1}
    }

    minor_cost = estimator.estimate_cost(minor)["estimated_cost"]
    severe_cost = estimator.estimate_cost(severe)["estimated_cost"]

    assert severe_cost > minor_cost


def test_custom_costs():
    """Test using custom repair costs."""
    custom_costs = RepairCosts(scratch=100.0, dent=200.0)
    estimator = CostEstimator(base_costs=custom_costs)

    classification = {
        "severity": "minor",
        "damage_count": 1,
        "damage_types": {"scratch": 1}
    }

    result = estimator.estimate_cost(classification)
    assert result["estimated_cost"] > 0


def test_cost_summary():
    """Test cost summary generation."""
    estimator = CostEstimator()
    estimate = {
        "estimated_cost": 500.0,
        "min_cost": 400.0,
        "max_cost": 600.0,
        "currency": "USD",
        "parts_cost": 300.0,
        "labor_cost": 200.0
    }

    summary = estimator.get_cost_summary(estimate)
    assert "500" in summary
    assert "USD" in summary
