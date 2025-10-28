"""
Cost estimation module for vehicle damage repair.
Provides rule-based cost calculations with configurable pricing.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class RepairCosts:
    """Base repair costs for different damage types (in USD)."""
    scratch: float = 150.0
    dent: float = 300.0
    crack: float = 450.0
    shatter: float = 800.0
    broken_part: float = 1200.0
    default: float = 400.0


class CostEstimator:
    """
    Estimates vehicle repair costs based on damage assessment.

    Uses configurable base costs and severity multipliers
    to provide cost estimates.
    """

    def __init__(self, base_costs: RepairCosts = None, currency: str = "USD"):
        """
        Initialize cost estimator.

        Args:
            base_costs: Custom base repair costs
            currency: Currency code for cost display
        """
        self.costs = base_costs if base_costs else RepairCosts()
        self.currency = currency

        # Severity multipliers
        self.severity_mult = {
            "minor": 1.0,
            "moderate": 1.5,
            "severe": 2.3,
            "critical": 3.5
        }

    def estimate_cost(
        self,
        damage_classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Estimate repair cost based on damage classification.

        Args:
            damage_classification: Output from DamageClassifier

        Returns:
            Dictionary with cost estimates and breakdown
        """
        severity = damage_classification.get("severity")
        damage_types = damage_classification.get("damage_types", {})
        damage_count = damage_classification.get("damage_count", 0)

        if not severity or damage_count == 0:
            return {
                "estimated_cost": 0,
                "min_cost": 0,
                "max_cost": 0,
                "currency": self.currency,
                "breakdown": {},
                "labor_cost": 0,
                "parts_cost": 0
            }

        # Calculate base cost from damage types
        parts_cost = 0
        breakdown = {}

        for dmg_type, count in damage_types.items():
            base = self._get_base_cost(dmg_type)
            type_cost = base * count
            parts_cost += type_cost
            breakdown[dmg_type] = {
                "count": count,
                "unit_cost": base,
                "total": type_cost
            }

        # Apply severity multiplier
        mult = self.severity_mult.get(severity, 1.0)
        adjusted_parts = parts_cost * mult

        # Calculate labor cost (typically 40-60% of parts)
        labor_rate = 0.5
        labor_cost = adjusted_parts * labor_rate

        # Total estimate
        total_est = adjusted_parts + labor_cost

        # Add confidence ranges
        uncertainty = 0.25  # +/- 25%
        min_cost = total_est * (1 - uncertainty)
        max_cost = total_est * (1 + uncertainty)

        return {
            "estimated_cost": round(total_est, 2),
            "min_cost": round(min_cost, 2),
            "max_cost": round(max_cost, 2),
            "currency": self.currency,
            "breakdown": breakdown,
            "labor_cost": round(labor_cost, 2),
            "parts_cost": round(adjusted_parts, 2),
            "severity_multiplier": mult
        }

    def _get_base_cost(self, damage_type: str) -> float:
        """
        Get base cost for a damage type.

        Args:
            damage_type: Type of damage

        Returns:
            Base repair cost
        """
        # Map damage type to cost
        type_lower = damage_type.lower()

        if "scratch" in type_lower:
            return self.costs.scratch
        elif "dent" in type_lower:
            return self.costs.dent
        elif "crack" in type_lower:
            return self.costs.crack
        elif "shatter" in type_lower:
            return self.costs.shatter
        elif "broken" in type_lower or "missing" in type_lower:
            return self.costs.broken_part
        else:
            return self.costs.default

    def batch_estimate(
        self,
        classifications: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Estimate costs for multiple damage assessments.

        Args:
            classifications: List of damage classification results

        Returns:
            List of cost estimates
        """
        return [self.estimate_cost(c) for c in classifications]

    def get_cost_summary(self, estimate: Dict[str, Any]) -> str:
        """
        Generate human-readable cost summary.

        Args:
            estimate: Cost estimate dictionary

        Returns:
            Formatted cost summary string
        """
        if estimate["estimated_cost"] == 0:
            return "No damage detected"

        min_val = estimate["min_cost"]
        max_val = estimate["max_cost"]
        est_val = estimate["estimated_cost"]
        curr = estimate["currency"]

        summary = f"Estimated repair cost: {est_val:,.0f} {curr}\n"
        summary += f"Range: {min_val:,.0f} - {max_val:,.0f} {curr}\n"
        summary += f"Parts: {estimate['parts_cost']:,.0f} {curr}\n"
        summary += f"Labor: {estimate['labor_cost']:,.0f} {curr}"

        return summary
