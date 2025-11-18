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

    def estimate_cost_with_area(
        self,
        damage_classification: Dict[str, Any],
        damage_area: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Estimate repair cost with precise area calculations from segmentation.
        
        Args:
            damage_classification: Output from DamageClassifier
            damage_area: Area calculations from SAM segmentation
            
        Returns:
            Enhanced cost estimate with area-based pricing
        """
        base_estimate = self.estimate_cost(damage_classification)
        
        if not damage_area or damage_area.get('total_pixels', 0) == 0:
            return base_estimate
        
        # Get area-based adjustments
        total_pixels = damage_area.get('total_pixels', 0)
        image_area = damage_classification.get('image_area', 1)
        area_ratio = total_pixels / image_area
        
        # Area-based cost multipliers
        if area_ratio > 0.1:  # Large damage area
            area_multiplier = 1.8
        elif area_ratio > 0.05:  # Medium damage area
            area_multiplier = 1.4
        elif area_ratio > 0.01:  # Small damage area
            area_multiplier = 1.1
        else:  # Very small damage area
            area_multiplier = 1.0
        
        # Adjust costs based on area
        enhanced_estimate = base_estimate.copy()
        enhanced_estimate['estimated_cost'] *= area_multiplier
        enhanced_estimate['min_cost'] *= area_multiplier
        enhanced_estimate['max_cost'] *= area_multiplier
        enhanced_estimate['parts_cost'] *= area_multiplier
        enhanced_estimate['area_multiplier'] = area_multiplier
        enhanced_estimate['damage_area_pixels'] = total_pixels
        enhanced_estimate['damage_area_ratio'] = area_ratio
        
        # Add area information to breakdown
        if 'breakdown' in enhanced_estimate:
            enhanced_estimate['breakdown']['area_analysis'] = {
                'total_pixels': total_pixels,
                'area_ratio': area_ratio,
                'area_multiplier': area_multiplier
            }
        
        return enhanced_estimate

    def estimate_cost_with_semantic(
        self,
        base_estimate: Dict[str, Any],
        semantic_predictions: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Enhance cost estimate with semantic analysis confidence.
        
        Args:
            base_estimate: Base cost estimate
            semantic_predictions: CLIP semantic predictions with confidence scores
            
        Returns:
            Enhanced cost estimate with semantic confidence adjustments
        """
        if not semantic_predictions:
            return base_estimate
        
        enhanced_estimate = base_estimate.copy()
        
        # Calculate semantic confidence boost
        max_confidence = max(semantic_predictions.values()) if semantic_predictions else 0.0
        avg_confidence = sum(semantic_predictions.values()) / len(semantic_predictions) if semantic_predictions else 0.0
        
        # Confidence-based adjustments
        if max_confidence > 0.8:
            confidence_multiplier = 1.2  # High confidence
        elif max_confidence > 0.6:
            confidence_multiplier = 1.1  # Medium confidence
        elif max_confidence > 0.4:
            confidence_multiplier = 1.05  # Low confidence
        else:
            confidence_multiplier = 1.0  # Very low confidence
        
        # Apply confidence boost to costs
        enhanced_estimate['estimated_cost'] *= confidence_multiplier
        enhanced_estimate['min_cost'] *= confidence_multiplier
        enhanced_estimate['max_cost'] *= confidence_multiplier
        enhanced_estimate['labor_cost'] *= confidence_multiplier
        enhanced_estimate['semantic_confidence'] = max_confidence
        enhanced_estimate['semantic_multiplier'] = confidence_multiplier
        
        # Add semantic information to breakdown
        if 'breakdown' in enhanced_estimate:
            enhanced_estimate['breakdown']['semantic_analysis'] = {
                'top_prediction_confidence': max_confidence,
                'avg_prediction_confidence': avg_confidence,
                'confidence_multiplier': confidence_multiplier,
                'predicted_classes': semantic_predictions
            }
        
        return enhanced_estimate

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
