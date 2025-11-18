"""
Unified Damage Analyzer - MVP Version

Combines damage classification and cost estimation in one module.
Simplified for the MVP without external dependencies.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np


class DamageSeverity(str, Enum):
    """Damage severity levels."""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class DamageType(str, Enum):
    """Common vehicle damage types."""
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    SHATTER = "shatter"
    BROKEN_GLASS = "broken_glass"
    PAINT_DAMAGE = "paint_damage"
    RUST = "rust"
    UNKNOWN = "unknown"


class DamageAnalyzer:
    """
    Unified analyzer for damage classification and cost estimation.
    
    Simplifies the pipeline by combining severity assessment and
    cost calculation in a single pass.
    """
    
    # Base repair costs (USD)
    BASE_COSTS = {
        DamageType.SCRATCH: 150.0,
        DamageType.DENT: 300.0,
        DamageType.CRACK: 450.0,
        DamageType.SHATTER: 800.0,
        DamageType.BROKEN_GLASS: 600.0,
        DamageType.PAINT_DAMAGE: 200.0,
        DamageType.RUST: 350.0,
        DamageType.UNKNOWN: 250.0,
    }
    
    # Severity multipliers
    SEVERITY_MULTIPLIERS = {
        DamageSeverity.MINOR: 1.0,
        DamageSeverity.MODERATE: 1.5,
        DamageSeverity.SEVERE: 2.3,
        DamageSeverity.CRITICAL: 3.5,
    }
    
    def __init__(
        self,
        currency: str = "USD",
        labor_rate: float = 0.5
    ):
        """
        Initialize damage analyzer.
        
        Args:
            currency: Currency code for costs
            labor_rate: Labor cost as ratio of parts (0.5 = 50%)
        """
        self.currency = currency
        self.labor_rate = labor_rate
    
    def analyze(
        self,
        detections: List[Dict[str, Any]],
        image_shape: tuple
    ) -> Dict[str, Any]:
        """
        Analyze detections: classify severity and estimate costs.
        
        Args:
            detections: List of detection dictionaries from YOLO
            image_shape: (height, width) of image
            
        Returns:
            Complete analysis with severity and cost estimate
        """
        if not detections or len(detections) == 0:
            return self._empty_result()
        
        # Step 1: Classify severity
        severity_info = self._classify_severity(detections, image_shape)
        
        # Step 2: Estimate costs
        cost_info = self._estimate_costs(detections, severity_info['severity'])
        
        # Combine results
        return {
            **severity_info,
            **cost_info,
            'currency': self.currency
        }
    
    def _classify_severity(
        self,
        detections: List[Dict[str, Any]],
        image_shape: tuple
    ) -> Dict[str, Any]:
        """
        Classify damage severity based on detections.
        
        Args:
            detections: List of detections
            image_shape: Image dimensions
            
        Returns:
            Severity classification info
        """
        img_height, img_width = image_shape[:2]
        total_img_area = img_height * img_width
        
        # Calculate metrics
        total_damage_area = 0
        confidences = []
        damage_types = {}
        
        for det in detections:
            # Area calculation
            if 'area' in det:
                total_damage_area += det['area']
            elif 'bbox' in det:
                bbox = det['bbox']
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    area = abs((x2 - x1) * (y2 - y1))
                    total_damage_area += area
            
            # Confidence
            if 'confidence' in det:
                confidences.append(det['confidence'])
            
            # Type distribution
            damage_type = self._normalize_damage_type(
                det.get('class_name', det.get('class', 'unknown'))
            )
            damage_types[damage_type] = damage_types.get(damage_type, 0) + 1
        
        # Calculate ratios
        area_ratio = total_damage_area / total_img_area if total_img_area > 0 else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        damage_count = len(detections)
        
        # Determine severity
        severity = self._determine_severity(area_ratio, damage_count, avg_confidence)
        
        return {
            'severity': severity,
            'damage_count': damage_count,
            'damage_types': damage_types,
            'total_damage_area': round(total_damage_area, 2),
            'area_ratio': round(area_ratio, 4),
            'avg_confidence': round(avg_confidence, 3)
        }
    
    def _determine_severity(
        self,
        area_ratio: float,
        damage_count: int,
        avg_confidence: float
    ) -> str:
        """
        Determine severity level from metrics.
        
        Args:
            area_ratio: Ratio of damaged area to total
            damage_count: Number of damage instances
            avg_confidence: Average detection confidence
            
        Returns:
            Severity level string
        """
        # Thresholds
        if area_ratio >= 0.20 or damage_count >= 10:
            return DamageSeverity.CRITICAL
        elif area_ratio >= 0.08 or damage_count >= 5:
            return DamageSeverity.SEVERE
        elif area_ratio >= 0.02 or damage_count >= 2:
            return DamageSeverity.MODERATE
        else:
            return DamageSeverity.MINOR
    
    def _estimate_costs(
        self,
        detections: List[Dict[str, Any]],
        severity: str
    ) -> Dict[str, Any]:
        """
        Estimate repair costs.
        
        Args:
            detections: List of detections
            severity: Severity level
            
        Returns:
            Cost estimate breakdown
        """
        # Calculate parts cost
        parts_cost = 0
        breakdown = {}
        
        for det in detections:
            damage_type = self._normalize_damage_type(
                det.get('class_name', det.get('class', 'unknown'))
            )
            
            # Get base cost
            base_cost = self.BASE_COSTS.get(damage_type, self.BASE_COSTS[DamageType.UNKNOWN])
            
            # Track in breakdown
            if damage_type not in breakdown:
                breakdown[damage_type] = {
                    'count': 0,
                    'unit_cost': base_cost,
                    'subtotal': 0
                }
            
            breakdown[damage_type]['count'] += 1
            breakdown[damage_type]['subtotal'] += base_cost
            parts_cost += base_cost
        
        # Apply severity multiplier
        severity_mult = self.SEVERITY_MULTIPLIERS.get(
            severity,
            self.SEVERITY_MULTIPLIERS[DamageSeverity.MODERATE]
        )
        adjusted_parts = parts_cost * severity_mult
        
        # Calculate labor
        labor_cost = adjusted_parts * self.labor_rate
        
        # Total cost
        total_cost = adjusted_parts + labor_cost
        
        # Cost range (±20%)
        min_cost = total_cost * 0.8
        max_cost = total_cost * 1.2
        
        return {
            'cost_estimate': {
                'total': round(total_cost, 2),
                'min': round(min_cost, 2),
                'max': round(max_cost, 2),
                'parts': round(adjusted_parts, 2),
                'labor': round(labor_cost, 2),
                'currency': self.currency
            },
            'cost_breakdown': breakdown,
            'severity_multiplier': severity_mult
        }
    
    def _normalize_damage_type(self, type_str: str) -> str:
        """
        Normalize damage type string to standard enum.
        
        Args:
            type_str: Raw damage type string
            
        Returns:
            Normalized damage type
        """
        type_lower = str(type_str).lower()
        
        # Map common variations
        if 'scratch' in type_lower or 'scuff' in type_lower:
            return DamageType.SCRATCH
        elif 'dent' in type_lower or 'ding' in type_lower:
            return DamageType.DENT
        elif 'crack' in type_lower:
            return DamageType.CRACK
        elif 'shatter' in type_lower or 'broken' in type_lower:
            if 'glass' in type_lower:
                return DamageType.BROKEN_GLASS
            return DamageType.SHATTER
        elif 'glass' in type_lower:
            return DamageType.BROKEN_GLASS
        elif 'paint' in type_lower:
            return DamageType.PAINT_DAMAGE
        elif 'rust' in type_lower or 'corrosion' in type_lower:
            return DamageType.RUST
        else:
            return DamageType.UNKNOWN
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty analysis result."""
        return {
            'severity': None,
            'damage_count': 0,
            'damage_types': {},
            'total_damage_area': 0,
            'area_ratio': 0,
            'avg_confidence': 0,
            'cost_estimate': {
                'total': 0,
                'min': 0,
                'max': 0,
                'parts': 0,
                'labor': 0
            },
            'cost_breakdown': {},
            'severity_multiplier': 1.0,
            'currency': self.currency
        }
    
    def add_severity_to_detections(
        self,
        detections: List[Dict[str, Any]],
        severity: str
    ) -> List[Dict[str, Any]]:
        """
        Add severity field to each detection.
        
        Args:
            detections: List of detections
            severity: Overall severity level
            
        Returns:
            Detections with severity added
        """
        for det in detections:
            det['severity'] = severity
        return detections


# Convenience function
def analyze_damage(
    detections: List[Dict[str, Any]],
    image_shape: tuple,
    currency: str = "USD"
) -> Dict[str, Any]:
    """
    Quick damage analysis.
    
    Args:
        detections: YOLO detections
        image_shape: Image dimensions
        currency: Currency code
        
    Returns:
        Complete analysis
    """
    analyzer = DamageAnalyzer(currency=currency)
    return analyzer.analyze(detections, image_shape)
