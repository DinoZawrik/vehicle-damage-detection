"""
Damage severity classification module.
Estimates severity of vehicle damage based on detection results.
"""

from typing import List, Dict, Any
from enum import Enum
import numpy as np


class DamageSeverity(str, Enum):
    """Enumeration for damage severity levels."""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class DamageClassifier:
    """
    Classifier for assessing vehicle damage severity.

    Uses a combination of damage area, count, and confidence
    to determine overall severity level.
    """

    def __init__(
        self,
        minor_threshold: float = 0.02,
        moderate_threshold: float = 0.08,
        severe_threshold: float = 0.20
    ):
        """
        Initialize the damage classifier.

        Args:
            minor_threshold: Area ratio threshold for minor damage
            moderate_threshold: Area ratio threshold for moderate damage
            severe_threshold: Area ratio threshold for severe damage
        """
        self.minor_th = minor_threshold
        self.moderate_th = moderate_threshold
        self.severe_th = severe_threshold

    def classify_damage(
        self,
        detections: List[Dict[str, Any]],
        image_shape: tuple
    ) -> Dict[str, Any]:
        """
        Classify damage severity based on detection results.

        Args:
            detections: List of detection dictionaries with bbox and confidence
            image_shape: Shape of the image (height, width)

        Returns:
            Dictionary with severity level and detailed metrics
        """
        if not detections:
            return {
                "severity": None,
                "damage_count": 0,
                "total_damage_area": 0,
                "area_ratio": 0,
                "avg_confidence": 0,
                "damage_types": {}
            }

        img_height, img_width = image_shape[:2]
        total_img_area = img_height * img_width

        # Calculate total damaged area
        total_damage_area = 0
        confidences = []
        damage_types = {}

        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                total_damage_area += area

            conf = det.get("confidence", 0)
            confidences.append(conf)

            # Track damage type distribution
            dmg_type = det.get("class", "unknown")
            damage_types[dmg_type] = damage_types.get(dmg_type, 0) + 1

        area_ratio = total_damage_area / total_img_area
        avg_conf = np.mean(confidences) if confidences else 0
        damage_cnt = len(detections)

        # Determine severity based on multiple factors
        severity = self._determine_severity(
            area_ratio, damage_cnt, avg_conf
        )

        return {
            "severity": severity.value if severity else None,
            "damage_count": damage_cnt,
            "total_damage_area": int(total_damage_area),
            "area_ratio": float(area_ratio),
            "avg_confidence": float(avg_conf),
            "damage_types": damage_types
        }

    def _determine_severity(
        self,
        area_ratio: float,
        damage_count: int,
        avg_confidence: float
    ) -> DamageSeverity:
        """
        Internal method to determine severity level.

        Args:
            area_ratio: Ratio of damaged area to total image area
            damage_count: Number of detected damages
            avg_confidence: Average detection confidence

        Returns:
            DamageSeverity enum value
        """
        # Weight by confidence - lower confidence reduces effective ratio
        weighted_ratio = area_ratio * avg_confidence

        # Multiple small damages can indicate severe impact
        count_factor = 1.0
        if damage_count > 5:
            count_factor = 1.3
        elif damage_count > 3:
            count_factor = 1.15

        effective_ratio = weighted_ratio * count_factor

        # Classify based on effective ratio
        if effective_ratio < self.minor_th:
            return DamageSeverity.MINOR
        elif effective_ratio < self.moderate_th:
            return DamageSeverity.MODERATE
        elif effective_ratio < self.severe_th:
            return DamageSeverity.SEVERE
        else:
            return DamageSeverity.CRITICAL

    def batch_classify(
        self,
        detections_list: List[List[Dict[str, Any]]],
        image_shapes: List[tuple]
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple images at once.

        Args:
            detections_list: List of detection lists
            image_shapes: List of image shapes

        Returns:
            List of classification results
        """
        results = []
        for dets, shape in zip(detections_list, image_shapes):
            result = self.classify_damage(dets, shape)
            results.append(result)
        return results
