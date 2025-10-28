"""
Main inference pipeline for vehicle damage detection and analysis.
Combines detection, classification, and cost estimation.
"""

import cv2
import numpy as np
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import time

from .yolo_detector import YOLODamageDetector
from .damage_classifier import DamageClassifier
from .cost_estimator import CostEstimator


class DamageAnalysisPipeline:
    """
    End-to-end pipeline for vehicle damage analysis.

    Performs detection, severity assessment, and cost estimation
    in a single workflow.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        device: str = "auto"
    ):
        """
        Initialize the damage analysis pipeline.

        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            device: Device to run inference ('cpu', 'cuda', or 'auto')
        """
        print(f"Initializing damage analysis pipeline...")

        # Initialize components
        self.detector = YOLODamageDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device
        )

        self.classifier = DamageClassifier()
        self.cost_estimator = CostEstimator()

        print("Pipeline ready!")

    def analyze_image(
        self,
        image_path: Union[str, Path, np.ndarray],
        visualize: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a single image for vehicle damage.

        Args:
            image_path: Path to image or numpy array
            visualize: Whether to generate visualization

        Returns:
            Complete analysis results including detections,
            classification, and cost estimate
        """
        start_time = time.time()

        # Load image if path provided
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
        else:
            image = image_path

        # Step 1: Detect damage
        detection_results = self.detector.detect(image)

        # Step 2: Classify severity
        classification = self.classifier.classify_damage(
            detection_results["detections"],
            image.shape
        )

        # Step 3: Estimate cost
        cost_estimate = self.cost_estimator.estimate_cost(classification)

        # Generate visualization if requested
        visualized_img = None
        if visualize and len(detection_results["detections"]) > 0:
            visualized_img = self.detector.visualize_detections(
                image,
                detection_results["detections"]
            )

        inference_time = time.time() - start_time

        # Compile complete results
        results = {
            "detection": {
                "num_detections": len(detection_results["detections"]),
                "detections": detection_results["detections"],
                "inference_time": detection_results["inference_time"]
            },
            "classification": classification,
            "cost_estimate": cost_estimate,
            "visualization": visualized_img,
            "total_time": inference_time,
            "image_shape": image.shape
        }

        return results

    def analyze_batch(
        self,
        image_paths: List[Union[str, Path]],
        visualize: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple images.

        Args:
            image_paths: List of image paths
            visualize: Whether to generate visualizations

        Returns:
            List of analysis results
        """
        results = []

        for img_path in image_paths:
            try:
                result = self.analyze_image(img_path, visualize)
                result["image_path"] = str(img_path)
                result["status"] = "success"
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": str(img_path),
                    "status": "error",
                    "error": str(e)
                })

        return results

    def get_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of analysis.

        Args:
            results: Analysis results dictionary

        Returns:
            Formatted summary string
        """
        det = results["detection"]
        cls = results["classification"]
        cost = results["cost_estimate"]

        summary = "=== Vehicle Damage Analysis Summary ===\n\n"

        # Detection info
        summary += f"Detections: {det['num_detections']} damage(s) found\n"

        if det['num_detections'] > 0:
            summary += f"Severity: {cls['severity'].upper()}\n"
            summary += f"Damage coverage: {cls['area_ratio']*100:.1f}% of image\n"
            summary += f"Confidence: {cls['avg_confidence']*100:.1f}%\n\n"

            # Damage types
            summary += "Damage types detected:\n"
            for dtype, count in cls['damage_types'].items():
                summary += f"  - {dtype}: {count}\n"

            summary += f"\n{self.cost_estimator.get_cost_summary(cost)}\n"

            summary += f"\nProcessing time: {results['total_time']:.2f}s"
        else:
            summary += "\nNo damage detected in the image."

        return summary

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Union[str, Path],
        save_visualization: bool = True
    ):
        """
        Save analysis results to disk.

        Args:
            results: Analysis results
            output_dir: Directory to save outputs
            save_visualization: Whether to save visualized image
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save visualization
        if save_visualization and results.get("visualization") is not None:
            vis_path = output_dir / "detection_visualization.jpg"
            cv2.imwrite(str(vis_path), results["visualization"])

        # Save text summary
        summary = self.get_summary(results)
        summary_path = output_dir / "analysis_summary.txt"
        summary_path.write_text(summary)

        print(f"Results saved to {output_dir}")
