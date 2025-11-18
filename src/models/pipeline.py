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
from .sam_segmentor import SAMSegmentor
from .clip_analyzer import CLIPAnalyzer


class DamageAnalysisPipeline:
    """
    End-to-end pipeline for vehicle damage analysis.

    Performs detection, severity assessment, and cost estimation
    in a single workflow.
    """

    def __init__(
        self,
        model_path: str = "yolov9n.pt",
        conf_threshold: float = 0.35,
        device: str = "auto",
        enable_sam_segmentation: bool = True,
        sam_model_type: str = "vit_b",
        enable_clip_analysis: bool = True,
        clip_model_name: str = "ViT-B/32"
    ):
        """
        Initialize the damage analysis pipeline with YOLOv9n, SAM segmentation, and CLIP analysis.

        Args:
            model_path: Path to YOLO model weights (YOLOv9n)
            conf_threshold: Confidence threshold for detections (0.35 for accuracy)
            device: Device to run inference ('cpu', 'cuda', or 'auto')
            enable_sam_segmentation: Whether to enable SAM segmentation
            sam_model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            enable_clip_analysis: Whether to enable CLIP semantic analysis
            clip_model_name: CLIP model name ('ViT-B/32', 'ViT-L/14', etc.)
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
        
        # Initialize SAM for segmentation if enabled
        self.sam_segmentor = None
        if enable_sam_segmentation:
            try:
                self.sam_segmentor = SAMSegmentor(
                    model_type=sam_model_type,
                    device=device
                )
                print("SAM segmentation enabled")
            except Exception as e:
                print(f"Warning: Could not initialize SAM segmentation: {e}")
                print("Continuing with YOLO detection only")
        
        # Initialize CLIP for semantic analysis if enabled
        self.clip_analyzer = None
        if enable_clip_analysis:
            try:
                self.clip_analyzer = CLIPAnalyzer(
                    model_name=clip_model_name,
                    device=device
                )
                print("CLIP semantic analysis enabled")
            except Exception as e:
                print(f"Warning: Could not initialize CLIP analysis: {e}")
                print("Continuing without semantic analysis")

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

        # Step 2: Optional SAM segmentation for precise masks
        segmentation_results = None
        if self.sam_segmentor is not None and len(detection_results["detections"]) > 0:
            try:
                # Extract boxes from YOLO detections
                boxes = []
                class_names = []
                confidences = []
                
                for detection in detection_results["detections"]:
                    if 'box' in detection:
                        # Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
                        x_center, y_center, width, height = detection['box']
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        boxes.append([x1, y1, x2, y2])
                    elif 'bbox' in detection:
                        boxes.append(detection['bbox'])
                    
                    class_names.append(detection.get('class_name', 'unknown'))
                    confidences.append(detection.get('confidence', 0.0))
                
                # Perform SAM segmentation
                segmentation_results = self.sam_segmentor.segment_from_boxes(
                    image=image,
                    boxes=boxes,
                    class_names=class_names,
                    confidences=confidences
                )
                
                # Post-process masks
                segmentation_results['masks'] = self.sam_segmentor.process_masks(
                    segmentation_results['masks']
                )
                
                # Calculate precise damage areas
                damage_area = self.sam_segmentor.calculate_damage_area(
                    segmentation_results['masks'],
                    image.shape
                )
                segmentation_results['damage_area'] = damage_area
                
            except Exception as e:
                print(f"Warning: SAM segmentation failed: {e}")
                segmentation_results = None

        # Step 3: Classify severity
        classification = self.classifier.classify_damage(
            detection_results["detections"],
            image.shape
        )

        # Step 4: Optional CLIP semantic analysis
        semantic_results = None
        if self.clip_analyzer is not None and len(detection_results["detections"]) > 0:
            try:
                semantic_results = self.clip_analyzer.semantic_analysis(
                    image=image,
                    detections=detection_results["detections"],
                    ensemble_method="weighted"
                )
                
                # Update classification with semantic analysis
                if semantic_results['ensemble_predictions']:
                    # Boost confidence based on semantic analysis
                    confidence_boost = semantic_results.get('confidence_boost', {})
                    if confidence_boost:
                        avg_boost = np.mean(list(confidence_boost.values()))
                        classification['avg_confidence'] *= avg_boost
                        classification['severity'] = self._update_severity_with_semantic(
                            classification['severity'],
                            classification['avg_confidence']
                        )
                
            except Exception as e:
                print(f"Warning: CLIP semantic analysis failed: {e}")
                semantic_results = None

        # Step 5: Estimate cost (enhanced with segmentation and semantic data if available)
        cost_estimate = self.cost_estimator.estimate_cost(classification)
        
        # Enhance cost estimation with precise area data if available
        if segmentation_results is not None and 'damage_area' in segmentation_results:
            cost_estimate = self.cost_estimator.estimate_cost_with_area(
                classification,
                segmentation_results['damage_area']
            )
            
            # Further enhance with semantic analysis if available
            if semantic_results is not None:
                cost_estimate = self.cost_estimator.estimate_cost_with_semantic(
                    cost_estimate,
                    semantic_results['ensemble_predictions']
                )

        # Generate visualization if requested
        visualized_img = None
        if visualize and len(detection_results["detections"]) > 0:
            if self.sam_segmentor is not None and segmentation_results is not None:
                # Use SAM segmentation visualization
                visualized_img = self.sam_segmentor.visualize_segmentation(
                    image=image,
                    results=segmentation_results
                )
            elif self.clip_analyzer is not None and semantic_results is not None:
                # Use enhanced visualization with semantic information
                visualized_img = self._visualize_with_semantic_info(
                    image=image,
                    detections=detection_results["detections"],
                    semantic_results=semantic_results
                )
            else:
                # Fall back to YOLO bbox visualization
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
        
        # Add segmentation results if available
        if segmentation_results is not None:
            results["segmentation"] = {
                "masks": segmentation_results["masks"],
                "scores": segmentation_results["scores"],
                "damage_area": segmentation_results.get("damage_area", {}),
                "inference_time": segmentation_results["inference_time"]
            }
        
        # Add semantic analysis results if available
        if semantic_results is not None:
            results["semantic_analysis"] = {
                "ensemble_predictions": semantic_results["ensemble_predictions"],
                "confidence_boost": semantic_results["confidence_boost"],
                "analysis_time": semantic_results["analysis_time"],
                "detailed_analysis": semantic_results["semantic_analysis"]
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
    
    def _update_severity_with_semantic(self, original_severity: str, confidence: float) -> str:
        """
        Update severity level based on semantic analysis confidence.
        
        Args:
            original_severity: Original severity level
            confidence: Updated confidence score
            
        Returns:
            Updated severity level
        """
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "moderate"
        elif confidence >= 0.3:
            return "low"
        else:
            return "minimal"
    
    def _visualize_with_semantic_info(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        semantic_results: Dict[str, Any]
    ) -> np.ndarray:
        """
        Create visualization with semantic analysis information.
        
        Args:
            image: Original image
            detections: YOLO detections
            semantic_results: CLIP semantic analysis results
            
        Returns:
            Annotated image with semantic information
        """
        annotated_image = image.copy()
        
        # Color map for different damage types
        color_map = {
            'scratch': (0, 255, 255),      # Yellow
            'dent': (0, 165, 255),         # Orange
            'crack': (0, 0, 255),          # Red
            'shatter': (255, 0, 0),        # Blue
            'rust': (128, 0, 128),         # Purple
            'broken_part': (255, 165, 0),  # Orange
            'paint_damage': (128, 128, 128), # Gray
            'smash': (255, 0, 255),        # Pink
            'glass_damage': (0, 255, 0),   # Green
            'light_damage': (255, 255, 255) # White
        }
        
        for i, detection in enumerate(detections):
            if i >= len(semantic_results['semantic_analysis']):
                continue
                
            # Get bbox
            if 'bbox' in detection:
                x1, y1, x2, y2 = detection['bbox']
            elif 'box' in detection:
                x_center, y_center, w, h = detection['box']
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2
            else:
                continue
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get semantic analysis for this region
            semantic_data = semantic_results['semantic_analysis'][i]
            ensemble_result = semantic_data['ensemble_result']
            
            # Choose color based on semantic analysis
            predicted_class = ensemble_result['top_class']
            color = color_map.get(predicted_class, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Create label with semantic information
            confidence = ensemble_result['top_confidence']
            label = f"{predicted_class}: {confidence:.2f}"
            
            # Add semantic confidence boost info
            confidence_boost = semantic_results['confidence_boost'].get(f"region_{i}", 1.0)
            if confidence_boost > 1.1:
                label += f" (boost: {confidence_boost:.1f}x)"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated_image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Add top semantic descriptions as tooltip
            clip_analysis = semantic_data['clip_analysis']
            if clip_analysis:
                top_description = clip_analysis[0]['description']
                # Truncate description if too long
                if len(top_description) > 30:
                    top_description = top_description[:27] + "..."
                
                # Draw small info box
                info_text = f"CLIP: {top_description}"
                (info_width, info_height), _ = cv2.getTextSize(
                    info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                cv2.rectangle(
                    annotated_image,
                    (x1, y2),
                    (x1 + info_width, y2 + info_height + 5),
                    color,
                    -1
                )
                cv2.putText(
                    annotated_image,
                    info_text,
                    (x1, y2 + info_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
        
        return annotated_image
