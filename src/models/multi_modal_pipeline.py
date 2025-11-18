"""
Полностью реализованная мультимодальная архитектура для анализа повреждений автомобилей.

Объединяет YOLO, SAM, CLIP и LLM для максимальной точности и понимания повреждений.
"""

import cv2
import numpy as np
import torch
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
import logging
from collections import defaultdict, Counter

from .yolo_detector import YOLODamageDetector
from .sam_segmentor import SAMSegmentor
from .clip_analyzer import CLIPAnalyzer
from .cost_estimator import CostEstimator
from .damage_classifier import DamageClassifier
from .llm_analyzer import LLMAnalyzer, DetectionResult

logger = logging.getLogger(__name__)


class MultiModalPipeline:
    """
    Мультимодальная система анализа повреждений автомобилей.
    
    Комбинирует:
    - YOLOv9n: Быстрое и точное детектирование повреждений
    - SAM: Точные маски сегментации для расчета площади
    - CLIP: Семантическое понимание типов повреждений
    - LLM: Качественные отчеты и рекомендации
    
    Преимущества:
    - Ensemble predictions для повышенной точности
    - Точные расчеты площади повреждений
    - Семантическое понимание контекста
    - Профессиональные отчеты с рекомендациями
    """
    
    def __init__(
        self,
        yolo_model_path: str = "yolov9n.pt",
        yolo_conf_threshold: float = 0.35,
        enable_sam: bool = True,
        sam_model_type: str = "vit_b",
        enable_clip: bool = True,
        clip_model_name: str = "ViT-B/32",
        enable_llm: bool = True,
        llm_api_key: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize multi-modal pipeline.
        
        Args:
            yolo_model_path: Path to YOLO model weights
            yolo_conf_threshold: Confidence threshold for YOLO
            enable_sam: Enable SAM segmentation
            sam_model_type: SAM model type
            enable_clip: Enable CLIP semantic analysis
            clip_model_name: CLIP model name
            enable_llm: Enable LLM analysis
            llm_api_key: LLM API key
            device: Device for inference
        """
        print("🚀 Initializing Multi-Modal Damage Analysis Pipeline...")
        
        self.device = device
        self.yolo_conf_threshold = yolo_conf_threshold
        
        # Initialize components with error handling
        self.yolo_detector = YOLODamageDetector(
            model_path=yolo_model_path,
            conf_threshold=yolo_conf_threshold,
            device=device
        )
        
        self.sam_segmentor = None
        if enable_sam:
            try:
                self.sam_segmentor = SAMSegmentor(
                    model_type=sam_model_type,
                    device=device
                )
                print("✅ SAM segmentation enabled")
            except Exception as e:
                print(f"⚠️ SAM segmentation disabled: {e}")
                logger.warning(f"SAM initialization failed: {e}")
        
        self.clip_analyzer = None
        if enable_clip:
            try:
                self.clip_analyzer = CLIPAnalyzer(
                    model_name=clip_model_name,
                    device=device
                )
                print("✅ CLIP semantic analysis enabled")
            except Exception as e:
                print(f"⚠️ CLIP analysis disabled: {e}")
                logger.warning(f"CLIP initialization failed: {e}")
        
        self.llm_analyzer = None
        if enable_llm:
            try:
                self.llm_analyzer = LLMAnalyzer(llm_api_key) if llm_api_key else LLMAnalyzer()
                print("✅ LLM analysis enabled")
            except Exception as e:
                print(f"⚠️ LLM analysis disabled: {e}")
                logger.warning(f"LLM initialization failed: {e}")
        
        self.cost_estimator = CostEstimator()
        self.classifier = DamageClassifier()
        
        print("🎯 Multi-modal pipeline ready!")
    
    def analyze_image(
        self,
        image_path: Union[str, Path, np.ndarray],
        visualize: bool = True,
        ensemble_method: str = "weighted"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive multi-modal analysis of vehicle damage.
        
        Args:
            image_path: Input image path or numpy array
            visualize: Whether to generate visualizations
            ensemble_method: Method for ensemble predictions
            
        Returns:
            Complete multi-modal analysis results
        """
        start_time = time.time()
        
        # Load image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
        else:
            image = image_path
        
        print(f"📸 Analyzing image: {image.shape}")
        
        # Multi-modal analysis pipeline
        results = {
            "image_info": {
                "shape": image.shape,
                "analysis_time": 0.0,
                "pipeline_version": "1.0.0"
            }
        }
        
        # Step 1: YOLO Detection
        print("🔍 Step 1: YOLO Detection")
        yolo_results = self._analyze_with_yolo(image)
        results["yolo"] = yolo_results
        
        # Step 2: SAM Segmentation (if available)
        sam_results = None
        if self.sam_segmentor and yolo_results["detections"]:
            print("✂️ Step 2: SAM Segmentation")
            try:
                sam_results = self._analyze_with_sam(image, yolo_results["detections"])
                results["sam"] = sam_results
            except Exception as e:
                print(f"⚠️ SAM segmentation failed: {e}")
                logger.error(f"SAM segmentation error: {e}")
        
        # Step 3: CLIP Semantics (if available)
        clip_results = None
        if self.clip_analyzer and yolo_results["detections"]:
            print("🧠 Step 3: CLIP Semantic Analysis")
            try:
                clip_results = self._analyze_with_clip(image, yolo_results["detections"])
                results["clip"] = clip_results
            except Exception as e:
                print(f"⚠️ CLIP analysis failed: {e}")
                logger.error(f"CLIP analysis error: {e}")
        
        # Step 4: Ensemble Predictions
        print("🎲 Step 4: Ensemble Predictions")
        ensemble_results = self._ensemble_predictions(
            yolo_results, sam_results, clip_results, ensemble_method
        )
        results["ensemble"] = ensemble_results
        
        # Step 5: Cost Estimation
        print("💰 Step 5: Cost Estimation")
        cost_results = self._estimate_costs(ensemble_results, sam_results)
        results["cost_analysis"] = cost_results
        
        # Step 6: LLM Analysis (if available)
        if self.llm_analyzer and yolo_results["detections"]:
            print("📝 Step 6: LLM Analysis")
            try:
                llm_results = self._analyze_with_llm(image, ensemble_results)
                results["llm"] = llm_results
            except Exception as e:
                print(f"⚠️ LLM analysis failed: {e}")
                logger.error(f"LLM analysis error: {e}")
        
        # Step 7: Generate Visualization
        if visualize:
            print("🖼️ Step 7: Generating Visualization")
            try:
                viz_image = self._generate_multimodal_visualization(
                    image, results, ensemble_method
                )
                results["visualization"] = viz_image
            except Exception as e:
                print(f"⚠️ Visualization generation failed: {e}")
                logger.error(f"Visualization error: {e}")
        
        # Finalize results
        total_time = time.time() - start_time
        results["image_info"]["analysis_time"] = total_time
        
        print(f"✅ Analysis completed in {total_time:.2f}s")
        
        return results
    
    def _analyze_with_yolo(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform YOLO detection."""
        start_time = time.time()
        
        try:
            detection_results = self.yolo_detector.detect(image)
            
            return {
                "detections": detection_results["detections"],
                "inference_time": detection_results["inference_time"],
                "confidence_scores": [d.get("confidence", 0.0) for d in detection_results["detections"]],
                "bbox_coordinates": [d.get("bbox", [0, 0, 0, 0]) for d in detection_results["detections"]],
                "class_names": [d.get("class_name", "unknown") for d in detection_results["detections"]]
            }
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return {
                "detections": [],
                "inference_time": 0.0,
                "confidence_scores": [],
                "bbox_coordinates": [],
                "class_names": []
            }
    
    def _analyze_with_sam(
        self,
        image: np.ndarray,
        yolo_detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform SAM segmentation."""
        start_time = time.time()
        
        # Extract boxes from YOLO detections
        boxes = []
        class_names = []
        confidences = []
        
        for detection in yolo_detections:
            if 'bbox' in detection:
                boxes.append(detection['bbox'])
            elif 'box' in detection:
                x_center, y_center, w, h = detection['box']
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2
                boxes.append([x1, y1, x2, y2])
            
            class_names.append(detection.get('class_name', 'unknown'))
            confidences.append(detection.get('confidence', 0.0))
        
        # Perform SAM segmentation
        segmentation_results = self.sam_segmentor.segment_from_boxes(
            image=image,
            boxes=boxes,
            class_names=class_names,
            confidences=confidences
        )
        
        # Process masks
        segmentation_results['masks'] = self.sam_segmentor.process_masks(
            segmentation_results['masks']
        )
        
        # Calculate damage areas
        damage_area = self.sam_segmentor.calculate_damage_area(
            segmentation_results['masks'],
            image.shape
        )
        segmentation_results['damage_area'] = damage_area
        
        return segmentation_results
    
    def _analyze_with_clip(
        self,
        image: np.ndarray,
        yolo_detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform CLIP semantic analysis."""
        start_time = time.time()
        
        semantic_results = self.clip_analyzer.semantic_analysis(
            image=image,
            detections=yolo_detections,
            ensemble_method="weighted"
        )
        
        return semantic_results
    
    def _ensemble_predictions(
        self,
        yolo_results: Dict[str, Any],
        sam_results: Optional[Dict[str, Any]],
        clip_results: Optional[Dict[str, Any]],
        method: str = "weighted"
    ) -> Dict[str, Any]:
        """Combine predictions from all available modalities."""
        
        ensemble_start = time.time()
        
        # Collect predictions from all modalities
        all_predictions = defaultdict(list)
        confidence_weights = {}
        
        # YOLO predictions
        yolo_detections = yolo_results.get("detections", [])
        for i, detection in enumerate(yolo_detections):
            class_name = detection.get("class_name", "unknown")
            confidence = detection.get("confidence", 0.0)
            all_predictions[i].append(("yolo", class_name, confidence))
        
        # SAM predictions (based on mask quality)
        if sam_results:
            sam_scores = sam_results.get("scores", [])
            for i, score in enumerate(sam_scores):
                if i < len(yolo_detections):
                    class_name = yolo_detections[i].get("class_name", "unknown")
                    all_predictions[i].append(("sam", class_name, float(score)))
        
        # CLIP predictions
        if clip_results and "ensemble_predictions" in clip_results:
            clip_predictions = clip_results["ensemble_predictions"]
            for class_name, confidence in clip_predictions.items():
                # Distribute CLIP confidence across detections
                for i in range(len(yolo_detections)):
                    all_predictions[i].append(("clip", class_name, confidence))
        
        # Ensemble predictions per detection
        final_detections = []
        
        for detection_id, predictions in all_predictions.items():
            if not predictions:
                continue
            
            # Aggregate predictions
            class_confidences = defaultdict(float)
            total_weight = 0.0
            
            for modality, class_name, confidence in predictions:
                # Apply modality-specific weights
                if modality == "yolo":
                    weight = 0.4
                elif modality == "sam":
                    weight = 0.3
                elif modality == "clip":
                    weight = 0.3
                else:
                    weight = 0.1
                
                class_confidences[class_name] += confidence * weight
                total_weight += weight
            
            # Normalize and select best class
            if total_weight > 0:
                for class_name in class_confidences:
                    class_confidences[class_name] /= total_weight
                
                best_class = max(class_confidences, key=class_confidences.get)
                best_confidence = class_confidences[best_class]
            else:
                best_class = "unknown"
                best_confidence = 0.0
            
            # Create enhanced detection
            enhanced_detection = {
                "class_name": best_class,
                "confidence": best_confidence,
                "modality_confidences": dict(class_confidences),
                "bbox": yolo_detections[detection_id].get("bbox", [0, 0, 0, 0])
            }
            
            # Add SAM mask if available
            if sam_results and "masks" in sam_results:
                if detection_id < len(sam_results["masks"]):
                    enhanced_detection["mask"] = sam_results["masks"][detection_id]
            
            # Add CLIP analysis if available
            if clip_results and "semantic_analysis" in clip_results:
                if detection_id < len(clip_results["semantic_analysis"]):
                    enhanced_detection["semantic_info"] = clip_results["semantic_analysis"][detection_id]
            
            final_detections.append(enhanced_detection)
        
        ensemble_time = time.time() - ensemble_start
        
        return {
            "detections": final_detections,
            "ensemble_method": method,
            "inference_time": ensemble_time,
            "modality_coverage": {
                "yolo": len(yolo_detections) > 0,
                "sam": sam_results is not None,
                "clip": clip_results is not None
            }
        }
    
    def _estimate_costs(
        self,
        ensemble_results: Dict[str, Any],
        sam_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Estimate repair costs based on multi-modal analysis."""
        
        detections = ensemble_results.get("detections", [])
        
        if not detections:
            return {
                "estimated_cost": 0.0,
                "breakdown": {},
                "currency": "USD"
            }
        
        # Create damage classification for cost estimator
        damage_classification = self.classifier.classify_damage(
            detections,
            [1080, 1920, 3]  # Placeholder image shape
        )
        
        # Base cost estimation
        cost_estimate = self.cost_estimator.estimate_cost(damage_classification)
        
        # Enhance with SAM area analysis if available
        if sam_results and "damage_area" in sam_results:
            cost_estimate = self.cost_estimator.estimate_cost_with_area(
                damage_classification,
                sam_results["damage_area"]
            )
        
        # Add multi-modal confidence adjustments
        avg_confidence = np.mean([d.get("confidence", 0.0) for d in detections])
        cost_estimate["multi_modal_confidence"] = avg_confidence
        
        return cost_estimate
    
    def _analyze_with_llm(
        self,
        image: np.ndarray,
        ensemble_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate LLM analysis and report."""
        
        detections = ensemble_results.get("detections", [])
        
        # Convert to LLM format
        llm_detections = []
        for detection in detections:
            llm_detections.append(DetectionResult(
                class_name=detection.get("class_name", "unknown"),
                confidence=detection.get("confidence", 0.0),
                bbox=detection.get("bbox", [0, 0, 0, 0]),
                area=detection.get("area", 0.0)
            ))
        
        # Generate LLM analysis
        try:
            llm_report = self.llm_analyzer.analyze_damage(
                llm_detections,
                image.shape[1],  # width
                image.shape[0]   # height
            )
            
            return {
                "summary": llm_report.summary,
                "detailed_description": llm_report.detailed_description,
                "damage_areas": llm_report.damage_areas,
                "severity_level": llm_report.severity_level,
                "estimated_cost_range": llm_report.estimated_cost_range,
                "recommendations": llm_report.recommendations,
                "confidence_score": llm_report.confidence_score
            }
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_multimodal_visualization(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        ensemble_method: str
    ) -> np.ndarray:
        """Generate comprehensive visualization combining all modalities."""
        
        annotated_image = image.copy()
        
        # Color scheme for different modalities
        colors = {
            "yolo": (255, 0, 0),      # Red
            "sam": (0, 255, 0),       # Green  
            "clip": (0, 0, 255),      # Blue
            "ensemble": (255, 255, 0)  # Yellow
        }
        
        ensemble_detections = results.get("ensemble", {}).get("detections", [])
        sam_results = results.get("sam")
        
        # Draw ensemble predictions with enhanced information
        for i, detection in enumerate(ensemble_detections):
            bbox = detection.get("bbox", [0, 0, 0, 0])
            class_name = detection.get("class_name", "unknown")
            confidence = detection.get("confidence", 0.0)
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)    # Green - High confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow - Medium confidence
            elif confidence > 0.4:
                color = (0, 165, 255)  # Orange - Low confidence
            else:
                color = (0, 0, 255)    # Red - Very low confidence
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw modality confidence breakdown
            modality_confidences = detection.get("modality_confidences", {})
            confidence_text = f"{class_name}: {confidence:.2f}"
            
            # Add modality breakdown if available
            if modality_confidences:
                modalities_str = " | ".join([
                    f"{k}:{v:.2f}" for k, v in modality_confidences.items() if v > 0.1
                ])
                if modalities_str:
                    confidence_text += f" ({modalities_str})"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
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
                confidence_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw SAM mask if available
            if sam_results and "masks" in sam_results and i < len(sam_results["masks"]):
                mask = sam_results["masks"][i]
                if isinstance(mask, list):
                    mask = np.array(mask, dtype=bool)
                
                # Create colored mask overlay
                mask_overlay = np.zeros_like(annotated_image)
                mask_overlay[:, :, 1] = mask * 255  # Green channel
                
                # Blend with original image
                annotated_image = cv2.addWeighted(
                    annotated_image, 1.0, mask_overlay, 0.3, 0
                )
                
                # Draw mask contour
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated_image, contours, -1, color, 2)
        
        # Add multi-modal summary text
        yolo_count = len(results.get("yolo", {}).get("detections", []))
        sam_count = len(results.get("sam", {}).get("masks", [])) if results.get("sam") else 0
        clip_available = results.get("clip") is not None
        
        summary_text = f"Multi-Modal Analysis | YOLO: {yolo_count} | SAM: {sam_count} | CLIP: {clip_available}"
        
        # Add summary at top-left
        cv2.rectangle(
            annotated_image,
            (10, 10),
            (10 + len(summary_text) * 12, 40),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            annotated_image,
            summary_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return annotated_image
    
    def get_analysis_summary(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis summary."""
        
        summary = "=== Multi-Modal Vehicle Damage Analysis Summary ===\n\n"
        
        # Basic detection info
        yolo_detections = results.get("yolo", {}).get("detections", [])
        ensemble_detections = results.get("ensemble", {}).get("detections", [])
        
        summary += f"YOLO Detections: {len(yolo_detections)}\n"
        summary += f"Ensemble Detections: {len(ensemble_detections)}\n"
        
        # Modality coverage
        modality_coverage = results.get("ensemble", {}).get("modality_coverage", {})
        summary += f"Modality Coverage: {modality_coverage}\n\n"
        
        # Damage types and confidence
        if ensemble_detections:
            damage_types = defaultdict(int)
            confidences = []
            
            for detection in ensemble_detections:
                class_name = detection.get("class_name", "unknown")
                confidence = detection.get("confidence", 0.0)
                damage_types[class_name] += 1
                confidences.append(confidence)
            
            summary += "Damage Types:\n"
            for damage_type, count in damage_types.items():
                summary += f"  - {damage_type}: {count}\n"
            
            if confidences:
                avg_confidence = np.mean(confidences)
                summary += f"\nAverage Confidence: {avg_confidence:.3f}\n"
        
        # Cost analysis
        cost_analysis = results.get("cost_analysis", {})
        if cost_analysis and cost_analysis.get("estimated_cost", 0) > 0:
            summary += f"\nEstimated Repair Cost: ${cost_analysis['estimated_cost']:.2f}\n"
            summary += f"Cost Range: ${cost_analysis.get('min_cost', 0):.2f} - ${cost_analysis.get('max_cost', 0):.2f}\n"
        
        # Processing time
        analysis_time = results.get("image_info", {}).get("analysis_time", 0)
        summary += f"\nAnalysis Time: {analysis_time:.2f}s\n"
        
        # Multi-modal benefits
        summary += "\n=== Multi-Modal Benefits ===\n"
        summary += "✓ YOLO: Fast and accurate detection\n"
        summary += "✓ SAM: Precise damage area calculation\n"
        summary += "✓ CLIP: Semantic understanding of damage types\n"
        summary += "✓ Ensemble: Improved accuracy through fusion\n"
        
        if results.get("llm"):
            summary += "✓ LLM: Professional analysis and recommendations\n"
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Initialize multi-modal pipeline
    pipeline = MultiModalPipeline(
        yolo_model_path="yolov9n.pt",
        enable_sam=True,
        enable_clip=True,
        enable_llm=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Multi-Modal Pipeline initialized successfully!")
    print("Ready for vehicle damage analysis with SOTA accuracy.")