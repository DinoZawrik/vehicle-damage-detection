"""
Simple Detection Pipeline - MVP Version

Simplified YOLO-only pipeline for vehicle damage detection.
No SAM, CLIP, or LLM - just straightforward YOLO detection.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time


class DetectionResult:
    """Simple detection result structure."""
    
    def __init__(self):
        self.detections: List[Dict] = []
        self.image_shape: Tuple[int, int] = (0, 0)
        self.processing_time: float = 0.0
        self.model_name: str = "yolov8n"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        return {
            'detections': self.detections,
            'image_shape': {
                'height': self.image_shape[0],
                'width': self.image_shape[1]
            },
            'processing_time': round(self.processing_time, 3),
            'model': self.model_name
        }


class SimpleDetectionPipeline:
    """
    MVP: YOLO-only detection pipeline.
    
    Simple, fast, and effective damage detection without complexity.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize simple YOLO pipeline.
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold (0.0-1.0)
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model: Optional[YOLO] = None
        self.is_loaded = False
        
        print(f"🚗 Initializing SimpleDetectionPipeline")
        print(f"   Model: {model_path}")
        print(f"   Confidence: {conf_threshold}")
        print(f"   Device: {device}")
    
    def load_model(self):
        """Load YOLO model (lazy loading)."""
        if self.is_loaded:
            return
        
        print(f"📦 Loading YOLO model from {self.model_path}...")
        start_time = time.time()
        
        try:
            # Check if model file exists
            model_file = Path(self.model_path)
            if not model_file.exists() and not model_file.is_absolute():
                # Try in models directory
                model_file = Path("models") / self.model_path
                if not model_file.exists():
                    print(f"⚠️  Model file not found, downloading {self.model_path}...")
            
            self.model = YOLO(str(model_file) if model_file.exists() else self.model_path)
            self.model.to(self.device)
            
            load_time = time.time() - start_time
            self.is_loaded = True
            print(f"✅ Model loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def detect(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        return_annotated: bool = False
    ) -> DetectionResult:
        """
        Detect damage in image.
        
        Args:
            image: Input image as numpy array (BGR format)
            conf: Override confidence threshold (optional)
            return_annotated: Return annotated image with bboxes
            
        Returns:
            DetectionResult with detected damages
        """
        # Ensure model is loaded
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        # Prepare result
        result = DetectionResult()
        result.image_shape = (image.shape[0], image.shape[1])
        
        # Run detection
        confidence = conf if conf is not None else self.conf_threshold
        
        results = self.model(
            image,
            conf=confidence,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Parse results
        result.detections = self._parse_results(results)
        result.processing_time = time.time() - start_time
        
        if return_annotated and len(results) > 0:
            result.annotated_image = results[0].plot()
        
        return result
    
    def _parse_results(self, results) -> List[Dict]:
        """
        Parse YOLO results into simple format.
        
        Args:
            results: YOLO detection results
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if not results or len(results) == 0:
            return detections
        
        result = results[0]
        
        # Check if any detections
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Extract detections
        boxes = result.boxes
        
        for i in range(len(boxes)):
            box = boxes[i]
            
            # Get coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)
            
            # Get confidence and class
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            # Get class name
            class_name = result.names[class_id] if result.names else f"class_{class_id}"
            
            # Calculate area
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': round(confidence, 3),
                'class_id': class_id,
                'class_name': class_name,
                'area': round(area, 2),
                'center': [
                    round((x1 + x2) / 2, 2),
                    round((y1 + y2) / 2, 2)
                ]
            }
            
            detections.append(detection)
        
        return detections
    
    def detect_from_path(self, image_path: str) -> DetectionResult:
        """
        Detect damage from image file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            DetectionResult with detected damages
        """
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        return self.detect(image)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_path': self.model_path,
            'confidence_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'model_type': 'YOLOv8n'
        }


# Convenience function for quick detection
def detect_damage(
    image_path: str,
    model_path: str = "yolov8n.pt",
    conf: float = 0.35
) -> DetectionResult:
    """
    Quick damage detection from image path.
    
    Args:
        image_path: Path to image
        model_path: Path to YOLO model
        conf: Confidence threshold
        
    Returns:
        DetectionResult
    """
    pipeline = SimpleDetectionPipeline(model_path=model_path, conf_threshold=conf)
    return pipeline.detect_from_path(image_path)
