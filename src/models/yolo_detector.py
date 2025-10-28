"""
YOLOv11/v8 Damage Detection Module

This module handles the core damage detection using YOLO models.
Supports both YOLOv8 and YOLOv11 for comparison.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch


class YOLODamageDetector:
    """
    Wrapper class for YOLO-based vehicle damage detection.

    Attributes:
        model: Loaded YOLO model
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        device: Device for inference (cuda/cpu)
    """

    def __init__(
        self,
        model_path: str = "yolov11m.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to trained YOLO weights or model name
            conf_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for Non-Maximum Suppression
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"Model loaded successfully on {self.device}")

        # Class names (will be loaded from model)
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}

    def predict(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        augment: bool = False
    ) -> Dict:
        """
        Run inference on a single image.

        Args:
            image: Input image (numpy array, BGR format)
            conf: Override confidence threshold
            iou: Override IOU threshold
            augment: Use test-time augmentation

        Returns:
            Dictionary containing:
                - boxes: List of bounding boxes [x1, y1, x2, y2]
                - scores: Confidence scores
                - classes: Class IDs
                - class_names: Class names
                - image_shape: Original image shape
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold

        # Run inference
        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            augment=augment,
            device=self.device,
            verbose=False
        )

        # Parse results
        detections = self._parse_results(results[0], image.shape)

        return detections

    def predict_batch(
        self,
        images: List[np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[Dict]:
        """
        Run inference on multiple images.

        Args:
            images: List of input images
            conf: Confidence threshold
            iou: IOU threshold

        Returns:
            List of detection dictionaries (one per image)
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold

        # Batch inference
        results = self.model.predict(
            images,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )

        # Parse all results
        all_detections = []
        for result, image in zip(results, images):
            detections = self._parse_results(result, image.shape)
            all_detections.append(detections)

        return all_detections

    def _parse_results(self, result, image_shape: Tuple[int, int, int]) -> Dict:
        """
        Parse YOLO results into structured format.

        Args:
            result: YOLO result object
            image_shape: Shape of input image (H, W, C)

        Returns:
            Parsed detection dictionary
        """
        boxes = []
        scores = []
        class_ids = []
        class_names = []

        if result.boxes is not None and len(result.boxes) > 0:
            # Extract bounding boxes
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            boxes = boxes_xyxy.tolist()

            # Extract confidence scores
            scores = result.boxes.conf.cpu().numpy().tolist()

            # Extract class IDs
            class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()

            # Get class names
            class_names = [self.class_names.get(cls_id, f"class_{cls_id}") for cls_id in class_ids]

        return {
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'class_names': class_names,
            'num_detections': len(boxes),
            'image_shape': image_shape
        }

    def visualize(
        self,
        image: np.ndarray,
        detections: Dict,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> np.ndarray:
        """
        Visualize detections on image.

        Args:
            image: Original image
            detections: Detection dictionary from predict()
            save_path: Path to save annotated image
            show: Whether to display image

        Returns:
            Annotated image
        """
        annotated = image.copy()

        # Draw each detection
        for box, score, class_name in zip(
            detections['boxes'],
            detections['scores'],
            detections['class_names']
        ):
            x1, y1, x2, y2 = map(int, box)

            # Choose color based on class (you can customize)
            color = self._get_class_color(class_name)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {score:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Label background
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # Save if requested
        if save_path:
            cv2.imwrite(save_path, annotated)
            print(f"Saved annotated image to {save_path}")

        # Show if requested
        if show:
            cv2.imshow("Detections", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated

    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Get consistent color for each class.

        Args:
            class_name: Name of the class

        Returns:
            BGR color tuple
        """
        # Predefined colors for common damage types
        color_map = {
            'scratch': (0, 255, 255),      # Yellow
            'dent': (0, 165, 255),         # Orange
            'crack': (0, 0, 255),          # Red
            'shattered': (255, 0, 0),      # Blue
            'broken': (128, 0, 128),       # Purple
        }

        # Return predefined color or generate from hash
        if class_name.lower() in color_map:
            return color_map[class_name.lower()]
        else:
            # Generate color from hash for consistency
            import hashlib
            hash_val = int(hashlib.md5(class_name.encode()).hexdigest(), 16)
            return (
                hash_val % 256,
                (hash_val // 256) % 256,
                (hash_val // 65536) % 256
            )

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        project: str = "runs/detect",
        name: str = "train",
        **kwargs
    ):
        """
        Train the YOLO model.

        Args:
            data_yaml: Path to data configuration YAML
            epochs: Number of training epochs
            batch_size: Batch size
            image_size: Input image size
            project: Project directory for saving results
            name: Experiment name
            **kwargs: Additional training arguments for YOLO
        """
        print(f"Starting training for {epochs} epochs...")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            project=project,
            name=name,
            device=self.device,
            **kwargs
        )

        print(f"Training completed! Results saved to {project}/{name}")
        return results

    def evaluate(self, data_yaml: str, **kwargs):
        """
        Evaluate model on validation set.

        Args:
            data_yaml: Path to data configuration YAML
            **kwargs: Additional evaluation arguments

        Returns:
            Evaluation metrics
        """
        print("Running evaluation...")
        metrics = self.model.val(data=data_yaml, device=self.device, **kwargs)
        return metrics

    def export(self, format: str = "onnx", **kwargs):
        """
        Export model to different formats.

        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            **kwargs: Additional export arguments

        Returns:
            Path to exported model
        """
        print(f"Exporting model to {format}...")
        path = self.model.export(format=format, **kwargs)
        print(f"Model exported to {path}")
        return path


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = YOLODamageDetector(
        model_path="yolov11m.pt",  # or path to your trained model
        conf_threshold=0.25
    )

    # Load test image
    test_image = cv2.imread("path/to/test/image.jpg")

    if test_image is not None:
        # Run detection
        detections = detector.predict(test_image)

        print(f"Detected {detections['num_detections']} damages:")
        for class_name, score in zip(detections['class_names'], detections['scores']):
            print(f"  - {class_name}: {score:.3f}")

        # Visualize
        annotated = detector.visualize(
            test_image,
            detections,
            save_path="output/annotated.jpg",
            show=True
        )
    else:
        print("Please provide a valid test image path")
