"""
SAM (Segment Anything Model) для точной сегментации повреждений автомобиля.

Интегрирует SAM для преобразования bbox детекций YOLO в точные маски сегментации.
"""

import torch
import numpy as np
import cv2
from typing import Union, List, Dict, Tuple, Optional, Any
from pathlib import Path
import time
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from .yolo_detector import YOLODamageDetector


class SAMSegmentor:
    """
    Класс для работы с SAM (Segment Anything Model) для точной сегментации повреждений.
    
    Поддерживает:
    - Загрузку предобученных моделей SAM (ViT-B, ViT-L, ViT-H)
    - Сегментацию по bounding box координатам от YOLO
    - Постобработку масок и расчет площади повреждений
    - Визуализацию масок на изображениях
    """
    
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize SAM segmentor.
        
        Args:
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            checkpoint_path: Path to SAM checkpoint (если None - скачает автоматически)
            device: Device for inference ('cuda', 'cpu', or None for auto)
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.model = None
        self.predictor = None
        
        print(f"Initializing SAM segmentor with {model_type} model...")
        self.load_model()
        print(f"SAM model loaded successfully on {self.device}")
    
    def load_model(self):
        """
        Load SAM model based on specified type.
        
        Supported models:
        - vit_b: ViT-B/16 (91M parameters, fastest)
        - vit_l: ViT-L/16 (308M parameters, balanced)
        - vit_h: ViT-H/16 (636M parameters, most accurate)
        """
        # Default checkpoints (будут автоматически скачаны при первом использовании)
        checkpoint_urls = {
            'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec6bb.pth',
            'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
        }
        
        if self.checkpoint_path is None:
            # Используем стандартные пути для автоматической загрузки
            checkpoint_map = {
                'vit_b': 'sam_vit_b_01ec6bb.pth',
                'vit_l': 'sam_vit_l_0b3195.pth', 
                'vit_h': 'sam_vit_h_4b8939.pth'
            }
            self.checkpoint_path = checkpoint_map.get(self.model_type, checkpoint_map['vit_b'])
        
        # Load model
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.model = self.model.to(device=self.device)
        
        # Initialize predictor
        self.predictor = SamPredictor(self.model)
        
        print(f"SAM {self.model_type} model loaded on {self.device}")
    
    def segment_from_boxes(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
        class_names: Optional[List[str]] = None,
        confidences: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Segment damage areas using bounding boxes from YOLO detection.
        
        Args:
            image: Input image (numpy array, BGR format)
            boxes: List of bounding boxes [x1, y1, x2, y2]
            class_names: Optional list of class names for each box
            confidences: Optional list of confidence scores for each box
            
        Returns:
            Dictionary with segmentation results:
            - masks: List of binary masks
            - scores: Model confidence scores for masks
            - logits: Raw model outputs
            - boxes: Input boxes
            - class_names: Class names if provided
            - confidences: Confidence scores if provided
            - inference_time: Time taken for segmentation
        """
        if len(boxes) == 0:
            return {
                'masks': [],
                'scores': [],
                'logits': [],
                'boxes': [],
                'class_names': class_names or [],
                'confidences': confidences or [],
                'inference_time': 0.0
            }
        
        start_time = time.time()
        
        # Convert image to RGB for SAM
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Set image for predictor
        self.predictor.set_image(image_rgb)
        
        # Convert boxes to numpy array
        input_boxes = torch.tensor(boxes, dtype=torch.float, device=self.device)
        
        # Segment using boxes
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes,
            multimask_output=False  # Return single best mask per box
        )
        
        # Convert to numpy and lists
        masks_list = masks.cpu().numpy().astype(bool).tolist()
        scores_list = scores.cpu().numpy().tolist()
        logits_list = logits.cpu().numpy().tolist()
        
        inference_time = time.time() - start_time
        
        return {
            'masks': masks_list,
            'scores': scores_list,
            'logits': logits_list,
            'boxes': boxes,
            'class_names': class_names or ['unknown'] * len(boxes),
            'confidences': confidences or [0.0] * len(boxes),
            'inference_time': inference_time
        }
    
    def process_masks(
        self,
        masks: List[np.ndarray],
        min_area_threshold: int = 100,
        area_ratio_threshold: float = 0.001
    ) -> List[np.ndarray]:
        """
        Post-process masks to filter out noise and improve quality.
        
        Args:
            masks: List of binary masks
            min_area_threshold: Minimum area in pixels
            area_ratio_threshold: Minimum area as ratio of image size
            
        Returns:
            Filtered list of masks
        """
        if not masks:
            return []
        
        filtered_masks = []
        total_pixels = masks[0].shape[0] * masks[0].shape[1]
        min_area_ratio_pixels = int(total_pixels * area_ratio_threshold)
        
        for mask in masks:
            # Calculate mask area
            area = np.sum(mask)
            
            # Filter by minimum area
            if area >= min(min_area_threshold, min_area_ratio_pixels):
                # Optional: Apply morphological operations to clean up mask
                kernel = np.ones((3, 3), np.uint8)
                mask_cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
                
                filtered_masks.append(mask_cleaned.astype(bool))
            else:
                filtered_masks.append(mask)  # Keep original if doesn't meet threshold
        
        return filtered_masks
    
    def calculate_damage_area(
        self,
        masks: List[np.ndarray],
        image_shape: Tuple[int, int, int],
        pixel_to_cm_ratio: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate damage area metrics.
        
        Args:
            masks: List of binary masks
            image_shape: Original image shape (H, W, C)
            pixel_to_cm_ratio: Optional conversion ratio from pixels to cm
            
        Returns:
            Dictionary with area calculations:
            - total_pixels: Total damaged pixels
            - total_percentage: Percentage of image area damaged
            - individual_areas: Area for each damage instance
            - estimated_cm2: Estimated area in cm² if ratio provided
        """
        if not masks:
            return {
                'total_pixels': 0,
                'total_percentage': 0.0,
                'individual_areas': [],
                'estimated_cm2': 0.0 if pixel_to_cm_ratio else None
            }
        
        image_area = image_shape[0] * image_shape[1]
        individual_areas = []
        total_pixels = 0
        
        for mask in masks:
            area_pixels = np.sum(mask)
            total_pixels += area_pixels
            individual_areas.append({
                'pixels': area_pixels,
                'percentage': (area_pixels / image_area) * 100
            })
        
        total_percentage = (total_pixels / image_area) * 100
        
        # Calculate estimated area in cm² if ratio provided
        estimated_cm2 = None
        if pixel_to_cm_ratio is not None:
            estimated_cm2 = total_pixels * (pixel_to_cm_ratio ** 2)
        
        return {
            'total_pixels': total_pixels,
            'total_percentage': total_percentage,
            'individual_areas': individual_areas,
            'estimated_cm2': estimated_cm2
        }
    
    def visualize_segmentation(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        alpha: float = 0.6,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize segmentation masks on original image.
        
        Args:
            image: Original image (BGR format)
            results: Segmentation results from segment_from_boxes
            alpha: Transparency level for masks (0-1)
            save_path: Optional path to save visualization
            
        Returns:
            Annotated image with segmentation masks
        """
        annotated_image = image.copy()
        
        masks = results.get('masks', [])
        class_names = results.get('class_names', [])
        scores = results.get('scores', [])
        
        # Color map for different damage types
        color_map = {
            'scratch': (0, 255, 255),      # Yellow
            'dent': (0, 165, 255),         # Orange  
            'crack': (0, 0, 255),          # Red
            'shattered': (255, 0, 0),      # Blue
            'broken': (128, 0, 128),       # Purple
        }
        
        for i, (mask, class_name, score) in enumerate(zip(masks, class_names, scores)):
            # Convert mask to numpy array if needed
            if isinstance(mask, list):
                mask = np.array(mask, dtype=bool)
            
            # Get color for this damage type
            base_color = color_map.get(class_name.lower(), (255, 255, 255))  # White default
            
            # Create colored mask
            colored_mask = np.zeros_like(annotated_image)
            for c in range(3):
                colored_mask[:, :, c] = mask * base_color[c]
            
            # Blend with original image
            annotated_image = np.where(
                mask[:, :, np.newaxis],
                (alpha * colored_mask + (1 - alpha) * annotated_image).astype(np.uint8),
                annotated_image
            )
            
            # Add border around mask
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated_image, contours, -1, base_color, 2)
            
            # Add label
            if len(contours) > 0:
                # Find center of mass for label placement
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    label = f"{class_name}: {score:.2f}"
                    cv2.putText(
                        annotated_image,
                        label,
                        (cX - 20, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, annotated_image)
            print(f"Segmentation visualization saved to {save_path}")
        
        return annotated_image
    
    def get_rle_encoding(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Convert binary mask to RLE (Run Length Encoding) format.
        
        Args:
            mask: Binary mask
            
        Returns:
            RLE encoded mask
        """
        from pycocotools import mask as mask_util
        
        if isinstance(mask, list):
            mask = np.array(mask, dtype=np.uint8)
        
        rle = mask_util.encode(np.asfortranarray(mask))
        return {
            'counts': rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts'],
            'size': rle['size']
        }


# Example usage
if __name__ == "__main__":
    # Initialize SAM segmentor
    sam_segmentor = SAMSegmentor(
        model_type="vit_b",  # Use ViT-B for speed
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Example with dummy data
    print("SAM Segmentor initialized successfully!")
    print(f"Model type: {sam_segmentor.model_type}")
    print(f"Device: {sam_segmentor.device}")