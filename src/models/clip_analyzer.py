"""
CLIP Analyzer для семантического анализа повреждений автомобилей.

Использует CLIP модель для извлечения семантических признаков и сравнения с базой знаний повреждений.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import cv2
import logging
from pathlib import Path
from collections import Counter

# Импорт CLIP модели
CLIP_AVAILABLE = False
USE_OPEN_CLIP = False

try:
    import open_clip
    CLIP_AVAILABLE = True
    USE_OPEN_CLIP = True
except ImportError:
    try:
        from transformers import CLIPProcessor, CLIPModel
        CLIP_AVAILABLE = True
        USE_OPEN_CLIP = False
    except ImportError:
        CLIP_AVAILABLE = False
        USE_OPEN_CLIP = False

from .damage_descriptions import (
    DAMAGE_DESCRIPTIONS, 
    ALL_DESCRIPTIONS, 
    DESCRIPTION_TO_DAMAGE_CLASS,
    get_yolo_class_mapping,
    get_class_descriptions
)

logger = logging.getLogger(__name__)


class CLIPAnalyzer:
    """
    Класс для семантического анализа повреждений с помощью CLIP модели.
    
    Функции:
    - Загрузка CLIP модели (OpenCLIP или Transformers)
    - Извлечение регионов повреждений из изображения
    - Создание текстовых и визуальных эмбеддингов
    - Семантическая классификация повреждений
    - Улучшенная уверенность классификации через ensemble
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        pretrained: str = "openai",
        device: Optional[str] = None
    ):
        """
        Initialize CLIP analyzer.
        
        Args:
            model_name: CLIP model architecture ('ViT-B/32', 'ViT-L/14', etc.)
            pretrained: Pretrained weights ('openai', 'laion2b_s34b_b79d')
            device: Device for inference ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.model = None
        self.processor = None
        self.text_features = None
        
        if CLIP_AVAILABLE:
            self.load_model()
            self.precompute_text_features()
            logger.info(f"CLIP analyzer initialized with {model_name} on {self.device}")
        else:
            logger.error("CLIP model not available. Please install open-clip-torch or transformers")
            raise ImportError("CLIP model not available")
    
    def load_model(self):
        """Load CLIP model based on available library."""
        if USE_OPEN_CLIP:
            # Используем OpenCLIP
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            logger.info(f"Loaded OpenCLIP model: {self.model_name}")
        else:
            # Используем Transformers
            self.model = CLIPModel.from_pretrained(f"openai/{self.model_name.lower().replace('/', '_')}")
            self.processor = CLIPProcessor.from_pretrained(f"openai/{self.model_name.lower().replace('/', '_')}")
            self.model = self.model.to(self.device)
            logger.info(f"Loaded Transformers CLIP model: {self.model_name}")
    
    def precompute_text_features(self):
        """Pre-compute text features for all damage descriptions."""
        logger.info("Pre-computing text features for damage descriptions...")
        
        # Tokenize all descriptions
        if USE_OPEN_CLIP:
            text_tokens = open_clip.tokenize(ALL_DESCRIPTIONS)
        else:
            text_tokens = self.processor(
                text=ALL_DESCRIPTIONS,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        
        # Encode text to features
        with torch.no_grad():
            if USE_OPEN_CLIP:
                self.text_features = self.model.encode_text(text_tokens.to(self.device))
            else:
                self.text_features = self.model.get_text_features(**text_tokens.to(self.device))
            
            # Normalize features
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        
        logger.info(f"Pre-computed text features for {len(ALL_DESCRIPTIONS)} descriptions")
    
    def extract_damage_regions(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        padding: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extract damage regions from image based on YOLO detections.
        
        Args:
            image: Input image (numpy array, BGR format)
            detections: List of YOLO detections with bbox
            padding: Padding around bbox for better context
            
        Returns:
            List of extracted regions with metadata
        """
        regions = []
        height, width = image.shape[:2]
        
        for i, detection in enumerate(detections):
            # Get bbox coordinates
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
            
            # Add padding
            x1 = max(0, int(x1 - padding))
            y1 = max(0, int(y1 - padding))
            x2 = min(width, int(x2 + padding))
            y2 = min(height, int(y2 + padding))
            
            # Extract region
            region = image[y1:y2, x1:x2]
            
            # Skip if region is too small
            if region.size == 0:
                continue
            
            regions.append({
                'region': region,
                'bbox': [x1, y1, x2, y2],
                'original_detection': detection,
                'region_id': i,
                'class_name': detection.get('class_name', 'unknown'),
                'confidence': detection.get('confidence', 0.0)
            })
        
        logger.info(f"Extracted {len(regions)} damage regions")
        return regions
    
    def get_image_features(self, image: np.ndarray) -> torch.Tensor:
        """
        Get CLIP features for an image.
        
        Args:
            image: Input image (numpy array, BGR format)
            
        Returns:
            CLIP image features (normalized)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL
        image_pil = Image.fromarray(image_rgb)
        
        # Process image
        if USE_OPEN_CLIP:
            image_tensor = self.processor(image_pil).unsqueeze(0).to(self.device)
        else:
            image_tensor = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        
        # Get features
        with torch.no_grad():
            if USE_OPEN_CLIP:
                image_features = self.model.encode_image(image_tensor)
            else:
                image_features = self.model.get_image_features(**image_tensor)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def classify_region_semantic(
        self,
        region: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Classify a single damage region using CLIP semantic analysis.
        
        Args:
            region: Extracted damage region (numpy array)
            top_k: Number of top predictions to return
            
        Returns:
            List of top classifications with scores
        """
        # Get image features
        image_features = self.get_image_features(region)
        
        # Calculate similarities
        similarities = (image_features @ self.text_features.T).squeeze(0)
        
        # Get top-k results
        top_indices = similarities.topk(top_k).indices
        top_scores = similarities.topk(top_k).values
        
        classifications = []
        for idx, score in zip(top_indices, top_scores):
            description = ALL_DESCRIPTIONS[idx]
            damage_class = DESCRIPTION_TO_DAMAGE_CLASS[description]
            
            classifications.append({
                'description': description,
                'damage_class': damage_class,
                'similarity': float(score),
                'confidence': float(score)  # CLIP similarity as confidence
            })
        
        return classifications
    
    def semantic_analysis(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        ensemble_method: str = "weighted"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis of damage detections.
        
        Args:
            image: Input image (numpy array, BGR format)
            detections: List of YOLO detections
            ensemble_method: Method for combining predictions ('weighted', 'voting', 'max')
            
        Returns:
            Comprehensive semantic analysis results
        """
        start_time = torch.cuda.Event(enable=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Extract damage regions
        regions = self.extract_damage_regions(image, detections)
        
        if not regions:
            return {
                'semantic_analysis': [],
                'ensemble_predictions': {},
                'confidence_boost': {},
                'analysis_time': 0.0
            }
        
        # Analyze each region
        semantic_results = []
        all_predictions = []
        
        for region_data in regions:
            region = region_data['region']
            original_class = region_data['class_name']
            original_confidence = region_data['confidence']
            
            # Get CLIP classifications
            clip_classifications = self.classify_region_semantic(region, top_k=3)
            
            # Ensemble with YOLO prediction
            ensemble_result = self._ensemble_predictions(
                clip_classifications,
                original_class,
                original_confidence,
                method=ensemble_method
            )
            
            semantic_results.append({
                'region_id': region_data['region_id'],
                'bbox': region_data['bbox'],
                'original_detection': region_data['original_detection'],
                'clip_analysis': clip_classifications,
                'ensemble_result': ensemble_result
            })
            
            all_predictions.extend(ensemble_result['predictions'])
        
        # Aggregate results
        ensemble_predictions = self._aggregate_predictions(all_predictions)
        confidence_boost = self._calculate_confidence_boost(detections, semantic_results)
        
        if start_time:
            end_time.record()
            torch.cuda.synchronize()
            analysis_time = start_time.elapsed_time(1000) / 1000  # Convert to seconds
        else:
            analysis_time = 0.0
        
        return {
            'semantic_analysis': semantic_results,
            'ensemble_predictions': ensemble_predictions,
            'confidence_boost': confidence_boost,
            'analysis_time': analysis_time
        }
    
    def _ensemble_predictions(
        self,
        clip_classifications: List[Dict[str, Any]],
        yolo_class: str,
        yolo_confidence: float,
        method: str = "weighted"
    ) -> Dict[str, Any]:
        """
        Ensemble CLIP and YOLO predictions.
        
        Args:
            clip_classifications: CLIP classification results
            yolo_class: YOLO predicted class
            yolo_confidence: YOLO confidence score
            method: Ensemble method
            
        Returns:
            Ensembled predictions
        """
        predictions = {}
        
        # Add CLIP predictions
        for cls in clip_classifications:
            class_name = cls['damage_class']
            confidence = cls['confidence']
            predictions[class_name] = predictions.get(class_name, 0) + confidence
        
        # Add YOLO prediction with mapped class
        yolo_mapped_class = get_yolo_class_mapping(yolo_class)
        predictions[yolo_mapped_class] = predictions.get(yolo_mapped_class, 0) + yolo_confidence
        
        # Normalize predictions
        total_weight = sum(predictions.values())
        if total_weight > 0:
            predictions = {k: v / total_weight for k, v in predictions.items()}
        
        # Sort by confidence
        sorted_predictions = sorted(
            predictions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            'predictions': dict(sorted_predictions),
            'top_class': sorted_predictions[0][0] if sorted_predictions else 'unknown',
            'top_confidence': sorted_predictions[0][1] if sorted_predictions else 0.0,
            'method': method
        }
    
    def _aggregate_predictions(self, all_predictions: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate predictions from all regions."""
        class_scores = Counter()
        total_confidence = 0.0
        
        for pred_dict in all_predictions:
            for class_name, confidence in pred_dict.items():
                class_scores[class_name] += confidence
                total_confidence += confidence
        
        # Normalize
        if total_confidence > 0:
            aggregated = {
                class_name: score / total_confidence 
                for class_name, score in class_scores.items()
            }
        else:
            aggregated = {}
        
        return aggregated
    
    def _calculate_confidence_boost(
        self,
        yolo_detections: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate confidence boost from semantic analysis."""
        confidence_boost = {}
        
        for i, (yolo_det, semantic_result) in enumerate(zip(yolo_detections, semantic_results)):
            original_confidence = yolo_det.get('confidence', 0.0)
            ensemble_confidence = semantic_result['ensemble_result'].get('top_confidence', 0.0)
            
            # Calculate boost factor
            if original_confidence > 0:
                boost_factor = ensemble_confidence / original_confidence
            else:
                boost_factor = 1.0
            
            confidence_boost[f"region_{i}"] = boost_factor
        
        return confidence_boost
    
    def get_semantic_similarity(
        self,
        image: np.ndarray,
        descriptions: List[str]
    ) -> Dict[str, float]:
        """
        Calculate semantic similarity between image and text descriptions.
        
        Args:
            image: Input image
            descriptions: List of text descriptions
            
        Returns:
            Similarity scores for each description
        """
        image_features = self.get_image_features(image)
        
        # Tokenize descriptions
        if USE_OPEN_CLIP:
            text_tokens = open_clip.tokenize(descriptions)
        else:
            text_tokens = self.processor(
                text=descriptions,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        
        # Get text features
        with torch.no_grad():
            if USE_OPEN_CLIP:
                text_features = self.model.encode_text(text_tokens.to(self.device))
            else:
                text_features = self.model.get_text_features(**text_tokens.to(self.device))
            
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = (image_features @ text_features.T).squeeze(0)
        
        return {
            desc: float(sim) 
            for desc, sim in zip(descriptions, similarities)
        }


# Example usage
if __name__ == "__main__":
    try:
        # Initialize CLIP analyzer
        clip_analyzer = CLIPAnalyzer(
            model_name="ViT-B/32",
            pretrained="openai",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print("CLIP Analyzer initialized successfully!")
        print(f"Model: {clip_analyzer.model_name}")
        print(f"Device: {clip_analyzer.device}")
        print(f"Available descriptions: {len(ALL_DESCRIPTIONS)}")
        
    except ImportError as e:
        print(f"Error initializing CLIP analyzer: {e}")
        print("Please install required dependencies: pip install open-clip-torch")