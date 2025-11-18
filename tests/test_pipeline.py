"""
Test the MVP pipeline components.
"""

import pytest
import numpy as np
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simple_pipeline import SimpleDetectionPipeline
from src.models.damage_analyzer import DamageAnalyzer

# Mock YOLO model to avoid downloading/loading heavy model during tests
class MockYOLO:
    def __call__(self, image, conf=0.25):
        return [MockResult()]

class MockResult:
    def __init__(self):
        self.boxes = MockBoxes()
        self.names = {0: 'scratch', 1: 'dent'}

class MockBoxes:
    def __init__(self):
        # [x1, y1, x2, y2, conf, cls]
        self.data = np.array([
            [100, 100, 200, 200, 0.9, 0],  # scratch
            [300, 300, 400, 400, 0.8, 1]   # dent
        ])
        self.xyxy = self.data[:, :4]
        self.conf = self.data[:, 4]
        self.cls = self.data[:, 5]

    def cpu(self):
        return self

    def numpy(self):
        return self

@pytest.fixture
def pipeline():
    # We can't easily mock the internal YOLO load without patching, 
    # so we'll just test the analyzer or use a real model if available.
    # For unit tests, we should mock.
    pass

def test_damage_analyzer():
    """Test DamageAnalyzer logic."""
    analyzer = DamageAnalyzer()
    
    # Mock detections
    detections = [
        type('obj', (object,), {
            'class_name': 'scratch',
            'confidence': 0.9,
            'bbox': [100, 100, 200, 200]
        }),
        type('obj', (object,), {
            'class_name': 'dent',
            'confidence': 0.8,
            'bbox': [300, 300, 400, 400]
        })
    ]
    
    # Analyze
    result = analyzer.analyze(detections, image_shape=(1000, 1000))
    
    assert result['damage_count'] == 2
    assert result['damage_types']['scratch'] == 1
    assert result['damage_types']['dent'] == 1
    assert result['severity'] in ['minor', 'moderate', 'severe', 'critical']
    assert result['cost_estimate']['min'] > 0

def test_pipeline_integration():
    """Integration test with real model (if available)."""
    model_path = "models/yolov8n.pt"
    if not os.path.exists(model_path):
        pytest.skip("Model not found")
        
    pipeline = SimpleDetectionPipeline(model_path=model_path)
    
    # Create dummy image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Detect
    result = pipeline.detect(img)
    
    assert result is not None
    assert isinstance(result.detections, list)
