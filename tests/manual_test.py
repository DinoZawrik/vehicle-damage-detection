"""
Manual test script to verify pipeline without pytest.
"""
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simple_pipeline import SimpleDetectionPipeline
from src.models.damage_analyzer import DamageAnalyzer

def test_pipeline():
    print("Testing pipeline...")
    
    # Check if model exists
    model_path = "models/yolov8n.pt"
    if not os.path.exists(model_path):
        print(f"Skipping pipeline test: {model_path} not found")
        return

    try:
        pipeline = SimpleDetectionPipeline(model_path=model_path)
        print("Pipeline initialized.")
        
        # Create dummy image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Detect
        print("Running detection...")
        result = pipeline.detect(img)
        print(f"Detection successful. Found {len(result.detections)} objects.")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        raise

def test_analyzer():
    print("\nTesting analyzer...")
    try:
        analyzer = DamageAnalyzer()
        print("Analyzer initialized.")
        
        # Mock detections
        detections = [
            {
                'class_name': 'scratch',
                'confidence': 0.9,
                'bbox': [100, 100, 200, 200]
            },
            {
                'class_name': 'dent',
                'confidence': 0.8,
                'bbox': [300, 300, 400, 400]
            }
        ]
        
        # Analyze
        print("Running analysis...")
        result = analyzer.analyze(detections, image_shape=(1000, 1000))
        
        print("Analysis result:")
        print(f"  Severity: {result['severity']}")
        print(f"  Damage count: {result['damage_count']}")
        print(f"  Cost: {result['cost_estimate']['min']}-{result['cost_estimate']['max']} {result['cost_estimate']['currency']}")
        
    except Exception as e:
        print(f"Analyzer test failed: {e}")
        raise

if __name__ == "__main__":
    test_pipeline()
    test_analyzer()
    print("\n✅ Manual tests passed!")
