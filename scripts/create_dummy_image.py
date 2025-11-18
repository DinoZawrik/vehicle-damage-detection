"""
Create a dummy image for testing the pipeline.
"""

import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEST_SAMPLES_DIR = PROJECT_ROOT / "data" / "test_samples"

def create_dummy_image():
    """Create a simple image with shapes."""
    # Create black image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Draw a "car" (rectangle)
    cv2.rectangle(img, (200, 200), (600, 400), (100, 100, 100), -1)
    
    # Draw a "window" (rectangle)
    cv2.rectangle(img, (250, 220), (550, 280), (200, 200, 250), -1)
    
    # Draw a "wheel" (circle)
    cv2.circle(img, (280, 400), 40, (50, 50, 50), -1)
    cv2.circle(img, (520, 400), 40, (50, 50, 50), -1)
    
    # Draw a "scratch" (yellow line)
    cv2.line(img, (300, 350), (400, 360), (0, 255, 255), 2)
    
    # Draw a "dent" (gray circle)
    cv2.circle(img, (500, 320), 20, (80, 80, 80), -1)
    
    # Save
    output_path = TEST_SAMPLES_DIR / "dummy_car.jpg"
    cv2.imwrite(str(output_path), img)
    print(f"Created dummy image: {output_path}")
    return output_path

if __name__ == "__main__":
    create_dummy_image()
