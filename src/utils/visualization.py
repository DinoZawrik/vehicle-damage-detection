"""
Visualization utilities for damage detection results.

Simple functions to draw bounding boxes and display results.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


# Color palette for different damage types
DAMAGE_COLORS = {
    'scratch': (0, 255, 255),      # Yellow
    'dent': (0, 165, 255),         # Orange
    'crack': (0, 0, 255),          # Red
    'shatter': (255, 0, 0),        # Blue
    'broken_glass': (255, 0, 255), # Magenta
    'paint_damage': (0, 255, 0),   # Green
    'rust': (42, 42, 165),         # Brown
    'unknown': (128, 128, 128)     # Gray
}

# Severity colors
SEVERITY_COLORS = {
    'minor': (0, 255, 0),      # Green
    'moderate': (0, 165, 255),  # Orange
    'severe': (0, 0, 255),      # Red
    'critical': (128, 0, 128)   # Purple
}


def draw_detection_box(
    image: np.ndarray,
    bbox: List[float],
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a single detection box on image.
    
    Args:
        image: Image array (BGR)
        bbox: [x1, y1, x2, y2]
        label: Detection label
        confidence: Confidence score
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Image with drawn box
    """
    # Make a copy
    img = image.copy()
    
    # Extract coordinates
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label text
    label_text = f"{label}: {confidence:.2f}"
    
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, font, font_scale, font_thickness
    )
    
    # Draw label background
    cv2.rectangle(
        img,
        (x1, y1 - text_h - baseline - 10),
        (x1 + text_w + 10, y1),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        img,
        label_text,
        (x1 + 5, y1 - 5),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness
    )
    
    return img


def visualize_detections(
    image: np.ndarray,
    detections: List[Dict],
    show_confidence: bool = True,
    thickness: int = 2
) -> np.ndarray:
    """
    Visualize all detections on image.
    
    Args:
        image: Input image (BGR)
        detections: List of detection dictionaries
        show_confidence: Show confidence scores
        thickness: Box line thickness
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) != 4:
            continue
        
        # Get class name and color
        class_name = det.get('class_name', det.get('class', 'unknown'))
        color = DAMAGE_COLORS.get(class_name.lower(), DAMAGE_COLORS['unknown'])
        
        # Get confidence
        confidence = det.get('confidence', 0.0)
        
        # Get severity if available
        severity = det.get('severity')
        if severity:
            color = SEVERITY_COLORS.get(severity, color)
        
        # Draw box
        label = class_name
        if show_confidence:
            label = f"{class_name}: {confidence:.2f}"
        
        annotated = draw_detection_box(
            annotated,
            bbox,
            label,
            confidence,
            color,
            thickness
        )
    
    return annotated


def add_info_overlay(
    image: np.ndarray,
    info: Dict,
    position: str = 'top'
) -> np.ndarray:
    """
    Add information overlay to image.
    
    Args:
        image: Input image
        info: Information dictionary
        position: 'top' or 'bottom'
        
    Returns:
        Image with overlay
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # Prepare text lines
    lines = []
    
    if 'damage_count' in info:
        lines.append(f"Damages detected: {info['damage_count']}")
    
    if 'severity' in info and info['severity']:
        lines.append(f"Severity: {info['severity'].upper()}")
    
    if 'cost_estimate' in info:
        cost = info['cost_estimate']
        if isinstance(cost, dict):
            min_cost = cost.get('min', 0)
            max_cost = cost.get('max', 0)
            lines.append(f"Estimated cost: ${min_cost:.0f} - ${max_cost:.0f}")
    
    if 'processing_time' in info:
        lines.append(f"Processing time: {info['processing_time']:.2f}s")
    
    # Calculate overlay size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    line_height = 30
    padding = 20
    
    overlay_height = len(lines) * line_height + 2 * padding
    
    # Create semi-transparent overlay
    overlay = img.copy()
    
    if position == 'top':
        y_start = 0
        y_text_start = padding + 20
    else:
        y_start = h - overlay_height
        y_text_start = y_start + padding + 20
    
    cv2.rectangle(
        overlay,
        (0, y_start),
        (w, y_start + overlay_height),
        (0, 0, 0),
        -1
    )
    
    # Blend
    alpha = 0.6
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Draw text
    for i, line in enumerate(lines):
        y_pos = y_text_start + i * line_height
        cv2.putText(
            img,
            line,
            (padding, y_pos),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return img


def create_side_by_side(
    original: np.ndarray,
    annotated: np.ndarray,
    gap: int = 20
) -> np.ndarray:
    """
    Create side-by-side comparison.
    
    Args:
        original: Original image
        annotated: Annotated image
        gap: Gap between images in pixels
        
    Returns:
        Combined image
    """
    h1, w1 = original.shape[:2]
    h2, w2 = annotated.shape[:2]
    
    # Match heights
    max_h = max(h1, h2)
    
    if h1 < max_h:
        original = cv2.resize(original, (w1, max_h))
    if h2 < max_h:
        annotated = cv2.resize(annotated, (w2, max_h))
    
    # Create combined image
    combined = np.ones((max_h, w1 + w2 + gap, 3), dtype=np.uint8) * 255
    
    # Place images
    combined[:, :w1] = original
    combined[:, w1+gap:] = annotated
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (20, 40), font, 1, (0, 0, 0), 2)
    cv2.putText(combined, "Detected", (w1 + gap + 20, 40), font, 1, (0, 0, 0), 2)
    
    return combined


def save_visualization(
    image: np.ndarray,
    output_path: str,
    quality: int = 95
) -> bool:
    """
    Save visualization to file.
    
    Args:
        image: Image to save
        output_path: Output file path
        quality: JPEG quality (1-100)
        
    Returns:
        Success status
    """
    try:
        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def display_image(image: np.ndarray, window_name: str = "Image", wait: bool = True):
    """
    Display image in window.
    
    Args:
        image: Image to display
        window_name: Window title
        wait: Wait for key press
    """
    cv2.imshow(window_name, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
