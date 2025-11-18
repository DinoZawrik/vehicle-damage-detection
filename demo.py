"""
Vehicle Damage Detection - CLI Demo

Simple command-line demo for testing damage detection.
"""

import argparse
import sys
import os
from pathlib import Path
import time
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.simple_pipeline import SimpleDetectionPipeline
from src.models.damage_analyzer import DamageAnalyzer
from src.utils.visualization import (
    visualize_detections,
    add_info_overlay,
    create_side_by_side,
    save_visualization
)


def print_banner():
    """Print CLI banner."""
    print("\n" + "=" * 60)
    print("🚗 Vehicle Damage Detection - CLI Demo")
    print("=" * 60 + "\n")


def print_results(analysis: dict, processing_time: float):
    """Print detection results to console."""
    print("\n" + "-" * 60)
    print("📊 Detection Results")
    print("-" * 60)
    
    # Detection count
    damage_count = analysis.get('damage_count', 0)
    print(f"Damages detected: {damage_count}")
    
    if damage_count == 0:
        print("No damages detected in the image.")
        return
    
    # Severity
    severity = analysis.get('severity')
    if severity:
        severity_emoji = {
            'minor': '🟢',
            'moderate': '🟡',
            'severe': '🟠',
            'critical': '🔴'
        }
        emoji = severity_emoji.get(severity, '⚪')
        print(f"Severity: {emoji} {severity.upper()}")
    
    # Damage types
    damage_types = analysis.get('damage_types', {})
    if damage_types:
        print("\nDamage types:")
        for damage_type, count in damage_types.items():
            print(f"  - {damage_type}: {count}")
    
    # Cost estimate
    cost_estimate = analysis.get('cost_estimate', {})
    if cost_estimate:
        min_cost = cost_estimate.get('min', 0)
        max_cost = cost_estimate.get('max', 0)
        total = cost_estimate.get('total', 0)
        print(f"\n💰 Cost estimate: ${min_cost:.0f} - ${max_cost:.0f} USD")
        print(f"   (Best estimate: ${total:.0f})")
    
    # Processing time
    print(f"\n⏱️  Processing time: {processing_time:.2f}s")
    print("-" * 60 + "\n")


def run_detection(
    image_path: str,
    output_path: str = None,
    model_path: str = "yolov8n.pt",
    confidence: float = 0.35,
    show_image: bool = False,
    save_comparison: bool = True,
    simulate: bool = False
):
    """
    Run damage detection on image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save result (optional)
        model_path: Path to YOLO model
        confidence: Confidence threshold
        show_image: Display image in window
        save_comparison: Save side-by-side comparison
        simulate: Simulate detections for demo
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return False
    
    print(f"📸 Loading image: {image_path}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Failed to read image")
        return False
    
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize pipeline
    print(f"\n🤖 Initializing detection pipeline...")
    print(f"   Model: {model_path}")
    print(f"   Confidence threshold: {confidence}")
    
    try:
        pipeline = SimpleDetectionPipeline(
            model_path=model_path,
            conf_threshold=confidence,
            device="cpu"
        )
        
        analyzer = DamageAnalyzer(currency="USD")
        
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        return False
    
    # Run detection
    print("\n🔍 Running detection...")
    start_time = time.time()
    
    try:
        # Detect
        result = pipeline.detect(image)
        
        # Simulate detections if requested
        if simulate:
            print("⚠️  SIMULATION MODE: Injecting fake detections for demo")
            h, w = image.shape[:2]
            
            # Smart simulation based on filename
            filename = Path(image_path).name.lower()
            
            fake_detections = []
            
            if 'demo_car_1' in filename or 'demo_car_minor' in filename: # Red Toyota Corolla (Minor)
                # Simulate minor scratch on headlight
                fake_detections = [
                    {
                        'class_name': 'scratch',
                        'confidence': 0.85,
                        'bbox': [int(w*0.6), int(h*0.55), int(w*0.8), int(h*0.58)], # Headlight scratch
                        'area': (w*0.2) * (h*0.03)
                    }
                ]
            elif 'demo_car_severe' in filename: # Red Toyota Corolla (Severe)
                 # Simulate dent + scratch
                 fake_detections = [
                    {
                        'class_name': 'dent',
                        'confidence': 0.92,
                        'bbox': [int(w*0.25), int(h*0.65), int(w*0.45), int(h*0.75)], # Front bumper left
                        'area': (w*0.2) * (h*0.1)
                    },
                    {
                        'class_name': 'scratch',
                        'confidence': 0.88,
                        'bbox': [int(w*0.6), int(h*0.55), int(w*0.8), int(h*0.58)], # Headlight scratch
                        'area': (w*0.2) * (h*0.03)
                    },
                    {
                        'class_name': 'glass_shatter', # Will map to severe
                        'confidence': 0.95,
                        'bbox': [int(w*0.2), int(h*0.3), int(w*0.8), int(h*0.45)], # Windshield
                        'area': (w*0.6) * (h*0.15)
                    }
                ]
            elif 'real_scratch' in filename or 'user_scratch' in filename: # User provided scratch image
                 # Simulate long scratch along the side - adjusted to cover the visible scratch better
                 fake_detections = [
                    {
                        'class_name': 'scratch',
                        'confidence': 0.96,
                        'bbox': [int(w*0.1), int(h*0.45), int(w*0.9), int(h*0.75)], # Wide area covering the long scratch
                        'area': (w*0.8) * (h*0.3)
                    }
                ]
            elif 'accident' in filename or 'crash' in filename: # Generic crash
                 fake_detections = [
                    {
                        'class_name': 'smash', # Will map to severe
                        'confidence': 0.95,
                        'bbox': [int(w*0.3), int(h*0.4), int(w*0.7), int(h*0.8)],
                        'area': (w*0.4) * (h*0.4)
                    }
                ]
            else: # Default random
                fake_detections = [
                    {
                        'class_name': 'scratch',
                        'confidence': 0.87,
                        'bbox': [int(w*0.2), int(h*0.4), int(w*0.4), int(h*0.5)],
                        'area': (w*0.2) * (h*0.1)
                    },
                    {
                        'class_name': 'dent',
                        'confidence': 0.76,
                        'bbox': [int(w*0.6), int(h*0.6), int(w*0.7), int(h*0.7)],
                        'area': (w*0.1) * (h*0.1)
                    }
                ]
            
            result.detections = fake_detections
        
        # Analyze
        analysis = analyzer.analyze(
            result.detections,
            image_shape=(image.shape[0], image.shape[1])
        )
        
        processing_time = time.time() - start_time
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return False
    
    # Print results
    print_results(analysis, processing_time)
    
    # Visualize if there are detections or if requested
    if result.detections or output_path or show_image:
        print("🎨 Creating visualization...")
        
        # Add severity to detections
        detections_with_severity = analyzer.add_severity_to_detections(
            result.detections.copy(),
            analysis.get('severity')
        )
        
        # Create annotated image
        annotated = visualize_detections(
            image,
            detections_with_severity,
            show_confidence=True
        )
        
        # Add info overlay
        info = {
            'damage_count': analysis.get('damage_count'),
            'severity': analysis.get('severity'),
            'cost_estimate': analysis.get('cost_estimate'),
            'processing_time': processing_time
        }
        annotated = add_info_overlay(annotated, info, position='top')
        
        # Save output
        if output_path:
            # Create side-by-side comparison
            if save_comparison:
                comparison = create_side_by_side(image, annotated)
                success = save_visualization(comparison, output_path)
            else:
                success = save_visualization(annotated, output_path)
            
            if success:
                print(f"✅ Result saved to: {output_path}")
            else:
                print(f"❌ Failed to save result")
        
        # Display image
        if show_image:
            print("🖼️  Displaying result (press any key to close)...")
            cv2.imshow("Vehicle Damage Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vehicle Damage Detection CLI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection
  python demo.py --image car.jpg
  
  # Save result
  python demo.py --image car.jpg --output result.jpg
  
  # Adjust confidence threshold
  python demo.py --image car.jpg --confidence 0.5
  
  # Show result in window
  python demo.py --image car.jpg --show
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save result image (default: auto-generated)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolov8n.pt',
        help='Path to YOLO model (default: yolov8n.pt)'
    )
    
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.35,
        help='Confidence threshold (default: 0.35)'
    )
    
    parser.add_argument(
        '--show', '-s',
        action='store_true',
        help='Display result in window'
    )
    
    parser.add_argument(
        '--no-comparison',
        action='store_true',
        help='Save only annotated image (not side-by-side)'
    )

    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Simulate detections for demo purposes (useful if model is not trained on damages)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Auto-generate output path if not provided
    if args.output is None and not args.show:
        input_path = Path(args.image)
        output_dir = Path("examples/demo_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}_result_{timestamp}{input_path.suffix}"
        args.output = str(output_dir / output_filename)
    
    # Run detection
    success = run_detection(
        image_path=args.image,
        output_path=args.output,
        model_path=args.model,
        confidence=args.confidence,
        show_image=args.show,
        save_comparison=not args.no_comparison,
        simulate=args.simulate
    )
    
    if success:
        print("✅ Demo completed successfully!\n")
        return 0
    else:
        print("❌ Demo failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
