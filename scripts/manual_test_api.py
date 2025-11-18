#!/usr/bin/env python
"""
API Testing Script for Vehicle Damage Detection System
Checks health, detects sample images, and validates responses
"""

import requests
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
HEALTH_CHECK_ENDPOINT = f"{API_BASE_URL}/health"
DETECT_ENDPOINT = f"{API_BASE_URL}/detect"

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.BLUE}{'='*50}{Colors.END}")
    print(f"{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'='*50}{Colors.END}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ {text}{Colors.END}")

def check_health() -> bool:
    """Check API health status"""
    print_header("Health Check")

    try:
        response = requests.get(HEALTH_CHECK_ENDPOINT, timeout=5)

        if response.status_code == 200:
            data = response.json()
            print_success(f"API is healthy")
            print(f"  Status: {data.get('status')}")
            print(f"  Version: {data.get('version')}")
            print(f"  Model Status: {data.get('model_status')}")
            print(f"  LLM Status: {data.get('llm_analyzer_status')}")
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False

    except requests.ConnectionError:
        print_error(f"Could not connect to API at {API_BASE_URL}")
        print_info("Make sure the backend is running: python -m uvicorn src.api.main:app --reload")
        return False
    except Exception as e:
        print_error(f"Error checking health: {str(e)}")
        return False

def test_detection(image_path: str) -> bool:
    """Test damage detection on an image"""
    print_header(f"Detection Test: {Path(image_path).name}")

    if not Path(image_path).exists():
        print_error(f"Image file not found: {image_path}")
        return False

    try:
        print_info(f"Uploading image ({Path(image_path).stat().st_size / 1024:.1f} KB)...")

        with open(image_path, 'rb') as f:
            files = {'file': f}
            start_time = time.time()
            response = requests.post(DETECT_ENDPOINT, files=files, timeout=30)
            elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                result = data.get('result', {})

                print_success(f"Detection completed in {elapsed:.2f}s")
                print(f"\n  Detections:")

                detections = result.get('detections', {})
                num_detections = detections.get('num_detections', 0)
                print(f"    - Found: {num_detections} damage areas")

                if detections.get('class_names'):
                    print(f"    - Types: {', '.join(detections.get('class_names', []))}")
                    print(f"    - Confidence: {[f'{s:.2f}' for s in detections.get('scores', [])]}")

                # Classification
                classification = result.get('classification', {})
                if classification:
                    print(f"\n  Severity: {classification.get('severity', 'N/A')}")
                    print(f"  Confidence: {classification.get('confidence', 0)*100:.1f}%")

                # Cost Estimate
                cost = result.get('cost_estimate', {})
                if cost:
                    print(f"\n  Cost Estimate: ${cost.get('estimated_cost', 0):.2f}")
                    cost_range = cost.get('cost_range', {})
                    if cost_range:
                        print(f"  Range: ${cost_range.get('min', 0):.2f} - ${cost_range.get('max', 0):.2f}")

                print(f"\n  Processing Time: {result.get('processing_time', 0):.3f}s")

                return True
            else:
                print_error(f"Detection failed: {data.get('message', 'Unknown error')}")
                return False
        else:
            print_error(f"API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.Timeout:
        print_error("Request timeout - API took too long to respond")
        return False
    except Exception as e:
        print_error(f"Error during detection: {str(e)}")
        return False

def find_test_images() -> list:
    """Find test images in the project"""
    test_dirs = [
        Path("data/test_images"),
        Path("data/raw"),
        Path("uploads"),
    ]

    images = []
    for dir_path in test_dirs:
        if dir_path.exists():
            images.extend(dir_path.glob("*.jpg"))
            images.extend(dir_path.glob("*.png"))
            images.extend(dir_path.glob("*.jpeg"))

    return images[:3]  # Limit to 3 images

def main():
    """Main test runner"""
    print(f"\n{Colors.BLUE}Vehicle Damage Detection - API Test Suite{Colors.END}")
    print(f"{Colors.BLUE}Testing API at {API_BASE_URL}{Colors.END}")

    # Test 1: Health Check
    if not check_health():
        print_error("\nAPI health check failed. Cannot proceed with tests.")
        sys.exit(1)

    # Test 2: Find and test images
    print_header("Detection Tests")
    test_images = find_test_images()

    if not test_images:
        print_info("No test images found. To test detection:")
        print("  1. Add images to data/test_images/")
        print("  2. Run this script again")
        print("\nYou can also test via the web interface at http://localhost:3000")
    else:
        results = []
        for image_path in test_images:
            success = test_detection(str(image_path))
            results.append((image_path.name, success))

        # Summary
        print_header("Test Summary")
        passed = sum(1 for _, success in results if success)
        total = len(results)

        for name, success in results:
            status = f"{Colors.GREEN}PASS{Colors.END}" if success else f"{Colors.RED}FAIL{Colors.END}"
            print(f"  {name}: {status}")

        print(f"\nTotal: {passed}/{total} tests passed")

        if passed < total:
            sys.exit(1)

    # Final Summary
    print_header("Testing Complete")
    print_success("API is ready for use!")
    print(f"\nWeb Interface: http://localhost:3000")
    print(f"API Docs:     http://localhost:8000/docs")
    print(f"Health Check: http://localhost:8000/health\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_info("\n\nTesting interrupted by user")
        sys.exit(0)
