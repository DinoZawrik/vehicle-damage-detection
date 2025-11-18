#!/usr/bin/env python3
"""Простой тест YOLO детекции на реальных изображениях."""

import cv2
from pathlib import Path
from ultralytics import YOLO

def test_detection():
    """Тест YOLO на тестовых изображениях."""
    print("="*60)
    print("YOLO DAMAGE DETECTION TEST")
    print("="*60)

    # Загрузить pretrained YOLO модель
    print("\n[*] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("[OK] Model loaded")

    # Тестовые изображения
    test_dir = Path("data/test_images")
    test_images = list(test_dir.glob("*.jpg"))

    if not test_images:
        print("\n[ERROR] No test images found in data/test_images/")
        return 1

    print(f"\n[*] Found {len(test_images)} test images")

    # Тестировать каждое изображение
    for img_path in test_images:
        print(f"\n[*] Testing: {img_path.name}")

        # Загрузить изображение
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERROR] Cannot load image")
            continue

        print(f"    Shape: {img.shape}")

        # Запустить detection
        try:
            results = model.predict(str(img_path), conf=0.35, imgsz=640, verbose=False)

            if results:
                for result in results:
                    detections = result.boxes
                    if detections is not None and len(detections) > 0:
                        print(f"    [OK] Found {len(detections)} detections:")
                        for box in detections:
                            class_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = result.names[class_id]
                            print(f"        - {class_name}: {conf:.3f}")
                    else:
                        print(f"    [WARNING] No objects detected")
        except Exception as e:
            print(f"    [ERROR] {e}")

    print("\n" + "="*60)
    print("[OK] YOLO test completed")
    print("="*60)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(test_detection())
