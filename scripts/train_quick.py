#!/usr/bin/env python3
"""Быстрое обучение YOLOv8n для pet-проекта (20 эпох)."""

from ultralytics import YOLO
import torch
import os

def main():
    print("[*] Quick training YOLOv8n for vehicle damage detection...")
    print()

    # Проверить GPU
    print(f"[*] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Загрузить базовую модель
    print("[*] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    # Обучение
    print("[*] Starting training... (20 epochs)")
    print("[*] Dataset: data/yolo_dataset/data.yaml")
    print()

    results = model.train(
        data="data/yolo_dataset/data.yaml",
        epochs=20,  # Мало, но быстро
        imgsz=640,
        batch=4,  # Маленький batch для маленького датасета
        patience=5,  # Early stopping
        device=0 if torch.cuda.is_available() else "cpu",  # GPU если есть
        save=True,
        verbose=True,
        plots=True,  # Сохранить графики
        project="runs/train",
        name="damage_detection_v1"
    )

    print()
    print("[OK] Training completed!")
    print(f"[OK] Best model: runs/train/damage_detection_v1/weights/best.pt")
    print(f"[OK] Last model: runs/train/damage_detection_v1/weights/last.pt")

    # Валидация
    print()
    print("[*] Running validation...")
    metrics = model.val()
    print(f"[OK] mAP50: {metrics.box.map50:.4f}")
    print(f"[OK] mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
