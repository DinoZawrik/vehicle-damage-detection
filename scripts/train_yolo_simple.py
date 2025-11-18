#!/usr/bin/env python3
"""
Simple YOLO training script without matplotlib dependencies for NumPy compatibility
"""

import os
import sys
import yaml
import argparse
import logging
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Импорт Ultralytics YOLO
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    print("Error: ultralytics not found. Please install: pip install ultralytics")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_yolo_simple(
    data_yaml: str,
    model_path: str = "yolov9n.pt",
    epochs: int = 10,
    batch_size: int = 4,
    img_size: int = 640,
    device: str = "auto",
    project: str = "runs/train",
    name: str = "damage_detection_simple",
    freeze_backbone: bool = True,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0005
):
    """
    Простое обучение YOLO без визуализации для быстрого fine-tuning
    """
    
    print("🔧 Simple YOLO Training for Damage Detection")
    print("=" * 50)
    
    # Автоопределение устройства
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    
    try:
        # Загрузка модели
        print("\n📥 Loading pre-trained model...")
        model = YOLO(model_path)
        
        # Заморозка backbone для fine-tuning
        if freeze_backbone:
            print("🔒 Freezing backbone layers...")
            for name, param in model.model.named_parameters():
                if any(x in name for x in ['backbone', 'encoder']):
                    param.requires_grad = False
        
        print("✅ Model loaded successfully!")
        
        # Параметры обучения
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': device,
            'project': project,
            'name': name,
            'exist_ok': True,
            
            # Оптимизация
            'optimizer': 'AdamW',
            'lr0': learning_rate,
            'weight_decay': weight_decay,
            'momentum': 0.937,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Аугментация данных
            'augment': True,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.3,
            'translate': 0.2,
            'scale': 0.5,
            'shear': 0.05,
            'perspective': 0.0001,
            
            # Loss и метрики
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # Валидация
            'val': True,
            'save_period': 5,
            'verbose': True
        }
        
        print("\n🚀 Starting training...")
        
        # Запуск обучения
        results = model.train(**train_args)
        
        # Получение метрик
        metrics = results.results_dict
        
        print("\n📊 Training Results:")
        print(f"   Best model path: {results.save_dir}")
        
        # Извлечение ключевых метрик
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)
        mAP50 = metrics.get('metrics/mAP50(B)', 0)
        mAP50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   mAP50: {mAP50:.4f}")
        print(f"   mAP50-95: {mAP50_95:.4f}")
        
        # Сохранение результатов в JSON
        results_summary = {
            'model_path': str(results.save_dir / 'weights' / 'best.pt'),
            'metrics': {
                'precision': precision,
                'recall': recall,
                'mAP50': mAP50,
                'mAP50_95': mAP50_95
            },
            'training_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'img_size': img_size,
                'device': device,
                'freeze_backbone': freeze_backbone,
                'learning_rate': learning_rate
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Сохранение результатов
        results_file = Path(results.save_dir) / 'training_summary.json'
        with open(results_file, 'w') as f:
            import json
            json.dump(results_summary, f, indent=2)
        
        print(f"\n💾 Training summary saved to: {results_file}")
        
        return results_summary
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return {}

def validate_model(model_path: str, data_yaml: str, img_size: int = 640, device: str = "auto"):
    """Валидация модели"""
    
    try:
        print(f"\n🔍 Validating model: {model_path}")
        
        # Загрузка модели
        model = YOLO(model_path)
        
        # Автоопределение устройства
        if device == "auto":
            device = 0 if torch.cuda.is_available() else "cpu"
        
        # Валидация
        results = model.val(
            data=data_yaml,
            imgsz=img_size,
            device=device,
            verbose=True
        )
        
        # Извлечение метрик
        metrics = results.results_dict
        validation_results = {
            'precision': metrics.get('metrics/precision(B)', 0),
            'recall': metrics.get('metrics/recall(B)', 0),
            'mAP50': metrics.get('metrics/mAP50(B)', 0),
            'mAP50_95': metrics.get('metrics/mAP50-95(B)', 0)
        }
        
        print(f"\n📊 Validation Results:")
        print(f"   Precision: {validation_results['precision']:.4f}")
        print(f"   Recall: {validation_results['recall']:.4f}")
        print(f"   mAP50: {validation_results['mAP50']:.4f}")
        print(f"   mAP50-95: {validation_results['mAP50_95']:.4f}")
        
        return validation_results
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return {}

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple YOLO training for damage detection")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml configuration file")
    parser.add_argument("--model", type=str, default="yolov9n.pt",
                        help="Path to pre-trained model weights")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for training (auto, cpu, cuda)")
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="damage_detection_simple",
                        help="Experiment name")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone layers for fine-tuning")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Weight decay")
    parser.add_argument("--validate", type=str, help="Path to model for validation only")
    
    args = parser.parse_args()
    
    if args.validate:
        # Только валидация
        validate_model(args.validate, args.data, args.img_size, args.device)
    else:
        # Обучение
        results = train_yolo_simple(
            data_yaml=args.data,
            model_path=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            device=args.device,
            project=args.project,
            name=args.name,
            freeze_backbone=args.freeze_backbone,
            learning_rate=args.lr,
            weight_decay=args.weight_decay
        )
        
        if results:
            print(f"\n🎉 Training completed successfully!")
            print(f"Best model: {results['model_path']}")
            print(f"mAP50: {results['metrics']['mAP50']:.4f}")
            print(f"mAP50-95: {results['metrics']['mAP50_95']:.4f}")

if __name__ == "__main__":
    main()