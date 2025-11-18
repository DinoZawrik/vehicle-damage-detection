#!/usr/bin/env python3
"""
Скрипт для дообучения YOLOv9n модели на специализированном датасете повреждений автомобилей.

Функции:
- Загрузка предобученной модели YOLOv9n
- Fine-tuning на датасете повреждений
- Валидация и сохранение лучшей модели
- Интеграция обученной модели в pipeline
"""

import os
import sys
import yaml
import argparse
import logging
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
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


class YOLOTrainer:
    """
    Класс для обучения YOLOv9n модели на датасете повреждений автомобилей.
    """
    
    def __init__(
        self,
        data_yaml: str,
        model_path: str = "yolov9n.pt",
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        device: str = "auto",
        project: str = "runs/train",
        name: str = "damage_detection",
        freeze_backbone: bool = True,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        patience: int = 10
    ):
        """
        Initialize YOLO trainer.
        
        Args:
            data_yaml: Path to data.yaml configuration file
            model_path: Path to pre-trained model weights
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Input image size
            device: Device for training ('auto', 'cpu', 'cuda')
            project: Project directory for saving results
            name: Experiment name
            freeze_backbone: Whether to freeze backbone layers
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            patience: Early stopping patience
        """
        self.data_yaml = data_yaml
        self.model_path = model_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device
        self.project = project
        self.name = name
        self.freeze_backbone = freeze_backbone
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        
        # Auto-detect device
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.training_results = {}
        
        logger.info(f"YOLO Trainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Data: {self.data_yaml}")
    
    def load_model(self) -> bool:
        """
        Load pre-trained YOLO model.
        
        Returns:
            True if model loaded successfully
        """
        try:
            logger.info(f"Loading pre-trained model from {self.model_path}...")
            
            # Load model
            self.model = YOLO(self.model_path)
            
            # Freeze backbone if requested
            if self.freeze_backbone:
                self._freeze_backbone()
                logger.info("Backbone layers frozen for fine-tuning")
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _freeze_backbone(self):
        """Freeze backbone layers for fine-tuning."""
        # Freeze backbone (first few layers)
        for name, param in self.model.model.named_parameters():
            if any(x in name for x in ['backbone', 'encoder']):
                param.requires_grad = False
        
        logger.info("Frozen backbone parameters")
    
    def train(
        self,
        resume: bool = False,
        optimizer: str = "AdamW",
        lr_scheduler: str = "cosine",
        augment: bool = True,
        mosaic: float = 1.0,
        mixup: float = 0.1,
        copy_paste: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train the YOLO model.
        
        Args:
            resume: Whether to resume from last checkpoint
            optimizer: Optimizer type ('SGD', 'Adam', 'AdamW')
            lr_scheduler: Learning rate scheduler ('linear', 'cosine', 'warmup_cosine')
            augment: Whether to use data augmentation
            mosaic: Mosaic augmentation probability
            mixup: MixUp augmentation probability
            copy_paste: Copy-paste augmentation probability
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            if not self.load_model():
                return {}
        
        logger.info("Starting training...")
        
        # Training arguments
        train_args = {
            'data': self.data_yaml,
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.img_size,
            'device': self.device,
            'project': self.project,
            'name': self.name,
            'exist_ok': True,
            'resume': resume,
            
            # Optimization
            'optimizer': optimizer,
            'lr0': self.learning_rate,
            'weight_decay': self.weight_decay,
            'momentum': 0.937,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'lr_scheduler': lr_scheduler,
            
            # Data augmentation
            'augment': augment,
            'mosaic': mosaic,
            'mixup': mixup,
            'copy_paste': copy_paste,
            'hsv_h': 0.015,  # Hue augmentation
            'hsv_s': 0.7,    # Saturation augmentation
            'hsv_v': 0.4,    # Value augmentation
            'degrees': 0.3,  # Rotation augmentation
            'translate': 0.2, # Translation augmentation
            'scale': 0.5,    # Scale augmentation
            'shear': 0.05,   # Shear augmentation
            'perspective': 0.0001, # Perspective augmentation
            
            # Loss and metrics
            'box': 7.5,      # Box loss gain
            'cls': 0.5,      # Class loss gain
            'dfl': 1.5,      # Distribution focal loss gain
            'fl_gamma': 2.0, # Focal loss gamma
            
            # Early stopping
            'patience': self.patience,
            
            # Validation
            'val': True,
            'save_period': 10, # Save checkpoint every N epochs
            'verbose': True
        }
        
        try:
            # Start training
            results = self.model.train(**train_args)
            
            # Save training results
            self.training_results = {
                'model_path': str(self.model.model.saves_dir / 'weights' / 'best.pt'),
                'metrics': results.results_dict,
                'training_args': train_args,
                'start_time': datetime.now().isoformat()
            }
            
            logger.info("Training completed successfully!")
            logger.info(f"Best model saved to: {self.training_results['model_path']}")
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {}
    
    def validate(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate the trained model.
        
        Args:
            model_path: Path to model weights (if None, uses current model)
            
        Returns:
            Validation metrics
        """
        try:
            # Load model if path provided
            if model_path:
                validation_model = YOLO(model_path)
            else:
                validation_model = self.model
            
            logger.info("Running validation...")
            
            # Run validation
            metrics = validation_model.val(
                data=self.data_yaml,
                batch=self.batch_size,
                imgsz=self.img_size,
                device=self.device,
                verbose=True
            )
            
            # Extract metrics
            validation_results = {
                'precision': metrics.results_dict.get('metrics/precision(B)', 0),
                'recall': metrics.results_dict.get('metrics/recall(B)', 0),
                'mAP50': metrics.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50_95': metrics.results_dict.get('metrics/mAP50-95(B)', 0),
                'fitness': metrics.results_dict.get('fitness', 0)
            }
            
            logger.info("Validation completed!")
            logger.info(f"Precision: {validation_results['precision']:.4f}")
            logger.info(f"Recall: {validation_results['recall']:.4f}")
            logger.info(f"mAP50: {validation_results['mAP50']:.4f}")
            logger.info(f"mAP50-95: {validation_results['mAP50_95']:.4f}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return {}
    
    def export_model(
        self,
        model_path: Optional[str] = None,
        format: str = "onnx",
        opset: int = 17,
        dynamic: bool = True
    ) -> str:
        """
        Export model to different formats.
        
        Args:
            model_path: Path to model weights (if None, uses best model)
            format: Export format ('onnx', 'torchscript', 'tflite', 'pb', 'coreml', 'paddle')
            opset: ONNX opset version
            dynamic: Whether to use dynamic input shapes
            
        Returns:
            Path to exported model
        """
        try:
            # Load model if path provided
            if model_path:
                export_model = YOLO(model_path)
            else:
                export_model = self.model
            
            logger.info(f"Exporting model to {format} format...")
            
            # Export model
            export_path = export_model.export(
                format=format,
                opset=opset,
                dynamic=dynamic,
                imgsz=self.img_size,
                half=False
            )
            
            logger.info(f"Model exported to: {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            return ""
    
    def plot_training_results(self, save_dir: Optional[str] = None):
        """
        Plot training results and metrics.
        
        Args:
            save_dir: Directory to save plots (if None, uses model's save directory)
        """
        if not self.training_results:
            logger.warning("No training results to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            save_dir = save_dir or str(Path(self.training_results['model_path']).parent)
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            
            # Extract metrics
            metrics = self.training_results['metrics']
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('YOLO Training Results', fontsize=16)
            
            # Training losses
            epochs = list(range(len(metrics.get('train/box_loss', []))))
            
            # Box loss
            if 'train/box_loss' in metrics:
                axes[0, 0].plot(epochs, metrics['train/box_loss'], 'b-', label='Train', linewidth=2)
                axes[0, 0].plot(epochs, metrics.get('val/box_loss', []), 'r-', label='Val', linewidth=2)
                axes[0, 0].set_title('Box Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Classification loss
            if 'train/cls_loss' in metrics:
                axes[0, 1].plot(epochs, metrics['train/cls_loss'], 'b-', label='Train', linewidth=2)
                axes[0, 1].plot(epochs, metrics.get('val/cls_loss', []), 'r-', label='Val', linewidth=2)
                axes[0, 1].set_title('Classification Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # DFL loss
            if 'train/dfl_loss' in metrics:
                axes[0, 2].plot(epochs, metrics['train/dfl_loss'], 'b-', label='Train', linewidth=2)
                axes[0, 2].plot(epochs, metrics.get('val/dfl_loss', []), 'r-', label='Val', linewidth=2)
                axes[0, 2].set_title('DFL Loss')
                axes[0, 2].set_xlabel('Epoch')
                axes[0, 2].set_ylabel('Loss')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
            
            # Precision and Recall
            if 'metrics/precision(B)' in metrics:
                axes[1, 0].plot(epochs, metrics['metrics/precision(B)'], 'g-', linewidth=2, label='Precision')
                axes[1, 0].plot(epochs, metrics.get('metrics/recall(B)', []), 'orange', linewidth=2, label='Recall')
                axes[1, 0].set_title('Precision & Recall')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # mAP scores
            if 'metrics/mAP50(B)' in metrics:
                axes[1, 1].plot(epochs, metrics['metrics/mAP50(B)'], 'purple', linewidth=2, label='mAP50')
                axes[1, 1].plot(epochs, metrics.get('metrics/mAP50-95(B)', []), 'red', linewidth=2, label='mAP50-95')
                axes[1, 1].set_title('mAP Scores')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # Learning rate
            if 'lr/pg0' in metrics:
                axes[1, 2].plot(epochs, metrics['lr/pg0'], 'brown', linewidth=2)
                axes[1, 2].set_title('Learning Rate')
                axes[1, 2].set_xlabel('Epoch')
                axes[1, 2].set_ylabel('LR')
                axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = save_dir / 'training_plots.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting training results: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary and statistics.
        
        Returns:
            Model summary dictionary
        """
        if self.model is None:
            return {}
        
        try:
            # Model info
            model_info = {
                'model_type': self.model.__class__.__name__,
                'model_size': self.model.model.__sizeof__() / 1024 / 1024,  # MB
                'parameters': sum(p.numel() for p in self.model.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.model.parameters() if p.requires_grad),
                'input_size': self.img_size,
                'device': self.device
            }
            
            logger.info("Model summary:")
            logger.info(f"  Type: {model_info['model_type']}")
            logger.info(f"  Size: {model_info['model_size']:.2f} MB")
            logger.info(f"  Parameters: {model_info['parameters']:,}")
            logger.info(f"  Trainable: {model_info['trainable_parameters']:,}")
            logger.info(f"  Input size: {model_info['input_size']}x{model_info['input_size']}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return {}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train YOLOv9n on damage detection dataset")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml configuration file")
    parser.add_argument("--model", type=str, default="yolov9n.pt",
                        help="Path to pre-trained model weights")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for training (auto, cpu, cuda)")
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="damage_detection",
                        help="Experiment name")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone layers for fine-tuning")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--export", type=str, choices=['onnx', 'torchscript', 'tflite'],
                        help="Export format for trained model")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLOTrainer(
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
        weight_decay=args.weight_decay,
        patience=args.patience
    )
    
    print("YOLOv9n Damage Detection Training")
    print("=" * 50)
    
    # Load model
    if not trainer.load_model():
        print("Failed to load model. Exiting...")
        return
    
    # Get model summary
    trainer.get_model_summary()
    
    # Train model
    print("\nStarting training...")
    results = trainer.train(
        resume=args.resume,
        augment=not args.no_augment
    )
    
    if not results:
        print("Training failed. Exiting...")
        return
    
    # Validate model
    print("\nRunning validation...")
    validation_metrics = trainer.validate()
    
    # Plot training results
    print("\nGenerating training plots...")
    trainer.plot_training_results()
    
    # Export model if requested
    if args.export:
        print(f"\nExporting model to {args.export} format...")
        export_path = trainer.export_model(format=args.export)
        if export_path:
            print(f"Model exported to: {export_path}")
    
    print("\nTraining completed successfully!")
    print(f"Best model: {results['model_path']}")
    print(f"Validation mAP50: {validation_metrics.get('mAP50', 0):.4f}")
    print(f"Validation mAP50-95: {validation_metrics.get('mAP50-95', 0):.4f}")


if __name__ == "__main__":
    main()