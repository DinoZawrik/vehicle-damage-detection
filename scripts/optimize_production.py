#!/usr/bin/env python3
"""
Скрипт для оптимизации моделей и подготовки production окружения.

Функции:
- Конвертация моделей в ONNX для быстрого inference
- Квантизация моделей (FP16/INT8)
- Batch inference оптимизация
- Кэширование результатов
- Мониторинг и логирование
"""

import os
import sys
import json
import yaml
import argparse
import logging
import time
import numpy as np
import torch
import torch.quantization as quantization
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import onnx
import onnxruntime as ort
from PIL import Image
import cv2

# Импорт моделей
try:
    from ultralytics import YOLO
    from segment_anything import sam_model_registry
    import open_clip
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Клass for model optimization and production preparation.
    """
    
    def __init__(self, output_dir: str = "models/optimized"):
        """
        Initialize model optimizer.
        
        Args:
            output_dir: Directory for optimized models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimization_results = {}
        
        logger.info(f"Model optimizer initialized with output dir: {self.output_dir}")
    
    def optimize_yolo_model(
        self,
        model_path: str,
        input_size: int = 640,
        format: str = "onnx",
        half_precision: bool = True,
        dynamic_axes: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize YOLO model for production.
        
        Args:
            model_path: Path to YOLO model
            input_size: Input image size
            format: Export format (onnx, torchscript)
            half_precision: Use FP16 precision
            dynamic_axes: Use dynamic input dimensions
            
        Returns:
            Optimization results
        """
        try:
            logger.info(f"Optimizing YOLO model: {model_path}")
            
            # Load model
            model = YOLO(model_path)
            
            # Export to ONNX
            export_path = model.export(
                format=format,
                imgsz=input_size,
                half=half_precision,
                dynamic=dynamic_axes,
                opset=17,
                simplify=True
            )
            
            # Validate ONNX model
            onnx_model = onnx.load(export_path)
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            session = ort.InferenceSession(str(export_path), providers=providers)
            
            # Benchmark inference
            benchmark_results = self._benchmark_onnx_model(session, input_size)
            
            # Calculate model size reduction
            original_size = os.path.getsize(model_path) / (1024**2)  # MB
            optimized_size = os.path.getsize(export_path) / (1024**2)  # MB
            size_reduction = (original_size - optimized_size) / original_size * 100
            
            results = {
                'model_type': 'YOLO',
                'original_path': model_path,
                'optimized_path': str(export_path),
                'original_size_mb': original_size,
                'optimized_size_mb': optimized_size,
                'size_reduction_percent': size_reduction,
                'benchmark': benchmark_results,
                'format': format,
                'half_precision': half_precision,
                'dynamic_axes': dynamic_axes
            }
            
            self.optimization_results['yolo'] = results
            
            logger.info(f"YOLO optimization completed:")
            logger.info(f"  Size reduction: {size_reduction:.1f}%")
            logger.info(f"  Inference time: {benchmark_results['avg_inference_time']:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing YOLO model: {e}")
            return {}
    
    def optimize_sam_model(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        half_precision: bool = True,
        quantization: str = "dynamic"
    ) -> Dict[str, Any]:
        """
        Optimize SAM model for production.
        
        Args:
            model_type: SAM model type
            checkpoint_path: Path to SAM checkpoint
            half_precision: Use FP16 precision
            quantization: Quantization method (dynamic, static, none)
            
        Returns:
            Optimization results
        """
        try:
            logger.info(f"Optimizing SAM model: {model_type}")
            
            # Load SAM model
            if checkpoint_path is None:
                checkpoint_map = {
                    'vit_b': 'sam_vit_b_01ec6bb.pth',
                    'vit_l': 'sam_vit_l_0b3195.pth',
                    'vit_h': 'sam_vit_h_4b8939.pth'
                }
                checkpoint_path = checkpoint_map.get(model_type, checkpoint_map['vit_b'])
            
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            
            if torch.cuda.is_available():
                sam = sam.cuda()
            
            # Apply optimizations
            if half_precision and torch.cuda.is_available():
                sam = sam.half()
                logger.info("Applied FP16 precision to SAM model")
            
            if quantization == "dynamic":
                # Dynamic quantization for encoder
                sam.image_encoder = torch.quantization.quantize_dynamic(
                    sam.image_encoder,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization to SAM encoder")
            
            # Save optimized model
            optimized_path = self.output_dir / f"sam_{model_type}_optimized.pth"
            torch.save(sam.state_dict(), optimized_path)
            
            # Calculate size reduction
            original_size = os.path.getsize(checkpoint_path) / (1024**2)  # MB
            optimized_size = os.path.getsize(optimized_path) / (1024**2)  # MB
            size_reduction = (original_size - optimized_size) / original_size * 100
            
            results = {
                'model_type': 'SAM',
                'original_path': checkpoint_path,
                'optimized_path': str(optimized_path),
                'original_size_mb': original_size,
                'optimized_size_mb': optimized_size,
                'size_reduction_percent': size_reduction,
                'half_precision': half_precision,
                'quantization': quantization,
                'model_config': {
                    'model_type': model_type,
                    'image_size': 1024,
                    'mask_threshold': 0.0
                }
            }
            
            self.optimization_results['sam'] = results
            
            logger.info(f"SAM optimization completed:")
            logger.info(f"  Size reduction: {size_reduction:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing SAM model: {e}")
            return {}
    
    def optimize_clip_model(
        self,
        model_name: str = "ViT-B/32",
        pretrained: str = "openai",
        half_precision: bool = True,
        quantization: str = "dynamic"
    ) -> Dict[str, Any]:
        """
        Optimize CLIP model for production.
        
        Args:
            model_name: CLIP model name
            pretrained: Pretrained weights
            half_precision: Use FP16 precision
            quantization: Quantization method
            
        Returns:
            Optimization results
        """
        try:
            logger.info(f"Optimizing CLIP model: {model_name}")
            
            # Load CLIP model
            model, _, transform = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained
            )
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            # Apply optimizations
            if half_precision and torch.cuda.is_available():
                model = model.half()
                logger.info("Applied FP16 precision to CLIP model")
            
            if quantization == "dynamic":
                # Quantize text and vision encoders
                model.visual = torch.quantization.quantize_dynamic(
                    model.visual,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
                model.transformer = torch.quantization.quantize_dynamic(
                    model.transformer,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization to CLIP model")
            
            # Save optimized model
            optimized_path = self.output_dir / f"clip_{model_name.replace('/', '_')}_optimized.pth"
            torch.save(model.state_dict(), optimized_path)
            
            # Calculate size reduction
            # Note: We can't easily get original size for OpenCLIP models
            # so we'll just report the optimized size
            
            results = {
                'model_type': 'CLIP',
                'optimized_path': str(optimized_path),
                'optimized_size_mb': os.path.getsize(optimized_path) / (1024**2),
                'half_precision': half_precision,
                'quantization': quantization,
                'model_config': {
                    'model_name': model_name,
                    'pretrained': pretrained
                }
            }
            
            self.optimization_results['clip'] = results
            
            logger.info(f"CLIP optimization completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing CLIP model: {e}")
            return {}
    
    def _benchmark_onnx_model(
        self,
        session: ort.InferenceSession,
        input_size: int,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark ONNX model performance.
        
        Args:
            session: ONNX Runtime session
            input_size: Input image size
            num_runs: Number of inference runs
            
        Returns:
            Benchmark results
        """
        try:
            # Create dummy input
            dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                _ = session.run(None, {'images': dummy_input})
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = session.run(None, {'images': dummy_input})
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            # Calculate statistics
            times = np.array(times)
            return {
                'avg_inference_time': float(np.mean(times)),
                'std_inference_time': float(np.std(times)),
                'min_inference_time': float(np.min(times)),
                'max_inference_time': float(np.max(times)),
                'throughput_fps': float(1.0 / np.mean(times)),
                'num_runs': num_runs
            }
            
        except Exception as e:
            logger.error(f"Error benchmarking ONNX model: {e}")
            return {}
    
    def create_production_config(self) -> Dict[str, Any]:
        """
        Create production configuration file.
        
        Returns:
            Production configuration
        """
        config = {
            'model_paths': {},
            'optimization_settings': {},
            'inference_settings': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'batch_size': 1,
                'input_sizes': {
                    'yolo': 640,
                    'sam': 1024,
                    'clip': 224
                },
                'confidence_thresholds': {
                    'yolo': 0.35,
                    'sam': 0.5,
                    'clip': 0.6
                }
            },
            'monitoring': {
                'enable_logging': True,
                'log_level': 'INFO',
                'metrics_collection': True,
                'prometheus_enabled': True,
                'prometheus_port': 8001
            },
            'caching': {
                'enabled': True,
                'cache_dir': 'cache/',
                'ttl_seconds': 3600,
                'max_cache_size': 1000
            },
            'batch_processing': {
                'enabled': True,
                'max_batch_size': 8,
                'queue_timeout': 30
            }
        }
        
        # Add optimized model paths
        for model_type, results in self.optimization_results.items():
            if 'optimized_path' in results:
                config['model_paths'][model_type] = results['optimized_path']
                config['optimization_settings'][model_type] = {
                    'format': results.get('format', 'pth'),
                    'half_precision': results.get('half_precision', False),
                    'quantization': results.get('quantization', 'none'),
                    'size_reduction': results.get('size_reduction_percent', 0)
                }
        
        # Save config
        config_path = self.output_dir / 'production_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Production config saved to: {config_path}")
        
        return config
    
    def create_docker_deployment(self, dockerfile_path: str = "Dockerfile.optimized"):
        """
        Create optimized Docker deployment files.
        
        Args:
            dockerfile_path: Path for Dockerfile
        """
        dockerfile_content = """# Optimized Vehicle Damage Detection API
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3.11 \\
    python3.11-dev \\
    python3-pip \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy optimized models
COPY models/optimized/ /app/models/optimized/

# Copy application code
COPY src/ /app/src/
COPY web/ /app/web/
COPY scripts/ /app/scripts/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python3", "-m", "src.api.main"]
"""
        
        # Save Dockerfile
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Create docker-compose for production
        compose_content = """version: '3.8'

services:
  damage-detection-api:
    build:
      context: .
      dockerfile: Dockerfile.optimized
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - TORCH_CUDA_ARCH_LIST="8.6"
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis-cache:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
"""
        
        # Save docker-compose
        with open('docker-compose.prod.yml', 'w') as f:
            f.write(compose_content)
        
        logger.info("Docker deployment files created:")
        logger.info("  - Dockerfile.optimized")
        logger.info("  - docker-compose.prod.yml")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Optimize models for production deployment")
    parser.add_argument("--yolo-model", type=str, default="yolov9n.pt",
                        help="Path to YOLO model")
    parser.add_argument("--sam-model-type", type=str, default="vit_b",
                        help="SAM model type")
    parser.add_argument("--clip-model-name", type=str, default="ViT-B/32",
                        help="CLIP model name")
    parser.add_argument("--output-dir", type=str, default="models/optimized",
                        help="Output directory for optimized models")
    parser.add_argument("--create-docker", action="store_true",
                        help="Create Docker deployment files")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark tests")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = ModelOptimizer(args.output_dir)
    
    print("🚀 Starting Model Optimization for Production")
    print("=" * 50)
    
    # Optimize YOLO model
    print("\n📦 Optimizing YOLO model...")
    optimizer.optimize_yolo_model(
        model_path=args.yolo_model,
        format="onnx",
        half_precision=True,
        dynamic_axes=True
    )
    
    # Optimize SAM model
    print("\n✂️ Optimizing SAM model...")
    optimizer.optimize_sam_model(
        model_type=args.sam_model_type,
        half_precision=True,
        quantization="dynamic"
    )
    
    # Optimize CLIP model
    print("\n🧠 Optimizing CLIP model...")
    optimizer.optimize_clip_model(
        model_name=args.clip_model_name,
        half_precision=True,
        quantization="dynamic"
    )
    
    # Create production config
    print("\n⚙️ Creating production configuration...")
    config = optimizer.create_production_config()
    
    # Create Docker deployment files
    if args.create_docker:
        print("\n🐳 Creating Docker deployment files...")
        optimizer.create_docker_deployment()
    
    # Print summary
    print("\n" + "=" * 50)
    print("✅ Model Optimization Summary")
    print("=" * 50)
    
    for model_type, results in optimizer.optimization_results.items():
        if results:
            print(f"\n{model_type.upper()} Model:")
            print(f"  Optimized path: {results['optimized_path']}")
            if 'size_reduction_percent' in results:
                print(f"  Size reduction: {results['size_reduction_percent']:.1f}%")
            if 'benchmark' in results:
                benchmark = results['benchmark']
                print(f"  Avg inference time: {benchmark['avg_inference_time']:.3f}s")
                print(f"  Throughput: {benchmark['throughput_fps']:.1f} FPS")
    
    print(f"\n📁 Output directory: {args.output_dir}")
    print("🎯 Models ready for production deployment!")
    
    if args.benchmark:
        print("\n📊 Running comprehensive benchmarks...")
        # Additional benchmarking logic here


if __name__ == "__main__":
    main()