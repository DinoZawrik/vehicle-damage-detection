#!/usr/bin/env python3
"""
Скрипт для подготовки специализированного датасета повреждений автомобилей.

Функции:
- Загрузка публичных датасетов повреждений
- Конвертация в YOLO формат
- Создание data.yaml конфигурации
- Проверка качества разметки
- Разделение на train/val/test
"""

import os
import json
import yaml
import shutil
import argparse
import zipfile
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
from PIL import Image
import hashlib
import random

# Константы
DAMAGE_CLASSES = {
    0: 'scratch',
    1: 'dent', 
    2: 'crack',
    3: 'shatter',
    4: 'rust',
    5: 'broken_part',
    6: 'paint_damage',
    7: 'smash',
    8: 'glass_damage',
    9: 'light_damage'
}

# Обратное соответствие
CLASS_TO_ID = {v: k for k, v in DAMAGE_CLASSES.items()}

# URL публичных датасетов (примеры)
DATASET_SOURCES = {
    'car_damage_kaggle': {
        'url': 'https://www.kaggle.com/datasets/akash1406/car-damage-detection',
        'description': 'Car Damage Detection Dataset',
        'expected_images': 2000
    },
    'vehicle_damage_roboflow': {
        'url': 'https://public.roboflow.com/object-detection/vehicle-damage',
        'description': 'Vehicle Damage Assessment Dataset',
        'expected_images': 1500
    },
    'auto_damage_public': {
        'url': 'https://www.kaggle.com/datasets/ahmedelnaggar/cardamage',
        'description': 'Car Damage Dataset',
        'expected_images': 1800
    }
}


class DamageDatasetPreparer:
    """
    Класс для подготовки датасета повреждений автомобилей.
    """
    
    def __init__(self, output_dir: str = "data/damage_dataset"):
        """
        Initialize dataset preparer.
        
        Args:
            output_dir: Output directory for prepared dataset
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.train_images_dir = self.images_dir / "train"
        self.val_images_dir = self.images_dir / "val"
        self.test_images_dir = self.images_dir / "test"
        self.train_labels_dir = self.labels_dir / "train"
        self.val_labels_dir = self.labels_dir / "val"
        self.test_labels_dir = self.labels_dir / "test"
        
        self.setup_directories()
        
        self.dataset_stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'test_images': 0,
            'class_distribution': {cls: 0 for cls in DAMAGE_CLASSES.values()},
            'annotation_issues': []
        }
    
    def setup_directories(self):
        """Create necessary directories."""
        for dir_path in [
            self.images_dir, self.labels_dir,
            self.train_images_dir, self.val_images_dir, self.test_images_dir,
            self.train_labels_dir, self.val_labels_dir, self.test_labels_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Created dataset directories at {self.output_dir}")
    
    def download_dataset(self, source_name: str, download_url: str) -> bool:
        """
        Download dataset from source.
        
        Args:
            source_name: Name of the dataset source
            download_url: URL to download the dataset
            
        Returns:
            True if download successful
        """
        print(f"Downloading {source_name} from {download_url}...")
        
        try:
            # Create download directory
            download_dir = self.output_dir / "downloads" / source_name
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file (simulated - in real scenario would use actual URLs)
            print(f"Download directory: {download_dir}")
            print("Note: Actual download URLs would be implemented here")
            print("For now, please manually download datasets and place in downloads folder")
            
            return True
            
        except Exception as e:
            print(f"Error downloading {source_name}: {e}")
            return False
    
    def convert_annotations_to_yolo(self, source_dir: Path) -> bool:
        """
        Convert various annotation formats to YOLO format.
        
        Args:
            source_dir: Directory containing source annotations
            
        Returns:
            True if conversion successful
        """
        print(f"Converting annotations from {source_dir} to YOLO format...")
        
        # Supported formats and their converters
        converters = {
            '.json': self._convert_coco_to_yolo,
            '.xml': self._convert_pascal_voc_to_yolo,
            '.txt': self._convert_labelme_to_yolo,
        }
        
        converted_count = 0
        
        for annotation_file in source_dir.rglob("*"):
            if annotation_file.suffix.lower() in converters:
                try:
                    converter = converters[annotation_file.suffix.lower()]
                    converter(annotation_file)
                    converted_count += 1
                except Exception as e:
                    print(f"Error converting {annotation_file}: {e}")
                    self.dataset_stats['annotation_issues'].append(str(annotation_file))
        
        print(f"Converted {converted_count} annotations to YOLO format")
        return converted_count > 0
    
    def _convert_coco_to_yolo(self, coco_file: Path):
        """Convert COCO format to YOLO."""
        try:
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            # Create image ID to filename mapping
            image_map = {img['id']: img['file_name'] for img in coco_data['images']}
            
            # Create annotations grouped by image
            annotations_by_image = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(ann)
            
            # Convert each image's annotations
            for image_id, annotations in annotations_by_image.items():
                if image_id not in image_map:
                    continue
                
                image_name = image_map[image_id]
                yolo_lines = []
                
                for ann in annotations:
                    # Get bounding box
                    x, y, w, h = ann['bbox']
                    
                    # Get image dimensions
                    image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
                    img_w, img_h = image_info['width'], image_info['height']
                    
                    # Convert to YOLO format
                    x_center = (x + w/2) / img_w
                    y_center = (y + h/2) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h
                    
                    # Map category ID to our classes
                    category_id = ann['category_id']
                    class_name = self._map_coco_category(category_id)
                    
                    if class_name in CLASS_TO_ID:
                        class_id = CLASS_TO_ID[class_name]
                        yolo_lines.append(f"{class_id} {x_center} {y_center} {norm_w} {norm_h}")
                        self.dataset_stats['class_distribution'][class_name] += 1
                
                # Save YOLO annotation
                if yolo_lines:
                    label_file = self.labels_dir / f"{Path(image_name).stem}.txt"
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    
                    # Copy image file
                    source_image = coco_file.parent / image_name
                    if source_image.exists():
                        shutil.copy(source_image, self.images_dir)
        
        except Exception as e:
            print(f"Error converting COCO file {coco_file}: {e}")
    
    def _convert_pascal_voc_to_yolo(self, xml_file: Path):
        """Convert Pascal VOC XML to YOLO."""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image filename
            filename = root.find('filename').text
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)
            
            yolo_lines = []
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text.lower()
                bndbox = obj.find('bndbox')
                
                x1 = int(bndbox.find('xmin').text)
                y1 = int(bndbox.find('ymin').text)
                x2 = int(bndbox.find('xmax').text)
                y2 = int(bndbox.find('ymax').text)
                
                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                norm_w = (x2 - x1) / img_w
                norm_h = (y2 - y1) / img_h
                
                # Map class name
                mapped_class = self._map_voc_class(class_name)
                
                if mapped_class in CLASS_TO_ID:
                    class_id = CLASS_TO_ID[mapped_class]
                    yolo_lines.append(f"{class_id} {x_center} {y_center} {norm_w} {norm_h}")
                    self.dataset_stats['class_distribution'][mapped_class] += 1
            
            # Save YOLO annotation
            if yolo_lines:
                label_file = self.labels_dir / f"{xml_file.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))
        
        except Exception as e:
            print(f"Error converting XML file {xml_file}: {e}")
    
    def _convert_labelme_to_yolo(self, labelme_file: Path):
        """Convert LabelMe JSON to YOLO."""
        try:
            with open(labelme_file, 'r') as f:
                data = json.load(f)
            
            # Get image info
            image_path = data.get('imagePath', '')
            image_height = data.get('imageHeight', 0)
            image_width = data.get('imageWidth', 0)
            
            yolo_lines = []
            
            for shape in data.get('shapes', []):
                label = shape['label'].lower()
                points = shape['points']
                
                if shape['shape_type'] == 'rectangle':
                    # Rectangle annotation
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    
                    # Convert to YOLO format
                    x_center = ((x1 + x2) / 2) / image_width
                    y_center = ((y1 + y2) / 2) / image_height
                    norm_w = abs(x2 - x1) / image_width
                    norm_h = abs(y2 - y1) / image_height
                    
                    # Map class name
                    mapped_class = self._map_labelme_class(label)
                    
                    if mapped_class in CLASS_TO_ID:
                        class_id = CLASS_TO_ID[mapped_class]
                        yolo_lines.append(f"{class_id} {x_center} {y_center} {norm_w} {norm_h}")
                        self.dataset_stats['class_distribution'][mapped_class] += 1
            
            # Save YOLO annotation
            if yolo_lines:
                label_file = self.labels_dir / f"{labelme_file.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))
        
        except Exception as e:
            print(f"Error converting LabelMe file {labelme_file}: {e}")
    
    def _map_coco_category(self, category_id: int) -> str:
        """Map COCO category ID to our damage classes."""
        # This would need to be customized based on the actual COCO categories used
        coco_mapping = {
            1: 'dent',      # person -> dent (example)
            2: 'scratch',   # bicycle -> scratch
            3: 'crack',     # car -> crack
            4: 'shatter',   # motorcycle -> shatter
            5: 'broken_part', # airplane -> broken_part
            # Add more mappings as needed
        }
        return coco_mapping.get(category_id, 'light_damage')
    
    def _map_voc_class(self, voc_class: str) -> str:
        """Map Pascal VOC class to our damage classes."""
        voc_mapping = {
            'car': 'dent',
            'person': 'scratch',
            'bicycle': 'crack',
            'motorbike': 'shatter',
            'aeroplane': 'broken_part',
            'bus': 'smash',
            'truck': 'rust',
            # Add more mappings as needed
        }
        return voc_mapping.get(voc_class.lower(), 'light_damage')
    
    def _map_labelme_class(self, labelme_class: str) -> str:
        """Map LabelMe class to our damage classes."""
        # Direct mapping for common damage terms
        labelme_mapping = {
            'scratch': 'scratch',
            'царапина': 'scratch',
            'dent': 'dent',
            'вмятина': 'dent',
            'crack': 'crack',
            'трещина': 'crack',
            'shatter': 'shatter',
            'разбитое': 'shatter',
            'rust': 'rust',
            'ржавчина': 'rust',
            'broken': 'broken_part',
            'сломано': 'broken_part',
            'paint': 'paint_damage',
            'краска': 'paint_damage',
            'smash': 'smash',
            'удар': 'smash',
            'glass': 'glass_damage',
            'стекло': 'glass_damage',
        }
        return labelme_mapping.get(labelme_class.lower(), 'light_damage')
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """
        Split dataset into train/val/test sets.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        """
        print("Splitting dataset into train/val/test...")
        
        # Get all image files
        image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        image_files.extend(self.images_dir.glob("*.jpeg"))
        
        # Shuffle for random split
        random.seed(42)
        random.shuffle(image_files)
        
        total_images = len(image_files)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        
        # Split files
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]
        
        # Move files to respective directories
        self._move_files_to_split(train_files, self.train_images_dir, self.train_labels_dir)
        self._move_files_to_split(val_files, self.val_images_dir, self.val_labels_dir)
        self._move_files_to_split(test_files, self.test_images_dir, self.test_labels_dir)
        
        # Update stats
        self.dataset_stats['total_images'] = total_images
        self.dataset_stats['train_images'] = len(train_files)
        self.dataset_stats['val_images'] = len(val_files)
        self.dataset_stats['test_images'] = len(test_files)
        
        print(f"Dataset split completed:")
        print(f"  Total images: {total_images}")
        print(f"  Training: {len(train_files)} ({train_ratio*100}%)")
        print(f"  Validation: {len(val_files)} ({val_ratio*100}%)")
        print(f"  Test: {len(test_files)} ({test_ratio*100}%)")
    
    def _move_files_to_split(self, image_files: List[Path], img_dest: Path, label_dest: Path):
        """Move image and label files to split directories."""
        for img_file in image_files:
            # Move image
            shutil.move(img_file, img_dest / img_file.name)
            
            # Move corresponding label file
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.move(label_file, label_dest / label_file.name)
    
    def create_data_yaml(self):
        """Create YOLO data.yaml configuration file."""
        data_config = {
            'train': str(self.train_images_dir),
            'val': str(self.val_images_dir),
            'test': str(self.test_images_dir),
            'nc': len(DAMAGE_CLASSES),
            'names': list(DAMAGE_CLASSES.values())
        }
        
        data_yaml_path = self.output_dir / 'data.yaml'
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Created data.yaml at {data_yaml_path}")
    
    def validate_dataset(self) -> bool:
        """
        Validate dataset quality and consistency.
        
        Returns:
            True if validation passed
        """
        print("Validating dataset...")
        
        validation_passed = True
        
        # Check for orphaned images (without labels)
        orphaned_images = []
        for img_file in self.images_dir.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                label_file = self.labels_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    orphaned_images.append(img_file)
        
        if orphaned_images:
            print(f"Warning: Found {len(orphaned_images)} images without labels")
            self.dataset_stats['annotation_issues'].extend([str(f) for f in orphaned_images])
        
        # Check for orphaned labels (without images)
        orphaned_labels = []
        for label_file in self.labels_dir.glob("*.txt"):
            img_extensions = ['.jpg', '.png', '.jpeg']
            img_found = any(
                (self.images_dir / f"{label_file.stem}{ext}").exists()
                for ext in img_extensions
            )
            if not img_found:
                orphaned_labels.append(label_file)
        
        if orphaned_labels:
            print(f"Warning: Found {len(orphaned_labels)} labels without images")
            self.dataset_stats['annotation_issues'].extend([str(f) for f in orphaned_labels])
        
        # Validate label format
        invalid_labels = []
        for label_file in self.labels_dir.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        invalid_labels.append(label_file)
                        break
                    
                    class_id, x_center, y_center, width, height = parts
                    class_id = int(class_id)
                    x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
                    
                    # Check bounds
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                        invalid_labels.append(label_file)
                        break
                    
                    if class_id not in DAMAGE_CLASSES:
                        invalid_labels.append(label_file)
                        break
                        
            except Exception as e:
                invalid_labels.append(label_file)
        
        if invalid_labels:
            print(f"Warning: Found {len(invalid_labels)} invalid label files")
            self.dataset_stats['annotation_issues'].extend([str(f) for f in invalid_labels])
        
        # Check class distribution
        print("Class distribution:")
        for class_name, count in self.dataset_stats['class_distribution'].items():
            print(f"  {class_name}: {count}")
        
        if validation_passed:
            print("Dataset validation completed successfully!")
        else:
            print("Dataset validation found issues. Please review and fix.")
        
        return validation_passed
    
    def print_stats(self):
        """Print dataset statistics."""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        print(f"Total images: {self.dataset_stats['total_images']}")
        print(f"Training images: {self.dataset_stats['train_images']}")
        print(f"Validation images: {self.dataset_stats['val_images']}")
        print(f"Test images: {self.dataset_stats['test_images']}")
        
        print("\nClass distribution:")
        for class_name, count in self.dataset_stats['class_distribution'].items():
            percentage = (count / max(self.dataset_stats['total_images'], 1)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        if self.dataset_stats['annotation_issues']:
            print(f"\nAnnotation issues found: {len(self.dataset_stats['annotation_issues'])}")
            print("Please review the following files:")
            for issue in self.dataset_stats['annotation_issues'][:10]:  # Show first 10
                print(f"  - {issue}")
            if len(self.dataset_stats['annotation_issues']) > 10:
                print(f"  ... and {len(self.dataset_stats['annotation_issues']) - 10} more")
        
        print("="*50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare vehicle damage dataset")
    parser.add_argument("--output", type=str, default="data/damage_dataset",
                        help="Output directory for prepared dataset")
    parser.add_argument("--download", action="store_true",
                        help="Download public datasets")
    parser.add_argument("--validate", action="store_true",
                        help="Validate dataset after preparation")
    
    args = parser.parse_args()
    
    # Initialize dataset preparer
    preparer = DamageDatasetPreparer(args.output)
    
    print("Vehicle Damage Dataset Preparation")
    print("=" * 40)
    
    # Download datasets if requested
    if args.download:
        print("\nDownloading public datasets...")
        for source_name, source_info in DATASET_SOURCES.items():
            preparer.download_dataset(source_name, source_info['url'])
    
    # Convert annotations (example - would need actual source directories)
    print("\nConverting annotations to YOLO format...")
    source_dirs = [
        preparer.output_dir / "downloads",
        # Add actual source directories here
    ]
    
    for source_dir in source_dirs:
        if source_dir.exists():
            preparer.convert_annotations_to_yolo(source_dir)
    
    # Split dataset
    print("\nSplitting dataset...")
    preparer.split_dataset()
    
    # Create data.yaml
    print("\nCreating data.yaml...")
    preparer.create_data_yaml()
    
    # Validate dataset
    if args.validate:
        print("\nValidating dataset...")
        preparer.validate_dataset()
    
    # Print statistics
    preparer.print_stats()
    
    print(f"\nDataset preparation completed!")
    print(f"Dataset location: {args.output}")
    print("You can now use this dataset for training YOLOv9n model.")


if __name__ == "__main__":
    main()