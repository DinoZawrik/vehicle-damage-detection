#!/usr/bin/env python3
"""
Script for downloading car damage detection dataset from Kaggle
"""

import os
import subprocess
import sys
from pathlib import Path

def check_kaggle_setup():
    """Check if Kaggle is properly configured"""
    try:
        # Check if kaggle.json exists
        kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
        if not kaggle_config.exists():
            print("❌ Kaggle API credentials not found!")
            print("Please:")
            print("1. Sign up for Kaggle: https://www.kaggle.com/")
            print("2. Go to Account -> API -> Create New API Token")
            print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
            return False
        
        # Test kaggle command
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Kaggle CLI not installed or not working")
            print("Install with: pip install kaggle")
            return False
            
        print("✅ Kaggle is properly configured")
        return True
        
    except Exception as e:
        print(f"❌ Error checking Kaggle setup: {e}")
        return False

def download_car_damage_dataset():
    """Download car damage detection dataset"""
    datasets_to_try = [
        {
            'name': 'car-damage-detection',
            'url': 'datalawyer/car-damage-detection',
            'description': 'Car Damage Detection Dataset'
        },
        {
            'name': 'vehicle-damage-assessment',
            'url': 'roboflow/vehicle-damage-assessment',
            'description': 'Vehicle Damage Assessment'
        },
        {
            'name': 'auto-damage-detection',
            'url': 'imranispeed/auto-damage-detection',
            'description': 'Automobile Damage Detection'
        }
    ]
    
    base_dir = Path('data/kaggle_dataset')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in datasets_to_try:
        print(f"\n📥 Trying to download: {dataset['description']}")
        print(f"   Dataset URL: {dataset['url']}")
        
        try:
            # Create dataset directory
            dataset_dir = base_dir / dataset['name']
            dataset_dir.mkdir(exist_ok=True)
            
            # Download dataset
            cmd = ['kaggle', 'datasets', 'download', '-d', dataset['url'], '-p', str(dataset_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully downloaded {dataset['description']}")
                
                # Extract if needed
                if (dataset_dir / f"{dataset['url'].split('/')[-1]}.zip").exists():
                    print("📦 Extracting dataset...")
                    import zipfile
                    with zipfile.ZipFile(dataset_dir / f"{dataset['url'].split('/')[-1]}.zip", 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
                    print("✅ Dataset extracted")
                
                return dataset_dir
                
            else:
                print(f"❌ Failed to download: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error downloading {dataset['description']}: {e}")
    
    return None

def organize_dataset(dataset_dir):
    """Organize downloaded dataset into YOLO format"""
    if not dataset_dir or not dataset_dir.exists():
        print("❌ No dataset directory found")
        return False
    
    # Create YOLO format structure
    yolo_dir = Path('data/yolo_dataset_kaggle')
    images_train = yolo_dir / 'images' / 'train'
    images_val = yolo_dir / 'images' / 'val'
    labels_train = yolo_dir / 'labels' / 'train'
    labels_val = yolo_dir / 'labels' / 'val'
    
    for dir_path in [images_train, images_val, labels_train, labels_val]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Organizing dataset in YOLO format: {yolo_dir}")
    
    # Look for images and annotations
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    annotation_extensions = ['.txt', '.xml', '.json']
    
    images_found = []
    annotations_found = []
    
    # Recursively find images and annotations
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in image_extensions:
                images_found.append(file_path)
            elif file_path.suffix.lower() in annotation_extensions:
                annotations_found.append(file_path)
    
    print(f"📸 Found {len(images_found)} images")
    print(f"📝 Found {len(annotations_found)} annotation files")
    
    if len(images_found) == 0:
        print("❌ No images found in dataset")
        return False
    
    # Copy images (80% train, 20% val)
    import random
    random.shuffle(images_found)
    
    split_idx = int(0.8 * len(images_found))
    train_images = images_found[:split_idx]
    val_images = images_found[split_idx:]
    
    def copy_with_progress(src_files, dst_dir, desc):
        copied = 0
        for src_file in src_files:
            try:
                dst_file = dst_dir / src_file.name
                import shutil
                shutil.copy2(src_file, dst_file)
                copied += 1
                if copied % 10 == 0:
                    print(f"   {desc}: {copied}/{len(src_files)} files copied")
            except Exception as e:
                print(f"   Warning: Could not copy {src_file}: {e}")
        
        print(f"✅ {desc}: {copied}/{len(src_files)} files copied successfully")
        return copied
    
    train_copied = copy_with_progress(train_images, images_train, "Train images")
    val_copied = copy_with_progress(val_images, images_val, "Validation images")
    
    # Create data.yaml
    classes = ['scratch', 'dent', 'crack', 'shatter', 'rust', 'broken_part', 'paint_damage']
    data_yaml = yolo_dir / 'data.yaml'
    
    with open(data_yaml, 'w') as f:
        f.write(f"""train: {images_train.absolute()}
val: {images_val.absolute()}

nc: {len(classes)}
names: {classes}
""")
    
    print(f"✅ Created YOLO dataset structure:")
    print(f"   Train images: {train_copied}")
    print(f"   Val images: {val_copied}")
    print(f"   Classes: {classes}")
    print(f"   Data.yaml: {data_yaml}")
    
    return True

def main():
    print("🚗 Car Damage Detection Dataset Preparation")
    print("=" * 50)
    
    # Check Kaggle setup
    if not check_kaggle_setup():
        print("\n💡 Alternative: You can manually download datasets from:")
        print("   - https://www.kaggle.com/datasets/datalawyer/car-damage-detection")
        print("   - https://www.kaggle.com/datasets/imranispeed/auto-damage-detection")
        print("   - https://public.roboflow.com/classification/car-damage")
        sys.exit(1)
    
    # Download dataset
    print("\n📥 Downloading dataset...")
    dataset_dir = download_car_damage_dataset()
    
    if not dataset_dir:
        print("\n❌ No datasets could be downloaded")
        print("💡 Try manual download and place images in data/kaggle_dataset/")
        sys.exit(1)
    
    # Organize dataset
    print("\n📁 Organizing dataset...")
    if organize_dataset(dataset_dir):
        print("\n✅ Dataset preparation completed!")
        print(f"📁 Dataset location: data/yolo_dataset_kaggle/")
    else:
        print("\n❌ Dataset organization failed")

if __name__ == "__main__":
    main()