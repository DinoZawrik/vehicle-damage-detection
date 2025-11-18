#!/usr/bin/env python3
"""
Simple script to create a minimal car damage dataset for testing
"""

import os
import shutil
from pathlib import Path
import random

def create_simple_dataset():
    """Create a simple dataset with existing test images"""
    
    # Dataset structure
    base_dir = Path('data/simple_test_dataset')
    images_train = base_dir / 'images' / 'train'
    images_val = base_dir / 'images' / 'val'
    labels_train = base_dir / 'labels' / 'train'
    labels_val = base_dir / 'labels' / 'val'
    
    for dir_path in [images_train, images_val, labels_train, labels_val]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("🚗 Creating simple test dataset...")
    
    # Source images
    source_dirs = [
        Path('data/test_images'),
        Path('uploads')
    ]
    
    images_found = []
    
    # Collect images from source directories
    for source_dir in source_dirs:
        if source_dir.exists():
            for img_file in source_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    images_found.append(img_file)
    
    if len(images_found) == 0:
        print("❌ No images found in source directories")
        return False
    
    print(f"📸 Found {len(images_found)} images")
    
    # Shuffle and split
    random.shuffle(images_found)
    split_idx = int(0.8 * len(images_found))
    train_images = images_found[:split_idx]
    val_images = images_found[split_idx:]
    
    print(f"📊 Split: {len(train_images)} train, {len(val_images)} val")
    
    # Copy images and create dummy annotations
    classes = ['scratch', 'dent', 'crack', 'shatter', 'rust', 'broken_part', 'paint_damage']
    
    def copy_images_with_annotations(src_images, img_dst, label_dst, desc):
        copied = 0
        for src_img in src_images:
            try:
                # Copy image
                dst_img = img_dst / src_img.name
                shutil.copy2(src_img, dst_img)
                
                # Create dummy annotation
                label_file = label_dst / f"{src_img.stem}.txt"
                
                # Random number of damage instances (1-3)
                num_damages = random.randint(1, 3)
                
                with open(label_file, 'w', encoding='utf-8') as f:
                    for _ in range(num_damages):
                        # Random class
                        class_id = random.randint(0, 6)  # 0-6 for 7 classes
                        
                        # Random bbox (normalized)
                        x_center = random.uniform(0.2, 0.8)
                        y_center = random.uniform(0.2, 0.8)
                        width = random.uniform(0.1, 0.4)
                        height = random.uniform(0.1, 0.4)
                        
                        f.write(f"{class_id} {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}\n")
                
                copied += 1
                
            except Exception as e:
                print(f"   Warning: Could not process {src_img}: {e}")
        
        print(f"✅ {desc}: {copied} images processed")
        return copied
    
    train_processed = copy_images_with_annotations(train_images, images_train, labels_train, "Train")
    val_processed = copy_images_with_annotations(val_images, images_val, labels_val, "Validation")
    
    # Create data.yaml
    data_yaml = base_dir / 'data.yaml'
    with open(data_yaml, 'w', encoding='utf-8') as f:
        f.write(f"""train: {images_train.absolute()}
val: {images_val.absolute()}

nc: {len(classes)}
names: {classes}
""")
    
    print(f"\n✅ Simple dataset created!")
    print(f"📁 Location: {base_dir}")
    print(f"📊 Statistics:")
    print(f"   Train images: {train_processed}")
    print(f"   Validation images: {val_processed}")
    print(f"   Total images: {train_processed + val_processed}")
    print(f"   Classes: {len(classes)}")
    print(f"   Data.yaml: {data_yaml}")
    
    return base_dir

def create_demo_images():
    """Create demo images directory for testing"""
    demo_dir = Path('data/demo_images')
    demo_dir.mkdir(exist_ok=True)
    
    source_dirs = [Path('data/test_images'), Path('uploads')]
    
    demo_count = 0
    for source_dir in source_dirs:
        if source_dir.exists():
            for img_file in source_dir.glob('*.{jpg,jpeg,png,JPG,JPEG,PNG}'):
                if demo_count < 10:  # Limit to 10 demo images
                    try:
                        dst_file = demo_dir / img_file.name
                        shutil.copy2(img_file, dst_file)
                        demo_count += 1
                    except Exception as e:
                        print(f"Warning: Could not copy {img_file}: {e}")
    
    if demo_count > 0:
        print(f"🎬 Created demo images: {demo_dir} ({demo_count} images)")
    
    return demo_count

def main():
    print("🔧 Creating Simple Test Dataset")
    print("=" * 40)
    
    # Create simple dataset
    dataset_dir = create_simple_dataset()
    
    # Create demo images
    demo_count = create_demo_images()
    
    if dataset_dir:
        print(f"\n🎉 Dataset creation completed!")
        print(f"\n📋 Next steps:")
        print(f"1. Use dataset for training: {dataset_dir}/data.yaml")
        print(f"2. Test with demo images: data/demo_images/")
        print(f"3. Run training: python scripts/train_yolo.py --data {dataset_dir}/data.yaml")
    else:
        print(f"\n❌ Dataset creation failed")
    
    if demo_count > 0:
        print(f"4. Test SAM segmentation with demo images!")

if __name__ == "__main__":
    main()