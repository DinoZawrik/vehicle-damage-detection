"""Быстрая подготовка YOLO датасета для обучения pet-проекта."""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np

# Пути
BASE_DIR = Path(".")
TEST_IMAGES_DIR = BASE_DIR / "data" / "test_images"
DATASET_DIR = BASE_DIR / "data" / "yolo_dataset"
TRAIN_IMAGES = DATASET_DIR / "images" / "train"
TRAIN_LABELS = DATASET_DIR / "labels" / "train"
VAL_IMAGES = DATASET_DIR / "images" / "val"
VAL_LABELS = DATASET_DIR / "labels" / "val"

def create_directories():
    """Создать необходимые директории."""
    for path in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
        path.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directories created: {DATASET_DIR}")

def copy_and_augment_images():
    """Скопировать изображения из test_images и создать их варианты."""
    image_files = [f for f in TEST_IMAGES_DIR.glob("*.jpg") if f.is_file()]

    if not image_files:
        print("[WARNING] No images in test_images/")
        return 0

    count = 0
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Оригинальное изображение в train
        train_img_path = TRAIN_IMAGES / img_path.name
        cv2.imwrite(str(train_img_path), img)
        print(f"[OK] Copied: {img_path.name} -> train/")
        count += 1

        # Создать 2 варианта через аугментацию
        # Вариант 1: brightness
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * 1.2  # Увеличить brightness
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        aug1 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        aug1_path = TRAIN_IMAGES / f"{img_path.stem}_bright{img_path.suffix}"
        cv2.imwrite(str(aug1_path), aug1)
        print(f"[OK] Augmentation: brightness -> train/")
        count += 1

        # Вариант 2: gamma correction (darker)
        inv_gamma = 1.0 / 0.8
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        aug2 = cv2.LUT(img, table)

        aug2_path = TRAIN_IMAGES / f"{img_path.stem}_dark{img_path.suffix}"
        cv2.imwrite(str(aug2_path), aug2)
        print(f"[OK] Augmentation: gamma -> train/")
        count += 1

    # Скопировать первое изображение в val
    if image_files:
        img = cv2.imread(str(image_files[0]))
        val_img_path = VAL_IMAGES / image_files[0].name
        cv2.imwrite(str(val_img_path), img)
        print(f"[OK] Copied to val: {image_files[0].name}")

    return count

def create_annotations(num_images):
    """Создать YOLO аннотации для изображений."""
    # Простая аннотация: одно повреждение в центре каждого изображения
    # Класс 0 = scratch, bbox: center_x, center_y, width, height (нормализованные 0-1)

    annotation = "0 0.5 0.5 0.3 0.3\n"  # class_id, x_center, y_center, width, height

    # Для train изображений
    for img_file in TRAIN_IMAGES.glob("*.jpg"):
        label_file = TRAIN_LABELS / f"{img_file.stem}.txt"
        with open(label_file, 'w') as f:
            f.write(annotation)

    # Для val изображений
    for img_file in VAL_IMAGES.glob("*.jpg"):
        label_file = VAL_LABELS / f"{img_file.stem}.txt"
        with open(label_file, 'w') as f:
            f.write(annotation)

    print(f"[OK] Created annotations for {num_images} images")

def create_data_yaml():
    """Создать data.yaml для YOLO."""
    yaml_content = """# YOLO Dataset Configuration
path: data/yolo_dataset
train: images/train
val: images/val

# Number of classes
nc: 7

# Class names
names:
  0: scratch
  1: dent
  2: crack
  3: shatter
  4: rust
  5: broken_part
  6: paint_damage
"""

    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"[OK] Created data.yaml: {yaml_path}")

def main():
    print("[*] Nachaem podgotovku dataseta...")
    print()

    create_directories()
    num_images = copy_and_augment_images()

    if num_images > 0:
        create_annotations(num_images)
        create_data_yaml()
        print()
        print(f"[OK] Gotovo! Podgotovleno {num_images} izobrazheniy dlya obucheniya")
        print(f"     Train: {TRAIN_IMAGES}")
        print(f"     Val: {VAL_IMAGES}")
    else:
        print("[ERROR] Ne udalos podgotovit dataset - net izobrazheniy")

if __name__ == "__main__":
    main()
