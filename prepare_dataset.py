"""
Dataset Preparation Script for Vehicle Classification Project

This script organizes your vehicle dataset into the required folder structure:
    traffic_data/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── valid/
        ├── images/
        └── labels/

Supported dataset formats:
1. YOLO format: dataset/train/images/, dataset/train/labels/ (with bounding box annotations)
2. Class-based folders: dataset/car/, dataset/bike/, dataset/truck/, dataset/bus/
3. Single folder with class prefixes: dataset/car_001.jpg, bike_002.jpg
4. Already split: dataset/train/, dataset/valid/ or dataset/test/

Usage:
    python prepare_dataset.py <dataset_path>
    python prepare_dataset.py <dataset_path> --split 0.8
    python prepare_dataset.py <dataset_path> --format yolo
"""

import os
import sys
import shutil
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Target class mapping (4 classes for this project)
CLASS_NAMES = ['car', 'bike', 'truck', 'bus']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# YOLO dataset class mapping (from data_1.yaml - 21 classes to 4 classes)
# Maps original YOLO class indices to our 4 target classes
YOLO_CLASS_MAPPING = {
    0: None,   # ambulance -> skip (or map to car)
    1: None,   # army vehicle -> skip
    2: None,   # auto rickshaw -> skip (3-wheeler)
    3: 1,      # bicycle -> bike
    4: 3,      # bus -> bus
    5: 0,      # car -> car
    6: 2,      # garbagevan -> truck
    7: 2,      # human hauler -> truck
    8: 3,      # minibus -> bus
    9: 0,      # minivan -> car
    10: 1,     # motorbike -> bike
    11: 2,     # pickup -> truck
    12: 0,     # policecar -> car
    13: None,  # rickshaw -> skip
    14: 1,     # scooter -> bike
    15: 0,     # suv -> car
    16: 0,     # taxi -> car
    17: None,  # three wheelers -CNG- -> skip
    18: 2,     # truck -> truck
    19: 0,     # van -> car
    20: None,  # wheelbarrow -> skip
}

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory recursively."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(directory.rglob(f'*{ext}'))
        images.extend(directory.rglob(f'*{ext.upper()}'))
    return images


def detect_dataset_format(dataset_path: Path) -> str:
    """Detect the format of the dataset."""
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    subdir_names = [d.name.lower() for d in subdirs]

    # Check if it's YOLO format (has train/images and train/labels structure)
    for subdir in subdirs:
        if subdir.name.lower() in ['train', 'training', 'valid', 'validation', 'val', 'test']:
            images_dir = subdir / 'images'
            labels_dir = subdir / 'labels'
            if images_dir.exists() and labels_dir.exists():
                # Check if labels are YOLO format (multiple values per line)
                label_files = list(labels_dir.glob('*.txt'))
                if label_files:
                    with open(label_files[0], 'r') as f:
                        first_line = f.readline().strip()
                        if first_line and len(first_line.split()) >= 5:
                            return 'yolo'

    # Check if it's already split (train/valid/test folders)
    if any(name in ['train', 'training', 'valid', 'validation', 'val', 'test'] for name in subdir_names):
        return 'pre_split'

    # Check if it has class folders
    class_folders = [name for name in subdir_names if any(cls in name for cls in CLASS_NAMES)]
    if class_folders:
        return 'class_folders'

    # Check for images with class prefixes in root
    images = get_image_files(dataset_path)
    if images:
        for img in images[:10]:  # Check first 10 images
            name_lower = img.stem.lower()
            if any(cls in name_lower for cls in CLASS_NAMES):
                return 'prefix_naming'

    # Default: try to use folder names as classes
    if subdirs:
        return 'custom_folders'

    return 'flat'


def parse_yolo_label(label_path: Path) -> Optional[int]:
    """Parse YOLO label file and return the most common class (mapped to our 4 classes)."""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            return None

        # Count occurrences of each mapped class
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # car, bike, truck, bus

        for line in lines:
            parts = line.strip().split()
            if parts:
                yolo_class = int(parts[0])
                mapped_class = YOLO_CLASS_MAPPING.get(yolo_class)
                if mapped_class is not None:
                    class_counts[mapped_class] += 1

        # Return the most common class, or None if no valid classes found
        if sum(class_counts.values()) == 0:
            return None

        return max(class_counts, key=class_counts.get)
    except Exception as e:
        print(f"Error parsing {label_path}: {e}")
        return None


def process_yolo_dataset(dataset_path: Path, output_path: Path):
    """Process YOLO format dataset (with bounding box annotations)."""
    print("Detected format: YOLO object detection dataset")
    print("Mapping 21 classes to 4 target classes (car, bike, truck, bus)")

    create_output_structure(output_path)

    stats = {cls: {'train': 0, 'valid': 0} for cls in CLASS_NAMES}
    counters = {cls: 0 for cls in CLASS_NAMES}

    # Map common folder names
    train_folders = ['train', 'training']
    valid_folders = ['valid', 'validation', 'val', 'test']

    for subdir in dataset_path.iterdir():
        if not subdir.is_dir():
            continue

        folder_lower = subdir.name.lower()

        if folder_lower in train_folders:
            dest_type = 'train'
        elif folder_lower in valid_folders:
            dest_type = 'valid'
        else:
            continue

        images_dir = subdir / 'images'
        labels_dir = subdir / 'labels'

        if not images_dir.exists():
            print(f"Warning: Images directory not found: {images_dir}")
            continue

        print(f"\nProcessing {dest_type} split from {subdir}...")

        # Process each image
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Find corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            # Parse YOLO label and map to our classes
            class_idx = parse_yolo_label(label_path)

            if class_idx is None:
                continue

            class_name = CLASS_NAMES[class_idx]

            # Copy image and create simple label
            if copy_image_with_label(img_path, output_path / dest_type, class_idx, counters[class_name]):
                counters[class_name] += 1
                stats[class_name][dest_type] += 1

    return stats


def get_class_from_path(path: Path, format_type: str, folder_class_map: Dict[str, int] = None) -> int:
    """Determine class index from file path."""
    name_lower = path.stem.lower()
    parent_lower = path.parent.name.lower()
    
    # Try to match class from filename
    for cls_name, cls_idx in CLASS_MAP.items():
        if cls_name in name_lower:
            return cls_idx
    
    # Try to match class from parent folder
    for cls_name, cls_idx in CLASS_MAP.items():
        if cls_name in parent_lower:
            return cls_idx
    
    # Use custom folder mapping if provided
    if folder_class_map and parent_lower in folder_class_map:
        return folder_class_map[parent_lower]
    
    return -1  # Unknown class


def create_output_structure(base_path: Path):
    """Create the required output directory structure."""
    dirs = [
        base_path / 'train' / 'images',
        base_path / 'train' / 'labels',
        base_path / 'valid' / 'images',
        base_path / 'valid' / 'labels',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_image_with_label(src_image: Path, dst_dir: Path, class_idx: int, counter: int) -> bool:
    """Copy image to destination and create label file."""
    # Generate unique filename
    ext = src_image.suffix.lower()
    new_name = f"{CLASS_NAMES[class_idx]}_{counter:05d}{ext}"
    
    dst_image = dst_dir / 'images' / new_name
    dst_label = dst_dir / 'labels' / f"{CLASS_NAMES[class_idx]}_{counter:05d}.txt"
    
    try:
        shutil.copy2(src_image, dst_image)
        with open(dst_label, 'w') as f:
            f.write(str(class_idx))
        return True
    except Exception as e:
        print(f"Error copying {src_image}: {e}")
        return False


def process_class_folders(dataset_path: Path, output_path: Path, train_split: float = 0.8):
    """Process dataset with class-based folder structure."""
    print("Detected format: Class-based folders")
    
    create_output_structure(output_path)
    
    stats = {cls: {'train': 0, 'valid': 0} for cls in CLASS_NAMES}
    counters = {cls: 0 for cls in CLASS_NAMES}
    
    for subdir in dataset_path.iterdir():
        if not subdir.is_dir():
            continue
        
        # Determine class from folder name
        folder_lower = subdir.name.lower()
        class_idx = -1
        class_name = None
        
        for cls_name, idx in CLASS_MAP.items():
            if cls_name in folder_lower:
                class_idx = idx
                class_name = cls_name
                break
        
        if class_idx == -1:
            print(f"Warning: Could not determine class for folder '{subdir.name}', skipping...")
            continue
        
        # Get all images in this folder
        images = get_image_files(subdir)
        random.shuffle(images)
        
        # Split into train/valid
        split_idx = int(len(images) * train_split)
        train_images = images[:split_idx]
        valid_images = images[split_idx:]
        
        print(f"  {class_name}: {len(train_images)} train, {len(valid_images)} valid")
        
        # Copy images
        for img in train_images:
            if copy_image_with_label(img, output_path / 'train', class_idx, counters[class_name]):
                counters[class_name] += 1
                stats[class_name]['train'] += 1
        
        for img in valid_images:
            if copy_image_with_label(img, output_path / 'valid', class_idx, counters[class_name]):
                counters[class_name] += 1
                stats[class_name]['valid'] += 1
    
    return stats


def process_pre_split(dataset_path: Path, output_path: Path):
    """Process dataset that's already split into train/valid/test."""
    print("Detected format: Pre-split dataset")
    
    create_output_structure(output_path)
    
    stats = {cls: {'train': 0, 'valid': 0} for cls in CLASS_NAMES}
    counters = {cls: 0 for cls in CLASS_NAMES}
    
    # Map common folder names
    train_folders = ['train', 'training']
    valid_folders = ['valid', 'validation', 'val', 'test']
    
    for subdir in dataset_path.iterdir():
        if not subdir.is_dir():
            continue
        
        folder_lower = subdir.name.lower()
        
        if folder_lower in train_folders:
            dest_type = 'train'
        elif folder_lower in valid_folders:
            dest_type = 'valid'
        else:
            continue
        
        # Process images in this split
        images = get_image_files(subdir)
        
        for img in images:
            class_idx = get_class_from_path(img, 'pre_split')
            if class_idx == -1:
                print(f"Warning: Could not determine class for '{img.name}', skipping...")
                continue
            
            class_name = CLASS_NAMES[class_idx]
            if copy_image_with_label(img, output_path / dest_type, class_idx, counters[class_name]):
                counters[class_name] += 1
                stats[class_name][dest_type] += 1
    
    return stats


def process_flat_or_prefix(dataset_path: Path, output_path: Path, train_split: float = 0.8):
    """Process flat dataset or prefix-named dataset."""
    print("Detected format: Flat/Prefix-named dataset")
    
    create_output_structure(output_path)
    
    stats = {cls: {'train': 0, 'valid': 0} for cls in CLASS_NAMES}
    counters = {cls: 0 for cls in CLASS_NAMES}
    
    # Group images by class
    class_images = {cls: [] for cls in CLASS_NAMES}
    unknown_images = []
    
    images = get_image_files(dataset_path)
    
    for img in images:
        class_idx = get_class_from_path(img, 'prefix_naming')
        if class_idx >= 0:
            class_images[CLASS_NAMES[class_idx]].append(img)
        else:
            unknown_images.append(img)
    
    if unknown_images:
        print(f"Warning: {len(unknown_images)} images could not be classified")
    
    # Process each class
    for class_name, images in class_images.items():
        if not images:
            continue
        
        random.shuffle(images)
        split_idx = int(len(images) * train_split)
        train_images = images[:split_idx]
        valid_images = images[split_idx:]
        
        class_idx = CLASS_MAP[class_name]
        print(f"  {class_name}: {len(train_images)} train, {len(valid_images)} valid")
        
        for img in train_images:
            if copy_image_with_label(img, output_path / 'train', class_idx, counters[class_name]):
                counters[class_name] += 1
                stats[class_name]['train'] += 1
        
        for img in valid_images:
            if copy_image_with_label(img, output_path / 'valid', class_idx, counters[class_name]):
                counters[class_name] += 1
                stats[class_name]['valid'] += 1
    
    return stats


def print_stats(stats: Dict):
    """Print dataset statistics."""
    print("\n" + "="*50)
    print("Dataset Preparation Complete!")
    print("="*50)
    
    total_train = sum(s['train'] for s in stats.values())
    total_valid = sum(s['valid'] for s in stats.values())
    
    print(f"\n{'Class':<10} {'Train':<10} {'Valid':<10} {'Total':<10}")
    print("-"*40)
    for cls in CLASS_NAMES:
        train = stats[cls]['train']
        valid = stats[cls]['valid']
        total = train + valid
        print(f"{cls:<10} {train:<10} {valid:<10} {total:<10}")
    print("-"*40)
    print(f"{'Total':<10} {total_train:<10} {total_valid:<10} {total_train + total_valid:<10}")
    
    print("\nOutput structure:")
    print("  traffic_data/")
    print("  ├── train/")
    print("  │   ├── images/")
    print("  │   └── labels/")
    print("  └── valid/")
    print("      ├── images/")
    print("      └── labels/")


def main():
    parser = argparse.ArgumentParser(description='Prepare vehicle dataset for training')
    parser.add_argument('dataset_path', type=str, help='Path to your dataset')
    parser.add_argument('--output', '-o', type=str, default='traffic_data',
                        help='Output directory (default: traffic_data)')
    parser.add_argument('--split', '-s', type=float, default=0.8,
                        help='Train/validation split ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist!")
        sys.exit(1)
    
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Train/Valid split: {args.split}/{1-args.split:.2f}")
    print()
    
    # Detect format and process
    format_type = detect_dataset_format(dataset_path)

    if format_type == 'yolo':
        stats = process_yolo_dataset(dataset_path, output_path)
    elif format_type == 'class_folders':
        stats = process_class_folders(dataset_path, output_path, args.split)
    elif format_type == 'pre_split':
        stats = process_pre_split(dataset_path, output_path)
    else:
        stats = process_flat_or_prefix(dataset_path, output_path, args.split)

    print_stats(stats)

    print("\nYou can now run training with:")
    print("  python -m src.models.train")


if __name__ == '__main__':
    main()

