"""
Data Loading Module for Vehicle Type Classification
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from loguru import logger


class ConfigLoader:
    """Load and manage configuration files"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def get(self, key: str, default=None):
        """Get configuration value by key (supports nested keys with dot notation)"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


class VehicleDataset(Dataset):
    """Custom Dataset for Vehicle Images"""
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        class_mapping: Dict[str, int] = None,
        config: ConfigLoader = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.config = config or ConfigLoader()
        
        # Default class mapping
        self.class_mapping = class_mapping or {
            'car': 0,
            'bike': 1,
            'truck': 2,
            'bus': 3
        }
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        
        # Load image paths and labels
        self.images, self.labels = self._load_data()
        logger.info(f"Loaded {len(self.images)} images from {data_dir}")
    
    def _load_data(self) -> Tuple[List[str], List[int]]:
        """Load image paths and corresponding labels"""
        images = []
        labels = []
        
        images_dir = self.data_dir / "images"
        labels_dir = self.data_dir / "labels"
        
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return images, labels
        
        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                # Try to find corresponding label file
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                if label_path.exists():
                    label = self._parse_label_file(label_path)
                    if label is not None:
                        images.append(str(img_path))
                        labels.append(label)
                else:
                    # Try to infer label from directory structure or filename
                    label = self._infer_label(img_path)
                    if label is not None:
                        images.append(str(img_path))
                        labels.append(label)
        
        return images, labels
    
    def _parse_label_file(self, label_path: Path) -> Optional[int]:
        """Parse YOLO-format label file to get class index"""
        try:
            with open(label_path, 'r') as f:
                content = f.readline().strip()
                if content:
                    class_idx = int(content.split()[0])
                    if class_idx < len(self.class_mapping):
                        return class_idx
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing label file {label_path}: {e}")
        return None
    
    def _infer_label(self, img_path: Path) -> Optional[int]:
        """Infer label from image path or filename"""
        filename = img_path.stem.lower()
        parent = img_path.parent.name.lower()
        
        for class_name, class_idx in self.class_mapping.items():
            if class_name in filename or class_name in parent:
                return class_idx
        return None
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            # Return a blank image as fallback
            img_size = self.config.get('image.width', 224)
            image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label
    
    def get_class_name(self, idx: int) -> str:
        """Get class name from index"""
        return self.idx_to_class.get(idx, "unknown")
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset"""
        distribution = {}
        for label in self.labels:
            class_name = self.get_class_name(label)
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


class DataLoaderFactory:
    """Factory class for creating data loaders"""
    
    def __init__(self, config: ConfigLoader = None):
        self.config = config or ConfigLoader()
    
    def create_data_loaders(
        self,
        train_transform=None,
        valid_transform=None,
        batch_size: int = None,
        num_workers: int = 4
    ) -> Tuple[TorchDataLoader, TorchDataLoader]:
        """Create training and validation data loaders"""
        
        batch_size = batch_size or self.config.get('training.batch_size', 32)
        
        train_path = self.config.get('data.train_path', 'traffic_data/train')
        valid_path = self.config.get('data.valid_path', 'traffic_data/valid')
        
        # Create datasets
        train_dataset = VehicleDataset(
            data_dir=train_path,
            transform=train_transform,
            config=self.config
        )

        valid_dataset = VehicleDataset(
            data_dir=valid_path,
            transform=valid_transform,
            config=self.config
        )

        # Check if datasets have data
        if len(train_dataset) == 0:
            raise ValueError(
                f"No training images found in '{train_path}/images'. "
                f"Please ensure the directory exists and contains images."
            )

        if len(valid_dataset) == 0:
            logger.warning(f"No validation images found in '{valid_path}/images'. Using training data for validation.")
            valid_dataset = train_dataset

        # Create data loaders
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True if len(train_dataset) > batch_size else False
        )

        valid_loader = TorchDataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        logger.info(f"Created data loaders - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

        return train_loader, valid_loader


class DataLoader:
    """Main data loader class for the project"""
    
    def __init__(self, config_path: str = None):
        self.config = ConfigLoader(config_path)
        self.factory = DataLoaderFactory(self.config)
    
    def load_yaml_config(self, yaml_path: str = None) -> Dict:
        """Load dataset YAML configuration"""
        yaml_path = yaml_path or self.config.get('data.yaml_path')
        
        try:
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Dataset YAML not found: {yaml_path}")
            return {}
    
    def get_data_loaders(
        self,
        train_transform=None,
        valid_transform=None,
        batch_size: int = None
    ) -> Tuple[TorchDataLoader, TorchDataLoader]:
        """Get training and validation data loaders"""
        return self.factory.create_data_loaders(
            train_transform=train_transform,
            valid_transform=valid_transform,
            batch_size=batch_size
        )
    
    def load_single_image(self, image_path: str) -> np.ndarray:
        """Load a single image for inference"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes (for Streamlit uploads)"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from bytes")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)