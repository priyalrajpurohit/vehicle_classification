"""
Data Preprocessing Module for Vehicle Type Classification
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from loguru import logger

from .data_loader import ConfigLoader


class ImageTransformer:
    """Image transformation utilities using OpenCV and Albumentations"""
    
    def __init__(self, config: ConfigLoader = None):
        self.config = config or ConfigLoader()
        self.img_width = self.config.get('image.width', 224)
        self.img_height = self.config.get('image.height', 224)
        self.mean = self.config.get('image.mean', [0.485, 0.456, 0.406])
        self.std = self.config.get('image.std', [0.229, 0.224, 0.225])
    
    def get_train_transforms(self) -> A.Compose:
        """Get training data augmentation pipeline"""
        aug_config = self.config.get('augmentation.train', {})
        
        transforms = [
            A.Resize(self.img_height, self.img_width),
            A.HorizontalFlip(p=0.5 if aug_config.get('horizontal_flip', True) else 0),
        ]
        
        if aug_config.get('rotation_limit', 0) > 0:
            transforms.append(
                A.Rotate(limit=aug_config['rotation_limit'], p=0.5)
            )
        
        if aug_config.get('brightness_limit', 0) > 0 or aug_config.get('contrast_limit', 0) > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=aug_config.get('brightness_limit', 0.2),
                    contrast_limit=aug_config.get('contrast_limit', 0.2),
                    p=0.5
                )
            )
        
        if aug_config.get('blur_limit', 0) > 0:
            transforms.append(
                A.GaussianBlur(blur_limit=aug_config['blur_limit'], p=0.3)
            )
        
        # Add additional augmentations
        transforms.extend([
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.05),
                A.GridDistortion(num_steps=5, distort_limit=0.05),
            ], p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                fill=0,
                p=0.2
            ),
        ])
        
        # Normalization and tensor conversion
        transforms.extend([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def get_valid_transforms(self) -> A.Compose:
        """Get validation/inference data transformation pipeline"""
        return A.Compose([
            A.Resize(self.img_height, self.img_width),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
    
    def get_inference_transforms(self) -> A.Compose:
        """Get inference transformation pipeline (same as validation)"""
        return self.get_valid_transforms()


class DataPreprocessor:
    """Data preprocessing utilities"""
    
    def __init__(self, config: ConfigLoader = None):
        self.config = config or ConfigLoader()
        self.transformer = ImageTransformer(config)
        self.img_width = self.config.get('image.width', 224)
        self.img_height = self.config.get('image.height', 224)
    
    def preprocess_image(
        self,
        image: np.ndarray,
        for_training: bool = False
    ) -> torch.Tensor:
        """
        Preprocess a single image for model input
        
        Args:
            image: Input image as numpy array (RGB format)
            for_training: Whether to apply training augmentations
        
        Returns:
            Preprocessed image as torch tensor
        """
        if for_training:
            transform = self.transformer.get_train_transforms()
        else:
            transform = self.transformer.get_valid_transforms()
        
        transformed = transform(image=image)
        return transformed['image']
    
    def preprocess_batch(
        self,
        images: List[np.ndarray],
        for_training: bool = False
    ) -> torch.Tensor:
        """Preprocess a batch of images"""
        preprocessed = [
            self.preprocess_image(img, for_training)
            for img in images
        ]
        return torch.stack(preprocessed)
    
    @staticmethod
    def resize_image(
        image: np.ndarray,
        width: int,
        height: int,
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """Resize image using OpenCV"""
        return cv2.resize(image, (width, height), interpolation=interpolation)
    
    @staticmethod
    def normalize_image(
        image: np.ndarray,
        mean: List[float] = None,
        std: List[float] = None
    ) -> np.ndarray:
        """Normalize image with mean and standard deviation"""
        mean = mean or [0.485, 0.456, 0.406]
        std = std or [0.229, 0.224, 0.225]

        image = image.astype(np.float32) / 255.0
        image = (image - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        return image.astype(np.float32)
    
    @staticmethod
    def denormalize_image(
        image: np.ndarray,
        mean: List[float] = None,
        std: List[float] = None
    ) -> np.ndarray:
        """Denormalize image for visualization"""
        mean = mean or [0.485, 0.456, 0.406]
        std = std or [0.229, 0.224, 0.225]
        
        image = image * np.array(std) + np.array(mean)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image)
    
    @staticmethod
    def remove_noise(
        image: np.ndarray,
        method: str = 'bilateral'
    ) -> np.ndarray:
        """Remove noise from image"""
        if method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        elif method == 'nlmeans':
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return image
    
    @staticmethod
    def sharpen_image(image: np.ndarray) -> np.ndarray:
        """Sharpen image using unsharp masking"""
        gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened
    
    @staticmethod
    def detect_edges(
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """Detect edges using Canny edge detector"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return cv2.Canny(gray, low_threshold, high_threshold)
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply a series of enhancement operations"""
        # Remove noise
        image = self.remove_noise(image, method='bilateral')
        
        # Apply CLAHE
        image = self.apply_clahe(image)
        
        # Sharpen
        image = self.sharpen_image(image)
        
        return image
    
    def validate_image(self, image: np.ndarray) -> Tuple[bool, str]:
        """Validate image for processing"""
        if image is None:
            return False, "Image is None"
        
        if not isinstance(image, np.ndarray):
            return False, "Image is not a numpy array"
        
        if len(image.shape) < 2:
            return False, "Image has invalid dimensions"
        
        if image.size == 0:
            return False, "Image is empty"
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False, f"Invalid number of channels: {image.shape[2]}"
        
        return True, "Valid image"


class BatchPreprocessor:
    """Batch preprocessing for training data"""
    
    def __init__(self, config: ConfigLoader = None):
        self.config = config or ConfigLoader()
        self.preprocessor = DataPreprocessor(config)
    
    def process_batch(
        self,
        images: List[np.ndarray],
        labels: List[int],
        augment: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of images and labels"""
        processed_images = self.preprocessor.preprocess_batch(images, for_training=augment)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return processed_images, labels_tensor
    
    def collate_fn(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function for DataLoader"""
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels