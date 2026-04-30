"""
Feature Engineering Module for Vehicle Type Classification
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torchvision import models
from loguru import logger

from src.data.data_loader import ConfigLoader


class FeatureExtractor:
    """Extract features from images using pre-trained models"""
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        pretrained: bool = True,
        device: str = None
    ):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(pretrained)
        self.model.eval()
        logger.info(f"Feature extractor initialized with {model_name} on {self.device}")
    
    def _load_model(self, pretrained: bool) -> nn.Module:
        """Load pre-trained model for feature extraction"""
        if self.model_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            # Remove the classifier to get features
            model.classifier = nn.Identity()
        elif self.model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Identity()
        elif self.model_name == 'mobilenet_v3':
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.mobilenet_v3_large(weights=weights)
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model.to(self.device)
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from a batch of images"""
        images = images.to(self.device)
        features = self.model(images)
        return features.cpu().numpy()
    
    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features"""
        feature_dims = {
            'efficientnet_b0': 1280,
            'resnet50': 2048,
            'mobilenet_v3': 960
        }
        return feature_dims.get(self.model_name, 1280)


class FeatureBuilder:
    """Build and combine various features for classification"""
    
    def __init__(self, config: ConfigLoader = None):
        self.config = config or ConfigLoader()
        self.feature_extractor = None
    
    def initialize_extractor(self, model_name: str = None):
        """Initialize the feature extractor"""
        model_name = model_name or self.config.get('model.name', 'efficientnet_b0')
        self.feature_extractor = FeatureExtractor(model_name=model_name)
    
    def extract_color_histogram(
        self,
        image: np.ndarray,
        bins: int = 32
    ) -> np.ndarray:
        """Extract color histogram features"""
        features = []
        
        for i in range(3):  # RGB channels
            hist = cv2.calcHist(
                [image],
                [i],
                None,
                [bins],
                [0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        return np.array(features)
    
    def extract_hog_features(
        self,
        image: np.ndarray,
        cell_size: Tuple[int, int] = (8, 8),
        block_size: Tuple[int, int] = (2, 2),
        num_bins: int = 9
    ) -> np.ndarray:
        """Extract HOG (Histogram of Oriented Gradients) features"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize to standard size
        gray = cv2.resize(gray, (128, 128))
        
        # Calculate HOG features
        win_size = gray.shape
        block_size_pixels = (cell_size[0] * block_size[0], cell_size[1] * block_size[1])
        block_stride = cell_size
        
        hog = cv2.HOGDescriptor(
            _winSize=win_size,
            _blockSize=block_size_pixels,
            _blockStride=block_stride,
            _cellSize=cell_size,
            _nbins=num_bins
        )
        
        features = hog.compute(gray)
        return features.flatten()
    
    def extract_texture_features(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Extract texture features using Gabor filters"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (128, 128))
        
        features = []
        
        # Apply Gabor filters with different orientations and frequencies
        for theta in np.arange(0, np.pi, np.pi / 4):
            for sigma in [1, 3]:
                for lambd in [np.pi / 4, np.pi / 2]:
                    kernel = cv2.getGaborKernel(
                        (21, 21),
                        sigma=sigma,
                        theta=theta,
                        lambd=lambd,
                        gamma=0.5,
                        psi=0,
                        ktype=cv2.CV_32F
                    )
                    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                    features.append(filtered.mean())
                    features.append(filtered.std())
        
        return np.array(features)
    
    def extract_shape_features(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Extract shape-based features"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        features = []
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Area
            area = cv2.contourArea(largest_contour)
            features.append(area)
            
            # Perimeter
            perimeter = cv2.arcLength(largest_contour, True)
            features.append(perimeter)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            features.append(aspect_ratio)
            
            # Extent
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            features.append(extent)
            
            # Solidity
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            features.append(solidity)
            
            # Hu moments
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend(hu_moments)
        else:
            # Return zeros if no contours found
            features = [0] * 12
        
        return np.array(features)
    
    def extract_edge_features(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Extract edge-based features"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (128, 128))
        
        # Sobel edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Canny edges
        canny = cv2.Canny(gray, 50, 150)
        
        features = [
            sobel_x.mean(),
            sobel_x.std(),
            sobel_y.mean(),
            sobel_y.std(),
            canny.mean(),
            canny.std(),
            np.sqrt(sobel_x**2 + sobel_y**2).mean()  # Gradient magnitude
        ]
        
        return np.array(features)
    
    def build_features(
        self,
        image: np.ndarray,
        include_deep: bool = True,
        include_traditional: bool = True
    ) -> np.ndarray:
        """Build comprehensive feature vector"""
        all_features = []
        
        if include_traditional:
            # Color features
            color_features = self.extract_color_histogram(image)
            all_features.append(color_features)
            
            # Texture features
            texture_features = self.extract_texture_features(image)
            all_features.append(texture_features)
            
            # Shape features
            shape_features = self.extract_shape_features(image)
            all_features.append(shape_features)
            
            # Edge features
            edge_features = self.extract_edge_features(image)
            all_features.append(edge_features)
        
        # Concatenate all features
        combined_features = np.concatenate(all_features) if all_features else np.array([])
        
        return combined_features
    
    def build_batch_features(
        self,
        images: List[np.ndarray],
        include_deep: bool = True,
        include_traditional: bool = True
    ) -> np.ndarray:
        """Build features for a batch of images"""
        batch_features = [
            self.build_features(img, include_deep, include_traditional)
            for img in images
        ]
        return np.array(batch_features)