"""
Unit tests for data loading and preprocessing
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path

from src.data.data_loader import ConfigLoader, VehicleDataset, DataLoaderFactory
from src.data.data_preprocessing import DataPreprocessor, ImageTransformer


class TestConfigLoader:
    """Tests for ConfigLoader"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = ConfigLoader()
        assert config.config is not None
        assert isinstance(config.config, dict)
    
    def test_get_nested_value(self):
        """Test getting nested configuration values"""
        config = ConfigLoader()
        image_width = config.get('image.width')
        assert image_width is not None
        assert isinstance(image_width, int)
    
    def test_get_default_value(self):
        """Test default value for missing keys"""
        config = ConfigLoader()
        value = config.get('nonexistent.key', default='default_value')
        assert value == 'default_value'


class TestImageTransformer:
    """Tests for ImageTransformer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.transformer = ImageTransformer()
        self.sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_train_transforms(self):
        """Test training transforms"""
        transforms = self.transformer.get_train_transforms()
        transformed = transforms(image=self.sample_image)
        
        assert 'image' in transformed
        assert isinstance(transformed['image'], torch.Tensor)
        assert transformed['image'].shape[0] == 3  # Channels first
    
    def test_valid_transforms(self):
        """Test validation transforms"""
        transforms = self.transformer.get_valid_transforms()
        transformed = transforms(image=self.sample_image)
        
        assert 'image' in transformed
        assert isinstance(transformed['image'], torch.Tensor)
    
    def test_output_dimensions(self):
        """Test output image dimensions"""
        transforms = self.transformer.get_valid_transforms()
        transformed = transforms(image=self.sample_image)
        
        expected_height = self.transformer.img_height
        expected_width = self.transformer.img_width
        
        assert transformed['image'].shape[1] == expected_height
        assert transformed['image'].shape[2] == expected_width


class TestDataPreprocessor:
    """Tests for DataPreprocessor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = DataPreprocessor()
        self.sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_preprocess_image(self):
        """Test single image preprocessing"""
        processed = self.preprocessor.preprocess_image(self.sample_image)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.dim() == 3
        assert processed.shape[0] == 3  # Channels
    
    def test_preprocess_batch(self):
        """Test batch preprocessing"""
        images = [self.sample_image for _ in range(4)]
        processed = self.preprocessor.preprocess_batch(images)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.dim() == 4
        assert processed.shape[0] == 4  # Batch size
    
    def test_resize_image(self):
        """Test image resizing"""
        resized = DataPreprocessor.resize_image(self.sample_image, 224, 224)
        
        assert resized.shape[0] == 224
        assert resized.shape[1] == 224
    
    def test_normalize_image(self):
        """Test image normalization"""
        normalized = DataPreprocessor.normalize_image(self.sample_image)
        
        assert normalized.dtype == np.float32
        assert normalized.min() >= -3  # Reasonable range after normalization
        assert normalized.max() <= 3
    
    def test_apply_clahe(self):
        """Test CLAHE application"""
        enhanced = DataPreprocessor.apply_clahe(self.sample_image)
        
        assert enhanced.shape == self.sample_image.shape
        assert enhanced.dtype == self.sample_image.dtype
    
    def test_validate_image_valid(self):
        """Test image validation with valid image"""
        is_valid, message = self.preprocessor.validate_image(self.sample_image)

        assert is_valid is True
        assert message == "Valid image"

    def test_validate_image_none(self):
        """Test image validation with None"""
        is_valid, message = self.preprocessor.validate_image(None)

        assert is_valid is False
        assert "None" in message


class TestVehicleDataset:
    """Tests for VehicleDataset"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock data structure
        images_dir = Path(self.temp_dir) / "images"
        labels_dir = Path(self.temp_dir) / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        # Create mock images and labels
        for i in range(5):
            # Create dummy image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = images_dir / f"image_{i}.jpg"
            import cv2
            cv2.imwrite(str(img_path), img)
            
            # Create label file
            label_path = labels_dir / f"image_{i}.txt"
            with open(label_path, 'w') as f:
                f.write(f"{i % 4} 0.5 0.5 0.5 0.5\n")
        
        self.temp_dir_path = self.temp_dir
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_loading(self):
        """Test dataset loading"""
        transformer = ImageTransformer()
        dataset = VehicleDataset(
            data_dir=self.temp_dir_path,
            transform=transformer.get_valid_transforms()
        )
        
        assert len(dataset) == 5
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__"""
        transformer = ImageTransformer()
        dataset = VehicleDataset(
            data_dir=self.temp_dir_path,
            transform=transformer.get_valid_transforms()
        )
        
        if len(dataset) > 0:
            image, label = dataset[0]
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, int)
    
    def test_class_distribution(self):
        """Test class distribution calculation"""
        dataset = VehicleDataset(data_dir=self.temp_dir_path)
        distribution = dataset.get_class_distribution()
        
        assert isinstance(distribution, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])