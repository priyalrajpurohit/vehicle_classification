"""
Unit tests for feature engineering
"""

import pytest
import numpy as np
import torch

from src.features.feature_builder import FeatureBuilder, FeatureExtractor


class TestFeatureExtractor:
    """Tests for FeatureExtractor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_feature_extractor_initialization(self):
        """Test feature extractor initialization"""
        extractor = FeatureExtractor(model_name='efficientnet_b0', pretrained=False)
        
        assert extractor.model is not None
        assert extractor.model_name == 'efficientnet_b0'
    
    def test_feature_dimension(self):
        """Test feature dimension getter"""
        extractor = FeatureExtractor(model_name='efficientnet_b0', pretrained=False)
        dim = extractor.get_feature_dim()
        
        assert dim == 1280  # EfficientNet-B0 feature dimension


class TestFeatureBuilder:
    """Tests for FeatureBuilder"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.builder = FeatureBuilder()
        self.sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_color_histogram(self):
        """Test color histogram extraction"""
        features = self.builder.extract_color_histogram(self.sample_image)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 96  # 32 bins * 3 channels
    
    def test_texture_features(self):
        """Test texture feature extraction"""
        features = self.builder.extract_texture_features(self.sample_image)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_shape_features(self):
        """Test shape feature extraction"""
        features = self.builder.extract_shape_features(self.sample_image)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 12
    
    def test_edge_features(self):
        """Test edge feature extraction"""
        features = self.builder.extract_edge_features(self.sample_image)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 7
    
    def test_build_features(self):
        """Test combined feature building"""
        features = self.builder.build_features(
            self.sample_image,
            include_deep=False,
            include_traditional=True
        )
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_batch_feature_building(self):
        """Test batch feature building"""
        images = [self.sample_image for _ in range(4)]
        features = self.builder.build_batch_features(
            images,
            include_deep=False,
            include_traditional=True
        )
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])