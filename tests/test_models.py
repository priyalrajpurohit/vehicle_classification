"""
Unit tests for model training, evaluation, and prediction
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path

from src.models.train import VehicleClassifier, Trainer, EarlyStopping
from src.models.predict import VehiclePredictor
from src.data.data_loader import ConfigLoader


class TestVehicleClassifier:
    """Tests for VehicleClassifier"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = VehicleClassifier(
            model_name='efficientnet_b0',
            num_classes=4,
            pretrained=False
        )
        
        assert model is not None
        assert model.num_classes == 4
    
    def test_model_forward(self):
        """Test model forward pass"""
        model = VehicleClassifier(
            model_name='efficientnet_b0',
            num_classes=4,
            pretrained=False
        )
        
        batch = torch.randn(2, 3, 224, 224)
        output = model(batch)
        
        assert output.shape == (2, 4)
    
    def test_freeze_backbone(self):
        """Test backbone freezing"""
        model = VehicleClassifier(
            model_name='efficientnet_b0',
            num_classes=4,
            pretrained=False
        )
        
        model.freeze_backbone()
        
        # Check that backbone parameters are frozen
        for param in model.backbone.parameters():
            assert param.requires_grad is False
    
    def test_unfreeze_backbone(self):
        """Test backbone unfreezing"""
        model = VehicleClassifier(
            model_name='efficientnet_b0',
            num_classes=4,
            pretrained=False
        )
        
        model.freeze_backbone()
        model.unfreeze_backbone()
        
        # Check that backbone parameters are trainable
        for param in model.backbone.parameters():
            assert param.requires_grad is True
    
    def test_trainable_params(self):
        """Test trainable parameter count"""
        model = VehicleClassifier(
            model_name='efficientnet_b0',
            num_classes=4,
            pretrained=False
        )
        
        num_params = model.get_num_trainable_params()
        assert num_params > 0
    
    @pytest.mark.parametrize("model_name", [
        'efficientnet_b0',
        'resnet50',
        'mobilenet_v3_large'
    ])
    def test_different_backbones(self, model_name):
        """Test different backbone architectures"""
        model = VehicleClassifier(
            model_name=model_name,
            num_classes=4,
            pretrained=False
        )
        
        batch = torch.randn(2, 3, 224, 224)
        output = model(batch)
        
        assert output.shape == (2, 4)


class TestEarlyStopping:
    """Tests for EarlyStopping"""
    
    def test_early_stopping_improvement(self):
        """Test early stopping with improvement"""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
        
        # Improving values
        assert early_stopping(1.0) is False
        assert early_stopping(0.9) is False
        assert early_stopping(0.8) is False
        assert early_stopping.counter == 0
    
    def test_early_stopping_no_improvement(self):
        """Test early stopping without improvement"""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
        
        early_stopping(1.0)
        early_stopping(1.0)
        early_stopping(1.0)
        
        assert early_stopping.counter == 2
        
        result = early_stopping(1.0)
        assert result is True  # Should stop
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode"""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='max')
        
        # Improving values (increasing for max mode)
        assert early_stopping(0.5) is False
        assert early_stopping(0.6) is False
        assert early_stopping(0.7) is False
        assert early_stopping.counter == 0


class TestTrainer:
    """Tests for Trainer"""
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = Trainer()
        
        assert trainer.device is not None
        assert trainer.model is None
    
    def test_setup_model(self):
        """Test model setup"""
        trainer = Trainer()
        trainer.setup_model(
            model_name='efficientnet_b0',
            num_classes=4,
            pretrained=False
        )
        
        assert trainer.model is not None
    
    def test_setup_training(self):
        """Test training setup"""
        trainer = Trainer()
        trainer.setup_model(pretrained=False)
        trainer.setup_training()
        
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.criterion is not None
    
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        trainer = Trainer()
        trainer.setup_model(pretrained=False)
        trainer.setup_training()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            trainer.save_model(temp_path)
            assert os.path.exists(temp_path)
            
            # Load model
            new_trainer = Trainer()
            new_trainer.setup_model(pretrained=False)
            new_trainer.setup_training()
            new_trainer.load_model(temp_path)
            
            assert new_trainer.model is not None
        finally:
            os.unlink(temp_path)


class TestVehiclePredictor:
    """Tests for VehiclePredictor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Create a temporary model
        self.temp_model_path = tempfile.NamedTemporaryFile(suffix='.pth', delete=False).name
        
        model = VehicleClassifier(
            model_name='efficientnet_b0',
            num_classes=4,
            pretrained=False
        )
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'model_name': 'efficientnet_b0',
                'num_classes': 4,
                'feature_dim': 1280
            }
        }
        torch.save(checkpoint, self.temp_model_path)
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        if os.path.exists(self.temp_model_path):
            os.unlink(self.temp_model_path)
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        predictor = VehiclePredictor(self.temp_model_path)
        
        assert predictor.model is not None
    
    def test_predict(self):
        """Test prediction"""
        predictor = VehiclePredictor(self.temp_model_path)
        result = predictor.predict(self.sample_image)
        
        assert 'class' in result
        assert 'confidence' in result
        assert result['class'] in ['car', 'bike', 'truck', 'bus']
        assert 0 <= result['confidence'] <= 1
    
    def test_predict_with_probabilities(self):
        """Test prediction with probabilities"""
        predictor = VehiclePredictor(self.temp_model_path)
        result = predictor.predict(self.sample_image, return_probs=True)
        
        assert 'probabilities' in result
        assert len(result['probabilities']) == 4
    
    def test_predict_from_bytes(self):
        """Test prediction from image bytes"""
        import cv2
        
        predictor = VehiclePredictor(self.temp_model_path)
        
        # Encode image to bytes
        _, encoded = cv2.imencode('.jpg', self.sample_image)
        image_bytes = encoded.tobytes()
        
        result = predictor.predict_from_bytes(image_bytes)
        
        assert 'class' in result
    
    def test_get_model_info(self):
        """Test getting model info"""
        predictor = VehiclePredictor(self.temp_model_path)
        info = predictor.get_model_info()
        
        assert 'model_name' in info
        assert 'num_classes' in info
        assert info['num_classes'] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])