"""
Model Training Module for Vehicle Type Classification
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm
from loguru import logger
import json
from datetime import datetime

from src.data.data_loader import ConfigLoader, DataLoaderFactory
from src.data.data_preprocessing import ImageTransformer


class VehicleClassifier(nn.Module):
    """CNN-based Vehicle Classifier using Transfer Learning"""
    
    def __init__(
        self,
        model_name: str = 'google/efficientnet-b0',
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super(VehicleClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load backbone model
        self.backbone, self.feature_dim = self._load_backbone(model_name, pretrained)
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        logger.info(f"Initialized {model_name} with {num_classes} classes")
    
    def _load_backbone(
        self,
        model_name: str,
        pretrained: bool
    ) -> Tuple[nn.Module, int]:
        """Load pre-trained backbone model"""
        
        if model_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            
        elif model_name == 'efficientnet_b3':
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b3(weights=weights)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet50(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            
        elif model_name == 'resnet101':
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet101(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            
        elif model_name == 'mobilenet_v3_large':
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.mobilenet_v3_large(weights=weights)
            feature_dim = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
            
        elif model_name == 'densenet121':
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.densenet121(weights=weights)
            feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return backbone, feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def freeze_backbone(self):
        """Freeze backbone weights for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """Model training orchestrator"""
    
    def __init__(self, config: ConfigLoader = None):
        self.config = config or ConfigLoader()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_model(
        self,
        model_name: str = None,
        num_classes: int = None,
        pretrained: bool = None,
        dropout_rate: float = None
    ):
        """Setup the model"""
        model_name = model_name or self.config.get('model.name', 'efficientnet_b0')
        num_classes = num_classes or self.config.get('model.num_classes', 4)
        pretrained = pretrained if pretrained is not None else self.config.get('model.pretrained', True)
        dropout_rate = dropout_rate or self.config.get('model.dropout_rate', 0.3)
        
        self.model = VehicleClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        logger.info(f"Model setup complete. Trainable params: {self.model.get_num_trainable_params():,}")
    
    def setup_training(
        self,
        learning_rate: float = None,
        weight_decay: float = None
    ):
        """Setup optimizer, scheduler, and loss function"""
        learning_rate = learning_rate or self.config.get('training.learning_rate', 0.001)
        weight_decay = weight_decay or self.config.get('training.weight_decay', 0.0001)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.get('training.reduce_lr_factor', 0.5),
            patience=self.config.get('training.reduce_lr_patience', 5),
            min_lr=self.config.get('training.min_lr', 1e-6)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info("Training setup complete")
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = None,
        save_best: bool = True
    ) -> Dict:
        """Full training loop"""
        epochs = epochs or self.config.get('training.epochs', 50)
        patience = self.config.get('training.early_stopping_patience', 10)
        
        early_stopping = EarlyStopping(patience=patience, mode='min')
        best_val_loss = float('inf')
        best_val_acc = 0.0
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                if save_best:
                    self.save_model()
                    logger.info("Saved best model")
            
            # Early stopping check
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.2f}%")
        
        return self.history
    
    def save_model(self, path: str = None):
        """Save model checkpoint"""
        path = path or self.config.get('model.save_path', 'models/best_model.pth')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': {
                'model_name': self.model.model_name,
                'num_classes': self.model.num_classes,
                'feature_dim': self.model.feature_dim
            }
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = None):
        """Load model checkpoint"""
        path = path or self.config.get('model.save_path', 'models/best_model.pth')
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Setup model if not already done
        if self.model is None:
            model_config = checkpoint.get('config', {})
            self.setup_model(
                model_name=model_config.get('model_name', 'efficientnet_b0'),
                num_classes=model_config.get('num_classes', 4)
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"Model loaded from {path}")
    
    def save_training_history(self, path: str = None):
        """Save training history to JSON"""
        path = path or 'models/training_history.json'
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Training history saved to {path}")


def main():
    """Main training function"""
    # Load configuration
    config = ConfigLoader()

    # Setup data loaders
    transformer = ImageTransformer(config)
    factory = DataLoaderFactory(config)

    try:
        train_loader, val_loader = factory.create_data_loaders(
            train_transform=transformer.get_train_transforms(),
            valid_transform=transformer.get_valid_transforms()
        )
    except ValueError as e:
        logger.error(f"Data loading error: {e}")
        logger.info("Please create the following directory structure:")
        logger.info("  traffic_data/")
        logger.info("  ├── train/")
        logger.info("  │   ├── images/    (put training images here)")
        logger.info("  │   └── labels/    (put label files here)")
        logger.info("  └── valid/")
        logger.info("      ├── images/    (put validation images here)")
        logger.info("      └── labels/    (put label files here)")
        logger.info("")
        logger.info("Label files should contain class index (0=car, 1=bike, 2=truck, 3=bus)")
        return None

    # Setup trainer
    trainer = Trainer(config)
    trainer.setup_model()
    trainer.setup_training()

    # Train model
    history = trainer.train(train_loader, val_loader)

    # Save training history
    trainer.save_training_history()

    return history


if __name__ == "__main__":
    main()