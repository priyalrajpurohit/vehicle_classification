"""
Model Evaluation Module for Vehicle Type Classification
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from loguru import logger
import json

from src.data.data_loader import ConfigLoader
from src.models.train import VehicleClassifier


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config: ConfigLoader = None):
        self.config = config or ConfigLoader()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = self.config.get('classes', ['car', 'bike', 'truck', 'bus'])
        self.results = {}
        
        logger.info(f"Evaluator initialized on device: {self.device}")
    
    def load_model(self, model_path: str = None):
        """Load trained model for evaluation"""
        model_path = model_path or self.config.get('model.save_path', 'models/best_model.pth')
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint.get('config', {})
        
        self.model = VehicleClassifier(
            model_name=model_config.get('model_name', 'efficientnet_b0'),
            num_classes=model_config.get('num_classes', 4),
            pretrained=False
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_predictions(
        self,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions for all samples in the data loader"""
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Getting predictions"):
                images = images.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return (
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities)
        )
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics['recall_per_class'] = recall_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics['f1_per_class'] = f1_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # ROC-AUC if probabilities are provided
        if y_prob is not None:
            try:
                # One-vs-rest ROC-AUC
                y_true_onehot = np.eye(len(self.class_names))[y_true]
                metrics['roc_auc_macro'] = roc_auc_score(
                    y_true_onehot, y_prob, average='macro', multi_class='ovr'
                )
                metrics['roc_auc_weighted'] = roc_auc_score(
                    y_true_onehot, y_prob, average='weighted', multi_class='ovr'
                )
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        return metrics
    
    def evaluate(
        self,
        data_loader: DataLoader,
        save_results: bool = True
    ) -> Dict:
        """Perform full evaluation"""
        if self.model is None:
            self.load_model()
        
        logger.info("Starting evaluation...")
        
        # Get predictions
        y_pred, y_true, y_prob = self.get_predictions(data_loader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        self.results = {
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'labels': y_true.tolist(),
            'probabilities': y_prob.tolist()
        }
        
        # Log results
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
        logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
        logger.info(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        
        if save_results:
            self.save_results()
        
        return self.results
    
    def plot_confusion_matrix(
        self,
        save_path: str = None,
        normalize: bool = True
    ):
        """Plot confusion matrix"""
        if 'metrics' not in self.results:
            logger.error("No evaluation results. Run evaluate() first.")
            return
        
        cm = np.array(self.results['metrics']['confusion_matrix'])
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_metrics_per_class(self, save_path: str = None):
        """Plot precision, recall, F1 per class"""
        if 'metrics' not in self.results:
            logger.error("No evaluation results. Run evaluate() first.")
            return
        
        metrics = self.results['metrics']
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, metrics['precision_per_class'], width, label='Precision')
        bars2 = ax.bar(x, metrics['recall_per_class'], width, label='Recall')
        bars3 = ax.bar(x + width, metrics['f1_per_class'], width, label='F1 Score')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Metrics per Class')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8
                )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics plot saved to {save_path}")
        
        plt.close()
    
    def plot_training_history(
        self,
        history: Dict = None,
        save_path: str = None
    ):
        """Plot training history"""
        if history is None:
            history_path = 'models/training_history.json'
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
            else:
                logger.error("No training history found")
                return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train Acc')
        axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Learning rate plot
        axes[2].plot(history['lr'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True)
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.close()
    
    def save_results(self, path: str = None):
        """Save evaluation results to JSON"""
        path = path or 'models/evaluation_results.json'
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {
            'metrics': self.results['metrics'],
            'class_names': self.class_names
        }
        
        with open(path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Evaluation results saved to {path}")
    
    def generate_report(self, output_dir: str = 'reports'):
        """Generate comprehensive evaluation report"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save plots
        self.plot_confusion_matrix(
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
        self.plot_metrics_per_class(
            save_path=os.path.join(output_dir, 'metrics_per_class.png')
        )
        self.plot_training_history(
            save_path=os.path.join(output_dir, 'training_history.png')
        )
        
        # Generate text report
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("VEHICLE TYPE CLASSIFICATION - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write("-" * 40 + "\n")
            metrics = self.results['metrics']
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Precision (weighted): {metrics['precision_weighted']:.4f}\n")
            f.write(f"Recall (macro): {metrics['recall_macro']:.4f}\n")
            f.write(f"Recall (weighted): {metrics['recall_weighted']:.4f}\n")
            f.write(f"F1 Score (macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}\n\n")
            
            f.write("PER-CLASS METRICS:\n")
            f.write("-" * 40 + "\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"\n{class_name.upper()}:\n")
                f.write(f"  Precision: {metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"  Recall: {metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1_per_class'][i]:.4f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        logger.info(f"Evaluation report generated in {output_dir}")