"""
Prediction Module for Vehicle Type Classification
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2
import torch
from PIL import Image
from loguru import logger

from src.data.data_loader import ConfigLoader
from src.data.data_preprocessing import DataPreprocessor, ImageTransformer
from src.models.train import VehicleClassifier


class VehiclePredictor:
    """Vehicle Type Prediction Interface"""
    
    def __init__(
        self,
        model_path: str = None,
        config: ConfigLoader = None
    ):
        self.config = config or ConfigLoader()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor = DataPreprocessor(config)
        self.transformer = ImageTransformer(config)
        
        self.class_names = self.config.get('classes', ['car', 'bike', 'truck', 'bus'])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        
        if model_path:
            self.load_model(model_path)
        
        logger.info(f"Predictor initialized on device: {self.device}")
    
    def load_model(self, model_path: str = None):
        """Load trained model for inference"""
        model_path = model_path or self.config.get('model.save_path', 'models/best_model.pth')
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
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
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference"""
        transform = self.transformer.get_inference_transforms()
        transformed = transform(image=image)
        return transformed['image'].unsqueeze(0)
    
    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_probs: bool = True
    ) -> Dict:
        """
        Predict vehicle type from image
        
        Args:
            image: Image path, numpy array, or PIL Image
            return_probs: Whether to return class probabilities
        
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            self.load_model()
        
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Preprocess
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = probabilities.max(1)
        
        predicted_class = self.idx_to_class[predicted_idx.item()]
        confidence_value = confidence.item()
        
        result = {
            'class': predicted_class,
            'class_idx': predicted_idx.item(),
            'confidence': confidence_value
        }
        
        if return_probs:
            result['probabilities'] = {
                self.idx_to_class[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        return_probs: bool = True
    ) -> List[Dict]:
        """Predict vehicle types for multiple images"""
        return [self.predict(img, return_probs) for img in images]
    
    def predict_from_bytes(
        self,
        image_bytes: bytes,
        return_probs: bool = True
    ) -> Dict:
        """Predict from image bytes (useful for web uploads)"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from bytes")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.predict(image, return_probs)
    
    def predict_with_visualization(
        self,
        image: Union[str, np.ndarray],
        return_probs: bool = True
    ) -> Tuple[Dict, np.ndarray]:
        """
        Predict and return visualization with prediction overlay
        
        Returns:
            Tuple of (prediction dict, annotated image)
        """
        # Load image if path
        if isinstance(image, str):
            original_image = cv2.imread(image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = image.copy()
        
        # Get prediction
        result = self.predict(original_image, return_probs)
        
        # Create annotated image
        annotated = original_image.copy()
        
        # Add prediction text
        label = f"{result['class'].upper()}: {result['confidence']*100:.1f}%"
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            annotated,
            (10, 10),
            (20 + text_width, 20 + text_height + baseline),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated,
            label,
            (15, 15 + text_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
        # Add probability bars if available
        if return_probs and 'probabilities' in result:
            bar_height = 20
            bar_max_width = 150
            y_start = 50 + text_height + baseline
            
            for i, (class_name, prob) in enumerate(result['probabilities'].items()):
                y_pos = y_start + i * (bar_height + 10)
                bar_width = int(prob * bar_max_width)
                
                # Draw background bar
                cv2.rectangle(
                    annotated,
                    (10, y_pos),
                    (10 + bar_max_width, y_pos + bar_height),
                    (50, 50, 50),
                    -1
                )
                
                # Draw probability bar
                color = (0, 255, 0) if class_name == result['class'] else (100, 100, 255)
                cv2.rectangle(
                    annotated,
                    (10, y_pos),
                    (10 + bar_width, y_pos + bar_height),
                    color,
                    -1
                )
                
                # Draw class label
                cv2.putText(
                    annotated,
                    f"{class_name}: {prob*100:.1f}%",
                    (15 + bar_max_width, y_pos + bar_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return result, annotated
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "model_name": self.model.model_name,
            "num_classes": self.model.num_classes,
            "feature_dim": self.model.feature_dim,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "class_names": self.class_names
        }
    
    def warmup(self, num_iterations: int = 3):
        """Warmup the model for faster inference"""
        if self.model is None:
            self.load_model()
        
        dummy_input = torch.randn(
            1, 3,
            self.config.get('image.height', 224),
            self.config.get('image.width', 224)
        ).to(self.device)
        
        logger.info("Warming up model...")
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
        logger.info("Model warmup complete")


class BatchPredictor:
    """Batch prediction with optimizations"""
    
    def __init__(
        self,
        model_path: str = None,
        config: ConfigLoader = None,
        batch_size: int = 32
    ):
        self.predictor = VehiclePredictor(model_path, config)
        self.batch_size = batch_size
        self.config = config or ConfigLoader()
    
    def predict_directory(
        self,
        directory: str,
        output_file: str = None
    ) -> List[Dict]:
        """Predict all images in a directory"""
        directory = Path(directory)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        image_paths = [
            str(p) for p in directory.iterdir()
            if p.suffix.lower() in valid_extensions
        ]
        
        logger.info(f"Found {len(image_paths)} images in {directory}")
        
        results = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_results = []
            
            for path in batch_paths:
                try:
                    result = self.predictor.predict(path)
                    result['image_path'] = path
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    batch_results.append({
                        'image_path': path,
                        'error': str(e)
                    })
            
            results.extend(batch_results)
            logger.info(f"Processed {min(i + self.batch_size, len(image_paths))}/{len(image_paths)} images")
        
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results


def main():
    """Main function for command-line prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vehicle Type Prediction")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--directory", type=str, help="Path to directory of images")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Path to model")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    if args.image:
        predictor = VehiclePredictor(args.model)
        result = predictor.predict(args.image)
        print(f"\nPrediction for {args.image}:")
        print(f"  Class: {result['class']}")
        print(f"  Confidence: {result['confidence']*100:.2f}%")
        print(f"  Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"    {class_name}: {prob*100:.2f}%")
    
    elif args.directory:
        batch_predictor = BatchPredictor(args.model)
        results = batch_predictor.predict_directory(args.directory, args.output)
        print(f"\nProcessed {len(results)} images")


if __name__ == "__main__":
    main()