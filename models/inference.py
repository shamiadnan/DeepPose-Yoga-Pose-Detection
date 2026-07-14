"""
DeepPose 2.0 — Inference Engine
=================================
Loads trained ResNet18 model and performs real-time pose classification
with GPU-to-CPU fallback support.
"""

import os
import sys
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PoseInferenceEngine:
    """Loads trained model and performs pose classification on frames.
    
    Handles model loading, preprocessing, and prediction with automatic
    GPU-to-CPU fallback if CUDA runs out of memory.
    
    Args:
        model_path: Path to saved model checkpoint (.pth file)
        class_names: List of class name strings (default: from config)
        device: Computing device (default: from config)
    
    Raises:
        FileNotFoundError: If model checkpoint file doesn't exist
    
    Example:
        >>> engine = PoseInferenceEngine("checkpoints/best_model.pth")
        >>> pose_name, confidence = engine.predict(frame)
        >>> print(f"{pose_name}: {confidence:.1%}")
    """
    
    def __init__(self, model_path: str = None, 
                 class_names: List[str] = None,
                 device: str = None):
        self.model_path = model_path or config.MODEL_PATH
        self.class_names = class_names or config.CLASSES
        self.device = device or config.DEVICE
        
        # Validate model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.model_path}\n"
                f"Please run the training pipeline first:\n"
                f"  python models/train.py"
            )
        
        # Load model
        self.model = self._load_model()
        
        # Inference transforms (no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD),
        ])
        
        print(f"[InferenceEngine] Model loaded from: {self.model_path}")
        print(f"[InferenceEngine] Device: {self.device}")
        print(f"[InferenceEngine] Classes: {len(self.class_names)}")
    
    def _load_model(self) -> nn.Module:
        """Load ResNet18 model with saved weights.
        
        Returns:
            Model in eval mode on the configured device
        """
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
        
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            # If GPU fails, fall back to CPU
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[InferenceEngine] GPU load failed, falling back to CPU")
                self.device = "cpu"
                state_dict = torch.load(self.model_path, map_location="cpu")
                model.load_state_dict(state_dict)
            else:
                raise
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        """Predict yoga pose from a BGR frame.
        
        Preprocesses the frame, runs inference through ResNet18, and returns
        the predicted class name with softmax confidence.
        
        Args:
            frame: BGR numpy array of shape (H, W, 3)
        
        Returns:
            Tuple of (pose_name: str, confidence: float)
            - pose_name: One of the 11 defined yoga class names
            - confidence: Softmax probability in [0.0, 1.0]
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = frame[:, :, ::-1]  # BGR to RGB
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transforms
        input_tensor = self.transform(pil_image).unsqueeze(0)  # Add batch dim
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        try:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)
        except RuntimeError as e:
            # GPU OOM fallback
            if "out of memory" in str(e).lower():
                print("[InferenceEngine] GPU OOM — falling back to CPU")
                self.device = "cpu"
                self.model = self.model.to("cpu")
                input_tensor = input_tensor.to("cpu")
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, dim=1)
            else:
                raise
        
        pose_name = self.class_names[predicted_idx.item()]
        conf_value = confidence.item()
        
        return pose_name, conf_value
    
    def predict_batch(self, frames: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Batch prediction for multiple frames.
        
        Args:
            frames: List of BGR numpy arrays
        
        Returns:
            List of (pose_name, confidence) tuples
        """
        results = []
        for frame in frames:
            results.append(self.predict(frame))
        return results
