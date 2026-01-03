"""Nutrition label detector (YOLO-based)"""

import torch
import torch.nn as nn


class NutritionLabelDetector:
    """Wrapper for YOLOv8 nutrition label detection"""
    
    def __init__(self, model_size='n', pretrained=True):
        """
        Args:
            model_size (str): YOLO model size ('n', 's', 'm', 'l', 'x')
            pretrained (bool): Load pretrained weights
        """
        self.model_size = model_size
        self.model = None
        
    def load_model(self, weights_path=None):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            if weights_path:
                self.model = YOLO(weights_path)
            else:
                self.model = YOLO(f'yolov8{self.model_size}.pt')
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")
    
    def train(self, data_yaml, epochs=100, img_size=640):
        """Train the model"""
        if self.model is None:
            self.load_model()
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=16
        )
        return results
    
    def predict(self, image_path):
        """Run inference"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = self.model(image_path)
        return results
    
    def export(self, format='onnx'):
        """Export model to different formats"""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        self.model.export(format=format)
