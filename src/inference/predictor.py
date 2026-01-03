"""Prediction interface for trained models"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np


class Predictor:
    """Base predictor class"""
    
    def __init__(self, model, transform, device='cuda'):
        self.model = model
        self.transform = transform
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, image):
        """Make prediction on a single image"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(image_tensor)
        
        return output
    
    @torch.no_grad()
    def predict_batch(self, images):
        """Make predictions on a batch of images"""
        if isinstance(images[0], str):
            images = [Image.open(img).convert('RGB') for img in images]
        
        image_tensors = torch.stack([
            self.transform(img) for img in images
        ]).to(self.device)
        
        outputs = self.model(image_tensors)
        
        return outputs


class FoodClassifierPredictor(Predictor):
    """Predictor for food classifier (processed/unprocessed)"""
    
    def __init__(self, model, transform, device='cuda', class_names=None):
        super().__init__(model, transform, device)
        self.class_names = class_names or ['processed', 'unprocessed']
    
    def predict(self, image):
        output = super().predict(image)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': probabilities[predicted_class].item(),
            'probabilities': {
                name: prob.item() 
                for name, prob in zip(self.class_names, probabilities)
            }
        }


class FoodRecognizerPredictor(Predictor):
    """Predictor for food recognizer (Food-101)"""
    
    def __init__(self, model, transform, device='cuda', class_names=None):
        super().__init__(model, transform, device)
        self.class_names = class_names or [f'class_{i}' for i in range(101)]
    
    def predict(self, image, top_k=5):
        output = super().predict(image)
        probabilities = torch.softmax(output, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': self.class_names[idx.item()],
                'confidence': prob.item()
            })
        
        return {
            'top_prediction': predictions[0],
            'top_k_predictions': predictions
        }


class MultiTaskPredictor:
    """Predictor for multi-task model"""
    
    def __init__(self, model, transform, device='cuda'):
        self.model = model
        self.transform = transform
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        outputs = self.model(image_tensor)
        
        # Process outputs
        classification_probs = torch.softmax(outputs['classification'], dim=1)[0]
        recognition_probs = torch.softmax(outputs['recognition'], dim=1)[0]
        portion_estimate = outputs['portion'][0].item()
        
        return {
            'classification': {
                'processed': classification_probs[0].item(),
                'unprocessed': classification_probs[1].item()
            },
            'recognition': {
                'class': torch.argmax(recognition_probs).item(),
                'confidence': torch.max(recognition_probs).item()
            },
            'portion': portion_estimate
        }
