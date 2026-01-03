"""Model adapter for Calorie Tracker integration"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import os


# Food-101 class names (subset shown for brevity)
FOOD_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    # ... (add all 101 classes)
]


class FoodMLAdapter:
    """Adapter for integrating ML models with Calorie Tracker"""
    
    def __init__(self, models_dir='models'):
        """
        Initialize the adapter with trained models
        
        Args:
            models_dir: Directory containing ONNX models
        """
        classifier_path = os.path.join(models_dir, 'classifier.onnx')
        recognizer_path = os.path.join(models_dir, 'recognizer.onnx')
        
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier model not found: {classifier_path}")
        if not os.path.exists(recognizer_path):
            raise FileNotFoundError(f"Recognizer model not found: {recognizer_path}")
        
        self.classifier = ort.InferenceSession(classifier_path)
        self.recognizer = ort.InferenceSession(recognizer_path)
    
    def analyze_food(self, image_path):
        """
        Analyze food image and return classification and recognition results
        
        Args:
            image_path: Path to food image
            
        Returns:
            dict: Analysis results with keys:
                - is_processed: bool
                - food_item: str
                - confidence: float
        """
        # Preprocess image
        img = self.preprocess_image(image_path)
        
        # Step 1: Classify (processed vs unprocessed)
        classification = self.classify_food(img)
        
        # Step 2: Recognize food item
        recognition = self.recognize_food(img)
        
        return {
            'is_processed': classification['is_processed'],
            'classification_confidence': classification['confidence'],
            'food_item': recognition['class'],
            'recognition_confidence': recognition['confidence'],
            'top_k_predictions': recognition.get('top_k', [])
        }
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to image file
            
        Returns:
            np.ndarray: Preprocessed image tensor
        """
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Transpose to CHW format and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array.astype(np.float32)
    
    def classify_food(self, img):
        """
        Classify food as processed or unprocessed
        
        Args:
            img: Preprocessed image tensor
            
        Returns:
            dict: Classification results
        """
        input_name = self.classifier.get_inputs()[0].name
        output = self.classifier.run(None, {input_name: img})[0]
        
        # Softmax
        exp_output = np.exp(output[0])
        probabilities = exp_output / np.sum(exp_output)
        
        is_processed = probabilities[0] > probabilities[1]
        confidence = float(probabilities[0] if is_processed else probabilities[1])
        
        return {
            'is_processed': bool(is_processed),
            'confidence': confidence
        }
    
    def recognize_food(self, img, top_k=5):
        """
        Recognize specific food item
        
        Args:
            img: Preprocessed image tensor
            top_k: Number of top predictions to return
            
        Returns:
            dict: Recognition results
        """
        input_name = self.recognizer.get_inputs()[0].name
        output = self.recognizer.run(None, {input_name: img})[0]
        
        # Softmax
        exp_output = np.exp(output[0])
        probabilities = exp_output / np.sum(exp_output)
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_probs = probabilities[top_indices]
        
        top_k_predictions = [
            {
                'class': FOOD_CLASSES[idx] if idx < len(FOOD_CLASSES) else f'class_{idx}',
                'confidence': float(prob)
            }
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return {
            'class': top_k_predictions[0]['class'],
            'confidence': top_k_predictions[0]['confidence'],
            'top_k': top_k_predictions
        }


def main():
    """Demo usage"""
    adapter = FoodMLAdapter(models_dir='exported_models')
    
    # Example usage
    result = adapter.analyze_food('test_image.jpg')
    print("Analysis Result:")
    print(f"  Is Processed: {result['is_processed']}")
    print(f"  Food Item: {result['food_item']}")
    print(f"  Confidence: {result['recognition_confidence']:.2%}")


if __name__ == "__main__":
    main()
