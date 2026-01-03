"""Test inference module"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.classifier import FoodClassifier
from src.data.augmentation import get_val_transforms
from src.inference.predictor import FoodClassifierPredictor
from src.inference.optimize import count_parameters, measure_model_size


def test_predictor():
    """Test predictor functionality"""
    # Create model
    model = FoodClassifier(num_classes=2, pretrained=False)
    model.eval()
    
    # Create transforms
    transform = get_val_transforms(224)
    
    # This would need an actual image file to test properly
    # For now, just test that the predictor can be instantiated
    predictor = FoodClassifierPredictor(
        model=model,
        transform=transform,
        device='cpu'
    )
    
    assert predictor is not None
    print("✓ Predictor test passed")


def test_model_optimization():
    """Test model optimization utilities"""
    model = FoodClassifier(num_classes=2, pretrained=False)
    
    # Count parameters
    params = count_parameters(model)
    assert params['total'] > 0
    print(f"✓ Parameter counting works: {params['total']:,} total params")
    
    # Measure size (approximately)
    # This would save/load a file, so we skip in basic testing
    print("✓ Model optimization tests passed")


if __name__ == "__main__":
    test_predictor()
    test_model_optimization()
    print("\nAll inference tests passed! ✓")
