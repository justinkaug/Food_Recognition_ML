"""Test model architectures"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.classifier import FoodClassifier
from src.models.recognizer import FoodRecognizer
from src.models.multitask import MultiTaskFoodModel


def test_food_classifier():
    """Test FoodClassifier model"""
    model = FoodClassifier(num_classes=2, pretrained=False)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"
    print("✓ FoodClassifier test passed")


def test_food_recognizer():
    """Test FoodRecognizer model"""
    model = FoodRecognizer(num_classes=101, pretrained=False)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    assert output.shape == (2, 101), f"Expected shape (2, 101), got {output.shape}"
    print("✓ FoodRecognizer test passed")


def test_multitask_model():
    """Test MultiTaskFoodModel"""
    model = MultiTaskFoodModel(num_food_classes=101, pretrained=False)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = model(dummy_input)
    
    assert 'classification' in outputs
    assert 'recognition' in outputs
    assert 'portion' in outputs
    assert outputs['classification'].shape == (2, 2)
    assert outputs['recognition'].shape == (2, 101)
    assert outputs['portion'].shape == (2, 1)
    print("✓ MultiTaskFoodModel test passed")


def test_freeze_unfreeze():
    """Test freeze/unfreeze functionality"""
    model = FoodClassifier(num_classes=2, pretrained=False)
    
    # Test freeze
    model.freeze_backbone()
    for param in model.backbone.features.parameters():
        assert not param.requires_grad
    
    # Test unfreeze
    model.unfreeze_backbone()
    for param in model.backbone.parameters():
        assert param.requires_grad
    
    print("✓ Freeze/unfreeze test passed")


if __name__ == "__main__":
    test_food_classifier()
    test_food_recognizer()
    test_multitask_model()
    test_freeze_unfreeze()
    print("\nAll model tests passed! ✓")
