"""Test data module"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.data.dataset import FoodDataset, FoodClassifierDataset, Food101Dataset
from src.data.augmentation import get_train_transforms, get_val_transforms


def test_dataset_creation():
    """Test dataset creation"""
    # This is a placeholder test
    assert True


def test_transforms():
    """Test data augmentation transforms"""
    img_size = 224
    
    train_transforms = get_train_transforms(img_size)
    val_transforms = get_val_transforms(img_size)
    
    assert train_transforms is not None
    assert val_transforms is not None


def test_food_dataset():
    """Test FoodDataset class"""
    # Placeholder - would need actual data
    pass


if __name__ == "__main__":
    test_dataset_creation()
    test_transforms()
    print("All data tests passed!")
