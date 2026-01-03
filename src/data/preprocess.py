"""Data preprocessing utilities"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_train_val_split(data_dir, val_split=0.2, random_state=42):
    """Create train/validation split"""
    images = list(Path(data_dir).glob('**/*.jpg'))
    train_imgs, val_imgs = train_test_split(
        images, 
        test_size=val_split, 
        random_state=random_state
    )
    return train_imgs, val_imgs


def organize_dataset(source_dir, dest_dir, class_names):
    """Organize dataset into class folders"""
    for class_name in class_names:
        os.makedirs(os.path.join(dest_dir, class_name), exist_ok=True)


def resize_images(input_dir, output_dir, size=(224, 224)):
    """Resize all images in a directory"""
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in Path(input_dir).glob('**/*.jpg'):
        img = Image.open(img_path)
        img_resized = img.resize(size)
        
        output_path = os.path.join(output_dir, img_path.name)
        img_resized.save(output_path)
