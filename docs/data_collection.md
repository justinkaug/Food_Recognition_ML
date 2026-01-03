# Data Collection Guide

## Overview

This guide provides instructions for collecting and preparing datasets for the Food Recognition ML project.

## Datasets Required

### 1. Food-101 Dataset (Food Recognition)
- **Purpose**: Train food recognition model
- **Size**: 101,000 images (101 classes, 1,000 images each)
- **Split**: 75,750 train / 25,250 test
- **Source**: [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

#### Download Instructions
```bash
python scripts/download_food101.py
```

---

### 2. Custom Processed/Unprocessed Dataset (Food Classification)
- **Purpose**: Train binary classifier
- **Target Size**: 2,000+ images (1,000+ per class)
- **Classes**: Processed (packaged), Unprocessed (raw)

#### Collection Sources
1. **Google Images**
   - Search terms: "packaged food", "processed food items", "raw vegetables", "fresh fruits"
   - Use Google Images Download tool

2. **Kaggle Datasets**
   - [Fruits 360](https://www.kaggle.com/moltean/fruits)
   - [Grocery Products](https://www.kaggle.com/datasets)

3. **Open Images Dataset**
   - [Open Images V7](https://storage.googleapis.com/openimages/web/index.html)

#### Collection Script (Google Images)
```bash
pip install google-images-download
google-images-download --keywords "packaged food,processed food" --limit 500
```

#### Manual Collection Tips
- Capture diverse lighting conditions
- Include various angles and perspectives
- Ensure clear images (no blur)
- Mix indoor/outdoor shots
- Include different brands and packaging styles

---

### 3. Nutrition Label Dataset (Object Detection)
- **Purpose**: Train label detection model
- **Target Size**: 2,000+ images
- **Annotations**: Bounding boxes (YOLO format)

#### Collection Strategy
1. **Personal Collection**
   - Take photos of nutrition labels on products
   - Use smartphone camera (good resolution)
   - Capture from different angles

2. **Online Sources**
   - Product images from e-commerce sites
   - Food packaging databases
   - Publicly available nutrition label images

3. **Synthetic Data**
   - Generate synthetic labels with variations
   - Use templates and randomization

#### Annotation Tools
- **LabelImg**: Recommended for YOLO format
  ```bash
  pip install labelImg
  labelImg
  ```

- **CVAT**: Web-based annotation platform
- **Roboflow**: Online annotation + augmentation

#### Annotation Guidelines
1. Draw tight bounding boxes around nutrition labels
2. Include "Nutrition Facts" header
3. Exclude decorative borders
4. Save in YOLO format (class x_center y_center width height)

---

## Data Organization

### Directory Structure
```
data/
├── raw/
│   ├── food-101/
│   ├── classifier/
│   │   ├── processed/
│   │   └── unprocessed/
│   └── nutrition_labels/
│
├── processed/
│   ├── classifier/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── nutrition_labels/
│       ├── images/
│       └── labels/
│
└── annotations/
    └── nutrition_labels.yaml
```

---

## Data Preprocessing

### 1. Image Resizing
```python
from PIL import Image

def resize_image(input_path, output_path, size=(224, 224)):
    img = Image.open(input_path)
    img_resized = img.resize(size)
    img_resized.save(output_path)
```

### 2. Train/Val/Test Split
```python
from sklearn.model_selection import train_test_split

# 70% train, 15% val, 15% test
train_val, test = train_test_split(images, test_size=0.15)
train, val = train_test_split(train_val, test_size=0.176)  # 0.15/0.85
```

### 3. Data Augmentation
Applied during training:
- Random horizontal flip
- Random rotation
- Color jitter
- Random crop
- Cutout

---

## Data Quality Checklist

- [ ] Images are clear and well-lit
- [ ] Minimum resolution: 224x224
- [ ] No duplicate images
- [ ] Balanced class distribution
- [ ] Proper train/val/test split
- [ ] Annotations are accurate (for detection)
- [ ] Files are properly organized
- [ ] No corrupted images

---

## Dataset Statistics

### Target Statistics
| Dataset | Classes | Train | Val | Test | Total |
|---------|---------|-------|-----|------|-------|
| Food-101 | 101 | 75,750 | - | 25,250 | 101,000 |
| Classifier | 2 | 1,400 | 300 | 300 | 2,000 |
| Detector | 1 | 1,400 | 300 | 300 | 2,000 |

---

## Legal and Ethical Considerations

1. **Copyright**: Ensure images are properly licensed
2. **Privacy**: Remove any personal information from images
3. **Attribution**: Credit original sources when required
4. **Commercial Use**: Check if dataset allows commercial use

---

## Resources

- [LabelImg](https://github.com/tzutalin/labelImg)
- [CVAT](https://github.com/opencv/cvat)
- [Roboflow](https://roboflow.com/)
- [Google Images Download](https://github.com/hardikvasa/google-images-download)
- [Food-101 Paper](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
