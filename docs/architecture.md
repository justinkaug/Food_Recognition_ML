# Model Architecture Documentation

## Overview

This document describes the architecture of the machine learning models used in the Food Recognition ML project.

## 1. Food Classifier (Binary Classification)

### Architecture
- **Backbone**: MobileNetV3-Small (pre-trained on ImageNet)
- **Input**: 224x224x3 RGB images
- **Output**: 2 classes (processed, unprocessed)

### Model Details
```
MobileNetV3-Small
├── Feature Extractor (16 inverted residual blocks)
│   └── Output: 576 channels
├── Global Average Pooling
│   └── Output: 576
├── Dropout (p=0.2)
└── Linear Layer
    └── Output: 2 classes
```

### Parameters
- Total: ~2M parameters
- Trainable (initial): ~500K (head only)
- Trainable (fine-tuned): ~2M (full model)

### Training Strategy
1. Phase 1: Freeze backbone, train head only (10 epochs)
2. Phase 2: Unfreeze, fine-tune entire model (40 epochs)

---

## 2. Food Recognizer (Multi-class Classification)

### Architecture
- **Backbone**: EfficientNet-B0 (pre-trained on ImageNet)
- **Input**: 224x224x3 RGB images
- **Output**: 101 food classes

### Model Details
```
EfficientNet-B0
├── Feature Extractor
│   └── Output: 1280 channels
├── Global Average Pooling
│   └── Output: 1280
├── Dropout (p=0.2)
├── Dense Layer (512 units, ReLU)
├── Dropout (p=0.5)
└── Dense Layer
    └── Output: 101 classes
```

### Parameters
- Total: ~5M parameters
- Memory: ~20MB (FP32), ~5MB (INT8)

### Data Augmentation
- Random horizontal flip
- Random rotation (±20°)
- Color jitter
- Random crop and resize
- Cutout/GridMask

---

## 3. Nutrition Label Detector (Object Detection)

### Architecture
- **Model**: YOLOv8-Nano
- **Input**: 640x640x3 RGB images
- **Output**: Bounding boxes + confidence scores

### Model Details
```
YOLOv8-Nano
├── Backbone (CSPDarknet)
│   ├── Conv layers
│   └── C2f modules
├── Neck (PAN - Path Aggregation Network)
│   ├── Feature pyramid
│   └── Multi-scale fusion
└── Detection Head
    ├── Box regression
    ├── Object classification
    └── Confidence scores
```

### Parameters
- Total: ~3M parameters
- Speed: ~150 FPS on GPU

---

## 4. Multi-Task Model

### Architecture
- **Shared Backbone**: EfficientNet-B0
- **Tasks**:
  1. Binary classification (processed/unprocessed)
  2. Food recognition (101 classes)
  3. Portion estimation (regression)

### Model Details
```
Shared Backbone (EfficientNet-B0)
├── Feature Extraction
│   └── Output: 1280 channels
│
├── Task 1: Classification Head
│   ├── Dense (256, ReLU)
│   └── Output: 2 classes
│
├── Task 2: Recognition Head
│   ├── Dense (512, ReLU)
│   └── Output: 101 classes
│
└── Task 3: Portion Head
    ├── Dense (128, ReLU)
    └── Output: 1 (portion size)
```

### Loss Function
```
Total Loss = w1 * L_classification + w2 * L_recognition + w3 * L_portion

where:
- L_classification: Cross-Entropy Loss
- L_recognition: Cross-Entropy Loss with Label Smoothing
- L_portion: MSE Loss
- Weights: w1=1.0, w2=1.0, w3=0.5
```

---

## Performance Metrics

| Model | Accuracy/mAP | Size | Latency (CPU) | Latency (GPU) |
|-------|-------------|------|---------------|---------------|
| Classifier | 88%+ | 8 MB | 30 ms | 5 ms |
| Recognizer | 82%+ Top-1 | 20 MB | 40 ms | 8 ms |
| Detector | 0.85+ mAP | 12 MB | 50 ms | 10 ms |
| Multi-Task | 85%+ Avg | 22 MB | 45 ms | 10 ms |

---

## Optimization Techniques

### 1. Quantization
- Dynamic quantization for Linear/Conv layers
- INT8 precision
- 4x size reduction, 2-3x speedup

### 2. Pruning
- Magnitude-based pruning
- Target: 30% sparsity
- Minimal accuracy loss (<1%)

### 3. Knowledge Distillation (Future)
- Teacher: Large model (EfficientNet-B3)
- Student: Small model (MobileNetV3)
- Target: Maintain accuracy with smaller model

---

## References

1. [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
2. [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
3. [YOLOv8 Documentation](https://docs.ultralytics.com/)
4. [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
