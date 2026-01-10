# 🍔 Food Recognition ML Project

> A comprehensive machine learning project for food classification, recognition, and nutrition label detection. Built to enhance the Calorie Tracker application with state-of-the-art computer vision models.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Project Overview

This project develops custom ML models for:

1. **Food Classification** - Classify images as processed (packaged) or unprocessed (raw)
2. **Food Recognition** - Identify specific food items from 101 categories
3. **Nutrition Label Detection** - Detect and localize nutrition labels on packaging
4. **Multi-Task Model** - Combined model handling all tasks simultaneously

**Target Accuracy**: 90%+ across all tasks
**Deployment**: Integration with Calorie Tracker backend API

---

## 🎯 Learning Goals

This project is designed to teach:

- ✅ End-to-end ML pipeline (data → model → deployment)
- ✅ Computer vision with CNNs
- ✅ Transfer learning and fine-tuning
- ✅ Object detection (YOLO)
- ✅ Multi-task learning
- ✅ Model optimization (quantization, pruning)
- ✅ Dataset creation and annotation
- ✅ MLOps practices
- ✅ Production deployment

**Resume Impact**: Portfolio project demonstrating practical ML engineering skills

---

## 🗂️ Project Structure

```
Food_Recognition_ML/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── .gitignore                        # Git ignore rules
│
├── notebooks/                        # Jupyter notebooks for experiments
│   ├── 01_data_exploration.ipynb    # EDA and visualization
│   ├── 02_baseline_model.ipynb      # Simple baseline
│   ├── 03_food_classifier.ipynb     # Processed/unprocessed classifier
│   ├── 04_food_recognizer.ipynb     # Food-101 recognition
│   ├── 05_label_detection.ipynb     # YOLO for labels
│   ├── 06_multitask_model.ipynb     # Combined model
│   └── 07_optimization.ipynb        # Model optimization
│
├── data/                             # Dataset directory
│   ├── raw/                         # Original unprocessed data
│   ├── processed/                   # Preprocessed data
│   ├── annotations/                 # Label files (YOLO format)
│   └── splits/                      # Train/val/test splits
│
├── src/                              # Source code
│   ├── __init__.py
│   │
│   ├── data/                        # Data processing
│   │   ├── __init__.py
│   │   ├── dataset.py              # PyTorch datasets
│   │   ├── augmentation.py         # Data augmentation
│   │   ├── download.py             # Dataset downloaders
│   │   └── preprocess.py           # Preprocessing utilities
│   │
│   ├── models/                      # Model architectures
│   │   ├── __init__.py
│   │   ├── classifier.py           # Food classifier
│   │   ├── recognizer.py           # Food recognizer
│   │   ├── detector.py             # Label detector (YOLO)
│   │   ├── multitask.py            # Multi-task model
│   │   └── backbones.py            # Backbone architectures
│   │
│   ├── training/                    # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop
│   │   ├── losses.py               # Custom loss functions
│   │   ├── metrics.py              # Evaluation metrics
│   │   └── callbacks.py            # Training callbacks
│   │
│   ├── inference/                   # Inference and deployment
│   │   ├── __init__.py
│   │   ├── predictor.py            # Prediction interface
│   │   ├── export.py               # Model export (ONNX, TFLite)
│   │   └── optimize.py             # Quantization, pruning
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       ├── logger.py               # Logging setup
│       └── visualization.py        # Plotting utilities
│
├── configs/                          # Configuration files
│   ├── classifier_config.yaml       # Classifier config
│   ├── recognizer_config.yaml       # Recognizer config
│   ├── detector_config.yaml         # Detector config
│   └── multitask_config.yaml        # Multi-task config
│
├── scripts/                          # Standalone scripts
│   ├── download_food101.py          # Download Food-101 dataset
│   ├── train_classifier.py          # Train classifier
│   ├── train_recognizer.py          # Train recognizer
│   ├── train_detector.py            # Train detector
│   ├── evaluate.py                  # Evaluation script
│   └── export_models.py             # Export for production
│
├── tests/                            # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_inference.py
│
├── models/                           # Saved model checkpoints
│   ├── classifier/
│   ├── recognizer/
│   ├── detector/
│   └── multitask/
│
├── docs/                             # Documentation
│   ├── architecture.md              # Model architecture docs
│   ├── data_collection.md           # Data collection guide
│   ├── training_guide.md            # Training instructions
│   ├── deployment.md                # Deployment guide
│   └── results.md                   # Results and metrics
│
└── integration/                      # Integration with Calorie Tracker
    ├── model_adapter.py             # Adapter for calorie tracker
    ├── test_integration.py          # Integration tests
    └── README.md                    # Integration instructions
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)
- 20GB+ disk space for datasets
- Git

### Installation

```bash
# Clone the repository
cd "D:\Python ML Projects"
cd Food_Recognition_ML

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Quick Start

```bash
# 1. Download Food-101 dataset
python scripts/download_food101.py

# 2. Explore data
jupyter notebook notebooks/01_data_exploration.ipynb

# 3. Train baseline model
python scripts/train_recognizer.py --config configs/recognizer_config.yaml

# 4. Evaluate model
python scripts/evaluate.py --model models/recognizer/best_model.pth
```

---

## 📚 Development Roadmap

### Phase 1: Foundation (Week 1-2) ✅
- [x] Project setup
- [x] Environment configuration
- [x] Download Food-101 dataset
- [x] Data exploration notebook
- [ ] Baseline model implementation

### Phase 2: Food Classifier (Week 3-6)
- [ ] Collect processed/unprocessed food images (1000+ each)
- [ ] Data augmentation pipeline
- [ ] MobileNetV3 transfer learning
- [ ] Training and evaluation
- [ ] Model export (ONNX)
- **Target**: 88%+ accuracy

### Phase 3: Food Recognizer (Week 7-10)
- [ ] Food-101 dataset preprocessing
- [ ] EfficientNet implementation
- [ ] Training with augmentation
- [ ] Hyperparameter tuning
- [ ] Evaluation on test set
- **Target**: 82%+ top-1 accuracy

### Phase 4: Label Detector (Week 11-14)
- [ ] Collect nutrition label images (2000+)
- [ ] Annotate with LabelImg
- [ ] YOLOv8 implementation
- [ ] Training and validation
- [ ] Inference optimization
- **Target**: 0.85+ mAP@0.5

### Phase 5: Multi-Task Model (Week 15-18)
- [ ] Design multi-task architecture
- [ ] Implement shared backbone
- [ ] Custom loss function
- [ ] Joint training
- [ ] Task balancing
- **Target**: 85%+ on all tasks

### Phase 6: Optimization & Deployment (Week 19-22)
- [ ] Model quantization (INT8)
- [ ] ONNX export
- [ ] TensorRT optimization
- [ ] TFLite conversion (mobile)
- [ ] Benchmarking (latency, size)
- [ ] Integration with Calorie Tracker
- **Target**: <50ms inference, <20MB model size

---

## 📊 Datasets

### 1. Food-101 Dataset
- **Size**: 101,000 images (101 food categories)
- **Split**: 75,750 train / 25,250 test
- **Source**: [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- **Use**: Food recognition training

### 2. Custom Processed/Unprocessed Dataset
- **Size**: 2,000+ images (to be collected)
- **Classes**: Processed (packaged), Unprocessed (raw)
- **Source**: Manual collection from Google Images, Kaggle
- **Use**: Binary food classification

### 3. Nutrition Label Dataset
- **Size**: 2,000+ images (to be collected)
- **Annotations**: Bounding boxes (YOLO format)
- **Source**: Manual collection and labeling
- **Use**: Object detection training

---

## 🏗️ Model Architectures

### 1. Food Classifier
```
MobileNetV3-Small (Pre-trained)
├── Feature Extractor (frozen initially)
├── Global Average Pooling
├── Dropout (0.2)
└── FC Layer (2 classes)

Parameters: ~2M
Input: 224x224x3
Output: [processed, unprocessed]
```

### 2. Food Recognizer
```
EfficientNet-B0 (Pre-trained)
├── Feature Extractor (fine-tuned)
├── Global Average Pooling
├── Dense Layer (512, ReLU)
├── Dropout (0.5)
└── Dense Layer (101 classes, Softmax)

Parameters: ~5M
Input: 224x224x3
Output: 101 food classes
```

### 3. Label Detector
```
YOLOv8-Nano
├── Backbone (CSPDarknet)
├── Neck (PAN)
└── Detection Head

Parameters: ~3M
Input: 640x640x3
Output: Bounding boxes + class scores
```

### 4. Multi-Task Model
```
Shared Backbone (EfficientNet-B0)
├── Classification Head (processed/unprocessed)
├── Recognition Head (101 food classes)
└── Portion Estimation Head (regression)

Parameters: ~6M
Input: 224x224x3
Output: Multiple task predictions
```

---

## 🎓 Learning Resources

### Online Courses (FREE)
1. **Fast.ai - Practical Deep Learning**
   - https://course.fast.ai/
   - Start here! Best for beginners

2. **Stanford CS231n - CNNs for Visual Recognition**
   - http://cs231n.stanford.edu/
   - Excellent theory + assignments

3. **PyTorch Official Tutorials**
   - https://pytorch.org/tutorials/
   - Learn PyTorch basics

### Papers to Read
1. **MobileNetV3** - Efficient mobile architectures
2. **EfficientNet** - Scaling CNNs efficiently
3. **YOLOv8** - Real-time object detection
4. **Food-101** - Food recognition benchmark

### Tools & Libraries
- **PyTorch** - Deep learning framework
- **Torchvision** - Pre-trained models & datasets
- **Ultralytics** - YOLOv8 implementation
- **Albumentations** - Image augmentation
- **Weights & Biases** - Experiment tracking
- **ONNX** - Model interoperability
- **TensorRT** - Inference optimization

---

## 📈 Expected Results

| Model | Metric | Target | Current |
|-------|--------|--------|---------|
| **Food Classifier** | Accuracy | 88% | TBD |
| **Food Recognizer** | Top-1 Accuracy | 82% | TBD |
| **Food Recognizer** | Top-5 Accuracy | 95% | TBD |
| **Label Detector** | mAP@0.5 | 0.85 | TBD |
| **Multi-Task Model** | Avg Accuracy | 85% | TBD |

### Inference Performance
- **Latency**: <50ms on CPU, <10ms on GPU
- **Model Size**: <20MB (quantized)
- **Memory**: <500MB during inference

---

## 🔗 Integration with Calorie Tracker

Once models are trained, they will replace the current heuristic-based system:

### Current System (Calorie_Tracker)
```python
# backend/src/recognizer/image_recognizer.py
# Heuristic-based classification (70-75% accuracy)
```

### After Integration
```python
# Use trained ML models
from food_recognition_ml import FoodClassifier, FoodRecognizer

classifier = FoodClassifier.load('models/classifier/best_model.onnx')
recognizer = FoodRecognizer.load('models/recognizer/best_model.onnx')

# 88%+ accuracy!
```

See [integration/README.md](integration/README.md) for detailed instructions.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test specific module
pytest tests/test_models.py

# With coverage
pytest --cov=src tests/
```

---

## 📝 Documentation

- **[Architecture](docs/architecture.md)** - Model architecture details
- **[Data Collection](docs/data_collection.md)** - How to collect and label data
- **[Training Guide](docs/training_guide.md)** - Training best practices
- **[Deployment](docs/deployment.md)** - Production deployment guide
- **[Results](docs/results.md)** - Experiment results and analysis

---

## 🤝 Contributing

This is a learning project! Contributions welcome:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **Food-101 Dataset** - ETH Zurich
- **PyTorch Team** - Excellent framework
- **Fast.ai** - Practical deep learning education
- **Ultralytics** - YOLOv8 implementation

---

## 📧 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

**Project Link**: [https://github.com/yourusername/Food_Recognition_ML](https://github.com/yourusername/Food_Recognition_ML)

---

## 🎯 Next Steps

1. **This Weekend**: 
   - Complete Fast.ai Lesson 1
   - Download Food-101 dataset
   - Run first notebook

2. **Next Week**:
   - Build baseline model
   - Achieve 70%+ accuracy
   - Start collecting custom data

3. **Month 1 Goal**:
   - Food classifier trained (88% accuracy)
   - Update resume with this project
   - Write blog post about learnings

**Let's build something amazing! 🚀**

---

## 📅 Progress Log

### January 3, 2026
- ✅ Created complete project structure
- ✅ Set up virtual environment (.venv)
- ✅ Created all source code modules (data, models, training, inference, utils)
- ✅ Implemented model architectures (Classifier, Recognizer, Detector, Multi-task)
- ✅ Created configuration files (YAML configs for all models)
- ✅ Wrote training scripts and evaluation scripts
- ✅ Set up testing framework with unit tests
- ✅ Created documentation (architecture.md, data_collection.md)
- ✅ Built integration adapter for Calorie Tracker
- ✅ Added requirements.txt with all dependencies
- ✅ Created setup.py for package installation
- ✅ Added .gitignore and MIT License

### January 4, 2026
- ✅ Installed all project dependencies in virtual environment
- ✅ Downloaded Food-101 dataset (101,000 images, 5GB)
- ✅ Verified dataset structure (101 food categories)
- ✅ Created data exploration notebook (01_data_exploration.ipynb)
- ✅ Analyzed dataset properties:
  - 750 train + 250 test images per class (perfectly balanced)
  - Variable image dimensions (will resize to 224x224)
  - Good variety across food categories
- 📝 **Next**: Build baseline model and establish performance benchmark

### January 9, 2026
- ✅ Started nutrition label data collection phase
- ✅ Set up SerpAPI integration for image scraping
- ✅ Created nutrition label notebook (01_is_nutrition_label.ipynb)
- ✅ Implemented automated image download function with Google Images API
- ✅ Configured YAML-based API key management
- ✅ Installed required packages (pyyaml, google-search-results, requests)
- ✅ Set up data directory structure for nutrition label images
- 🔄 **In Progress**: Collecting nutrition label dataset (target: 500-1000 images)
- 📝 **Next**: Collect negative examples (non-nutrition labels) and begin model training

---
