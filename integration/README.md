# Integration with Calorie Tracker

## Overview

This document provides instructions for integrating the trained ML models with the Calorie Tracker backend application.

## Integration Architecture

```
Calorie Tracker Backend (Flask)
├── API Endpoints
│   └── /api/food/analyze
│
└── ML Model Adapter
    ├── FoodClassifier (ONNX)
    ├── FoodRecognizer (ONNX)
    └── LabelDetector (ONNX)
```

---

## Setup

### 1. Export Models

First, export trained models to ONNX format:

```bash
python scripts/export_models.py \
    --model-path models/classifier/best_model.pth \
    --model-type classifier \
    --output-dir exported_models/

python scripts/export_models.py \
    --model-path models/recognizer/best_model.pth \
    --model-type recognizer \
    --output-dir exported_models/
```

### 2. Copy Models to Calorie Tracker

```bash
# Copy exported models to Calorie Tracker project
cp exported_models/*.onnx ../Calorie_Tracker/backend/models/
```

---

## Model Adapter Implementation

### File: `integration/model_adapter.py`

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

class FoodMLAdapter:
    """Adapter for integrating ML models with Calorie Tracker"""
    
    def __init__(self, models_dir='models'):
        self.classifier = ort.InferenceSession(f'{models_dir}/classifier.onnx')
        self.recognizer = ort.InferenceSession(f'{models_dir}/recognizer.onnx')
    
    def analyze_food(self, image_path):
        """Analyze food image and return results"""
        # Preprocess image
        img = self.preprocess_image(image_path)
        
        # Step 1: Classify (processed vs unprocessed)
        classification = self.classify_food(img)
        
        # Step 2: Recognize food item
        recognition = self.recognize_food(img)
        
        return {
            'is_processed': classification['is_processed'],
            'food_item': recognition['class'],
            'confidence': recognition['confidence']
        }
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def classify_food(self, img):
        """Classify as processed/unprocessed"""
        input_name = self.classifier.get_inputs()[0].name
        output = self.classifier.run(None, {input_name: img})[0]
        is_processed = output[0][0] > output[0][1]
        return {'is_processed': bool(is_processed)}
    
    def recognize_food(self, img):
        """Recognize food item"""
        input_name = self.recognizer.get_inputs()[0].name
        output = self.recognizer.run(None, {input_name: img})[0]
        class_id = np.argmax(output[0])
        confidence = float(output[0][class_id])
        return {
            'class': FOOD_CLASSES[class_id],
            'confidence': confidence
        }
```

---

## API Integration

### Update Calorie Tracker Backend

#### File: `backend/src/recognizer/image_recognizer.py`

Replace existing heuristic code with:

```python
from integration.model_adapter import FoodMLAdapter

class ImageRecognizer:
    def __init__(self):
        self.ml_adapter = FoodMLAdapter(models_dir='backend/models')
    
    def recognize(self, image_path):
        """Recognize food from image"""
        try:
            results = self.ml_adapter.analyze_food(image_path)
            return {
                'success': True,
                'food_item': results['food_item'],
                'is_processed': results['is_processed'],
                'confidence': results['confidence']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
```

---

## Testing Integration

### File: `integration/test_integration.py`

```python
import unittest
from model_adapter import FoodMLAdapter

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        self.adapter = FoodMLAdapter(models_dir='../exported_models')
    
    def test_analyze_food(self):
        result = self.adapter.analyze_food('test_images/apple.jpg')
        self.assertIn('is_processed', result)
        self.assertIn('food_item', result)
        self.assertIn('confidence', result)
    
    def test_processed_food(self):
        result = self.adapter.analyze_food('test_images/chips.jpg')
        self.assertTrue(result['is_processed'])
    
    def test_unprocessed_food(self):
        result = self.adapter.analyze_food('test_images/banana.jpg')
        self.assertFalse(result['is_processed'])

if __name__ == '__main__':
    unittest.main()
```

---

## Deployment Checklist

- [ ] Export all models to ONNX format
- [ ] Test models with ONNX Runtime
- [ ] Copy models to Calorie Tracker backend
- [ ] Implement model adapter
- [ ] Update ImageRecognizer class
- [ ] Write integration tests
- [ ] Test API endpoints
- [ ] Benchmark inference time
- [ ] Deploy to production
- [ ] Monitor model performance

---

## Performance Optimization

### 1. Model Quantization
```python
# Already done during export
# INT8 quantization reduces size by 4x
```

### 2. Batch Inference
```python
# Process multiple images at once
def analyze_batch(self, image_paths):
    imgs = [self.preprocess_image(path) for path in image_paths]
    batch = np.vstack(imgs)
    # Run inference on batch
```

### 3. Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def analyze_food(self, image_hash):
    # Cache results for frequently queried images
    pass
```

---

## Monitoring

### Track These Metrics
- Inference latency (p50, p95, p99)
- Model accuracy in production
- API error rates
- Resource usage (CPU, memory)

### Tools
- Prometheus for metrics
- Grafana for visualization
- Logging for debugging

---

## Rollback Plan

If ML models perform poorly:

1. Keep old heuristic-based system as fallback
2. Use confidence thresholds
3. Gradual rollout (A/B testing)

```python
if ml_result['confidence'] < 0.7:
    # Fallback to heuristic method
    result = fallback_recognize(image)
```

---

## Future Improvements

1. **Model Updates**: Retrain with user feedback
2. **Active Learning**: Collect edge cases
3. **Multi-Model Ensemble**: Combine predictions
4. **Real-Time Updates**: Hot-swap models without downtime

---

## Support

For issues or questions:
- Check integration tests
- Review logs in `logs/integration.log`
- Contact ML team
