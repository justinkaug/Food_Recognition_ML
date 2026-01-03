"""Evaluate trained models"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.classifier import FoodClassifier
from src.models.recognizer import FoodRecognizer
from src.data.augmentation import get_val_transforms
from src.training.metrics import calculate_accuracy, calculate_metrics
from src.utils.logger import setup_logger


def evaluate_classifier(model_path, test_loader, device='cuda'):
    """Evaluate food classifier"""
    logger = get_logger()
    
    # Load model
    model = FoodClassifier()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_labels)
    
    logger.info("=== Classifier Evaluation ===")
    logger.info(f"Accuracy: {metrics['precision']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1']:.4f}")
    
    return metrics


def evaluate_recognizer(model_path, test_loader, device='cuda'):
    """Evaluate food recognizer"""
    logger = get_logger()
    
    # Load model
    model = FoodRecognizer()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    # Calculate top-k accuracy
    top1_acc, top5_acc = calculate_accuracy(all_outputs, all_labels, topk=(1, 5))
    
    logger.info("=== Recognizer Evaluation ===")
    logger.info(f"Top-1 Accuracy: {top1_acc:.2f}%")
    logger.info(f"Top-5 Accuracy: {top5_acc:.2f}%")
    
    return {'top1_accuracy': top1_acc, 'top5_accuracy': top5_acc}


def main(args):
    """Main evaluation function"""
    logger = setup_logger('evaluation')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info(f"Evaluating model: {args.model}")
    
    # TODO: Load test data and run evaluation
    logger.info("Evaluation script ready. Implement data loading for your specific model.")


def get_logger():
    import logging
    return logging.getLogger('evaluation')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['classifier', 'recognizer', 'detector', 'multitask'],
        required=True,
        help='Type of model to evaluate'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        help='Path to test data directory'
    )
    
    args = parser.parse_args()
    main(args)
