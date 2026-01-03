"""Evaluation metrics"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def calculate_accuracy(predictions, targets, topk=(1,)):
    """Calculate top-k accuracy"""
    maxk = max(topk)
    batch_size = targets.size(0)
    
    _, pred = predictions.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    
    return res


def calculate_metrics(predictions, targets):
    """Calculate precision, recall, F1"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted'
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def get_confusion_matrix(predictions, targets, num_classes):
    """Calculate confusion matrix"""
    return confusion_matrix(targets, predictions, labels=range(num_classes))


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """Track multiple metrics during training"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, metric_name, value, n=1):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = AverageMeter()
        self.metrics[metric_name].update(value, n)
    
    def get_metrics(self):
        return {name: meter.avg for name, meter in self.metrics.items()}
    
    def reset(self):
        for meter in self.metrics.values():
            meter.reset()
