"""Custom loss functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning"""
    
    def __init__(self, task_weights=None):
        super(MultiTaskLoss, self).__init__()
        
        if task_weights is None:
            task_weights = {'classification': 1.0, 'recognition': 1.0, 'portion': 1.0}
        
        self.task_weights = task_weights
        self.classification_loss = nn.CrossEntropyLoss()
        self.recognition_loss = nn.CrossEntropyLoss()
        self.portion_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict with keys 'classification', 'recognition', 'portion'
            targets: Dict with corresponding target tensors
        """
        loss_dict = {}
        
        if 'classification' in predictions:
            loss_dict['classification'] = self.classification_loss(
                predictions['classification'], 
                targets['classification']
            )
        
        if 'recognition' in predictions:
            loss_dict['recognition'] = self.recognition_loss(
                predictions['recognition'],
                targets['recognition']
            )
        
        if 'portion' in predictions:
            loss_dict['portion'] = self.portion_loss(
                predictions['portion'],
                targets['portion']
            )
        
        # Weighted sum
        total_loss = sum(
            self.task_weights[task] * loss 
            for task, loss in loss_dict.items()
        )
        
        return total_loss, loss_dict


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing"""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_pred = F.log_softmax(pred, dim=1)
        
        # One-hot with smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_pred, dim=1))
