"""Multi-task model for combined food analysis"""

import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskFoodModel(nn.Module):
    """Multi-task model with shared backbone"""
    
    def __init__(self, num_food_classes=101, pretrained=True):
        super(MultiTaskFoodModel, self).__init__()
        
        # Shared backbone (EfficientNet-B0)
        backbone = models.efficientnet_b0(pretrained=pretrained)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Get feature dimension
        in_features = 1280  # EfficientNet-B0 output channels
        
        # Task-specific heads
        
        # Head 1: Binary classification (processed/unprocessed)
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2)
        )
        
        # Head 2: Food recognition (101 classes)
        self.recognition_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_food_classes)
        )
        
        # Head 3: Portion estimation (regression)
        self.portion_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # Shared feature extraction
        features = self.features(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Task-specific predictions
        classification = self.classifier_head(features)
        recognition = self.recognition_head(features)
        portion = self.portion_head(features)
        
        return {
            'classification': classification,
            'recognition': recognition,
            'portion': portion
        }
    
    def freeze_backbone(self):
        """Freeze shared feature extractor"""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True
