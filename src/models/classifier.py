"""Food classifier model (processed/unprocessed)"""

import torch
import torch.nn as nn
import torchvision.models as models


class FoodClassifier(nn.Module):
    """Binary classifier for processed/unprocessed food"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(FoodClassifier, self).__init__()
        
        # Load MobileNetV3-Small
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Modify classifier head
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze feature extractor layers"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
