"""Food recognizer model (Food-101)"""

import torch
import torch.nn as nn
import torchvision.models as models


class FoodRecognizer(nn.Module):
    """Multi-class classifier for food recognition"""
    
    def __init__(self, num_classes=101, pretrained=True):
        super(FoodRecognizer, self).__init__()
        
        # Load EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
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
