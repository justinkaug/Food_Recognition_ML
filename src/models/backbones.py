"""Backbone architectures for feature extraction"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name='efficientnet_b0', pretrained=True):
    """Get a backbone model by name"""
    
    if name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        out_channels = 1280
    elif name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        out_channels = 576
    elif name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=pretrained)
        out_channels = 960
    elif name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        out_channels = 2048
    else:
        raise ValueError(f"Unknown backbone: {name}")
    
    return model, out_channels


class BackboneFeatureExtractor(nn.Module):
    """Extract features from a backbone model"""
    
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True):
        super(BackboneFeatureExtractor, self).__init__()
        
        backbone, self.out_channels = get_backbone(backbone_name, pretrained)
        
        if 'efficientnet' in backbone_name:
            self.features = backbone.features
        elif 'mobilenet' in backbone_name:
            self.features = backbone.features
        elif 'resnet' in backbone_name:
            self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
