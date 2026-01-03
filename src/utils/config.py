"""Configuration management"""

import yaml
import os
from pathlib import Path


class Config:
    """Configuration class for loading and managing configs"""
    
    def __init__(self, config_path=None):
        self.config = {}
        if config_path:
            self.load(config_path)
    
    def load(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        return self.config
    
    def save(self, config_path):
        """Save configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def set(self, key, value):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __setitem__(self, key, value):
        self.set(key, value)
    
    def __repr__(self):
        return f"Config({self.config})"


def load_config(config_path):
    """Load configuration from file"""
    config = Config()
    config.load(config_path)
    return config


def create_default_config():
    """Create default configuration template"""
    default_config = {
        'model': {
            'name': 'efficientnet_b0',
            'num_classes': 101,
            'pretrained': True
        },
        'data': {
            'train_dir': 'data/train',
            'val_dir': 'data/val',
            'img_size': 224,
            'batch_size': 32,
            'num_workers': 4
        },
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'early_stopping_patience': 10
        },
        'paths': {
            'save_dir': 'models',
            'log_dir': 'logs'
        }
    }
    return default_config
