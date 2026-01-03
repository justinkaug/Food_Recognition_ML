"""Logging utilities"""

import logging
import sys
from datetime import datetime
import os


def setup_logger(name='food_recognition', log_file=None, level=logging.INFO):
    """Set up logger with file and console handlers"""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name='food_recognition'):
    """Get existing logger or create new one"""
    return logging.getLogger(name)


class TrainingLogger:
    """Logger for tracking training progress"""
    
    def __init__(self, log_dir='logs', experiment_name=None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f'{experiment_name}.log')
        self.logger = setup_logger(experiment_name, self.log_file)
    
    def log_epoch(self, epoch, metrics):
        """Log metrics for an epoch"""
        msg = f"Epoch {epoch}:"
        for key, value in metrics.items():
            msg += f" {key}={value:.4f}"
        self.logger.info(msg)
    
    def log_model_info(self, model):
        """Log model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {model.__class__.__name__}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def log_config(self, config):
        """Log configuration"""
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
