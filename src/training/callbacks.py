"""Training callbacks for monitoring and control"""

import os
import torch
from datetime import datetime


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class ModelCheckpoint:
    """Save model checkpoints during training"""
    
    def __init__(self, save_dir, monitor='val_loss', mode='min', save_best_only=True):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        
        os.makedirs(save_dir, exist_ok=True)
        
        if mode == 'min':
            self.best = float('inf')
            self.monitor_op = lambda x, y: x < y
        else:
            self.best = -float('inf')
            self.monitor_op = lambda x, y: x > y
    
    def __call__(self, model, epoch, metrics):
        current = metrics.get(self.monitor)
        
        if current is None:
            print(f"Warning: {self.monitor} not found in metrics")
            return
        
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.best = current
                filepath = os.path.join(self.save_dir, 'best_model.pth')
                torch.save(model.state_dict(), filepath)
                print(f'Saved best model to {filepath}')
        else:
            filepath = os.path.join(self.save_dir, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), filepath)


class LearningRateScheduler:
    """Custom learning rate scheduler"""
    
    def __init__(self, optimizer, mode='cosine', num_epochs=100, warmup_epochs=5):
        self.optimizer = optimizer
        self.mode = mode
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.initial_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            if self.mode == 'cosine':
                import math
                lr = 0.5 * self.initial_lr * (
                    1 + math.cos(math.pi * (epoch - self.warmup_epochs) / 
                                (self.num_epochs - self.warmup_epochs))
                )
            elif self.mode == 'step':
                lr = self.initial_lr * (0.1 ** (epoch // 30))
            else:
                lr = self.initial_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class TensorBoardLogger:
    """Log metrics to TensorBoard"""
    
    def __init__(self, log_dir='runs'):
        try:
            from torch.utils.tensorboard import SummaryWriter
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(f'{log_dir}/{timestamp}')
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
    
    def log_scalar(self, tag, value, step):
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag, values_dict, step):
        if self.enabled:
            self.writer.add_scalars(tag, values_dict, step)
    
    def close(self):
        if self.enabled:
            self.writer.close()
