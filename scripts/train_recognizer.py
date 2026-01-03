"""Train food recognizer (Food-101)"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.recognizer import FoodRecognizer
from src.data.dataset import Food101Dataset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main(args):
    """Train food recognizer"""
    
    # Load configuration
    config = load_config(args.config)
    logger = setup_logger('recognizer_training')
    
    logger.info("Starting Food Recognizer training...")
    logger.info(f"Configuration: {args.config}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data transforms
    train_transforms = get_train_transforms(config.get('data.img_size', 224))
    val_transforms = get_val_transforms(config.get('data.img_size', 224))
    
    # Datasets (placeholder - need to implement Food101Dataset properly)
    train_dataset = Food101Dataset(
        root_dir=config.get('data.data_dir'),
        transform=train_transforms
    )
    
    val_dataset = Food101Dataset(
        root_dir=config.get('data.data_dir'),
        transform=val_transforms
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('data.batch_size', 64),
        shuffle=True,
        num_workers=config.get('data.num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('data.batch_size', 64),
        shuffle=False,
        num_workers=config.get('data.num_workers', 4),
        pin_memory=True
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Model
    model = FoodRecognizer(
        num_classes=config.get('model.num_classes', 101),
        pretrained=config.get('model.pretrained', True)
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.get('training.label_smoothing', 0.1)
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('training.learning_rate', 0.001),
        weight_decay=config.get('training.weight_decay', 0.0001)
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('training.epochs', 100),
        eta_min=config.get('training.scheduler.min_lr', 0.00001)
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=config.get('paths.save_dir', 'models/recognizer')
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.train(
        num_epochs=config.get('training.epochs', 100),
        scheduler=scheduler
    )
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train food recognizer')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/recognizer_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args)
