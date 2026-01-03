"""Train nutrition label detector (YOLOv8)"""

import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.detector import NutritionLabelDetector
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main(args):
    """Train nutrition label detector"""
    
    # Load configuration
    config = load_config(args.config)
    logger = setup_logger('detector_training')
    
    logger.info("Starting Nutrition Label Detector training...")
    logger.info(f"Configuration: {args.config}")
    
    # Initialize detector
    detector = NutritionLabelDetector(
        model_size=config.get('model.size', 'n'),
        pretrained=config.get('model.pretrained', True)
    )
    
    # Load model
    detector.load_model()
    logger.info(f"Loaded YOLOv8-{config.get('model.size', 'n')} model")
    
    # Train
    logger.info("Starting training...")
    results = detector.train(
        data_yaml=config.get('data.data_yaml'),
        epochs=config.get('training.epochs', 100),
        img_size=config.get('data.img_size', 640)
    )
    
    logger.info("Training completed!")
    logger.info(f"Results saved to {config.get('paths.save_dir')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train nutrition label detector')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/detector_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args)
