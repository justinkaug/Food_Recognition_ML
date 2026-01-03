"""Export trained models for production deployment"""

import os
import sys
import argparse
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.classifier import FoodClassifier
from src.models.recognizer import FoodRecognizer
from src.inference.export import export_to_onnx, export_to_torchscript, test_onnx_inference
from src.inference.optimize import quantize_model_dynamic, measure_model_size, benchmark_inference
from src.utils.logger import setup_logger


def export_classifier(model_path, output_dir):
    """Export food classifier"""
    logger = setup_logger('export')
    
    # Load model
    model = FoodClassifier()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, 'classifier.onnx')
    export_to_onnx(model, dummy_input, onnx_path)
    logger.info(f"Exported to ONNX: {onnx_path}")
    
    # Export to TorchScript
    torchscript_path = os.path.join(output_dir, 'classifier.pt')
    export_to_torchscript(model, dummy_input, torchscript_path)
    logger.info(f"Exported to TorchScript: {torchscript_path}")
    
    # Test ONNX inference
    test_onnx_inference(onnx_path, dummy_input)
    
    # Quantize model
    quantized_model = quantize_model_dynamic(model)
    quantized_path = os.path.join(output_dir, 'classifier_quantized.pth')
    torch.save(quantized_model.state_dict(), quantized_path)
    logger.info(f"Quantized model saved: {quantized_path}")
    
    # Measure sizes
    original_size = measure_model_size(model)
    quantized_size = measure_model_size(quantized_model)
    
    logger.info(f"Original model size: {original_size:.2f} MB")
    logger.info(f"Quantized model size: {quantized_size:.2f} MB")
    logger.info(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    
    # Benchmark
    logger.info("Benchmarking original model...")
    original_stats = benchmark_inference(model, device='cpu')
    logger.info(f"Original model: {original_stats['avg_time_ms']:.2f} ms/image ({original_stats['fps']:.1f} FPS)")
    
    logger.info("Benchmarking quantized model...")
    quantized_stats = benchmark_inference(quantized_model, device='cpu')
    logger.info(f"Quantized model: {quantized_stats['avg_time_ms']:.2f} ms/image ({quantized_stats['fps']:.1f} FPS)")


def export_recognizer(model_path, output_dir):
    """Export food recognizer"""
    logger = setup_logger('export')
    
    # Load model
    model = FoodRecognizer()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, 'recognizer.onnx')
    export_to_onnx(model, dummy_input, onnx_path)
    logger.info(f"Exported to ONNX: {onnx_path}")
    
    # Export to TorchScript
    torchscript_path = os.path.join(output_dir, 'recognizer.pt')
    export_to_torchscript(model, dummy_input, torchscript_path)
    logger.info(f"Exported to TorchScript: {torchscript_path}")


def main(args):
    """Main export function"""
    logger = setup_logger('export')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Exporting model: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    if args.model_type == 'classifier':
        export_classifier(args.model_path, args.output_dir)
    elif args.model_type == 'recognizer':
        export_recognizer(args.model_path, args.output_dir)
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        return
    
    logger.info("Export completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export models for production')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['classifier', 'recognizer', 'detector', 'multitask'],
        required=True,
        help='Type of model to export'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='exported_models',
        help='Output directory for exported models'
    )
    
    args = parser.parse_args()
    main(args)
