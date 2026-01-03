"""Model optimization utilities (quantization, pruning)"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.nn.utils import prune
import copy


def quantize_model_dynamic(model, dtype=torch.qint8):
    """Apply dynamic quantization to model"""
    quantized_model = quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=dtype
    )
    return quantized_model


def quantize_model_static(model, data_loader, device='cuda'):
    """Apply static quantization to model"""
    model.eval()
    model.to('cpu')  # Quantization only works on CPU
    
    # Specify quantization configuration
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for quantization
    model_prepared = quantization.prepare(model, inplace=False)
    
    # Calibrate with representative data
    print("Calibrating model...")
    with torch.no_grad():
        for images, _ in data_loader:
            model_prepared(images)
    
    # Convert to quantized model
    model_quantized = quantization.convert(model_prepared, inplace=False)
    
    return model_quantized


def prune_model_magnitude(model, amount=0.3):
    """Prune model using magnitude-based pruning"""
    model_pruned = copy.deepcopy(model)
    
    # Apply pruning to all conv and linear layers
    for name, module in model_pruned.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make pruning permanent
    
    return model_pruned


def prune_model_structured(model, amount=0.3):
    """Prune model using structured pruning (entire filters)"""
    model_pruned = copy.deepcopy(model)
    
    for name, module in model_pruned.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module, 
                name='weight', 
                amount=amount, 
                n=2, 
                dim=0
            )
            prune.remove(module, 'weight')
    
    return model_pruned


def measure_model_size(model):
    """Measure model size in MB"""
    torch.save(model.state_dict(), 'temp_model.pth')
    import os
    size_mb = os.path.getsize('temp_model.pth') / (1024 * 1024)
    os.remove('temp_model.pth')
    return size_mb


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def benchmark_inference(model, input_size=(1, 3, 224, 224), device='cuda', num_runs=100):
    """Benchmark model inference time"""
    import time
    
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    fps = 1 / avg_time
    
    return {
        'avg_time_ms': avg_time * 1000,
        'fps': fps,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000
    }


def optimize_for_mobile(model, input_size=(1, 3, 224, 224)):
    """Optimize model for mobile deployment"""
    # Export to TorchScript
    model.eval()
    dummy_input = torch.randn(input_size)
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Optimize for mobile
    optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
    
    return optimized_model
