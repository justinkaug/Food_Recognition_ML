"""Model export utilities for deployment"""

import torch
import torch.nn as nn
import onnx
import onnxruntime


def export_to_onnx(model, dummy_input, output_path, opset_version=11):
    """Export PyTorch model to ONNX format"""
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify the model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Model exported to {output_path}")


def export_to_torchscript(model, dummy_input, output_path, method='trace'):
    """Export PyTorch model to TorchScript"""
    model.eval()
    
    if method == 'trace':
        traced_model = torch.jit.trace(model, dummy_input)
    elif method == 'script':
        traced_model = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    traced_model.save(output_path)
    print(f"Model exported to {output_path}")


def test_onnx_inference(onnx_path, dummy_input):
    """Test ONNX model inference"""
    session = onnxruntime.InferenceSession(onnx_path)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    ort_inputs = {input_name: dummy_input.cpu().numpy()}
    ort_outputs = session.run([output_name], ort_inputs)
    
    print(f"ONNX inference successful!")
    print(f"Output shape: {ort_outputs[0].shape}")
    
    return ort_outputs[0]


def convert_to_tflite(onnx_path, tflite_path):
    """Convert ONNX model to TensorFlow Lite"""
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph('temp_tf_model')
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model('temp_tf_model')
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model converted to TFLite: {tflite_path}")
        
    except ImportError:
        print("TensorFlow and onnx-tf required. Install with:")
        print("pip install tensorflow onnx-tf")


class ONNXPredictor:
    """Predictor using ONNX Runtime"""
    
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, input_tensor):
        """Run inference with ONNX model"""
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.cpu().numpy()
        
        ort_inputs = {self.input_name: input_tensor}
        ort_outputs = self.session.run([self.output_name], ort_inputs)
        
        return ort_outputs[0]
