# working/Cassava-Disease-Detection/src/cassava_classifier/pipelines/tensorrt_inference.py
"""
TensorRT inference module for Cassava disease classification
Supports both single models and the ensemble approach from the 3rd place solution
"""

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA context
import cv2
import torch
from albumentations import Compose, Normalize, Resize
from omegaconf import DictConfig
from pathlib import Path
import os
from typing import List, Tuple, Optional, Union
import time

class TensorRTInfer:
    def __init__(self, engine_path: str, model_config: DictConfig):
        self.engine_path = Path(engine_path)
        self.model_config = model_config
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self.device = "cuda"
        print(f"✅ TensorRT engine loaded: {self.engine_path.name}")

    def _load_engine(self):
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to load engine: {self.engine_path}")
            return engine

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = {}  # name → device ptr
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings[name] = int(device_mem)

            buf = {
                'name': name,
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
                'dtype': dtype
            }

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(buf)
            else:
                outputs.append(buf)

        print(f"  Input:  {inputs[0]['name']} {inputs[0]['shape']}")
        print(f"  Output: {outputs[0]['name']} {outputs[0]['shape']}")
        return inputs, outputs, bindings, stream

    def _infer_standard(self, image_path: str):
        input_data = self.preprocess(image_path)
        input_name = self.inputs[0]['name']
        output_name = self.outputs[0]['name']

        # Set shapes and addresses
        self.context.set_input_shape(input_name, input_data.shape)
        self.context.set_tensor_address(input_name, self.inputs[0]['device'])
        self.context.set_tensor_address(output_name, self.outputs[0]['device'])

        # Copy input
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        # Run
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()

        output = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        probs = torch.softmax(torch.tensor(output), dim=1).numpy()[0]
        pred_class = int(np.argmax(probs))
        return pred_class, probs

    def _infer_divided_image(self, image_path: str):
        parts = self.preprocess_divided_image(image_path)
        input_name = self.inputs[0]['name']
        output_name = self.outputs[0]['name']

        features_list = []
        for part in parts:
            self.context.set_input_shape(input_name, part.shape)
            self.context.set_tensor_address(input_name, self.inputs[0]['device'])
            self.context.set_tensor_address(output_name, self.outputs[0]['device'])

            np.copyto(self.inputs[0]["host"], part.ravel())
            cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
            self.stream.synchronize()

            feat = self.outputs[0]["host"].copy().reshape(self.outputs[0]["shape"])
            features_list.append(feat)

        weighted = np.mean(features_list, axis=0)
        probs = torch.softmax(torch.tensor(weighted), dim=1).numpy()[0]
        pred_class = int(np.argmax(probs))
        return pred_class, probs

    # dinfef er(self, image_path: str):
    #     is_divided = self.model_config.get("divide_image", False)
    #     if is_divided:
    #         return self._infer_divided_image(image_path)
    #     else:
    #         return self._infer_standard(image_path)

    def preprocess(self, image_path: str) -> np.ndarray:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = Compose([
            Resize(height=self.model_config.img_size, width=self.model_config.img_size),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        image = transform(image=image)["image"]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        else:
            image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image

    def preprocess_divided_image(self, image_path: str) -> List[np.ndarray]:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = Compose([
            Resize(height=self.model_config.img_size, width=self.model_config.img_size),
        ])
        augmented = transform(image=image)
        full_image = augmented["image"]
        parts = [
            full_image[:224, :224],
            full_image[:224, 224:],
            full_image[224:, :224],
            full_image[224:, 224:]
        ]
        normalized_parts = []
        normalize_transform = Compose([
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        for part in parts:
            normalized = normalize_transform(image=part)["image"]
            if len(normalized.shape) == 2:
                normalized = np.expand_dims(normalized, axis=0)
            else:
                normalized = np.transpose(normalized, (2, 0, 1))
            normalized = np.expand_dims(normalized, axis=0).astype(np.float32)
            normalized_parts.append(normalized)
        return normalized_parts

    def infer(self, image_path: str):
        return self._infer_standard(image_path)

    def benchmark(self, image_path: str, num_runs: int = 100) -> dict:
        """
        Benchmark inference performance.
        
        Args:
            image_path: Path to test image
            num_runs: Number of inference runs
            
        Returns:
            Dictionary with timing statistics
        """
        # Warm up
        for _ in range(10):
            self.infer(image_path)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.infer(image_path)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds
        
        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "fps": 1000 / np.mean(times)
        }

    def __del__(self):
        try:
            if self.stream is not None:
                self.stream.synchronize()
            for inp in self.inputs or []:
                inp["device"].free()
            for out in self.outputs or []:
                out["device"].free()
        except Exception:
            pass


class EnsembleTensorRTInfer:
    """
    Ensemble inference using multiple TensorRT engines.
    Implements the 3rd place solution's weighted ensemble approach.
    """
    
    def __init__(self, engine_paths: List[str], model_configs: List[DictConfig], weights: Optional[List[float]] = None):
        """
        Initialize ensemble of TensorRT engines.
        
        Args:
            engine_paths: List of paths to .trt engine files
            model_configs: List of model configurations (one per engine)
            weights: Weighting for each model in the ensemble (default: [0.4, 0.3, 0.3])
        """
        if weights is None:
            weights = [0.4, 0.3, 0.3]  # Default weights from competition solution
            
        if len(engine_paths) != len(model_configs) or len(engine_paths) != len(weights):
            raise ValueError("engine_paths, model_configs, and weights must have the same length")
            
        self.weights = np.array(weights) / np.sum(weights)  # Normalize weights
        self.models = []
        
        print(f"🚀 Initializing ensemble of {len(engine_paths)} TensorRT engines")
        print(f"  Using weights: {self.weights.tolist()}")
        
        for i, (path, config) in enumerate(zip(engine_paths, model_configs)):
            print(f"  Loading model {i+1}/{len(engine_paths)}: {Path(path).name}")
            self.models.append(TensorRTInfer(path, config))
        
        self.device = "cuda"
        self.class_names = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]

    def infer(self, image_path: str) -> Tuple[int, np.ndarray, List[np.ndarray]]:
        """
        Run ensemble inference on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (predicted_class, ensemble_probabilities, individual_model_probabilities)
        """
        print(f"🔮 Running ensemble inference on: {Path(image_path).name}")
        
        # Get predictions from each model
        all_probs = []
        model_names = ["vit_base_patch16_384", "vit_base_patch16_224-A", "vit_base_patch16_224-B"]
        
        for i, model in enumerate(self.models):
            pred_class, probs = model.infer(image_path)
            all_probs.append(probs)
            print(f"  Model {i+1} ({model_names[i]}): Class {pred_class} ({self.class_names[pred_class]}) | " +
                  f"Top prob: {np.max(probs):.4f}")
        
        # Weighted averaging of predictions
        weighted_sum = np.zeros_like(all_probs[0])
        for i, probs in enumerate(all_probs):
            weighted_sum += self.weights[i] * probs
        
        # Get final prediction
        final_probs = weighted_sum
        pred_class = int(np.argmax(final_probs))
        
        print(f"✅ Ensemble prediction: Class {pred_class} ({self.class_names[pred_class]})")
        print(f"   Final probabilities: {[f'{p:.4f}' for p in final_probs]}")
        
        return pred_class, final_probs, all_probs

    def benchmark(self, image_path: str, num_runs: int = 100) -> dict:
        """
        Benchmark ensemble inference performance.
        
        Args:
            image_path: Path to test image
            num_runs: Number of inference runs
            
        Returns:
            Dictionary with timing statistics
        """
        # Warm up
        for _ in range(10):
            self.infer(image_path)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.infer(image_path)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds
        
        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "fps": 1000 / np.mean(times),
            "models_count": len(self.models)
        }

    def __del__(self):
        """Clean up all models."""
        try:
            for model in self.models:
                del model
            self.models = None
        except Exception as e:
            print(f"Warning during ensemble cleanup: {e}")


def load_ensemble_engines(cfg: DictConfig, weights: Optional[List[float]] = None) -> EnsembleTensorRTInfer:
    """
    Helper function to load the ensemble of TensorRT engines based on config.
    
    Args:
        cfg: Configuration object
        weights: Optional custom weights for ensemble
        
    Returns:
        EnsembleTensorRTInfer instance
    """
    # Load model configurations
    model_configs = [
        OmegaConf.load("configs/model/model1.yaml"),  # vit_base_patch16_384
        OmegaConf.load("configs/model/model2.yaml"),  # vit_base_patch16_224 - A
        OmegaConf.load("configs/model/model3.yaml")   # vit_base_patch16_224 - B
    ]
    
    # Get engine paths
    engine_dirs = [
        Path(cfg.data.output_dir) / "models" / "model1",
        Path(cfg.data.output_dir) / "models" / "model2",
        Path(cfg.data.output_dir) / "models" / "model3",
    ]
    engine_paths = [str(d / "model.trt") for d in engine_dirs]
    
    # Verify all engines exist
    for path in engine_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"TensorRT engine not found: {path}. " +
                                   "Run training with run_full=true to generate engines.")
    
    return EnsembleTensorRTInfer(engine_paths, model_configs, weights)


# Example usage (can be removed or kept as documentation)
if __name__ == "__main__":
    import argparse
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description='TensorRT Inference for Cassava Disease Classification')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--config', type=str, default='config', help='Config name (e.g., config)')
    parser.add_argument('--config-path', type=str, default='../../../configs', help='Config directory')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble of all 3 models')
    parser.add_argument('--weights', type=float, nargs=3, default=[0.4, 0.3, 0.3], 
                        help='Ensemble weights for the 3 models')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark instead of single inference')
    
    args = parser.parse_args()

    # Use Hydra to compose config (respects 'defaults' in YAML)
    with initialize(version_base=None, config_path=args.config_path, job_name="trt_infer"):
        cfg = compose(config_name=args.config)

    # Rest of your logic unchanged...
    if args.ensemble:
        print("🚀 Loading ensemble of TensorRT engines...")
        ensemble = load_ensemble_engines(cfg, weights=args.weights)
        
        if args.benchmark:
            print("\n⏱️  Running benchmark...")
            results = ensemble.benchmark(args.image, num_runs=100)
            print(f"\n📊 Benchmark Results (Ensemble of {results['models_count']} models):")
            print(f"   Mean inference time: {results['mean_ms']:.2f} ms ± {results['std_ms']:.2f} ms")
            print(f"   FPS: {results['fps']:.2f}")
            print(f"   Range: {results['min_ms']:.2f} - {results['max_ms']:.2f} ms")
        else:
            pred_class, probs, _ = ensemble.infer(args.image)
            class_names = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
            
            print(f"\n{'='*50}")
            print(f"ENSEMBLE RESULT: Predicted class {pred_class} - {class_names[pred_class]}")
            print(f"{'='*50}")
            for i, prob in enumerate(probs):
                print(f"{class_names[i]}: {prob:.4f}")
    else:
        print("🚀 Loading single TensorRT engine...")
        model_config = OmegaConf.load("configs/model/model1.yaml")
        engine_path = Path(cfg.data.output_dir) / "models" / "model1" / "model.trt"
        
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}. Train model first.")
        
        infer = TensorRTInfer(str(engine_path), model_config)
        
        if args.benchmark:
            print("\n⏱️  Running benchmark...")
            results = infer.benchmark(args.image, num_runs=100)
            print(f"\n📊 Benchmark Results:")
            print(f"   Mean inference time: {results['mean_ms']:.2f} ms ± {results['std_ms']:.2f} ms")
            print(f"   FPS: {results['fps']:.2f}")
            print(f"   Range: {results['min_ms']:.2f} - {results['max_ms']:.2f} ms")
        else:
            pred_class, probs = infer.infer(args.image)
            class_names = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
            
            print(f"\n{'='*50}")
            print(f"SINGLE MODEL RESULT: Predicted class {pred_class} - {class_names[pred_class]}")
            print(f"{'='*50}")
            for i, prob in enumerate(probs):
                print(f"{class_names[i]}: {prob:.4f}")