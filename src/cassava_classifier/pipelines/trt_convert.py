# /kaggle/working/Cassava-Disease-Detection/src/cassava_classifier/pipelines/trt_convert.py

import argparse
from pathlib import Path

import onnx
import tensorrt as trt
import torch


def build_engine(onnx_file: Path, engine_file: Path, fp16: bool = False):
    assert torch.cuda.is_available(), "CUDA not available"
    _ = torch.zeros(1).cuda()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    onnx_model = onnx.load(onnx_file)
    input_proto = onnx_model.graph.input[0]
    shape = [d.dim_value for d in input_proto.type.tensor_type.shape.dim]
    _, c, h, w = shape
    print(f"ONNX input shape: (batch, {c}, {h}, {w})")

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)

    profile.set_shape(
        input_tensor.name,
        min=(1, c, h, w),
        opt=(1, c, h, w),
        max=(1, c, h, w),
    )
    config.add_optimization_profile(profile)
    print("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT build failed")

    engine_file.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_file, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT saved → {engine_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    build_engine(Path(args.onnx), Path(args.output), args.fp16)


if __name__ == "__main__":
    main()
