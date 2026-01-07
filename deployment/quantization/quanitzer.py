import onnx
from onnxconverter_common import float16
import os
models = {
    "model1": "triton/model1/1/model.onnx",
    "model2": "triton/model2/1/model.onnx",
    "model3": "triton/model3/1/model.onnx",
}
output_dir = "triton_quantizeds"
os.makedirs(output_dir, exist_ok=True)
for name, path in models.items():
    if not os.path.exists(path):
        continue

    print(f"converting {name} to fp16")
    output_path = os.path.join(output_dir, f"{name}_fp16.onnx")

    try:
        model = onnx.load(path)
        model_fp16 = float16.convert_float_to_float16(
            model, keep_io_types=True, disable_shape_infer=True
        )
        onnx.save(model_fp16, output_path)
        print(f"success: {output_path}")
    except Exception as e:
        print(f"failed {name}: {e}")
