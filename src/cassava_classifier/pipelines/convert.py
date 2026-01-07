import tempfile
from pathlib import Path

import onnx  # Add this import
import torch
import torch.serialization
from omegaconf import DictConfig

from ..models.model import CassavaLightningModule


def convert_to_onnx(checkpoint_path: str, output_path: str, model_config: DictConfig):
    torch.serialization.add_safe_globals([type(model_config)])
    model = CassavaLightningModule.load_from_checkpoint(
        checkpoint_path,
        model_config=model_config,
        map_location="cpu",
        weights_only=False,
    )
    model.eval()

    if model_config.get("divide_image", False):
        dummy_input = torch.randn(1, 3, model_config.img_size, model_config.img_size)
    else:
        dummy_input = torch.randn(1, 3, model_config.img_size, model_config.img_size)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_onnx = Path(tmpdir) / "model.onnx"

            torch.onnx.export(
                model,
                dummy_input,
                temp_onnx,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            onnx_model = onnx.load(temp_onnx)
            onnx.save(onnx_model, output_path, save_as_external_data=False)

        print(f"Model successfully converted to ONNX: {output_path}")
        return True
    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        return False
