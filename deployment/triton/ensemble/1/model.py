# triton/ensemble/1/model.py
import numpy as np
import triton_python_backend_utils as pb_utils
from albumentations import Resize, Normalize, Compose
import cv2

class TritonPythonModel:
    def initialize(self, args):
        # Preprocessing for different model sizes
        self.transform_384 = Compose([
            Resize(384, 384),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_448 = Compose([
            Resize(448, 448),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.weights = np.array([0.4, 0.3, 0.3], dtype=np.float32)

    def preprocess(self, image, size):
        image = cv2.resize(image, (size, size))
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5        # normalize
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        return image

    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            image = input_tensor.as_numpy()  # HWC uint8

            img384 = np.expand_dims(self.preprocess(image, 384), axis=0)
            img448 = np.expand_dims(self.preprocess(image, 448), axis=0)

            infer1 = pb_utils.InferenceRequest(
                model_name="model1",
                inputs=[pb_utils.Tensor("input", img384)],
                requested_output_names=["output"]
            )

            infer2 = pb_utils.InferenceRequest(
                model_name="model2",
                inputs=[pb_utils.Tensor("input", img448)],
                requested_output_names=["output"]
            )

            infer3 = pb_utils.InferenceRequest(
                model_name="model3",
                inputs=[pb_utils.Tensor("input", img448)],
                requested_output_names=["output"]
            )

            # ✅ Correct execution
            resp1 = infer1.exec()
            resp2 = infer2.exec()
            resp3 = infer3.exec()

            out1 = pb_utils.get_output_tensor_by_name(resp1, "output").as_numpy().squeeze()
            out2 = pb_utils.get_output_tensor_by_name(resp2, "output").as_numpy().squeeze()
            out3 = pb_utils.get_output_tensor_by_name(resp3, "output").as_numpy().squeeze()

            probs1 = self._softmax(out1)
            probs2 = self._softmax(out2)
            probs3 = self._softmax(out3)

            final_probs = (
                self.weights[0] * probs1 +
                self.weights[1] * probs2 +
                self.weights[2] * probs3
            )

            output_tensor = pb_utils.Tensor("ENSEMBLE_OUTPUT", final_probs)
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def finalize(self):
        pass