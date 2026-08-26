import os

from loguru import logger


class FeatureService:

    def __init__(self):
        backend = os.getenv("FEATURE_BACKEND", "onnx").lower()
        if backend in {"ascend", "ais_bench"}:
            from .feature_model_ascend import EfficientNetB7Ascend

            self.model = EfficientNetB7Ascend(
                om_path=os.getenv("FEATURE_OM_PATH"),
                input_name=os.getenv("FEATURE_OM_INPUT_NAME", "input"),
                input_shape=os.getenv("FEATURE_OM_INPUT_SHAPE", "1,3,600,600"),
                input_dtype=os.getenv("FEATURE_OM_INPUT_DTYPE", "float32"),
                output_dtype=os.getenv("FEATURE_OM_OUTPUT_DTYPE", None),
                output_shape=os.getenv("FEATURE_OM_OUTPUT_SHAPE", "1,64,19,19"),
                device_id=int(os.getenv("FEATURE_DEVICE_ID", "0")),
            )
            logger.info("Feature backend: ascend (ais_bench/OM)")
        elif backend == "onnx":
            from .feature_model_onnx import EfficientNetB7Onnx

            self.model = EfficientNetB7Onnx()
            logger.info("Feature backend: onnxruntime")
        elif backend == "pth":
            from .feature_model_pth import EfficientNetB7Pth

            self.model = EfficientNetB7Pth(os.getenv(
                "FEATURE_PTH_PATH",
                "resources/tf_efficientnet_b7_ns-1dbc32de.pth",
            ))
            logger.info("Feature backend: pytorch (PTH)")
        else:
            raise ValueError(f"Unsupported FEATURE_BACKEND: {backend}")

    async def feature(self, bytes) -> list[float]:
        return self.model.forward(bytes, 0)
