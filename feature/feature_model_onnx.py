from typing import List, Optional

import numpy as np
from torchvision import transforms

from .base_model_onnx import BaseOnnxModel


class EfficientNetB7Onnx(BaseOnnxModel):
    def __init__(
        self,
        onnx_model_path: str = "resources/tf_efficientnet_b7_ns-1dbc32de.onnx",
        providers: Optional[List[str]] = None,
    ):
        image_size = 600
        tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        super().__init__(
            onnx_model_path=onnx_model_path,
            transforms=tfms,
            providers=providers,
        )

    def decode(self, outputs: List[np.ndarray]) -> np.ndarray:
        y = outputs[0]
        if y.ndim == 4:
            y = y.mean(axis=(2, 3))
        elif y.ndim != 2:
            y = y.reshape((y.shape[0], -1))
        return y[0]
