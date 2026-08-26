import base64
import io

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms


class EfficientNetB7Pth(torch.nn.Module):
    def __init__(self, model_path="resources/tf_efficientnet_b7_ns-1dbc32de.pth"):
        super().__init__()
        original_model = timm.create_model(
            "tf_efficientnet_b7_ns",
            pretrained=False,
            checkpoint_path=model_path,
        )
        self.features = torch.nn.Sequential(*list(original_model.children())[:-5])

        image_size = 600
        self.tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = self.features.to(self.device)
        self.features.eval()

    def forward(self, img, mode=0):
        if mode == 0:
            x = Image.open(io.BytesIO(img))
        elif mode == 1:
            x = Image.open(io.BytesIO(base64.b64decode(img)))
        else:
            raise ValueError("mode must be 0 (bytes) or 1 (base64)")
        x = self.tfms(x.convert("RGB")).unsqueeze(0).to(self.device)
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        result = x.cpu().detach().numpy().flatten()
        result = result / np.linalg.norm(result)
        return result.tolist()
