from pathlib import Path
from functools import lru_cache

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import v2


class Classifier:
    def __init__(self, model_path: Path, device: torch.device | None = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        self.input_size = ckpt["input_size"]
        self.in_ch = ckpt["in_ch"]
        self.classes = ckpt["classes"]
        mean, std = ckpt["normalize"]["mean"], ckpt["normalize"]["std"]

        model = mobilenet_v3_small(weights=None)
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            self.in_ch, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=old.bias is not None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features,
                                        len(self.classes))
        model.load_state_dict(ckpt["model"])
        self.model = model.eval().to(self.device)

        steps = [v2.ToImage()]
        if self.in_ch == 1:
            steps.append(v2.Grayscale(num_output_channels=1))
        steps += [
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((self.input_size, self.input_size), antialias=True),
            v2.Normalize(mean=mean, std=std),
        ]
        self.tf = v2.Compose(steps)

        # Warmup — trigger MIOpen/cuDNN kernel compile now, not on first request
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_ch, self.input_size,
                                self.input_size, device=self.device)
            self.model(dummy)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> tuple[str, float]:
        x = self.tf(image).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(x), dim=1)[0]
        conf, idx = probs.max(dim=0)
        return self.classes[idx.item()], conf.item()

    @torch.no_grad()
    def predict_topk(self, image: Image.Image, k: int = 3):
        x = self.tf(image).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(x), dim=1)[0]
        top = probs.topk(min(k, len(self.classes)))
        return [(self.classes[i], float(p))
                for p, i in zip(top.values, top.indices)]