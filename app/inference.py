# inference.py
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import v2


def load_bundle(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    input_size = ckpt["input_size"]
    in_ch = ckpt["in_ch"]
    classes = ckpt["classes"]
    mean, std = ckpt["normalize"]["mean"], ckpt["normalize"]["std"]

    model = mobilenet_v3_small(weights=None)
    old = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_ch, old.out_channels, kernel_size=old.kernel_size,
        stride=old.stride, padding=old.padding, bias=old.bias is not None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((input_size, input_size), antialias=True),
        v2.Normalize(mean=mean, std=std),
    ])
    return model, tf, classes, input_size


@torch.no_grad()
def predict(image_path, model, tf, classes, input_size, device):
    img = Image.open(image_path).convert("L")   # RGB / RGBA / P / L -> single-channel
    x = tf(img).unsqueeze(0).to(device, non_blocking=True)
    probs = F.softmax(model(x), dim=1)
    conf, idx = probs.max(dim=1)
    return classes[idx.item()], conf.item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image-path", required=True)
    ap.add_argument("-m", "--model-path",
                    default="model/model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, classes, input_size = load_bundle(Path(args.model_path), device)

    # Warmup — forces MIOpen to compile kernels for our shape now, not on the
    # first user request.
    with torch.no_grad():
        model(torch.zeros(1, 1, input_size, input_size, device=device))

    label, conf = predict(args.image_path, model, tf, classes, input_size, device)
    print(f"{label} ({conf * 100:.2f}%)")