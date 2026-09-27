# train.py
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ----------------------------- utilities -----------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train MobileNetV3-Small on FashionMNIST.")
    p.add_argument("--data-dir", type=Path, default=Path("data/"))
    p.add_argument("--out-dir", type=Path, default=Path("model/"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5, help="early-stop patience (epochs)")
    p.add_argument("--input-size", type=int, default=28,
                   help="H=W fed to the network. 28 = native FMNIST; try 96/112 with MobileNet.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--resume", type=Path, default=None,
                   help="path to last.pt to resume training from")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "train.log")
    fh.setFormatter(fmt)
    logger.handlers[:] = [ch, fh]
    return logger


# ------------------------------- data -------------------------------

def make_transform(input_size: int) -> v2.Compose:
    # Single-channel input. Resize is a no-op when input_size == 28.
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((input_size, input_size), antialias=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])


def build_loaders(args, device, log):
    tf = make_transform(args.input_size)

    full_train = datasets.FashionMNIST(args.data_dir, train=True,
                                       download=True, transform=tf)
    test_set = datasets.FashionMNIST(args.data_dir, train=False,
                                     download=True, transform=tf)

    n_val = int(len(full_train) * args.val_frac)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    log.info(f"Split sizes -> train={n_train} val={n_val} test={len(test_set)}")

    pin = device.type == "cuda"
    common = dict(num_workers=args.num_workers, pin_memory=pin,
                  persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, **common)
    return train_loader, val_loader, test_loader


# ------------------------------- model -------------------------------

def build_model(num_classes: int, in_ch: int, pretrained: bool) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    # Replace the stem conv so it accepts `in_ch` channels instead of 3.
    old = model.features[0][0]
    new = nn.Conv2d(in_ch, old.out_channels,
                    kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=old.bias is not None)
    if pretrained:
        with torch.no_grad():
            # average the RGB filters -> single-channel init (Kaiming-ish scale kept)
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1))
    model.features[0][0] = new

    # New classification head
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


# ---------------------------- train / eval ----------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device,
                    epoch, log, log_every=100):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for step, (x, y) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type,
                                enabled=scaler is not None):
            logits = model(x)
            loss = loss_fn(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

        if step % log_every == 0 or step == len(loader):
            log.info(f"epoch {epoch} step {step:>4}/{len(loader)} "
                     f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

    return running_loss / total, correct / total, time.time() - t0


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        running_loss += loss_fn(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


# ---------------------------- checkpointing ----------------------------

def save_ckpt(path: Path, model, optimizer, scheduler, epoch, best_val, args):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val,
        "classes": CLASSES,
        "args": {k: str(v) for k, v in vars(args).items()},
    }, path)


# ------------------------------- main -------------------------------

def main():
    args = get_args()
    log = setup_logging(args.out_dir)
    set_seed(args.seed)
    device = pick_device()
    log.info(f"device={device} | args={vars(args)}")

    train_loader, val_loader, test_loader = build_loaders(args, device, log)

    model = build_model(num_classes=len(CLASSES), in_ch=1,
                        pretrained=not args.no_pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    start_epoch = 1
    best_val = float("inf")
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val_loss"]
        log.info(f"Resumed from {args.resume} at epoch {start_epoch}, "
                 f"best_val={best_val:.4f}")

    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_acc, secs = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, epoch, log)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        log.info(f"[epoch {epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                 f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                 f"| {secs:.1f}s")

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            epochs_no_improve = 0
            save_ckpt(args.out_dir / "best.pt", model, optimizer,
                      scheduler, epoch, best_val, args)
            log.info(f"  ↳ saved new best (val_loss={best_val:.4f})")
        else:
            epochs_no_improve += 1

        # Always keep last for resume
        save_ckpt(args.out_dir / "last.pt", model, optimizer,
                  scheduler, epoch, best_val, args)

        if epochs_no_improve >= args.patience:
            log.info(f"Early stopping at epoch {epoch} "
                     f"(no val improvement for {args.patience} epochs).")
            break

    # ---- final test using the best checkpoint ----
    best_ckpt = torch.load(args.out_dir / "best.pt", map_location=device,
                           weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    log.info(f"TEST | loss={test_loss:.4f} acc={test_acc:.4f}")

    # ---- persist config + a lightweight export for inference ----
    (args.out_dir / "config.json").write_text(
        json.dumps({**vars(args), "best_val_loss": best_val,
                    "test_loss": test_loss, "test_acc": test_acc,
                    "classes": CLASSES},
                   default=str, indent=2))

    torch.save({
        "model": model.state_dict(),
        "classes": CLASSES,
        "input_size": args.input_size,
        "in_ch": 1,
        "normalize": {"mean": [0.5], "std": [0.5]},
    }, args.out_dir / "model.pt")

    log.info(f"Artifacts written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()