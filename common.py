"""Phần dùng chung cho train.py, analyze_routing.py: chia dữ liệu, seed, đánh giá mọi chế độ trộn trong một lượt.

make_loaders chép nguyên từ qcnn_qconv4x3/train.py (cùng split 50.000 / 10.000 theo --split-seed, cùng aug).
"""
from __future__ import annotations

import argparse
import math
import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Bộ dữ liệu ảnh xám 28x28, 10 lớp; model tự resize về 16x16. "fashion" = Fashion-MNIST.
DATASETS = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}
DATASET_NAMES = {"mnist": "MNIST", "fashion": "Fashion-MNIST"}

AUG_INTERPOLATION = {
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "nearest": transforms.InterpolationMode.NEAREST,
}


def make_loaders(args: argparse.Namespace, device: torch.device):
    # Chia cố định 50.000 train / 10.000 validation
    split_order = torch.randperm(
        60_000, generator=torch.Generator().manual_seed(args.split_seed)
    ).tolist()
    validation_indices = split_order[:10_000]
    train_indices = split_order[10_000:]

    has_aug = args.aug_degrees > 0 or args.aug_translate > 0
    if has_aug:
        train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=args.aug_degrees,
                translate=(args.aug_translate, args.aug_translate),
                interpolation=AUG_INTERPOLATION[args.aug_interpolation],
                fill=0,
            ),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.ToTensor()

    validation_transform = transforms.ToTensor()

    dataset_cls = DATASETS[getattr(args, "dataset", "mnist")]
    train_base = dataset_cls(
        root=args.data_root, train=True, download=True, transform=train_transform
    )
    validation_base = dataset_cls(
        root=args.data_root, train=True, download=True, transform=validation_transform
    )

    pin = device.type == "cuda"
    train_generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        Subset(train_base, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    validation_loader = DataLoader(
        Subset(validation_base, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    return train_loader, validation_loader


@torch.no_grad()
def collect(model, loader: DataLoader, device: torch.device, progress=None) -> Dict[str, torch.Tensor]:
    """Một lượt eval: nhãn, đầu ra soft, g (batch, E), logit từng cây (batch, E, C; None với head fusion),
    loss soft từng ảnh. Trên CPU."""
    model.eval()
    labels, outputs, gates, expert_logits, losses = [], [], [], [], []
    for index, (images, target) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        target = target.to(device, non_blocking=device.type == "cuda")
        output = model(images, mode="soft")
        losses.append(F.cross_entropy(output, target, reduction="none").cpu())
        labels.append(target.cpu())
        outputs.append(output.float().cpu())
        gates.append(model.last["gates"].float().cpu())
        if model.last["expert_logits"] is not None:
            expert_logits.append(model.last["expert_logits"].float().cpu())
        if progress is not None:
            progress(index, len(loader))
    return {"labels": torch.cat(labels), "outputs": torch.cat(outputs), "gates": torch.cat(gates),
            "expert_logits": torch.cat(expert_logits) if expert_logits else None, "losses": torch.cat(losses)}


def predictions(model, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Dự đoán (nhãn) của mọi chế độ: soft, top1, uniform, tree<i>. Head fusion: chỉ soft."""
    out = {"soft": data["outputs"].argmax(dim=1)}
    gates, logits = data["gates"], data["expert_logits"]
    if logits is None:
        return out
    for mode in ("top1", "uniform"):
        out[mode] = model.combine(logits, gates, mode).argmax(dim=1)
    for index in range(logits.shape[1]):
        out[f"tree{index}"] = logits[:, index].argmax(dim=1)
    return out


def summarize(model, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Các số của một lượt eval (accuracy theo tỷ lệ 0..1). Head fusion: top1, uniform, từng cây, oracle = nan."""
    labels, gates = data["labels"], data["gates"]
    experts = gates.shape[1]
    preds = predictions(model, data)
    result = {"loss": data["losses"].mean().item()}
    for mode in ["soft", "top1", "uniform", *[f"tree{i}" for i in range(experts)]]:
        result[f"acc_{mode}"] = (preds[mode] == labels).float().mean().item() if mode in preds else math.nan
    if data["expert_logits"] is None:
        result["acc_oracle"] = math.nan
    else:
        any_correct = torch.stack([preds[f"tree{i}"] == labels for i in range(experts)], dim=1).any(dim=1)
        result["acc_oracle"] = any_correct.float().mean().item()
    chosen = gates.argmax(dim=1)
    for index in range(experts):
        result[f"gate_mean{index}"] = gates[:, index].mean().item()
        result[f"top1_share{index}"] = (chosen == index).float().mean().item()
    entropy = -(gates * gates.clamp_min(1e-12).log()).sum(dim=1) / math.log(experts) if experts > 1 else gates[:, 0] * 0
    result["gate_entropy"] = entropy.mean().item()
    return result
