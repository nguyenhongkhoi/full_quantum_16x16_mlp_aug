"""Dữ liệu: seed, MNIST / Fashion-MNIST, split cố định 50.000 train / 10.000 validation, augmentation."""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
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
    limit = getattr(args, "limit_samples", 0) or 0
    if limit > 0:  # smoke test: chỉ dùng `limit` ảnh đầu của train và của validation
        train_indices, validation_indices = train_indices[:limit], validation_indices[:limit]

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
