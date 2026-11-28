"""Classical CNN baselines on the same 16x16 MNIST input as the QCNN models.

Layout (SmallCNN, the layout of the earlier cnn1k..cnn5k and of Mammoth cnn-3k/5k/12k/19k):
    resize 28x28 -> 16x16 (bilinear, align_corners=False), clamp to [0, 1]
    Conv(1->a, 3x3, pad 1) BN ReLU MaxPool2      16x16 -> 8x8
    Conv(a->b, 3x3, pad 1) BN ReLU MaxPool2      8x8 -> 4x4
    Flatten (16b) -> Linear(16b, h) ReLU [Dropout] -> Linear(h, 10)       (h = b unless --hidden is given)
Parameters with h = b: 12a + 9ab + 16b^2 + 14b + 10 (BatchNorm running statistics are buffers, not counted).
The clamp is a no-op for bilinear resizing of [0, 1] images; it is kept to match the QCNN preprocessing exactly.

Layout (StandardCNN, --arch standard: a common VGG-style CNN, reference baseline ~85k parameters, preset std85k):
    Conv(1->a) BN ReLU, Conv(a->a) BN ReLU, MaxPool2              16x16 -> 8x8
    Conv(a->b) BN ReLU, Conv(b->b) BN ReLU, MaxPool2              8x8 -> 4x4
    Global average pool (b) -> Linear(b, h) ReLU [Dropout] -> Linear(h, 10)       (h = 128 unless --hidden)
Parameters: 9a^2 + 15a + 9ab + 9b^2 + 6b + bh + 11h + 10 (BN weight + bias counted). std85k: a=32, b=70, h=128
-> 84,754.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

IMAGE_SIZE = 16

# name: (a, b); hidden = b. Counts are checked in test_model.py.
PRESETS = {
    "cnn1k": (4, 6),      # 934     (= earlier cnn1k)
    "cnn2k": (6, 9),      # 1,990   (= earlier cnn2k)
    "cnn3k": (8, 11),     # 2,988   (= earlier cnn3k, Mammoth cnn-3k)
    "cnn5k": (8, 16),     # 5,578   (= earlier cnn5k, Mammoth cnn-5k)
    "cnn12k": (11, 24),   # 12,070  (= Mammoth cnn-12k)
    "cnn17k": (16, 28),   # 17,170  (size of qcnn + qconv4x3, 17,264)
    "cnn19k": (15, 30),   # 19,060  (= Mammoth cnn-19k)
    "cnn30k": (20, 38),   # 30,726
    "cnn50k": (24, 49),   # 49,984
    "cnn85k": (32, 64),   # 85,258
}
EXPECTED = {"cnn1k": 934, "cnn2k": 1990, "cnn3k": 2988, "cnn5k": 5578, "cnn12k": 12070, "cnn17k": 17170,
            "cnn19k": 19060, "cnn30k": 30726, "cnn50k": 49984, "cnn85k": 85258}


# StandardCNN presets: name: (a, b, hidden)
STANDARD_PRESETS = {
    "std85k": (32, 70, 128),  # 84,754
}
STANDARD_EXPECTED = {"std85k": 84754}
ARCHS = ("small", "standard")


def preprocess(images: torch.Tensor) -> torch.Tensor:
    """(batch, 1, 28, 28) in [0, 1] -> (batch, 1, 16, 16), as preprocess_full_quantum before flattening."""
    values = F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    return values.clamp(0.0, 1.0)


class SmallCNN(nn.Module):
    def __init__(self, widths: Sequence[int] = (8, 16), hidden: Optional[int] = None, dropout: float = 0.0,
                 num_classes: int = 10):
        super().__init__()
        first, second = widths
        hidden = hidden or second
        self.core = nn.Sequential(
            nn.Conv2d(1, first, 3, padding=1), nn.BatchNorm2d(first), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(first, second, 3, padding=1), nn.BatchNorm2d(second), nn.ReLU(), nn.MaxPool2d(2),
        )
        layers = [nn.Flatten(), nn.Linear(second * (IMAGE_SIZE // 4) ** 2, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.core(preprocess(images)))

    def count_parameters(self) -> Dict[str, int]:
        conv = sum(p.numel() for p in self.core.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return {"conv": conv, "head": head, "total": conv + head}


class StandardCNN(nn.Module):
    """VGG-style: two blocks of (conv-BN-ReLU) x 2 + max-pool, global average pool, MLP head."""

    def __init__(self, widths: Sequence[int] = (32, 70), hidden: Optional[int] = None, dropout: float = 0.0,
                 num_classes: int = 10):
        super().__init__()
        first, second = widths
        hidden = hidden or 128

        def conv(cin: int, cout: int) -> list:
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]

        self.core = nn.Sequential(
            *conv(1, first), *conv(first, first), nn.MaxPool2d(2),
            *conv(first, second), *conv(second, second), nn.MaxPool2d(2),
        )
        layers = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(second, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.core(preprocess(images)))

    def count_parameters(self) -> Dict[str, int]:
        conv = sum(p.numel() for p in self.core.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return {"conv": conv, "head": head, "total": conv + head}


def standard_count(a: int, b: int, h: int = 128) -> int:
    return 9 * a * a + 15 * a + 9 * a * b + 9 * b * b + 6 * b + b * h + 11 * h + 10


def resolve_widths(cfg: dict) -> tuple[int, int]:
    """--widths a b overrides --preset."""
    widths = cfg.get("widths")
    if widths:
        return int(widths[0]), int(widths[1])
    if cfg.get("arch", "small") == "standard":
        preset = cfg.get("preset") or "std85k"
        if preset not in STANDARD_PRESETS:
            raise ValueError(f"--arch standard: preset must be one of {list(STANDARD_PRESETS)} or --widths a b")
        a, b, _ = STANDARD_PRESETS[preset]
        return a, b
    preset = cfg.get("preset") or "cnn5k"
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; use one of {list(PRESETS)} or --widths a b")
    return PRESETS[preset]


def resolve_hidden(cfg: dict) -> int:
    """--hidden overrides; default 128 for standard (all STANDARD_PRESETS use 128), b for small."""
    if cfg.get("hidden"):
        return int(cfg["hidden"])
    if cfg.get("arch", "small") == "standard":
        return 128
    return resolve_widths(cfg)[1]


def build_model(cfg: dict) -> nn.Module:
    """From run arguments (vars(args)) or the args saved in a checkpoint config (old configs: arch small)."""
    if cfg.get("arch", "small") == "standard":
        return StandardCNN(widths=resolve_widths(cfg), hidden=resolve_hidden(cfg), dropout=cfg.get("dropout", 0.0))
    return SmallCNN(widths=resolve_widths(cfg), hidden=cfg.get("hidden"), dropout=cfg.get("dropout", 0.0))
