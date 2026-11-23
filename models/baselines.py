from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .bond2_tree import IMAGE_SIZE, _count_parameters


class SmallCNN(nn.Module):
    """R1: ClassicalCNN(parameter_matched=True) layout on the same 16x16 input as the quantum variants.

    widths=(8, 16) is that exact model; smaller widths keep the layout for size-matched comparisons.
    """

    def __init__(self, widths: Tuple[int, int] = (8, 16), num_classes: int = 10):
        super().__init__()
        first, second = widths
        self.core = nn.Sequential(
            nn.Conv2d(1, first, 3, padding=1),
            nn.BatchNorm2d(first),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(first, second, 3, padding=1),
            nn.BatchNorm2d(second),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(second * 4 * 4, second),
            nn.ReLU(inplace=True),
            nn.Linear(second, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                               align_corners=False)
        return self.head(self.core(values))

    def count_parameters(self) -> Dict[str, int]:
        return _count_parameters(self)
