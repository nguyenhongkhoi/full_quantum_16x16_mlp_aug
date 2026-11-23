"""Classical controls for the front-end models: the same ClassicalFront, with the circuit removed.

Both take the same 16x16 input as FusionQCNN (bilinear resize, values in [0, 1]) and use the same front end
(models/front.py): with --front deep12p, the same layers and the same kind of start (last conv at 0). The random
starting values of the first convs differ, as they are drawn at another point of the random stream. Its output
(theta, and phi for deep<C>p) becomes 1 or 2 image channels.

    EncoderLinear  front -> AvgPool 4x4 (16x16 -> 4x4) -> Linear(channels * 16, 10).
                   The smallest readout: almost all parameters are the encoder's.
                   deep12p: 1,646 + 330 = 1,976.
    EncoderCNN     front -> Conv(ch->a) BN ReLU MaxPool -> Conv(a->b) BN ReLU MaxPool -> Linear(16b, h) ReLU Dropout
                   -> Linear(h, 10). The circuit and its head are replaced by a classical CNN of about the same size.
                   deep12p, a=20, b=32, h=20: 1,646 + 16,746 = 18,392 (deep12p + circuit + projection 10: 18,418).
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from .front import ClassicalFront, parse_front
from .preprocessing import preprocess_full_quantum

TARGET_SIZE = (16, 16)


def _front_channels(front: str) -> int:
    kind, _, phase = parse_front(front)
    if kind == "none":
        raise ValueError("The classical controls need a front end (e.g. --front deep12p)")
    return 2 if phase else 1


class _ClassicalBase(nn.Module):
    def __init__(self, front: str):
        super().__init__()
        self.front_mode = front
        self.channels = _front_channels(front)
        self.front = ClassicalFront(front, *TARGET_SIZE)

    def encoded(self, images: torch.Tensor) -> torch.Tensor:
        """(batch, channels, 16, 16): theta, and phi if the front end has it."""
        values = preprocess_full_quantum(images, TARGET_SIZE, grayscale=True, mode="resize")
        theta, phi = self.front(values)
        maps = [theta] if phi is None else [theta, phi]
        return torch.stack(maps, dim=1).reshape(-1, self.channels, *TARGET_SIZE)

    def count_parameters(self) -> Dict[str, int]:
        front = sum(parameter.numel() for parameter in self.front.parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"quantum": 0, "front": front, "head": total - front, "total": total, "quantum_share": 0.0}


class EncoderLinear(_ClassicalBase):
    def __init__(self, front: str = "deep12p", num_classes: int = 10):
        super().__init__(front)
        self.head = nn.Sequential(nn.AvgPool2d(4), nn.Flatten(), nn.Linear(self.channels * 16, num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoded(images))


class EncoderCNN(_ClassicalBase):
    def __init__(self, front: str = "deep12p", widths: Sequence[int] = (20, 32), hidden: int = 20,
                 dropout: float = 0.2, num_classes: int = 10):
        super().__init__(front)
        first, second = widths
        self.body = nn.Sequential(
            nn.Conv2d(self.channels, first, 3, padding=1), nn.BatchNorm2d(first), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(first, second, 3, padding=1), nn.BatchNorm2d(second), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(second * 4 * 4, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.encoded(images)))
