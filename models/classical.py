"""Classical controls for the front-end models: the same ClassicalFront, with the circuit removed.

Both take the same 16x16 input as FusionQCNN (bilinear resize, values in [0, 1]) and use the same front end
(models/front.py): with --front deep12p, the same layers and the same kind of start (last conv at 0). The random
starting values of the first convs differ, as they are drawn at another point of the random stream. Its output
(theta, and phi for deep<C>p) becomes 1 or 2 image channels.

    full_quantum_16x16_level32_qconv (2026-11-13): the front end may also be the quantum encoder qconv<L>[x<S>][f]
    (models/front.py, QuantumFront): enc-linear with --front qconv2x2 is "quantum + linear" (220 + 330 = 550).

    EncoderLinear  front -> AvgPool PxP -> Linear(channels * (16/P)^2, 10), P = --linear-pool (default 4).
                   The smallest readout: almost all parameters are the encoder's.
                   P 4: deep12p 1,646 + 330 = 1,976. P 2 (2026-11-14): readout 1,290 (2 channels).
                   P 1: readout 5,130.
    EncoderCNN     front -> Conv(ch->a) BN ReLU MaxPool -> Conv(a->b) BN ReLU MaxPool -> Linear(16b, h) ReLU Dropout
                   -> Linear(h, 10). The circuit and its head are replaced by a classical CNN of about the same size.
                   deep12p, a=20, b=32, h=20: 1,646 + 16,746 = 18,392 (deep12p + circuit + projection 10: 18,418).
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from .front import front_circuit_parameters, front_has_phase, make_front
from .preprocessing import preprocess_full_quantum

TARGET_SIZE = (16, 16)


def _front_channels(front: str) -> int:
    if front == "none":
        raise ValueError("The encoder models need a front end (e.g. --front deep12p or --front qconv2x2)")
    return 2 if front_has_phase(front) else 1


class _ClassicalBase(nn.Module):
    def __init__(self, front: str):
        super().__init__()
        self.front_mode = front
        self.channels = _front_channels(front)
        self.front = make_front(front, *TARGET_SIZE)

    def encoded(self, images: torch.Tensor) -> torch.Tensor:
        """(batch, channels, 16, 16): theta, and phi if the front end has it."""
        values = preprocess_full_quantum(images, TARGET_SIZE, grayscale=True, mode="resize")
        theta, phi = self.front(values)
        maps = [theta] if phi is None else [theta, phi]
        return torch.stack(maps, dim=1).reshape(-1, self.channels, *TARGET_SIZE)

    def count_parameters(self) -> Dict[str, int]:
        front = sum(parameter.numel() for parameter in self.front.parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        front_quantum = front_circuit_parameters(self.front)
        return {"quantum": 0, "front": front, "front_quantum": front_quantum, "head": total - front, "total": total,
                "quantum_share": front_quantum / total}


class EncoderLinear(_ClassicalBase):
    def __init__(self, front: str = "deep12p", num_classes: int = 10, pool: int = 4):
        super().__init__(front)
        if pool not in (1, 2, 4, 8, 16):
            raise ValueError(f"linear pool must divide 16, got {pool}")
        cells = (TARGET_SIZE[0] // pool) * (TARGET_SIZE[1] // pool)
        self.head = nn.Sequential(nn.AvgPool2d(pool), nn.Flatten(), nn.Linear(self.channels * cells, num_classes))

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


class CNN3Branch(nn.Module):
    """Classical control with the SAME readout as FusionQCNN (linhead, 2026-11-15): the quantum part (encoder and
    the three trees) is replaced by three independent small CNNs, one per tree, each on the whole 16x16 image.

    Branch: Conv(1->a, 3x3) BN ReLU MaxPool2 -> Conv(a->b, 3x3) BN ReLU MaxPool2 -> Flatten (16b features), the
    layout of the classical cnn5k (SmallCNN). widths (8, 23): 1,821 parameters and 368 features per branch
    (a tree gives 372). The head is built exactly as in FusionQCNN for the same --head:
      mlp: per-branch Linear(368, projection) LayerNorm GELU Dropout, then Linear(3*projection, hidden) LayerNorm
           GELU Dropout Linear(hidden, 10). projection 10: total 17,291 (qcnn + qconv4x3: 17,264).
      linear3: per-branch Linear(368, 10), logits = mean: 16,533 (qcnn + qconv4x3: 16,506).
      linear-shared: one Linear(368, 10) for all branches: 9,153 (9,046).
    Input: the same preprocessing as FusionQCNN (bilinear resize to 16x16, values clamped to [0, 1]).
    """

    def __init__(self, widths: Sequence[int] = (8, 23), head: str = "mlp", projection: int = 10, hidden: int = 16,
                 dropout: float = 0.0, branches: int = 3, num_classes: int = 10):
        super().__init__()
        if head not in ("mlp", "linear3", "linear-shared"):
            raise ValueError(f"Unknown head {head!r}")
        first, second = widths
        self.head_mode = head
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, first, 3, padding=1), nn.BatchNorm2d(first), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(first, second, 3, padding=1), nn.BatchNorm2d(second), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
            )
            for _ in range(branches)
        ])
        features = second * (TARGET_SIZE[0] // 4) * (TARGET_SIZE[1] // 4)
        if head == "linear3":
            self.projections = nn.ModuleList([nn.Linear(features, num_classes) for _ in range(branches)])
            self.fusion = None
        elif head == "linear-shared":
            shared = nn.Linear(features, num_classes)
            self.projections = nn.ModuleList([shared] * branches)
            self.fusion = None
        else:
            self.projections = nn.ModuleList([
                nn.Sequential(nn.Linear(features, projection), nn.LayerNorm(projection), nn.GELU(),
                              nn.Dropout(dropout))
                for _ in range(branches)
            ])
            self.fusion = nn.Sequential(
                nn.Linear(branches * projection, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = preprocess_full_quantum(images, TARGET_SIZE, grayscale=True, mode="resize")
        grid = values.reshape(-1, 1, *TARGET_SIZE)
        outputs = [projection(branch(grid)) for branch, projection in zip(self.branches, self.projections)]
        if self.fusion is None:
            return torch.stack(outputs, dim=0).mean(dim=0)
        return self.fusion(torch.cat(outputs, dim=1))

    def count_parameters(self) -> Dict[str, int]:
        body = sum(parameter.numel() for parameter in self.branches.parameters())
        head = sum(parameter.numel() for module in (self.projections, self.fusion) if module is not None
                   for parameter in module.parameters())
        # "front" holds the CNN body (the part that replaces encoder + trees), so the runner prints it.
        return {"quantum": 0, "front": body, "front_quantum": 0, "head": head, "total": body + head,
                "quantum_share": 0.0}
