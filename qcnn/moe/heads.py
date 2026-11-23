"""Head đọc đặc trưng đo của mỗi cây (expert): Linear con, bỏ tương quan (nocorr), tương quan cùng trục
(samebasis / zz), quan sát Pauli cố định (obs10).
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import torch
from torch import nn


FINAL_QUBITS = 8


PAIRS = FINAL_QUBITS * (FINAL_QUBITS - 1) // 2


BRANCH_FEATURES = 3 * FINAL_QUBITS + 9 * PAIRS


AXES = "XYZ"


DEFAULT_OBS = ("Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "X0", "X4")


def observable_index(term: str, qubits: int = FINAL_QUBITS) -> int:
    """Vị trí của một quan sát trong đặc trưng PauliFeatures: "Z3" (một qubit) hoặc "X0Z5" (cặp)."""
    term = term.strip().upper()
    parts = [(term[i], int(term[i + 1])) for i in range(0, len(term), 2)] if len(term) in (2, 4) else []
    if not parts or any(axis not in AXES or not 0 <= qubit < qubits for axis, qubit in parts):
        raise ValueError(f"Bad observable {term!r}: use e.g. Z3 or X0Z5 (axes X/Y/Z, qubits 0-{qubits - 1})")
    if len(parts) == 1:
        axis, qubit = parts[0]
        return AXES.index(axis) * qubits + qubit
    (axis_i, first), (axis_j, second) = parts
    if first == second:
        raise ValueError(f"Bad observable {term!r}: the two qubits must differ")
    if first > second:
        (axis_i, first), (axis_j, second) = (axis_j, second), (axis_i, first)
    pair = list(combinations(range(qubits), 2)).index((first, second))
    return 3 * qubits + 9 * pair + 3 * AXES.index(axis_i) + AXES.index(axis_j)


class ObservableHead(nn.Module):
    """logit_c = scale_c * <O_c> + bias_c với num_classes quan sát cố định (scale khởi tạo 1, bias 0)."""

    def __init__(self, terms: Sequence[str], num_classes: int):
        super().__init__()
        if len(terms) != num_classes:
            raise ValueError(f"--obs-terms needs exactly {num_classes} observables, got {len(terms)}")
        self.terms = tuple(terms)
        self.register_buffer("index", torch.tensor([observable_index(term) for term in terms]), persistent=False)
        self.scale = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.index_select(1, self.index) * self.scale + self.bias


class SliceLinear(nn.Module):
    """Linear trên `width` đặc trưng đầu tiên (bloch8: 24 = X, Y, Z của 8 qubit cuối)."""

    def __init__(self, width: int, num_classes: int):
        super().__init__()
        self.width = width
        self.linear = nn.Linear(width, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features[:, :self.width])


class NoCorrLinear(nn.Module):
    """nocorr: Linear trên X, Y, Z 8 qubit cuối (24) + tầng side (96), bỏ 252 tương quan cặp -> Linear(120, 10)."""

    def __init__(self, side_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(3 * FINAL_QUBITS + side_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features = [24 Bloch | 252 tương quan | side]
        return self.linear(torch.cat((features[:, :3 * FINAL_QUBITS], features[:, BRANCH_FEATURES:]), dim=1))


class NoCorrSmall(nn.Module):
    """Head nhỏ hơn nocorr (thêm 2026-11-19), vẫn bỏ 252 tương quan cặp.

    side_axes "XYZ": 24 Bloch + 3 * side; "Z": 24 Bloch + chỉ <Z> tầng side (bloch() xếp X hết, Y hết, Z hết -> Z là
    `side` số cuối). rank 0: Linear(in, 10); rank r > 0: Linear(in, r, không bias) rồi Linear(r, 10) (hạng thấp)."""

    def __init__(self, side_features: int, num_classes: int, side_axes: str = "XYZ", rank: int = 0):
        super().__init__()
        if side_axes not in ("XYZ", "Z"):
            raise ValueError(f"side_axes must be XYZ or Z, got {side_axes!r}")
        self.side_qubits = side_features // 3
        self.side_axes = side_axes
        width = 3 * FINAL_QUBITS + (side_features if side_axes == "XYZ" else self.side_qubits)
        self.linear = (nn.Linear(width, num_classes) if rank <= 0 else
                       nn.Sequential(nn.Linear(width, rank, bias=False), nn.Linear(rank, num_classes)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        side = features[:, BRANCH_FEATURES:]
        if self.side_axes == "Z":
            side = side[:, 2 * self.side_qubits:]
        return self.linear(torch.cat((features[:, :3 * FINAL_QUBITS], side), dim=1))


def same_axis_pair_indices(axes: str) -> list:
    """Vị trí của tương quan cùng trục A_i A_j (A thuộc `axes`) của 28 cặp trong vector PauliFeatures."""
    return [observable_index(f"{axis}{i}{axis}{j}") for i, j in combinations(range(FINAL_QUBITS), 2) for axis in axes]


class SameAxisLinear(nn.Module):
    """Head riêng mỗi cây (thêm 2026-11-19), giữ tương quan cùng trục, bỏ tương quan chéo trục.

    samebasis (axes "XYZ"): 24 Bloch + 84 (XX, YY, ZZ của 28 cặp) + side -> Linear(204, 10): đúng các số lấy được
    từ 3 cấu hình đo toàn cục X, Y, Z (mỗi cấu hình cho mọi A_i và A_i A_j). zz (axes "Z"): 24 + 28 ZZ + side ->
    Linear(148, 10). Bỏ các tương quan chéo trục (XY, XZ, ...) cần thêm cấu hình đo."""

    def __init__(self, side_features: int, num_classes: int, axes: str):
        super().__init__()
        index = list(range(3 * FINAL_QUBITS)) + same_axis_pair_indices(axes) + \
            list(range(BRANCH_FEATURES, BRANCH_FEATURES + side_features))
        self.register_buffer("index", torch.tensor(index), persistent=False)
        self.linear = nn.Linear(len(index), num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features.index_select(1, self.index))
