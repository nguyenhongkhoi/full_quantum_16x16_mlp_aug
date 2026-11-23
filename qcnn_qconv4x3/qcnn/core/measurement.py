"""Phép đo của một cây (giá trị kỳ vọng chính xác).

- tầng giữa: vector Bloch của 32 qubit độc lập sau tầng 3 của cây (hàm bloch bên dưới, 96 số);
- 8 qubit cuối: PauliFeatures (pauli.py): 24 kỳ vọng <X>,<Y>,<Z> một qubit + 252 tương quan cặp = 276.
"""
from __future__ import annotations

import torch

from .pauli import PauliFeatures

FINAL_QUBITS = 8
PAIRS = FINAL_QUBITS * (FINAL_QUBITS - 1) // 2
BRANCH_FEATURES = 3 * FINAL_QUBITS + 9 * PAIRS  # 276


def bloch(rho: torch.Tensor) -> torch.Tensor:
    """(batch, qubits, 2, 2) -> (batch, 3 * qubits): mọi X, rồi mọi Y, rồi mọi Z. rho = (I + xX + yY + zZ) / 2."""
    x = 2 * rho[..., 0, 1].real
    y = -2 * rho[..., 0, 1].imag
    z = (rho[..., 0, 0] - rho[..., 1, 1]).real
    return torch.cat((x, y, z), dim=1)


__all__ = ["PauliFeatures", "bloch", "FINAL_QUBITS", "PAIRS", "BRANCH_FEATURES"]
