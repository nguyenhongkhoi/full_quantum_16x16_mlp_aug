"""Chuyển đổi ma trận mật độ 1 qubit <-> vector Bloch (X, Y, Z)."""
from __future__ import annotations

import torch


def bloch(rho: torch.Tensor) -> torch.Tensor:
    """(batch, qubits, 2, 2) -> (batch, 3 * qubits): all X, then all Y, then all Z. rho = (I + xX + yY + zZ) / 2."""
    x = 2 * rho[..., 0, 1].real
    y = -2 * rho[..., 0, 1].imag
    z = (rho[..., 0, 0] - rho[..., 1, 1]).real
    return torch.cat((x, y, z), dim=1)


def density_from_bloch(vector: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 2, 2) complex64: (I + xX + yY + zZ) / 2 (ngược với bloch() ở trên)."""
    x, y, z = vector.float().unbind(-1)
    zero = torch.zeros_like(x)
    r00 = torch.complex((1 + z) / 2, zero)
    r11 = torch.complex((1 - z) / 2, zero)
    r01 = torch.complex(x / 2, -y / 2)
    r10 = torch.complex(x / 2, y / 2)
    return torch.stack((torch.stack((r00, r01), -1), torch.stack((r10, r11), -1)), -2)
