"""Mã hóa 256 pixel thành 256 qubit lá của các cây:
    |q_p> = cos(pi theta_p / 2)|0> + exp(i pi phi_p) sin(pi theta_p / 2)|1>
(angle_ry_encode trong models/encoding.py và bước thêm pha trong FusionQCNN.forward, không đổi)."""
from __future__ import annotations

import math
from typing import Optional

import torch


def angle_ry_encode(values: torch.Tensor, max_angle: float = math.pi, clamp: bool = True) -> torch.Tensor:
    """(batch, sites) -> (batch, sites, 2) số phức cos(theta/2)|0> + sin(theta/2)|1>, theta = max_angle * value.

    clamp: kẹp giá trị về [0, 1] trước. Khi có encoder thì tắt, để đầu ra của encoder có gradient ở mọi chỗ."""
    theta = (values.clamp(0.0, 1.0) if clamp else values) * max_angle
    states = torch.stack((torch.cos(theta / 2.0), torch.sin(theta / 2.0)), dim=-1)
    return states.to(torch.complex64)


def encode_qubits(theta: torch.Tensor, phi: Optional[torch.Tensor], clamp: bool) -> torch.Tensor:
    """RY(pi theta) rồi RZ(pi phi) (sai khác pha toàn cục: nhân exp(i pi phi) vào |1>)."""
    qubits = angle_ry_encode(theta, clamp=clamp)
    if phi is not None:
        qubits = torch.stack((qubits[..., 0], qubits[..., 1] * torch.polar(torch.ones_like(phi), torch.pi * phi)),
                             dim=-1)
    return qubits
