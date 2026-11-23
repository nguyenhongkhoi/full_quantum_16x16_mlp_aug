"""Mã hóa góc RY cho pixel."""
from __future__ import annotations

import math

import torch


def angle_ry_encode(values: torch.Tensor, max_angle: float = math.pi, clamp: bool = True) -> torch.Tensor:
    """Encode normalized scalar values into single-qubit RY states.

    Args:
        values: Tensor shaped ``(batch, sites)`` with values usually in ``[0, 1]``.
        max_angle: Value multiplier used for ``theta``.
        clamp: Clamp values to ``[0, 1]`` first. The classical front end turns this off, so its output keeps a
            gradient everywhere.

    Returns:
        Complex tensor shaped ``(batch, sites, 2)`` containing
        ``cos(theta/2)|0> + sin(theta/2)|1>``.
    """
    theta = (values.clamp(0.0, 1.0) if clamp else values) * max_angle
    states = torch.stack((torch.cos(theta / 2.0), torch.sin(theta / 2.0)), dim=-1)
    return states.to(torch.complex64)
