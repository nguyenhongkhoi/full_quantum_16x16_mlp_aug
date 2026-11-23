"""Unitary 1 và 2 qubit học được (U = exp(iH), H Hermite)."""
from __future__ import annotations

import torch
from torch import nn


def _scaled(parameter: torch.Tensor, scale: float) -> torch.Tensor:
    """param_scale = 1 (the default) returns the parameter itself, so earlier models compute exactly as before."""
    return parameter if scale == 1.0 else parameter * scale


class LearnedUnitary1Q(nn.Module):
    """Learned 1-qubit unitary parameterized by matrix exponential.

    param_scale: the matrix uses param_scale * the stored parameters (see fusion_model.scale_learning_rate)."""

    def __init__(self, init_scale: float = 1e-2, real_only: bool = False):
        super().__init__()
        self.real_only = real_only
        self.param_scale = 1.0
        self.raw_real = nn.Parameter(init_scale * torch.randn(2, 2))
        self.raw_imag = None if real_only else nn.Parameter(init_scale * torch.randn(2, 2))

    def matrix(self) -> torch.Tensor:
        raw_real = _scaled(self.raw_real, self.param_scale)
        if self.real_only:
            skew = raw_real - raw_real.T
            return torch.matrix_exp(skew).to(torch.complex64)

        a = torch.complex(raw_real, _scaled(self.raw_imag, self.param_scale))
        hermitian = 0.5 * (a + a.conj().T)
        return torch.matrix_exp(1j * hermitian)

    def forward(self) -> torch.Tensor:
        return self.matrix()


class LearnedUnitary2Q(nn.Module):
    """Learned 2-qubit unitary parameterized by matrix exponential.

    Complex mode learns an unconstrained complex matrix A, forms a Hermitian
    H=(A+A^dagger)/2, then returns exp(iH). Real-only mode returns an
    orthogonal matrix exp(A-A^T), kept as complex for downstream consistency.
    param_scale: the matrix uses param_scale * the stored parameters (see fusion_model.scale_learning_rate).
    """

    def __init__(self, init_scale: float = 1e-2, real_only: bool = False):
        super().__init__()
        self.real_only = real_only
        self.param_scale = 1.0
        self.raw_real = nn.Parameter(init_scale * torch.randn(4, 4))
        self.raw_imag = None if real_only else nn.Parameter(init_scale * torch.randn(4, 4))

    def matrix(self) -> torch.Tensor:
        raw_real = _scaled(self.raw_real, self.param_scale)
        if self.real_only:
            skew = raw_real - raw_real.T
            return torch.matrix_exp(skew).to(torch.complex64)

        a = torch.complex(raw_real, _scaled(self.raw_imag, self.param_scale))
        hermitian = 0.5 * (a + a.conj().T)
        return torch.matrix_exp(1j * hermitian)

    def forward(self) -> torch.Tensor:
        return self.matrix()
