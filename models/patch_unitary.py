from __future__ import annotations

import math

import torch
from torch import nn


class LearnedUnitary2Q(nn.Module):
    """Two-qubit unitary exp(iH) with a non-redundant Hermitian parameterization (16 parameters).

    balanced=True centres the gate on sqrt(iSWAP), so a kept qubit depends on both input qubits
    from the first step; init_scale is the noise around that centre.
    """

    def __init__(self, init_scale: float = 0.1, balanced: bool = True):
        super().__init__()
        upper = torch.triu_indices(4, 4, offset=1)
        self.register_buffer("upper_i", upper[0], persistent=False)
        self.register_buffer("upper_j", upper[1], persistent=False)

        off_diagonal_scale = init_scale / math.sqrt(2.0)
        self.diagonal = nn.Parameter(init_scale * torch.randn(4))
        self.raw_real_upper = nn.Parameter(off_diagonal_scale * torch.randn(upper.shape[1]))
        self.raw_imag_upper = nn.Parameter(off_diagonal_scale * torch.randn(upper.shape[1]))
        if balanced:
            # sqrt(iSWAP) = exp(i*pi/8*(XX+YY)), so H[1,2] = H[2,1] = pi/4.
            # triu_indices order is (0,1),(0,2),(0,3),(1,2),(1,3),(2,3): entry (1,2) has index 3.
            with torch.no_grad():
                self.raw_real_upper[3] += math.pi / 4.0

    def forward(self) -> torch.Tensor:
        real = torch.diag(self.diagonal)
        real = real.index_put((self.upper_i, self.upper_j), self.raw_real_upper)
        real = real.index_put((self.upper_j, self.upper_i), self.raw_real_upper)
        imag = torch.zeros_like(real)
        imag = imag.index_put((self.upper_i, self.upper_j), self.raw_imag_upper)
        imag = imag.index_put((self.upper_j, self.upper_i), -self.raw_imag_upper)
        return torch.matrix_exp(1j * torch.complex(real, imag))


class LearnedUnitary1Q(nn.Module):
    """One-qubit unitary exp(iH) with the same parameterization as LearnedUnitary2Q (4 parameters)."""

    def __init__(self, init_scale: float = 1.0):
        super().__init__()
        self.diagonal = nn.Parameter(init_scale * torch.randn(2))
        self.raw_real_upper = nn.Parameter(init_scale / math.sqrt(2.0) * torch.randn(1))
        self.raw_imag_upper = nn.Parameter(init_scale / math.sqrt(2.0) * torch.randn(1))

    def forward(self) -> torch.Tensor:
        upper = torch.complex(self.raw_real_upper[0], self.raw_imag_upper[0])
        first = torch.stack((self.diagonal[0].to(upper.dtype), upper))
        second = torch.stack((upper.conj(), self.diagonal[1].to(upper.dtype)))
        return torch.matrix_exp(1j * torch.stack((first, second)))
