from __future__ import annotations

import math

import torch
from torch import nn


class _LearnedUnitary(nn.Module):
    """Unitary exp(iH) with a non-redundant Hermitian parameterization.

    A Hermitian d x d matrix has d**2 independent real values. The former
    implementation stored two full d x d real matrices, then symmetrized them;
    half of those stored directions had no effect on H. This implementation
    stores its real diagonal and the real/imaginary parts of the strict upper
    triangle directly, preserving the same family of complex unitaries.
    """

    def __init__(self, dimension: int, init_scale: float = 1e-2, real_only: bool = False):
        super().__init__()
        self.dimension = dimension
        self.real_only = real_only
        upper = torch.triu_indices(dimension, dimension, offset=1)
        self.register_buffer("upper_i", upper[0], persistent=False)
        self.register_buffer("upper_j", upper[1], persistent=False)

        if real_only:
            # A real orthogonal matrix exp(K) needs d(d-1)/2 skew entries.
            self.raw_skew_upper = nn.Parameter(
                math.sqrt(2.0) * init_scale * torch.randn(upper.shape[1])
            )
            self.diagonal = None
            self.raw_real_upper = None
            self.raw_imag_upper = None
        else:
            self.diagonal = nn.Parameter(init_scale * torch.randn(dimension))
            off_diagonal_scale = init_scale / math.sqrt(2.0)
            self.raw_real_upper = nn.Parameter(
                off_diagonal_scale * torch.randn(upper.shape[1])
            )
            self.raw_imag_upper = nn.Parameter(
                off_diagonal_scale * torch.randn(upper.shape[1])
            )
            self.raw_skew_upper = None

    def matrix(self) -> torch.Tensor:
        d = self.dimension
        if self.real_only:
            skew = torch.zeros((d, d), dtype=self.raw_skew_upper.dtype,
                               device=self.raw_skew_upper.device)
            skew = skew.index_put((self.upper_i, self.upper_j), self.raw_skew_upper)
            skew = skew.index_put((self.upper_j, self.upper_i), -self.raw_skew_upper)
            return torch.matrix_exp(skew).to(torch.complex64)

        real = torch.diag(self.diagonal)
        real = real.index_put((self.upper_i, self.upper_j), self.raw_real_upper)
        real = real.index_put((self.upper_j, self.upper_i), self.raw_real_upper)
        imag = torch.zeros_like(real)
        imag = imag.index_put((self.upper_i, self.upper_j), self.raw_imag_upper)
        imag = imag.index_put((self.upper_j, self.upper_i), -self.raw_imag_upper)
        hermitian = torch.complex(real, imag)
        return torch.matrix_exp(1j * hermitian)

    def forward(self) -> torch.Tensor:
        return self.matrix()


class LearnedUnitary1Q(_LearnedUnitary):
    """Learned 1-qubit unitary parameterized by an independent generator."""

    def __init__(self, init_scale: float = 1e-2, real_only: bool = False):
        super().__init__(dimension=2, init_scale=init_scale, real_only=real_only)


class LearnedUnitary2Q(_LearnedUnitary):
    """Learned 2-qubit unitary parameterized by an independent generator."""

    def __init__(self, init_scale: float = 1e-2, real_only: bool = False):
        super().__init__(dimension=4, init_scale=init_scale, real_only=real_only)
