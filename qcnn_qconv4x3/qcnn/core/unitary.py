from __future__ import annotations

import torch
from torch import nn


def _scaled(parameter: torch.Tensor, scale: float) -> torch.Tensor:
    """param_scale = 1 (mặc định) trả về chính tham số, nên các model cũ tính giống hệt trước."""
    return parameter if scale == 1.0 else parameter * scale


class LearnedUnitary1Q(nn.Module):
    """Unitary 1 qubit học được, tham số hóa bằng hàm mũ ma trận: U = exp(iH), H = (A + A^dagger) / 2, A phức 2x2
    (8 số thực).

    param_scale: ma trận dùng param_scale * tham số lưu trữ (xem fusion_model.scale_learning_rate)."""

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
    """Unitary 2 qubit học được, tham số hóa bằng hàm mũ ma trận.

    Chế độ phức học một ma trận phức A 4x4 không ràng buộc (32 số thực), lập H = (A + A^dagger) / 2 (Hermite),
    rồi trả về exp(iH). Chế độ chỉ thực trả về ma trận trực giao exp(A - A^T), để dạng phức cho đồng nhất.
    param_scale: ma trận dùng param_scale * tham số lưu trữ (xem fusion_model.scale_learning_rate).
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
