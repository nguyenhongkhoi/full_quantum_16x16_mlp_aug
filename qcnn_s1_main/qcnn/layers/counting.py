"""Đếm tham số theo bậc tự do thật (FREE): unitary 2 qubit 16, 1 qubit 4 (bỏ pha toàn cục: 15 / 3).
Xem CACH_DEM_THAM_SO.txt.
"""
from __future__ import annotations

from .unitary import LearnedUnitary1Q, LearnedUnitary2Q


def free_parameter_count(module, drop_phase: bool = False) -> int:
    """Số tham số THẬT (bậc tự do) của module (thêm 2026-11-20).

    LearnedUnitary1Q/2Q lưu ma trận phức A d x d (2 d^2 số thực) nhưng chỉ dùng H = (A + A^dagger)/2, U = exp(iH):
    phần phản Hermite của A không ảnh hưởng gì (gradient 0). Bậc tự do thật là d^2 (4 cho 1 qubit, 16 cho 2 qubit),
    d^2 - 1 nếu bỏ pha toàn cục (drop_phase: 3, 15); real_only: d(d-1)/2. Tham số khác đếm như cũ.
    Module dùng chung (head shared) chỉ đếm một lần."""
    if module is None:
        return 0
    total = 0
    for sub in module.modules():
        if isinstance(sub, (LearnedUnitary1Q, LearnedUnitary2Q)):
            d = sub.raw_real.shape[0]
            total += d * (d - 1) // 2 if sub.real_only else d * d - (1 if drop_phase else 0)
        else:
            total += sum(parameter.numel() for parameter in sub.parameters(recurse=False))
    return total
