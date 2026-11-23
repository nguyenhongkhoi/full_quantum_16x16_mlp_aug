"""Thứ tự lá của ba cây (spatial_permutation trong models/fusion_model.py, không đổi).

Cây ghép các lá liên tiếp, nên thứ tự quét quyết định pixel nào được ghép trước:
  row: lần lượt từng hàng của ảnh 16x16; column: từng cột; morton: đường chữ Z (khối 2x2, 4x4, 8x8).
"""
from __future__ import annotations

import math

import torch

SPATIAL_ORDERS = ("row", "column", "morton")


def spatial_permutation(height: int, width: int, order: str) -> torch.Tensor:
    """Sắp lại vị trí mà không đổi pixel; Morton gom các khối vuông 2D, angle<d> quét theo các đường nghiêng d độ."""
    if order.startswith("angle"):
        return angled_scan(height, width, float(order[len("angle"):]))
    shift = 0
    if order.startswith("morton_shift"):
        shift = int(order[len("morton_shift"):])
        order = "morton"
    if order not in {"row", "column", "morton"}:
        raise ValueError(f"Unknown spatial order: {order}")
    sites = list(range(height * width))
    if order == "column":
        sites.sort(key=lambda i: (i % width, i // width))
    elif order == "morton":
        def code(i):
            y, x = divmod(i, width)
            y, x = (y + shift) % height, (x + shift) % width
            return sum(((x >> b) & 1) << (2 * b) |
                       ((y >> b) & 1) << (2 * b + 1)
                       for b in range(max(height, width).bit_length()))
        sites.sort(key=code)
    return torch.tensor(sites)


def angled_scan(height: int, width: int, degrees: float) -> torch.Tensor:
    """Đi qua các vị trí theo các đường song song nghiêng một góc, lần lượt từng đường. Góc 0 là quét theo hàng."""
    cosine, sine = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))

    def key(i):
        y, x = divmod(i, width)
        x, y = x - (width - 1) / 2, y - (height - 1) / 2
        return math.floor(y * cosine - x * sine), x * cosine + y * sine, i

    return torch.tensor(sorted(range(height * width), key=key))
