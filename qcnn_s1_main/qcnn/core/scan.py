"""Thứ tự quét pixel -> qubit của cây (row, column, morton, góc nghiêng)."""
from __future__ import annotations

import math

import torch


SPATIAL_ORDERS = ("row", "column", "morton")


# ---- thứ tự quét (chép nguyên từ linhead/models/fusion_model.py) ----
def spatial_permutation(height: int, width: int, order: str) -> torch.Tensor:
    """Reorder sites without changing pixels; Morton groups dyadic 2D patches, angle<d> scans lines at d degrees."""
    if order.startswith("angle"):
        return angled_scan(height, width, float(order[len("angle"):]))
    shift = 0
    if order.startswith("morton_shift"):
        # Morton order of the image shifted cyclically by `shift` pixels in x and y: the same kind of tree, with its
        # block borders moved. With shift 4 the 4x4 blocks stay, and the middle 8 columns and rows form blocks.
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
    """Visit sites along parallel lines at this angle, line by line. Angle 0 is the row order."""
    cosine, sine = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))

    def key(i):
        y, x = divmod(i, width)
        x, y = x - (width - 1) / 2, y - (height - 1) / 2
        return math.floor(y * cosine - x * sine), x * cosine + y * sine, i

    return torch.tensor(sorted(range(height * width), key=key))
