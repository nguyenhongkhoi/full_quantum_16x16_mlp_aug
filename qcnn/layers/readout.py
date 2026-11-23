"""Readout cổ điển (head) sau ba cây (như FusionQCNN trong full_quantum_16x16_level32_qconv và các head tuyến tính
của full_quantum_16x16_level32_qconv_linhead; thứ tự lớp và số ngẫu nhiên không đổi).

head "mlp" (model 2c, 11.948 tham số với projection 10, hidden 16):
    mỗi cây:  Linear(372, projection) LayerNorm GELU Dropout        -> projection số
    gộp:      nối 3 x projection -> Linear(., hidden) LayerNorm GELU Dropout Linear(hidden, 10)
head "linear3": mỗi cây một Linear(372, 10); logit = trung bình của ba cây (11.190).
head "linear-shared": một Linear(372, 10) dùng chung cho ba cây; logit = trung bình (3.730).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

HEADS = ("mlp", "linear3", "linear-shared")


def build_head(head: str, trees: int, features: int, projection: int, hidden: int, dropout: float,
               num_classes: int) -> tuple[nn.ModuleList, Optional[nn.Module]]:
    """(projections, fusion). fusion là None với head tuyến tính (logit = trung bình đầu ra của từng cây)."""
    if head not in HEADS:
        raise ValueError(f"Unknown head {head!r}; use one of {HEADS}")
    if head == "linear3":
        return nn.ModuleList([nn.Linear(features, num_classes) for _ in range(trees)]), None
    if head == "linear-shared":
        shared = nn.Linear(features, num_classes)
        return nn.ModuleList([shared] * trees), None
    projections = nn.ModuleList([
        nn.Sequential(nn.Linear(features, projection), nn.LayerNorm(projection), nn.GELU(), nn.Dropout(dropout))
        for _ in range(trees)
    ])
    fusion = nn.Sequential(
        nn.Linear(trees * projection, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, num_classes),
    )
    return projections, fusion


def combine(branches: list[torch.Tensor], fusion: Optional[nn.Module]) -> torch.Tensor:
    """Đầu ra head của từng cây -> logit."""
    if fusion is None:
        return torch.stack(branches, dim=0).mean(dim=0)
    return fusion(torch.cat(branches, dim=1))
