from __future__ import annotations

from typing import Sequence

from torch import nn

from .baselines import SmallCNN
from .bond2_tree import Bond2TreeClassifier, OneQubitTreeClassifier, is_head_parameter

VARIANTS = ("b0", "b1", "b2", "tree1q", "cnn5k", "cnn3k", "cnn2k", "cnn1k")
EXPECTED_PARAMETERS = {"b0": 866, "b1": 1922, "b2": 3066, "tree1q": 3060,
                       "cnn5k": 5578, "cnn3k": 2988, "cnn2k": 1990, "cnn1k": 934}
CNN_WIDTHS = {"cnn5k": (8, 16), "cnn3k": (8, 11), "cnn2k": (6, 9), "cnn1k": (4, 6)}
DEFAULT_INIT = {"b0": "balanced", "b1": "balanced", "b2": "balanced", "tree1q": "identity"}


def build_model(variant: str, shifts: Sequence[int] = (0,), init: str | None = None) -> nn.Module:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if variant in {"b0", "b1", "b2"}:
        return Bond2TreeClassifier(variant, shifts=shifts, init=init or DEFAULT_INIT[variant])
    if tuple(shifts) != (0,):
        raise ValueError("--shifts is only supported by b0, b1 and b2")
    if variant == "tree1q":
        return OneQubitTreeClassifier(init=init or DEFAULT_INIT[variant])
    if init is not None:
        raise ValueError("--init does not apply to CNN variants")
    return SmallCNN(CNN_WIDTHS[variant])


__all__ = ["DEFAULT_INIT", "EXPECTED_PARAMETERS", "VARIANTS", "build_model", "is_head_parameter"]
