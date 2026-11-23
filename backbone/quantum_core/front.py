"""A classical front end that rewrites the 16x16 pixel values before they are encoded as qubits.

The circuit still gets 256 qubits, one per pixel, in the same scan orders. The front end only changes the state
given to each pixel's qubit, computed from a window of the 16x16 image around it. It is shared by the three trees.

Modes (all convs 3x3, padding 1):
    none      no front end; the model is level32.
    conv1     Conv2d(1, 1): 10 parameters.
    cnn<C>    x + Conv(C->1)(GELU(Conv(1->C)(x))): 19*C + 1 parameters (cnn4: 77, cnn8: 153). Window 5x5.
    deep<C>   x + Conv(C->1)(GELU(Conv(C->C)(GELU(Conv(1->C)(x))))): 9C^2 + 20C + 1 (deep12: 1.537). Window 7x7.
    deep<C>p  as deep<C>, with a second output channel phi: each qubit is RZ(pi * phi) RY(pi * theta)|0>, so the
              encoder sets the full Bloch direction of every qubit, not only its RY angle. 9C^2 + 29C + 2
              (deep12p: 1.646).

Every mode starts as the identity (the last conv starts at 0; conv1 starts as centre weight 1), and phi starts at 0,
so the model starts with the same outputs as level32. Its creation runs under a forked CPU random state, so the
rest of the model gets the same initial values, and training draws the same random numbers, as without it.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def parse_front(mode: str) -> tuple[str, int, bool]:
    """mode -> (kind, channels, has_phase)."""
    if mode in ("none", "conv1"):
        return mode, 0, False
    for kind in ("cnn", "deep"):
        if mode.startswith(kind):
            rest = mode[len(kind):]
            phase = kind == "deep" and rest.endswith("p")
            rest = rest[:-1] if phase else rest
            if rest.isdigit() and int(rest) > 0:
                return kind, int(rest), phase
    raise ValueError(f"Unknown front end {mode!r}; use none, conv1, cnn<C>, deep<C> or deep<C>p (e.g. deep12p)")


class ClassicalFront(nn.Module):
    def __init__(self, mode: str, height: int, width: int):
        super().__init__()
        kind, channels, phase = parse_front(mode)
        if kind == "none":
            raise ValueError("ClassicalFront needs a mode other than none")
        self.mode, self.height, self.width, self.phase = mode, height, width, phase
        # The model is built on the CPU, so only the CPU random state is drawn from; it is restored afterwards.
        # Do not call torch.manual_seed here: it would also reset the CUDA state, which fork_rng does not restore.
        with torch.random.fork_rng(devices=[]):
            if kind == "conv1":
                self.net = nn.Conv2d(1, 1, 3, padding=1)
                with torch.no_grad():
                    self.net.weight.zero_()
                    self.net.weight[0, 0, 1, 1] = 1.0
                    self.net.bias.zero_()
                self.residual = False
                return
            last = nn.Conv2d(channels, 2 if phase else 1, 3, padding=1)
            with torch.no_grad():
                last.weight.zero_()
                last.bias.zero_()
            layers = [nn.Conv2d(1, channels, 3, padding=1), nn.GELU()]
            if kind == "deep":
                layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.GELU()]
            self.net = nn.Sequential(*layers, last)
            self.residual = True

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """(batch, H*W) row-major pixel values -> (theta, phi), each (batch, H*W); phi is None without phase.

        theta is not clamped: RY takes any angle."""
        grid = values.reshape(-1, 1, self.height, self.width)
        out = self.net(grid)
        theta = out[:, :1]
        if self.residual:
            theta = grid + theta
        phi = out[:, 1].flatten(start_dim=1) if self.phase else None
        return theta.flatten(start_dim=1), phi
