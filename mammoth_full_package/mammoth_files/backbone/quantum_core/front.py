"""Front ends that rewrite the 16x16 pixel values before they are encoded as qubits.

The circuit still gets 256 qubits, one per pixel, in the same scan orders. The front end only changes the state
given to each pixel's qubit, computed from a window of the 16x16 image around it. It is shared by the three trees.

Classical modes (all convs 3x3, padding 1), ClassicalFront:
    none      no front end; the model is level32.
    conv1     Conv2d(1, 1): 10 parameters.
    cnn<C>    x + Conv(C->1)(GELU(Conv(1->C)(x))): 19*C + 1 parameters (cnn4: 77, cnn8: 153). Window 5x5.
    cnn<C>p   as cnn<C>, with a second output channel phi: 28C + 2 (cnn8p: 226). Window 5x5.
    deep<C>   x + Conv(C->1)(GELU(Conv(C->C)(GELU(Conv(1->C)(x))))): 9C^2 + 20C + 1 (deep12: 1.537). Window 7x7.
    deep<C>p  as deep<C>, with a second output channel phi: each qubit is RZ(pi * phi) RY(pi * theta)|0>, so the
              encoder sets the full Bloch direction of every qubit, not only its RY angle. 9C^2 + 29C + 2
              (deep12p: 1.646).

Quantum modes (mammoth_qconv_front and full_quantum_16x16_level32_qconv, 2026-11-13), QuantumFront:
    qconv<L>[x<S>][h<H>][f]  S stages (default 1) of a small 9-qubit circuit with L layers on the 3x3 window around
              each pixel, measured, then a Linear(27, 2) per pixel. Window (2S+1)x(2S+1). Trainable parameters per
              stage 27L + 56 (qconv2x2: 220). With "f" the circuit angles are frozen at their random start (a
              buffer), only the readout learns (qconv2x2f: 112).
              With h<H> (2026-11-14) the readout is Linear(27, H) GELU Linear(H, 2) instead: 27L + 30H + 2 per
              stage (qconv2x2h16: 1,072). Without h the module and its random draws are exactly as before.

Every mode starts as the identity (the last conv / Linear starts at 0; conv1 starts as centre weight 1), and phi
starts at 0, so the model starts with the same outputs as level32. Its creation runs under a forked CPU random state,
so the rest of the model gets the same initial values, and training draws the same random numbers, as without it.
"""
from __future__ import annotations

import math
import re
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def parse_front(mode: str) -> tuple[str, int, bool]:
    """Classical mode -> (kind, channels, has_phase)."""
    if mode in ("none", "conv1"):
        return mode, 0, False
    for kind in ("cnn", "deep"):
        if mode.startswith(kind):
            rest = mode[len(kind):]
            phase = rest.endswith("p")
            rest = rest[:-1] if phase else rest
            if rest.isdigit() and int(rest) > 0:
                return kind, int(rest), phase
    raise ValueError(f"Unknown front end {mode!r}; use none, conv1, cnn<C>[p], deep<C>[p] or qconv<L>[x<S>][h<H>][f]")


def parse_qconv(mode: str) -> Optional[tuple[int, int, bool, int]]:
    """qconv<L>[x<S>][h<H>][f] -> (layers, stages, frozen, hidden); hidden 0 = plain Linear(27, 2) readout.
    None if mode is not a quantum front end."""
    match = re.fullmatch(r"qconv(\d+)(?:x(\d+))?(?:h(\d+))?(f?)", mode)
    if match is None:
        return None
    layers, stages, hidden = int(match.group(1)), int(match.group(2) or 1), int(match.group(3) or 0)
    if layers < 1 or stages < 1:
        raise ValueError(f"{mode!r}: layers and stages must be at least 1")
    if match.group(3) is not None and hidden < 1:
        raise ValueError(f"{mode!r}: the readout hidden size must be at least 1")
    return layers, stages, match.group(4) == "f", hidden


def front_has_phase(mode: str) -> bool:
    """Whether the front end also outputs phi."""
    return parse_qconv(mode) is not None or parse_front(mode)[2]


def front_circuit_parameters(front: Optional[nn.Module]) -> int:
    """Trainable circuit angles of a QuantumFront (its Linear readouts are classical); 0 for other front ends."""
    if not isinstance(front, QuantumFront):
        return 0
    return sum(stage.angles.numel() for stage in front.stages if isinstance(stage.angles, nn.Parameter))


def make_front(mode: str, height: int, width: int) -> nn.Module:
    if parse_qconv(mode) is not None:
        return QuantumFront(mode, height, width)
    return ClassicalFront(mode, height, width)


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


# ---- quantum front end ----

PATCH_QUBITS = 9  # the 3x3 window, row-major; qubit 0 is the most significant bit of the state index


def grid_edges(size: int = 3) -> list[tuple[int, int]]:
    """Nearest-neighbour pairs of a size x size grid (12 for 3x3)."""
    edges = []
    for row in range(size):
        for column in range(size):
            here = row * size + column
            if column + 1 < size:
                edges.append((here, here + 1))
            if row + 1 < size:
                edges.append((here, here + size))
    return edges


def cz_signs(qubits: int, edges) -> torch.Tensor:
    """Diagonal of the product of CZ on these edges: (-1)^(sum over edges of b_i b_j), as a real vector."""
    index = torch.arange(2 ** qubits)
    bits = [(index >> (qubits - 1 - q)) & 1 for q in range(qubits)]
    parity = torch.zeros_like(index)
    for a, b in edges:
        parity = parity ^ (bits[a] & bits[b])
    return (1 - 2 * parity).to(torch.float32)


def cis(radius: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """radius * exp(i angle); radius may be negative (torch.polar is meant for non-negative abs)."""
    return torch.complex(radius * torch.cos(angle), radius * torch.sin(angle))


def rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    """(..., 3) angles (a, b, c) -> (..., 2, 2) complex RZ(a) RY(b) RZ(c)."""
    a, b, c = angles.unbind(-1)
    cos, sin = torch.cos(b / 2), torch.sin(b / 2)
    plus, minus = (a + c) / 2, (a - c) / 2
    m00 = cis(cos, -plus)
    m01 = cis(-sin, -minus)
    m10 = cis(sin, minus)
    m11 = cis(cos, plus)
    return torch.stack((torch.stack((m00, m01), -1), torch.stack((m10, m11), -1)), -2)


def apply_one_qubit(state: torch.Tensor, gate: torch.Tensor, qubit: int, qubits: int) -> torch.Tensor:
    """state (n, 2^qubits), gate (2, 2) on this qubit."""
    view = state.reshape(state.shape[0], 2 ** qubit, 2, 2 ** (qubits - 1 - qubit))
    return torch.einsum("ab,nibj->niaj", gate, view).reshape(state.shape)


def bloch_of_qubits(state: torch.Tensor, qubits: int) -> torch.Tensor:
    """state (n, 2^qubits) -> (n, 3 * qubits): X, Y, Z of every qubit, in the order X0 Y0 Z0 X1 Y1 Z1 ..."""
    out = []
    for qubit in range(qubits):
        view = state.reshape(state.shape[0], 2 ** qubit, 2, 2 ** (qubits - 1 - qubit))
        zero, one = view[:, :, 0], view[:, :, 1]
        coherence = (zero * one.conj()).sum(dim=(1, 2))  # rho_01
        z = (zero.abs() ** 2 - one.abs() ** 2).sum(dim=(1, 2))
        out += [2 * coherence.real, -2 * coherence.imag, z]
    return torch.stack(out, dim=1)


class QuantumConvStage(nn.Module):
    """One quantum 'convolution': for every pixel, a 9-qubit circuit on its 3x3 window, shared by all pixels.

    Each window qubit starts as RZ(pi phi) RY(pi theta)|0> of its pixel (outside the image: |0>). Then L layers,
    each: one rotation RZ RY RZ per qubit (own angles), then CZ on the 12 grid edges. X, Y, Z of all 9 qubits
    (27 values, exact) go through a Linear(27, 2) that starts at 0: theta += out[0], phi += out[1].
    hidden > 0: the readout is Linear(27, hidden) GELU Linear(hidden, 2), the last layer starting at 0.
    """

    def __init__(self, layers: int, frozen: bool, hidden: int = 0):
        super().__init__()
        angles = torch.empty(layers, PATCH_QUBITS, 3).uniform_(-math.pi, math.pi)
        if frozen:
            self.register_buffer("angles", angles)
        else:
            self.angles = nn.Parameter(angles)
        if hidden:
            self.readout = nn.Sequential(nn.Linear(3 * PATCH_QUBITS, hidden), nn.GELU(), nn.Linear(hidden, 2))
            last = self.readout[-1]
        else:
            self.readout = last = nn.Linear(3 * PATCH_QUBITS, 2)
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
        self.register_buffer("signs", cz_signs(PATCH_QUBITS, grid_edges()), persistent=False)

    def features(self, theta: torch.Tensor, phi: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """theta, phi (batch, H*W) -> (batch, H*W, 27) measured values of each pixel's window circuit."""
        batch = theta.shape[0]
        windows = [F.unfold(x.reshape(batch, 1, height, width), 3, padding=1)  # (batch, 9, H*W)
                   .transpose(1, 2).reshape(-1, PATCH_QUBITS) for x in (theta, phi)]
        theta_w, phi_w = windows
        amp0 = cis(torch.cos(torch.pi * theta_w / 2), torch.zeros_like(phi_w))
        amp1 = cis(torch.sin(torch.pi * theta_w / 2), torch.pi * phi_w)
        qubits = torch.stack((amp0, amp1), dim=-1)  # (n, 9, 2)
        gates = rotation_matrices(self.angles)  # (L, 9, 2, 2)
        # The first rotation layer acts on a product state, so apply it to each qubit before the tensor product.
        qubits = torch.einsum("qab,nqb->nqa", gates[0], qubits)
        state = qubits[:, 0]
        for qubit in range(1, PATCH_QUBITS):
            state = (state[:, :, None] * qubits[:, qubit, None, :]).reshape(state.shape[0], -1)
        signs = self.signs.to(state.dtype)
        for layer in range(gates.shape[0]):
            if layer > 0:
                for qubit in range(PATCH_QUBITS):
                    state = apply_one_qubit(state, gates[layer, qubit], qubit, PATCH_QUBITS)
            state = state * signs
        return bloch_of_qubits(state, PATCH_QUBITS).reshape(batch, height * width, -1)

    def forward(self, theta, phi, height, width):
        out = self.readout(self.features(theta, phi, height, width))
        return theta + out[..., 0], phi + out[..., 1]


class QuantumFront(nn.Module):
    """S QuantumConvStage in a row; the first gets the pixel values as theta and phi = 0."""

    def __init__(self, mode: str, height: int, width: int):
        super().__init__()
        layers, stages, frozen, hidden = parse_qconv(mode)
        self.mode, self.height, self.width, self.phase = mode, height, width, True
        self.layers, self.frozen, self.hidden = layers, frozen, hidden
        # Same as ClassicalFront: draw only from a forked CPU random state.
        with torch.random.fork_rng(devices=[]):
            self.stages = nn.ModuleList([QuantumConvStage(layers, frozen, hidden) for _ in range(stages)])

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta, phi = values, torch.zeros_like(values)
        for stage in self.stages:
            if self.training and torch.is_grad_enabled():
                # Recompute the 9-qubit states in backward instead of keeping them (about 64 MB per state for a
                # batch of 64); the result does not change.
                theta, phi = checkpoint(stage, theta, phi, self.height, self.width, use_reentrant=False,
                                        preserve_rng_state=False)
            else:
                theta, phi = stage(theta, phi, self.height, self.width)
        return theta, phi
