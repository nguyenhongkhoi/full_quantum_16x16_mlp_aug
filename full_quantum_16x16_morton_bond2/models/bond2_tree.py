"""Tree models without the 8-qubit stage: bond-2 Morton tree (b0/b1/b2) and 1-qubit tree (tree1q)."""
from __future__ import annotations

import math
from itertools import product
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .unitary import LearnedUnitary2Q

IMAGE_SIZE = 16
TAP_MERGE_INDEX = 2  # Merges are levels B, C, D, E, F; index 2 (D) leaves 16 nodes of 4x4 pixels.


def spatial_permutation(size: int, order: str) -> torch.Tensor:
    """Reorder sites without changing pixels; Morton groups dyadic 2D patches."""
    if order not in {"row", "morton"}:
        raise ValueError(f"Unknown spatial order: {order}")
    sites = list(range(size * size))
    if order == "morton":
        def code(i):
            y, x = divmod(i, size)
            return sum(((x >> b) & 1) << (2 * b) | ((y >> b) & 1) << (2 * b + 1)
                       for b in range(size.bit_length()))
        sites.sort(key=code)
    return torch.tensor(sites)


def encode_images(images: torch.Tensor, shift: int = 0) -> torch.Tensor:
    """Resize to 16x16, optionally shift cyclically, and RY-encode: (batch, 256, 2) complex states."""
    values = F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                           align_corners=False).clamp(0.0, 1.0)
    if shift:
        values = torch.roll(values, shifts=(shift, shift), dims=(2, 3))
    theta = values.flatten(start_dim=1) * math.pi
    return torch.stack((torch.cos(theta / 2.0), torch.sin(theta / 2.0)), dim=-1).to(torch.complex64)


def _normalize_hermitize(rho: torch.Tensor) -> torch.Tensor:
    trace = rho.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real.clamp_min(1e-8)
    rho = rho / trace[..., None, None]
    return 0.5 * (rho + rho.conj().transpose(-1, -2))


def _pair_density(rho: torch.Tensor) -> torch.Tensor:
    """Join neighbouring nodes into product states: (batch, n, d, d) -> (batch, n/2, d*d, d*d)."""
    batch, nodes, dim, _ = rho.shape
    if nodes % 2 != 0:
        raise ValueError(f"Expected an even node count, got {nodes}")
    pairs = torch.einsum("bnij,bnkl->bnikjl", rho[:, 0::2], rho[:, 1::2])
    return pairs.reshape(batch, nodes // 2, dim * dim, dim * dim)


def two_qubit_paulis() -> torch.Tensor:
    """The 15 two-qubit Pauli operators except II: (15, 4, 4)."""
    identity = torch.eye(2, dtype=torch.complex64)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    return torch.stack([torch.kron(a, b) for a, b in product((identity, x, y, z), repeat=2)][1:])


class Bond2Leaf(nn.Module):
    """Level A: two pixel qubits -> one 2-qubit node, nothing traced out."""

    def __init__(self, init_scale: float = 1e-2, balanced: bool = True):
        super().__init__()
        self.gate = LearnedUnitary2Q(init_scale=init_scale, balanced=balanced)

    def forward(self, qubits: torch.Tensor) -> torch.Tensor:
        pairs = torch.einsum("bni,bnj->bnij", qubits[:, 0::2], qubits[:, 1::2]).flatten(start_dim=2)
        states = pairs @ self.gate().T  # Row-vector form of (gate @ state) for every node.
        return torch.einsum("bni,bnj->bnij", states, states.conj())


class Bond2Merge(nn.Module):
    """Levels B-F: children [a0 a1] and [b0 b1] -> one node holding qubits (a1, b0).

    Gate order matters. The cross gate on (a1, b0) runs first so that the kept pair is correlated;
    the local gates then pull information from a0 and b1 before those are traced out. A cross gate
    placed last would only act on a product of two 1-qubit states (6 real numbers).
    """

    def __init__(self, init_scale: float = 1e-2, balanced: bool = True):
        super().__init__()
        self.cross = LearnedUnitary2Q(init_scale=init_scale, balanced=balanced)
        self.left = LearnedUnitary2Q(init_scale=init_scale, balanced=balanced)
        self.right = LearnedUnitary2Q(init_scale=init_scale, balanced=balanced)

    def unitary(self) -> torch.Tensor:
        cross = self.cross()
        identity = torch.eye(2, dtype=cross.dtype, device=cross.device)
        return torch.kron(self.left(), self.right()) @ torch.kron(torch.kron(identity, cross), identity)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        pairs = _pair_density(rho)
        batch, nodes, _, _ = pairs.shape
        unitary = self.unitary()
        pairs = unitary @ pairs @ unitary.conj().T
        # Indices are row (a0, a1, b0, b1) then column; a0 and b1 are traced out.
        kept = torch.einsum("bnaijdakld->bnijkl", pairs.reshape(batch, nodes, *([2] * 8)))
        return _normalize_hermitize(kept.reshape(batch, nodes, 4, 4))


class Bond2Core(nn.Module):
    """256 qubits -> 128 (A) -> 64, 32, 16 (tap), 8, 4 two-qubit nodes (B-F)."""

    def __init__(self, init_scale: float = 1e-2, balanced: bool = True):
        super().__init__()
        self.leaf = Bond2Leaf(init_scale=init_scale, balanced=balanced)
        self.merges = nn.ModuleList(
            [Bond2Merge(init_scale=init_scale, balanced=balanced) for _ in range(5)]
        )

    def forward(self, qubits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rho = self.leaf(qubits)
        tap = None
        for index, merge in enumerate(self.merges):
            rho = merge(rho)
            if index == TAP_MERGE_INDEX:
                tap = rho
        return tap, rho


class Bond2Head(nn.Module):
    """b0: final features only. b1/b2: also the 4x4 tap, projected by one Linear(15,6) shared by all cells."""

    def __init__(self, variant: str, num_classes: int = 10):
        super().__init__()
        if variant not in {"b0", "b1", "b2"}:
            raise ValueError(f"Unknown bond-2 variant: {variant}")
        self.project = None if variant == "b0" else nn.Linear(15, 6)
        feature_dim = 60 if variant == "b0" else 16 * 6 + 60
        # Parameter-free: upper-level features can be small because of the partial traces.
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        if variant == "b2":
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 16),
                nn.LayerNorm(16),
                nn.GELU(),
                nn.Linear(16, num_classes),
            )
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, tap_features: torch.Tensor, final_features: torch.Tensor) -> torch.Tensor:
        features = final_features.flatten(start_dim=1)
        if self.project is not None:
            features = torch.cat((self.project(tap_features).flatten(start_dim=1), features), dim=1)
        return self.classifier(self.norm(features))


class Bond2TreeClassifier(nn.Module):
    """One Morton tree. Each entry of `shifts` is one pass with the SAME gates and head; logits are averaged."""

    def __init__(self, variant: str, shifts: Sequence[int] = (0,), init: str = "balanced",
                 init_scale: float = 1e-2, num_classes: int = 10):
        super().__init__()
        if not shifts:
            raise ValueError("Provide at least one shift")
        self.shifts = tuple(shifts)
        self.register_buffer("sites", spatial_permutation(IMAGE_SIZE, "morton"), persistent=False)
        self.register_buffer("paulis", two_qubit_paulis(), persistent=False)
        self.core = Bond2Core(init_scale=init_scale, balanced=init == "balanced")
        self.head = Bond2Head(variant, num_classes=num_classes)

    def _pauli_features(self, rho: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bnij,pji->bnp", rho, self.paulis).real

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits_sum = None
        for shift in self.shifts:
            qubits = encode_images(images, shift).index_select(1, self.sites)
            tap, final = self.core(qubits)
            logits = self.head(self._pauli_features(tap), self._pauli_features(final))
            logits_sum = logits if logits_sum is None else logits_sum + logits
        return logits_sum / len(self.shifts)

    def count_parameters(self) -> Dict[str, int]:
        return _count_parameters(self)


class OneQubitBlock(nn.Module):
    """Same operation as QuantumConvPoolBlock in the 2-branch run: conv, pool, keep the first qubit."""

    def __init__(self, init_scale: float = 1e-2, balanced: bool = False):
        super().__init__()
        self.conv = LearnedUnitary2Q(init_scale=init_scale, balanced=balanced)
        self.pool = LearnedUnitary2Q(init_scale=init_scale, balanced=balanced)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        pairs = _pair_density(rho)
        batch, nodes, _, _ = pairs.shape
        unitary = self.pool() @ self.conv()
        pairs = unitary @ pairs @ unitary.conj().T
        kept = torch.einsum("bnijkj->bnik", pairs.reshape(batch, nodes, 2, 2, 2, 2))
        return _normalize_hermitize(kept)


def _bloch_features(rho: torch.Tensor) -> torch.Tensor:
    x = 2.0 * rho[..., 0, 1].real
    y = -2.0 * rho[..., 0, 1].imag
    z = rho[..., 0, 0].real - rho[..., 1, 1].real
    return torch.stack((x, y, z), dim=-1).flatten(start_dim=1)


class OneQubitTreeClassifier(nn.Module):
    """R2: the 2-branch run without the 8-qubit stage. Only that stage and re-upload are removed."""

    def __init__(self, init: str = "identity", init_scale: float = 1e-2, num_classes: int = 10):
        super().__init__()
        self.register_buffer("site_orders", torch.stack([
            spatial_permutation(IMAGE_SIZE, order) for order in ("row", "morton")
        ]), persistent=False)
        self.branches = nn.ModuleList([
            nn.ModuleDict({
                "core": nn.ModuleList([
                    OneQubitBlock(init_scale=init_scale, balanced=init == "balanced") for _ in range(5)
                ]),
                "head": nn.Sequential(
                    nn.Linear(48 + 24, 16),
                    nn.LayerNorm(16),
                    nn.GELU(),
                    nn.Linear(16, num_classes),
                ),
            })
            for _ in range(2)
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        qubits = encode_images(images)
        logits_sum = None
        for index, branch in enumerate(self.branches):
            ordered = qubits.index_select(1, self.site_orders[index])
            rho = torch.einsum("bni,bnj->bnij", ordered, ordered.conj())
            tap = None
            for block_index, block in enumerate(branch["core"]):
                rho = block(rho)
                if block_index == 3:  # After block 4 of 5, 16 sites remain.
                    tap = rho
            features = torch.cat((_bloch_features(tap), _bloch_features(rho)), dim=1)
            logits = branch["head"](features)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        return logits_sum / len(self.branches)

    def count_parameters(self) -> Dict[str, int]:
        return _count_parameters(self)


def is_head_parameter(name: str) -> bool:
    return name.startswith("head.") or ".head." in name


def _count_parameters(model: nn.Module) -> Dict[str, int]:
    core, head = 0, 0
    for name, parameter in model.named_parameters():
        if is_head_parameter(name):
            head += parameter.numel()
        else:
            core += parameter.numel()
    return {"core": core, "head": head, "total": core + head}
