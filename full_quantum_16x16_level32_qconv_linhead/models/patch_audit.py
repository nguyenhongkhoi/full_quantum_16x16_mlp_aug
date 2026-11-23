"""s9_meas and its controls: the same patches and the same head, with the feature layer removed step by step."""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .patch_unitary import LearnedUnitary1Q, LearnedUnitary2Q

IMAGE_SIZE = 16
CORES = ("quantum", "product", "encoding", "pixel", "local_mlp")


@dataclass(frozen=True)
class AuditConfig:
    core: str                               # One of CORES.
    patch: Tuple[int, int] = (3, 4)         # Qubits per patch = rows * cols.
    grid: Tuple[int, int] = (8, 4)          # Patch positions.
    projection: int = 4                     # Output size of the Linear shared by all patches.
    hidden: int = 16                        # MLP readout width.
    local_hidden: int = 16                  # Hidden width of the local_mlp core.


def patch_positions(image_size: int, patch_size: int, count: int) -> List[int]:
    """`count` evenly spread start positions, first at 0 and last touching the image edge."""
    span = image_size - patch_size
    return [round(index * span / (count - 1)) for index in range(count)]


def patch_site_indices(config: AuditConfig) -> torch.Tensor:
    """Flat pixel index of every qubit of every patch, row-major: (patches, qubits)."""
    rows = patch_positions(IMAGE_SIZE, config.patch[0], config.grid[0])
    cols = patch_positions(IMAGE_SIZE, config.patch[1], config.grid[1])
    return torch.tensor([
        [(top + dy) * IMAGE_SIZE + left + dx
         for dy, dx in product(range(config.patch[0]), range(config.patch[1]))]
        for top, left in product(rows, cols)
    ])


def lattice_bonds(rows: int, cols: int) -> Dict[str, List[Tuple[int, int]]]:
    """Nearest-neighbour bonds grouped so that bonds inside a group never share a qubit."""
    groups: Dict[str, List[Tuple[int, int]]] = {"h_even": [], "h_odd": [], "v_even": [], "v_odd": []}
    for row, col in product(range(rows), range(cols)):
        site = row * cols + col
        if col + 1 < cols:
            groups["h_even" if col % 2 == 0 else "h_odd"].append((site, site + 1))
        if row + 1 < rows:
            groups["v_even" if row % 2 == 0 else "v_odd"].append((site, site + cols))
    return {name: bonds for name, bonds in groups.items() if bonds}


def amplitude_indices(n_qubits: int, targets: Tuple[int, ...]) -> torch.Tensor:
    """Basis indices grouped by the target qubits: (2**(n-len(targets)), 2**len(targets)).

    Qubit 0 is the most significant bit, and the first target is the most significant bit of the
    second index, which matches torch.kron(gate_on_first, gate_on_second).
    """
    positions = [n_qubits - 1 - target for target in targets]
    rest = [bit for bit in range(n_qubits) if bit not in positions]
    rows = torch.arange(2 ** len(rest))
    columns = torch.arange(2 ** len(targets))
    base = torch.zeros_like(rows)
    for index, bit in enumerate(rest):
        base = base + (((rows >> index) & 1) << bit)
    offset = torch.zeros_like(columns)
    for index, bit in enumerate(positions):
        offset = offset + (((columns >> (len(targets) - 1 - index)) & 1) << bit)
    return base[:, None] + offset[None, :]


class GroupedAmplitudes(nn.Module):
    """Applies an operation to selected qubits without leaving the flat statevector layout."""

    def __init__(self, n_qubits: int, targets: Tuple[int, ...]):
        super().__init__()
        grouped = amplitude_indices(n_qubits, targets)
        self.group_size = grouped.shape[1]
        self.register_buffer("flat", grouped.flatten(), persistent=False)
        self.register_buffer("inverse", torch.argsort(grouped.flatten()), persistent=False)

    def split(self, states: torch.Tensor) -> torch.Tensor:
        return states[:, self.flat].reshape(states.shape[0], -1, self.group_size)

    def merge(self, grouped: torch.Tensor) -> torch.Tensor:
        return grouped.flatten(start_dim=1)[:, self.inverse]


def product_state(values: torch.Tensor) -> torch.Tensor:
    """RY-encode pixels: (states, qubits) real values -> (states, 2**qubits) complex amplitudes."""
    theta = values * math.pi
    cos, sin = torch.cos(theta / 2.0), torch.sin(theta / 2.0)
    state = torch.stack((cos[:, 0], sin[:, 0]), dim=-1)
    for qubit in range(1, values.shape[1]):
        factor = torch.stack((cos[:, qubit], sin[:, qubit]), dim=-1)
        state = (state[:, :, None] * factor[:, None, :]).flatten(start_dim=1)
    return state.to(torch.complex64)


class PatchCircuit(nn.Module):
    """The s9_meas circuit: encode, then one two-qubit gate per bond group, shared by all patches."""

    def __init__(self, config: AuditConfig, init_scale: float, balanced: bool):
        super().__init__()
        self.rows, self.cols = config.patch
        self.n_qubits = self.rows * self.cols
        self.bond_groups = lattice_bonds(self.rows, self.cols)
        # Same module layout as s9_meas, so the old checkpoints load and the seed draws the same gates.
        self.gates = nn.ModuleList([
            nn.ModuleDict({name: LearnedUnitary2Q(init_scale=init_scale, balanced=balanced)
                           for name in self.bond_groups})
        ])
        self.bond_amplitudes = nn.ModuleDict({
            name: nn.ModuleList([GroupedAmplitudes(self.n_qubits, bond) for bond in bonds])
            for name, bonds in self.bond_groups.items()
        })
        self.qubit_amplitudes = nn.ModuleList(
            [GroupedAmplitudes(self.n_qubits, (qubit,)) for qubit in range(self.n_qubits)]
        )

    def evolve(self, values: torch.Tensor) -> torch.Tensor:
        """(states, qubits) pixel values -> (states, 2**qubits) final amplitudes."""
        states = product_state(values)
        for name, amplitude_list in self.bond_amplitudes.items():
            gate = self.gates[0][name]()
            for amplitudes in amplitude_list:
                states = amplitudes.merge(amplitudes.split(states) @ gate.T)
        return states

    def measure_paulis(self, states: torch.Tensor) -> torch.Tensor:
        """X, Y, Z on every qubit, then ZZ on every lattice bond: (states, 3*qubits + bonds)."""
        x, y, z = [], [], []
        for amplitudes in self.qubit_amplitudes:
            grouped = amplitudes.split(states)
            cross = (grouped[:, :, 0].conj() * grouped[:, :, 1]).sum(dim=1)
            populations = grouped.abs() ** 2
            x.append(2.0 * cross.real)
            y.append(2.0 * cross.imag)
            z.append((populations[:, :, 0] - populations[:, :, 1]).sum(dim=1))
        zz = []
        for amplitude_list in self.bond_amplitudes.values():
            for amplitudes in amplitude_list:
                populations = amplitudes.split(states).abs() ** 2
                zz.append((populations[:, :, 0] - populations[:, :, 1]
                           - populations[:, :, 2] + populations[:, :, 3]).sum(dim=1))
        return torch.stack([*x, *y, *z, *zz], dim=-1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.measure_paulis(self.evolve(values))


class ProductCircuit(nn.Module):
    """PatchCircuit with every gate restricted to first ⊗ second, so nothing ever entangles.

    The state stays a product of one-qubit states, so it is computed with 2 amplitudes per qubit
    instead of 2**12. test_model.py compares the result with the full statevector.
    """

    def __init__(self, config: AuditConfig, init_scale: float = 1.0):
        super().__init__()
        self.n_qubits = config.patch[0] * config.patch[1]
        self.bond_groups = lattice_bonds(*config.patch)
        self.gates = nn.ModuleDict({
            name: nn.ModuleDict({"first": LearnedUnitary1Q(init_scale), "second": LearnedUnitary1Q(init_scale)})
            for name in self.bond_groups
        })
        for name, bonds in self.bond_groups.items():
            self.register_buffer(f"{name}_first", torch.tensor([bond[0] for bond in bonds]), persistent=False)
            self.register_buffer(f"{name}_second", torch.tensor([bond[1] for bond in bonds]), persistent=False)

    def evolve(self, values: torch.Tensor) -> torch.Tensor:
        """(states, qubits) pixel values -> (states, qubits, 2) one-qubit amplitudes."""
        theta = values * math.pi
        qubits = torch.stack((torch.cos(theta / 2.0), torch.sin(theta / 2.0)), dim=-1).to(torch.complex64)
        for name in self.bond_groups:
            first, second = getattr(self, f"{name}_first"), getattr(self, f"{name}_second")
            updated = qubits.clone()
            updated[:, first] = qubits[:, first] @ self.gates[name]["first"]().T
            updated[:, second] = qubits[:, second] @ self.gates[name]["second"]().T
            qubits = updated
        return qubits

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Same features in the same order as PatchCircuit: X, Y, Z on every qubit, then ZZ on every bond."""
        qubits = self.evolve(values)
        cross = qubits[:, :, 0].conj() * qubits[:, :, 1]
        populations = qubits.abs() ** 2
        z = populations[:, :, 0] - populations[:, :, 1]
        zz = [z[:, first] * z[:, second] for bonds in self.bond_groups.values() for first, second in bonds]
        return torch.cat((2.0 * cross.real, 2.0 * cross.imag, z, torch.stack(zz, dim=-1)), dim=-1)


class EncodingFeatures(nn.Module):
    """What PatchCircuit measures with no gates at all: cos, sin and neighbour products. Y is always 0, so it is left out."""

    def __init__(self, config: AuditConfig):
        super().__init__()
        bonds = [bond for group in lattice_bonds(*config.patch).values() for bond in group]
        self.register_buffer("first", torch.tensor([bond[0] for bond in bonds]), persistent=False)
        self.register_buffer("second", torch.tensor([bond[1] for bond in bonds]), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        theta = values * math.pi
        cos, sin = torch.cos(theta), torch.sin(theta)
        return torch.cat((sin, cos, cos[:, self.first] * cos[:, self.second]), dim=-1)


class PatchAuditModel(nn.Module):
    """Feature layer per patch, one Linear (or small MLP) shared by all patches, then the s9_meas MLP readout."""

    def __init__(self, config: AuditConfig, init: str = "balanced", init_scale: float = 0.1,
                 freeze_core: bool = False):
        super().__init__()
        if config.core not in CORES:
            raise ValueError(f"Unknown core: {config.core}")
        if init not in {"balanced", "random"}:
            raise ValueError(f"Unknown init: {init}")
        if config.core == "product" and init != "random":
            raise ValueError("The balanced gate is entangling, so c2_product needs --init random")
        if freeze_core and config.core not in {"quantum", "product"}:
            raise ValueError(f"Core {config.core} has no gates to freeze")
        self.config = config
        self.register_buffer("sites", patch_site_indices(config), persistent=False)
        n_qubits = config.patch[0] * config.patch[1]
        n_bonds = sum(len(bonds) for bonds in lattice_bonds(*config.patch).values())
        patches = config.grid[0] * config.grid[1]

        self.channels = nn.ModuleList()
        if config.core == "quantum":
            balanced = init == "balanced"
            self.channels.append(PatchCircuit(config, init_scale if balanced else 1.0, balanced))
            feature = nn.Identity()
            project = nn.Linear(3 * n_qubits + n_bonds, config.projection)
        elif config.core == "product":
            self.channels.append(ProductCircuit(config))
            feature = nn.Identity()
            project = nn.Linear(3 * n_qubits + n_bonds, config.projection)
        elif config.core == "encoding":
            feature = EncodingFeatures(config)
            project = nn.Linear(2 * n_qubits + n_bonds, config.projection)
        elif config.core == "pixel":
            feature = nn.Identity()
            project = nn.Linear(n_qubits, config.projection)
        else:
            feature = nn.Identity()
            project = nn.Sequential(nn.Linear(n_qubits, config.local_hidden), nn.GELU(),
                                    nn.Linear(config.local_hidden, config.projection))
        self.feature = feature
        self.head = nn.ModuleDict({"project": project})
        feature_dim = patches * config.projection
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.head["classifier"] = nn.Sequential(
            nn.Linear(feature_dim, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, 10),
        )
        if freeze_core:
            self.channels.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixels = F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                               align_corners=False).clamp(0.0, 1.0).flatten(start_dim=1)
        values = pixels[:, self.sites]
        batch, patches, qubits = values.shape
        flat = values.reshape(batch * patches, qubits)
        features = self.channels[0](flat) if len(self.channels) else self.feature(flat)
        projected = self.head["project"](features.reshape(batch, patches, -1)).flatten(start_dim=1)
        return self.head["classifier"](self.norm(projected))

    def count_parameters(self) -> Dict[str, int]:
        core = sum(parameter.numel() for parameter in self.channels.parameters())
        head = sum(parameter.numel() for parameter in self.head.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"core": core, "head": head, "total": core + head, "trainable": trainable}


def is_head_parameter(name: str) -> bool:
    return name.startswith("head.")
