"""Patch QCNN: exact statevector patches with interleaved gates, then a quantum merge tree."""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .unitary import LearnedUnitary2Q

IMAGE_SIZE = 16
TAP_NODES = 16
FINAL_NODES = 4


@dataclass(frozen=True)
class PatchConfig:
    patch: Tuple[int, int]      # Qubits per patch = rows * cols.
    grid: Tuple[int, int]       # Patch positions; both must be powers of two.
    layers: int = 1             # Number of [encode, gates] rounds inside a patch.
    channels: int = 1           # Independent copies of the whole quantum core.
    projection: int = 6         # Output size of each per-cell Linear.
    hidden: int = 16            # MLP readout width.
    grad_checkpoint: bool = False


@dataclass(frozen=True)
class MeasureConfig(PatchConfig):
    """No trace and no merge tree: every patch is measured directly."""

    projection: int = 4


def patch_positions(image_size: int, patch_size: int, count: int) -> List[int]:
    """`count` evenly spread start positions, first at 0 and last touching the image edge."""
    span = image_size - patch_size
    return [round(index * span / (count - 1)) for index in range(count)]


def patch_site_indices(config: PatchConfig) -> torch.Tensor:
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
    """[encode, gates] x layers on one patch, with one gate per bond group and layer, shared by all patches."""

    def __init__(self, config: PatchConfig, init_scale: float, balanced: bool):
        super().__init__()
        self.rows, self.cols = config.patch
        self.n_qubits = self.rows * self.cols
        self.layers = config.layers
        self.grad_checkpoint = config.grad_checkpoint
        self.bond_groups = lattice_bonds(self.rows, self.cols)
        self.gates = nn.ModuleList([
            nn.ModuleDict({name: LearnedUnitary2Q(init_scale=init_scale, balanced=balanced)
                           for name in self.bond_groups})
            for _ in range(self.layers)
        ])
        self.bond_amplitudes = nn.ModuleDict({
            name: nn.ModuleList([GroupedAmplitudes(self.n_qubits, bond) for bond in bonds])
            for name, bonds in self.bond_groups.items()
        })
        self.qubit_amplitudes = nn.ModuleList(
            [GroupedAmplitudes(self.n_qubits, (qubit,)) for qubit in range(self.n_qubits)]
        )
        row = self.rows // 2
        self.kept = (row * self.cols + self.cols // 2 - 1, row * self.cols + self.cols // 2)
        self.kept_amplitudes = GroupedAmplitudes(self.n_qubits, self.kept)

    def _reupload(self, states: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        theta = values * math.pi
        cos, sin = torch.cos(theta / 2.0), torch.sin(theta / 2.0)
        for qubit, amplitudes in enumerate(self.qubit_amplitudes):
            grouped = amplitudes.split(states)
            c, s = cos[:, qubit, None], sin[:, qubit, None]
            first = c * grouped[:, :, 0] - s * grouped[:, :, 1]
            second = s * grouped[:, :, 0] + c * grouped[:, :, 1]
            states = amplitudes.merge(torch.stack((first, second), dim=-1))
        return states

    def _gate_round(self, states: torch.Tensor, layer: int) -> torch.Tensor:
        for name, amplitude_list in self.bond_amplitudes.items():
            gate = self.gates[layer][name]()
            for amplitudes in amplitude_list:
                states = amplitudes.merge(amplitudes.split(states) @ gate.T)
        return states

    def _layer(self, states: torch.Tensor, values: torch.Tensor, layer: int) -> torch.Tensor:
        return self._gate_round(states if layer == 0 else self._reupload(states, values), layer)

    def evolve(self, values: torch.Tensor) -> torch.Tensor:
        """(states, qubits) pixel values -> (states, 2**qubits) final amplitudes."""
        states = product_state(values)
        for layer in range(self.layers):
            if self.grad_checkpoint and self.training:
                states = checkpoint(self._layer, states, values, layer,
                                    use_reentrant=False, preserve_rng_state=False)
            else:
                states = self._layer(states, values, layer)
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
        """(states, qubits) pixel values -> (states, 4, 4) density matrices of the two kept qubits."""
        states = self.evolve(values)
        kept = self.kept_amplitudes.split(states)
        rho = torch.einsum("nmr,nms->nrs", kept, kept.conj())
        return _normalize_hermitize(rho)


def _normalize_hermitize(rho: torch.Tensor) -> torch.Tensor:
    trace = rho.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real.clamp_min(1e-8)
    rho = rho / trace[..., None, None]
    return 0.5 * (rho + rho.conj().transpose(-1, -2))


class Bond2Merge(nn.Module):
    """Children [a0 a1] and [b0 b1] -> one node holding (a1, b0): cross gate first, then local gates, then trace."""

    def __init__(self, init_scale: float, balanced: bool):
        super().__init__()
        self.cross = LearnedUnitary2Q(init_scale=init_scale, balanced=balanced)
        self.left = LearnedUnitary2Q(init_scale=init_scale, balanced=balanced)
        self.right = LearnedUnitary2Q(init_scale=init_scale, balanced=balanced)

    def unitary(self) -> torch.Tensor:
        cross = self.cross()
        identity = torch.eye(2, dtype=cross.dtype, device=cross.device)
        return torch.kron(self.left(), self.right()) @ torch.kron(torch.kron(identity, cross), identity)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """(batch, 2*n, 4, 4) with paired neighbours -> (batch, n, 4, 4)."""
        batch, nodes, _, _ = rho.shape
        pairs = torch.einsum("bnij,bnkl->bnikjl", rho[:, 0::2], rho[:, 1::2])
        pairs = pairs.reshape(batch, nodes // 2, 16, 16)
        unitary = self.unitary()
        pairs = unitary @ pairs @ unitary.conj().T
        kept = torch.einsum("bnaijdakld->bnijkl", pairs.reshape(batch, nodes // 2, *([2] * 8)))
        return _normalize_hermitize(kept.reshape(batch, nodes // 2, 4, 4))


def two_qubit_paulis() -> torch.Tensor:
    """The 15 two-qubit Pauli operators except II: (15, 4, 4)."""
    identity = torch.eye(2, dtype=torch.complex64)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    return torch.stack([torch.kron(a, b) for a, b in product((identity, x, y, z), repeat=2)][1:])


def merge_axes(grid: Tuple[int, int]) -> List[int]:
    """Which grid axis each merge halves (0 rows, 1 columns), down to FINAL_NODES nodes."""
    rows, cols = grid
    axes = []
    while rows * cols > FINAL_NODES:
        if rows >= cols:
            axes.append(0)
            rows //= 2
        else:
            axes.append(1)
            cols //= 2
    return axes


class PatchChannel(nn.Module):
    """One quantum core: a patch circuit followed by a merge tree."""

    def __init__(self, config: PatchConfig, init_scale: float, balanced: bool):
        super().__init__()
        self.grid = config.grid
        self.circuit = PatchCircuit(config, init_scale, balanced)
        self.axes = merge_axes(config.grid)
        self.merges = nn.ModuleList([Bond2Merge(init_scale, balanced) for _ in self.axes])

    def forward(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(batch, patches, qubits) -> tap (batch, 16, 4, 4) and final (batch, 4, 4, 4)."""
        batch, patches, qubits = values.shape
        rho = self.circuit(values.reshape(batch * patches, qubits))
        rows, cols = self.grid
        rho = rho.reshape(batch, rows, cols, 4, 4)
        tap = rho.flatten(1, 2) if rows * cols == TAP_NODES else None
        for axis, merge in zip(self.axes, self.merges):
            # Pair neighbours along the merged axis, which has to be the last grid axis before flattening.
            if axis == 0:
                flat = rho.transpose(1, 2).flatten(1, 2)
                merged = merge(flat).reshape(batch, cols, rows // 2, 4, 4)
                rho = merged.transpose(1, 2)
                rows //= 2
            else:
                merged = merge(rho.flatten(1, 2)).reshape(batch, rows, cols // 2, 4, 4)
                rho = merged
                cols //= 2
            if rows * cols == TAP_NODES:
                tap = rho.flatten(1, 2)
        return tap, rho.flatten(1, 2)


class PatchHead(nn.Module):
    """Per-cell projections shared across cells, then an MLP or Linear readout."""

    def __init__(self, config: PatchConfig, readout: str, num_classes: int = 10):
        super().__init__()
        cell_features = 15 * config.channels
        self.project_tap = nn.Linear(cell_features, config.projection)
        self.project_final = nn.Linear(cell_features, config.projection)
        feature_dim = (TAP_NODES + FINAL_NODES) * config.projection
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        if readout == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, config.hidden),
                nn.LayerNorm(config.hidden),
                nn.GELU(),
                nn.Linear(config.hidden, num_classes),
            )
        elif readout == "linear":
            self.classifier = nn.Linear(feature_dim, num_classes)
        else:
            raise ValueError(f"Unknown readout: {readout}")

    def forward(self, tap: torch.Tensor, final: torch.Tensor) -> torch.Tensor:
        features = torch.cat((self.project_tap(tap).flatten(start_dim=1),
                              self.project_final(final).flatten(start_dim=1)), dim=1)
        return self.classifier(self.norm(features))


class PatchQCNN(nn.Module):
    def __init__(self, config: PatchConfig, readout: str = "mlp", init: str = "balanced",
                 init_scale: float = 0.1, freeze_core: bool = False):
        super().__init__()
        if init not in {"balanced", "random"}:
            raise ValueError(f"Unknown init: {init}")
        self.config = config
        balanced = init == "balanced"
        scale = init_scale if balanced else 1.0
        self.register_buffer("sites", patch_site_indices(config), persistent=False)
        self.register_buffer("paulis", two_qubit_paulis(), persistent=False)
        self.channels = nn.ModuleList([PatchChannel(config, scale, balanced) for _ in range(config.channels)])
        self.head = PatchHead(config, readout)
        if freeze_core:
            self.channels.requires_grad_(False)

    def _pauli_features(self, rho: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bnij,pji->bnp", rho, self.paulis).real

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixels = F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                               align_corners=False).clamp(0.0, 1.0).flatten(start_dim=1)
        values = pixels[:, self.sites]
        taps, finals = [], []
        for channel in self.channels:
            tap, final = channel(values)
            taps.append(self._pauli_features(tap))
            finals.append(self._pauli_features(final))
        return self.head(torch.cat(taps, dim=-1), torch.cat(finals, dim=-1))

    def count_parameters(self) -> Dict[str, int]:
        core = sum(parameter.numel() for parameter in self.channels.parameters())
        head = sum(parameter.numel() for parameter in self.head.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"core": core, "head": head, "total": core + head, "trainable": trainable}


class PatchMeasureQCNN(nn.Module):
    """Patch circuit -> many Paulis per patch, one Linear shared by all patches -> MLP or Linear readout."""

    def __init__(self, config: MeasureConfig, readout: str = "mlp", init: str = "balanced",
                 init_scale: float = 0.1, freeze_core: bool = False):
        super().__init__()
        if init not in {"balanced", "random"}:
            raise ValueError(f"Unknown init: {init}")
        if config.channels != 1:
            raise ValueError("MeasureConfig supports one channel")
        self.config = config
        balanced = init == "balanced"
        self.register_buffer("sites", patch_site_indices(config), persistent=False)
        circuit = PatchCircuit(config, init_scale if balanced else 1.0, balanced)
        self.channels = nn.ModuleList([circuit])
        n_bonds = sum(len(bonds) for bonds in circuit.bond_groups.values())
        patches = config.grid[0] * config.grid[1]
        self.head = nn.ModuleDict({"project": nn.Linear(3 * circuit.n_qubits + n_bonds, config.projection)})
        feature_dim = patches * config.projection
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        if readout == "mlp":
            self.head["classifier"] = nn.Sequential(
                nn.Linear(feature_dim, config.hidden),
                nn.LayerNorm(config.hidden),
                nn.GELU(),
                nn.Linear(config.hidden, 10),
            )
        elif readout == "linear":
            self.head["classifier"] = nn.Linear(feature_dim, 10)
        else:
            raise ValueError(f"Unknown readout: {readout}")
        if freeze_core:
            self.channels.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixels = F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                               align_corners=False).clamp(0.0, 1.0).flatten(start_dim=1)
        values = pixels[:, self.sites]
        batch, patches, qubits = values.shape
        circuit = self.channels[0]
        features = circuit.measure_paulis(circuit.evolve(values.reshape(batch * patches, qubits)))
        projected = self.head["project"](features.reshape(batch, patches, -1)).flatten(start_dim=1)
        return self.head["classifier"](self.norm(projected))

    count_parameters = PatchQCNN.count_parameters


def is_head_parameter(name: str) -> bool:
    return name.startswith("head.")
