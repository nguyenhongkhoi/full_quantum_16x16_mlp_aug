"""The 3-scan fusion QCNN from full_quantum_16x16_3scan_fusion_mlp16, always with re-upload."""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .encoding import angle_ry_encode
from .front import front_circuit_parameters, make_front
from .graph_layer import GraphZZLayer
from .pauli import PauliFeatures
from .preprocessing import preprocess_full_quantum
from .reupload import ReuploadQuantumTNCore

TARGET_SIZE = (16, 16)
FINAL_QUBITS = 8
PAIRS = FINAL_QUBITS * (FINAL_QUBITS - 1) // 2
BRANCH_FEATURES = 3 * FINAL_QUBITS + 9 * PAIRS
SPATIAL_ORDERS = ("row", "column", "morton")
SCAN_ANGLE_ORDERS = ("angle0", "angle60", "angle120")
PHASE_DEGREES = (0.0, 60.0, 120.0)


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


def position_phases(height: int, width: int, cycles: float, degrees: float) -> torch.Tensor:
    """exp(i phi) per pixel in row-major order: a plane wave of `cycles` periods across the image, at this direction."""
    rows, columns = torch.meshgrid(torch.arange(height) / height, torch.arange(width) / width, indexing="ij")
    direction = math.radians(degrees)
    phi = 2 * math.pi * cycles * (columns * math.cos(direction) + rows * math.sin(direction))
    return torch.polar(torch.ones_like(phi), phi).flatten().to(torch.complex64)


def bloch(rho: torch.Tensor) -> torch.Tensor:
    """(batch, qubits, 2, 2) -> (batch, 3 * qubits): all X, then all Y, then all Z. rho = (I + xX + yY + zZ) / 2."""
    x = 2 * rho[..., 0, 1].real
    y = -2 * rho[..., 0, 1].imag
    z = (rho[..., 0, 0] - rho[..., 1, 1]).real
    return torch.cat((x, y, z), dim=1)


class PauliTypeProjection(nn.Module):
    """One small Linear for every qubit (its X, Y, Z) and one for every pair (its 9 correlators), shared by all.

    Follows the PauliFeatures order: all X, all Y, all Z, then 9 correlators per pair.
    """

    def __init__(self, size: int):
        super().__init__()
        self.single = nn.Linear(3, size)
        self.pair = nn.Linear(9, size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        singles = features[:, :3 * FINAL_QUBITS].reshape(-1, 3, FINAL_QUBITS).transpose(1, 2)
        pairs = features[:, 3 * FINAL_QUBITS:].reshape(-1, PAIRS, 9)
        return torch.cat((self.single(singles).flatten(1), self.pair(pairs).flatten(1)), dim=1)


class ExpertMixture(nn.Module):
    """Small MLP experts chosen softly from the input, with no task label. Returns log probabilities of the mixture.

    Mixing probabilities makes the experts compete: an expert with a small weight gets a small gradient.
    Cross entropy still works, because log_softmax leaves log probabilities unchanged.
    """

    def __init__(self, inputs: int, hidden: int, num_classes: int, experts: int):
        super().__init__()
        self.gate = nn.Linear(inputs, experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, num_classes))
            for _ in range(experts)
        ])

    def weights(self, values: torch.Tensor) -> torch.Tensor:
        return self.gate(values).softmax(dim=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        log_weights = self.gate(values).log_softmax(dim=1)
        log_probabilities = torch.stack([expert(values).log_softmax(dim=1) for expert in self.experts], dim=1)
        return torch.logsumexp(log_weights[:, :, None] + log_probabilities, dim=1)


def scale_learning_rate(module: nn.Module, scale: float) -> None:
    """Make every parameter of `module` learn `scale` times faster under Adam, without changing what it computes.

    Each quantum layer uses value = param_scale * stored. The stored tensor is divided by scale here, so the value
    is unchanged (exactly, when scale is a power of two). Adam moves the stored tensor by about lr per step, so the
    value moves about scale * lr. Every module that owns parameters must support param_scale."""
    for owner in module.modules():
        own = list(owner.parameters(recurse=False))
        if not own:
            continue
        if not hasattr(owner, "param_scale"):
            raise TypeError(f"{type(owner).__name__} does not support param_scale")
        with torch.no_grad():
            for parameter in own:
                parameter.div_(scale)
        owner.param_scale = owner.param_scale * scale


class FusionQCNN(nn.Module):
    """Three independent trees, one per scan order; each projects to 16 values, then one shared MLP.

    With experts > 0, each tree projects to expert_projection values and an ExpertMixture replaces the shared MLP.
    With typed_projection set, one PauliTypeProjection shared by all trees replaces the per-tree Linear(276, 16).
    With side_level set, each tree also measures X, Y, Z of its qubits after the tree level with that many qubits.
    Inside the tree the qubits are independent, so these values describe that level exactly. projection sets the
    size each tree is projected to, 16 by default.
    With phase_cycles set, each tree adds a position phase to |1> of every pixel, in its own direction. Pixel values
    do not change, and a pixel that is exactly 0 or 1 is not affected, because the phase is then global.
    With graph_layer, each tree gets its own GraphZZLayer (8 neighbours, 10 parameters) before the tree; it starts
    as the identity. With circuit_lr_scale, every quantum parameter (trees, cores, re-upload, graph layers) learns
    that many times faster under Adam; the starting values do not change. With dropout > 0, dropout follows GELU in
    the per-tree projections and plain fusion MLP; it is active only during training and adds no parameters.
    With front other than "none", a ClassicalFront (models/front.py) shared by all trees rewrites the 256 pixel
    values before RY encoding; values are then not clamped. front may also be qconv<L>[x<S>][f] (2026-11-13), a
    small quantum circuit as the encoder (QuantumFront). It starts as the identity, and the rest of the model
    starts exactly as without it. count_parameters reports the encoder's circuit angles as front_quantum, and
    quantum_share counts them as quantum.
    """

    def __init__(self, num_classes: int = 10, hidden: int = 16, orders: Sequence[str] = SPATIAL_ORDERS,
                 experts: int = 0, expert_projection: int = 8, phase_cycles: Optional[float] = None,
                 typed_projection: Optional[int] = None, side_level: Optional[int] = None,
                 projection: Optional[int] = None, graph_layer: bool = False, circuit_lr_scale: float = 1.0,
                 dropout: float = 0.2, front: str = "none"):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if graph_layer and phase_cycles is not None:
            raise ValueError("graph_layer does not support phase_cycles")
        if front != "none" and graph_layer:
            raise ValueError("front does not support graph_layer (the graph layer clamps values to [0, 1])")
        if side_level is not None:
            if typed_projection or experts:
                raise ValueError("side_level works with the plain head only")
            if side_level not in (16, 32, 64, 128):
                raise ValueError(f"No tree level has {side_level} qubits")
        self.side_level = side_level
        branch_inputs = BRANCH_FEATURES + (3 * side_level if side_level else 0)
        self.register_buffer("site_permutations", torch.stack([
            spatial_permutation(*TARGET_SIZE, order) for order in orders
        ]), persistent=False)
        phases = None if phase_cycles is None else torch.stack([
            position_phases(*TARGET_SIZE, phase_cycles, degrees) for degrees in PHASE_DEGREES[:len(orders)]
        ])
        self.register_buffer("site_phases", phases, persistent=False)
        self.cores = nn.ModuleList([
            ReuploadQuantumTNCore(input_sites=TARGET_SIZE[0] * TARGET_SIZE[1], final_qubits=FINAL_QUBITS,
                                  init_scale=0.01, merge_rounds=2, entangling_type="strongly_entangling",
                                  n_layers=4)
            for _ in orders
        ])
        self.pauli = PauliFeatures(FINAL_QUBITS)
        if typed_projection:
            shared = PauliTypeProjection(typed_projection)
            self.projections = nn.ModuleList([shared] * len(orders))
            projected = (FINAL_QUBITS + PAIRS) * typed_projection
        else:
            projected = expert_projection if experts else (projection or hidden)
            self.projections = nn.ModuleList([
                nn.Sequential(nn.Linear(branch_inputs, projected), nn.LayerNorm(projected), nn.GELU(),
                              nn.Dropout(dropout))
                for _ in orders
            ])
        if experts:
            self.fusion = ExpertMixture(len(orders) * projected, hidden, num_classes, experts)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(len(orders) * projected, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        # Created last and drawing no random numbers, so everything above starts as without these options.
        self.graphs = nn.ModuleList([GraphZZLayer(*TARGET_SIZE) for _ in orders]) if graph_layer else None
        self.circuit_lr_scale = circuit_lr_scale
        if circuit_lr_scale != 1.0:
            scale_learning_rate(self.cores, circuit_lr_scale)
            if self.graphs is not None:
                scale_learning_rate(self.graphs, circuit_lr_scale)
        # Created under a forked random state (see models/front.py), so nothing above or in training changes.
        self.front_mode = front
        self.front = None if front == "none" else make_front(front, *TARGET_SIZE)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = preprocess_full_quantum(images, TARGET_SIZE, grayscale=True, mode="resize")
        phi = None
        if self.front is not None:
            values, phi = self.front(values)
        qubits = angle_ry_encode(values, clamp=self.front is None).to(images.device)
        if phi is not None:
            # RZ(pi * phi) up to a global phase: exp(i pi phi) on |1>. The re-upload angles use magnitudes only.
            qubits = torch.stack((qubits[..., 0], qubits[..., 1] * torch.polar(torch.ones_like(phi), torch.pi * phi)),
                                 dim=-1)
        branches = []
        for index, (core, projection) in enumerate(zip(self.cores, self.projections)):
            branch_qubits = qubits
            if self.site_phases is not None:
                branch_qubits = torch.stack((qubits[..., 0], qubits[..., 1] * self.site_phases[index]), dim=-1)
            permutation = self.site_permutations[index]
            branch_qubits = branch_qubits.index_select(1, permutation)
            branch_rho = None
            if self.graphs is not None:
                branch_rho = self.graphs[index](values.to(images.device)).index_select(1, permutation)
            # Keep only one branch's graph in GPU memory at a time; the result does not change.
            if self.side_level is not None:
                if self.training:
                    features = checkpoint(self._side_features, core, branch_qubits, branch_rho, use_reentrant=False,
                                          preserve_rng_state=False)
                else:
                    features = self._side_features(core, branch_qubits, branch_rho)
                branches.append(projection(features))
                continue
            if branch_rho is not None:
                raise ValueError("graph_layer needs side_level")
            if self.training:
                rho = checkpoint(core, branch_qubits, use_reentrant=False, preserve_rng_state=False)
            else:
                rho = core(branch_qubits)
            branches.append(projection(self.pauli(rho)))
        return self.fusion(torch.cat(branches, dim=1))

    def _side_features(self, core: ReuploadQuantumTNCore, qubits: torch.Tensor,
                       rho: Optional[torch.Tensor] = None) -> torch.Tensor:
        joint, kept = core.forward_levels(qubits, (self.side_level,), rho=rho)
        return torch.cat((self.pauli(joint), bloch(kept[self.side_level])), dim=1)

    def count_parameters(self) -> Dict[str, int]:
        quantum = sum(parameter.numel() for parameter in self.cores.parameters())
        if self.graphs is not None:
            quantum += sum(parameter.numel() for parameter in self.graphs.parameters())
        head = sum(parameter.numel() for module in (self.projections, self.fusion)
                   for parameter in module.parameters())
        front = 0 if self.front is None else sum(parameter.numel() for parameter in self.front.parameters())
        front_quantum = front_circuit_parameters(self.front)
        return {"quantum": quantum, "front": front, "front_quantum": front_quantum, "head": head,
                "total": quantum + front + head, "quantum_share": (quantum + front_quantum) / (quantum + front + head)}
