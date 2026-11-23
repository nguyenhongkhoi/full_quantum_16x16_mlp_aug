"""Checks that need no MNIST and no training. Copy next to Mammoth's main.py and run it from there.

mammoth_level32_front (2026-11-13): adds check_front for qcnn-fusion-level32-deep12p and its classical controls.
mammoth_qconv_front (2026-11-13): adds check_qconv for the quantum encoder (QuantumFront).
mammoth_full_package (2026-11-16): the timing report covers the backbones of the run queue."""
from __future__ import annotations

import argparse
import cmath
import math
import time
from argparse import Namespace

import torch

from backbone import get_backbone, get_backbone_names
from backbone.QuantumCL import EXPECTED_PARAMETERS
from backbone.quantum_core.encoding import angle_ry_encode
from backbone.quantum_core.fusion_model import (BRANCH_FEATURES, SCAN_ANGLE_ORDERS, SPATIAL_ORDERS, FusionQCNN,
                                                bloch, spatial_permutation)
from backbone.quantum_core.graph_layer import graph_reduced_states
from backbone.quantum_core.tn_core import _pure_to_density
from backbone.quantum_core.pauli import PauliFeatures
from backbone.quantum_core.front import QuantumConvStage, QuantumFront, grid_edges, rotation_matrices


def build(name: str):
    return get_backbone(Namespace(backbone=name, num_classes=10))


def check_registered_and_counts() -> None:
    names = get_backbone_names(names_only=True)
    for name, expected in EXPECTED_PARAMETERS.items():
        assert name in names, f"{name} is not registered"
        count = sum(parameter.numel() for parameter in build(name).parameters())
        assert count == expected, f"{name}: {count}"
    parts = FusionQCNN().count_parameters()
    assert parts == {"quantum": 4824, "head": 14378, "total": 19202}, parts
    print("OK  registered, parameter counts:", EXPECTED_PARAMETERS)


def check_scan_orders() -> None:
    """Scan angles only reorder the pixels: every pixel once, and the row order at angle 0."""
    orders = [spatial_permutation(16, 16, order) for order in SCAN_ANGLE_ORDERS]
    for order, sites in zip(SCAN_ANGLE_ORDERS, orders):
        assert sorted(sites.tolist()) == list(range(256)), f"{order} is not a permutation"
    assert torch.equal(orders[0], spatial_permutation(16, 16, "row")), "angle 0 differs from the row order"
    assert not torch.equal(orders[1], orders[2]) and not torch.equal(orders[0], orders[1]), "scan orders repeat"
    print("OK  scan angles are permutations of the same pixels:", SCAN_ANGLE_ORDERS)


def check_position_phase() -> None:
    """Zero cycles and pure black or white images must give the plain model; grey images must not."""
    def outputs(images: torch.Tensor, cycles) -> torch.Tensor:
        torch.manual_seed(0)
        return FusionQCNN(phase_cycles=cycles).eval()(images)

    grey = torch.rand(2, 1, 28, 28)
    with torch.no_grad():
        plain = outputs(grey, None)
        assert torch.allclose(outputs(grey, 0.0), plain, atol=1e-4), "zero cycles changes the output"
        assert not torch.allclose(outputs(grey, 2.0), plain, atol=1e-4), "the phase has no effect on grey pixels"
        for value in (0.0, 1.0):
            flat = torch.full((2, 1, 28, 28), value)
            assert torch.allclose(outputs(flat, 2.0), outputs(flat, None), atol=1e-4), f"phase changes pixels at {value}"
    print("OK  position phase: no effect at 0 cycles or on pure pixels, an effect on grey pixels")


def check_typed_readout() -> None:
    """Parameter split, one projection shared by the three trees, and the same features as the offline sweep."""
    model = FusionQCNN(typed_projection=4)
    parts = model.count_parameters()
    assert parts == {"quantum": 4824, "head": 7186, "total": 12010}, parts
    assert model.projections[0] is model.projections[1] is model.projections[2], "the projection is not shared"
    features = torch.randn(2, BRANCH_FEATURES)
    projected = model.projections[0](features)
    assert projected.shape == (2, 36 * 4), projected.shape
    first_qubit = model.projections[0].single(features[:, [0, 8, 16]])
    assert torch.allclose(projected[:, :4], first_qubit, atol=1e-6), "X, Y, Z of qubit 0 are not read together"
    print("OK  typed readout:", parts)


def check_side_level() -> None:
    """Level-32 readout: parameter split, an unchanged core output, the real state after three tree levels, Bloch
    vectors no longer than 1, and gradient checkpointing that changes nothing."""
    model = FusionQCNN(side_level=32, projection=12)
    parts = model.count_parameters()
    assert parts == {"quantum": 4824, "head": 14294, "total": 19118}, parts
    core = model.cores[0]
    qubits = angle_ry_encode(torch.rand(2, 256)).index_select(1, model.site_permutations[0])
    with torch.no_grad():
        joint, kept = core.forward_levels(qubits, (32,))
        assert torch.allclose(joint, core(qubits), atol=1e-6), "forward_levels changes the core output"
        rho = _pure_to_density(qubits)
        for block in core.blocks[:3]:
            rho = block(rho)
        assert torch.allclose(kept[32], rho, atol=1e-6), "level 32 is not the state after three tree levels"
        values = bloch(rho)
        assert values.shape == (2, 96), values.shape
        assert values.reshape(2, 3, 32).norm(dim=1).max().item() <= 1 + 1e-4, "Bloch vector longer than 1"
        assert torch.allclose(values[:, 64:], (rho[:, :, 0, 0] - rho[:, :, 1, 1]).real, atol=1e-6), "Z is wrong"
    images = torch.rand(2, 1, 28, 28)
    labels = torch.randint(0, 10, (2,))
    model.eval()
    plain = model(images)
    torch.nn.functional.cross_entropy(plain, labels).backward()
    expected = [parameter.grad.clone() for parameter in model.parameters()]
    model.zero_grad()
    model.train()
    checked = model(images)
    torch.nn.functional.cross_entropy(checked, labels).backward()
    assert torch.allclose(plain, checked, atol=1e-4), "checkpointed output differs"
    for parameter, reference in zip(model.parameters(), expected):
        assert torch.allclose(parameter.grad, reference, atol=1e-4), "checkpointed gradient differs"
    print("OK  level-32 readout:", parts)


def _same_start(**options) -> tuple[FusionQCNN, FusionQCNN]:
    torch.manual_seed(0)
    base = FusionQCNN(side_level=32, projection=12)
    torch.manual_seed(0)
    other = FusionQCNN(side_level=32, projection=12, **options)
    return base, other


def check_lr_scale() -> None:
    """qcnn-fusion-level32-lr4: same parameters, the same output at the start, and after one Adam step the quantum
    parameters move 4 times as far while the head moves the same."""
    base, fast = _same_start(circuit_lr_scale=4.0)
    assert fast.count_parameters() == {"quantum": 4824, "head": 14294, "total": 19118}, fast.count_parameters()
    images = torch.rand(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    base.eval(), fast.eval()
    with torch.no_grad():
        assert torch.allclose(base(images), fast(images), atol=1e-6), "the scaled model starts differently"

    def one_step(model):
        gate = model.cores[0].blocks[0].conv
        start, head_start = (gate.raw_real * gate.param_scale).detach().clone(), model.fusion[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        torch.nn.functional.cross_entropy(model(images), labels).backward()
        optimizer.step()
        return (gate.raw_real * gate.param_scale).detach() - start, model.fusion[0].weight.detach() - head_start

    quantum_base, head_base = one_step(base)
    quantum_fast, head_fast = one_step(fast)
    moved = quantum_base.abs() > 5e-4
    ratio = (quantum_fast[moved] / quantum_base[moved]).median().item()
    assert abs(ratio - 4.0) < 0.05, f"quantum step ratio {ratio}"
    assert torch.allclose(head_base, head_fast, atol=1e-6), "the head step changed"
    print(f"OK  circuit lr x4: same start, quantum step ratio {ratio:.3f}, same head step")


def check_four_trees() -> None:
    """qcnn-fusion-level32-4tree: parameter split, and the shifted Morton order keeps the 4x4 blocks while one
    64-pixel block is the middle 8x8 square."""
    model = FusionQCNN(orders=SPATIAL_ORDERS + ("morton_shift4",), side_level=32, projection=8)
    parts = model.count_parameters()
    assert parts == {"quantum": 6432, "head": 12730, "total": 19162}, parts
    morton = spatial_permutation(16, 16, "morton")
    shifted = spatial_permutation(16, 16, "morton_shift4")
    assert sorted(shifted.tolist()) == list(range(256)), "morton_shift4 is not a permutation"
    assert not torch.equal(morton, shifted), "morton_shift4 equals morton"
    blocks = lambda order, size: {frozenset(order[i:i + size].tolist()) for i in range(0, 256, size)}
    assert blocks(morton, 16) == blocks(shifted, 16), "the 4x4 blocks moved"
    middle = frozenset(y * 16 + x for y in range(4, 12) for x in range(4, 12))
    assert middle in blocks(shifted, 64), "the middle 8x8 square is not one block"
    print("OK  four trees:", parts)


def check_graph_layer() -> None:
    """The graph layer against a full 9-qubit statevector on a 3x3 grid, turning with the image by 90 degrees, and
    qcnn-fusion-level32-graph starting exactly as qcnn-fusion-level32."""
    torch.manual_seed(1)
    theta = torch.rand(1, 3, 3) * torch.pi
    gammas = torch.tensor([0.37, -0.81])
    fast = graph_reduced_states(theta, gammas)[0]
    angles = theta.flatten()
    psi = torch.ones(1, dtype=torch.complex128)
    for angle in angles:
        psi = torch.kron(psi, torch.tensor([torch.cos(angle / 2), torch.sin(angle / 2)], dtype=torch.complex128))
    bits = (torch.arange(512)[:, None] >> torch.arange(8, -1, -1)[None, :]) & 1
    z = 1 - 2 * bits.double()
    energy = torch.zeros(512, dtype=torch.float64)
    for i in range(9):
        for j in range(i + 1, 9):
            (yi, xi), (yj, xj) = divmod(i, 3), divmod(j, 3)
            dy, dx = abs(yi - yj), abs(xi - xj)
            if max(dy, dx) == 1:
                energy += float(gammas[0] if dy + dx == 1 else gammas[1]) * z[:, i] * z[:, j]
    psi = psi * torch.exp(-1j * energy)
    for site in range(9):
        amplitudes = psi.reshape([2] * 9).movedim(site, 0).reshape(2, -1)
        exact = amplitudes @ amplitudes.conj().T
        assert torch.allclose(fast[site].to(torch.complex128), exact, atol=1e-5), f"site {site} differs"
    grid = torch.rand(2, 16, 16) * torch.pi
    states = graph_reduced_states(grid, gammas).reshape(2, 16, 16, 2, 2)
    turned = graph_reduced_states(grid.rot90(1, (1, 2)), gammas).reshape(2, 16, 16, 2, 2)
    assert torch.allclose(states.rot90(1, (1, 2)), turned, atol=1e-5), "the layer does not turn with the image"
    base, graph = _same_start(graph_layer=True)
    assert graph.count_parameters() == {"quantum": 4854, "head": 14294, "total": 19148}, graph.count_parameters()
    images = torch.rand(2, 1, 28, 28)
    base.eval(), graph.eval()
    with torch.no_grad():
        assert torch.allclose(base(images), graph(images), atol=1e-5), "the graph model starts differently"
        graph.graphs[0].gammas.fill_(0.3)
        assert not torch.allclose(base(images), graph(images), atol=1e-4), "the graph layer has no effect"
    labels = torch.randint(0, 10, (2,))
    graph.zero_grad()
    plain = graph(images)
    torch.nn.functional.cross_entropy(plain, labels).backward()
    expected = [parameter.grad.clone() for parameter in graph.parameters()]
    graph.zero_grad()
    graph.train()
    checked = graph(images)
    torch.nn.functional.cross_entropy(checked, labels).backward()
    assert torch.allclose(plain, checked, atol=1e-4), "checkpointed output differs"
    for parameter, reference in zip(graph.parameters(), expected):
        assert torch.allclose(parameter.grad, reference, atol=1e-4), "checkpointed gradient differs"
    print("OK  graph layer: matches a 9-qubit statevector, turns with the image, starts as level32")


def check_expert_mixture() -> None:
    """Parameter split, and the output must be the log of a probability mixture."""
    model = FusionQCNN(experts=4)
    parts = model.count_parameters()
    assert parts == {"quantum": 4824, "head": 9076, "total": 13900}, parts
    log_probabilities = model.fusion(torch.randn(5, 24))
    assert torch.allclose(log_probabilities.exp().sum(dim=1), torch.ones(5), atol=1e-5), "not a probability mixture"
    assert torch.allclose(log_probabilities.log_softmax(dim=1), log_probabilities, atol=1e-5), "cross entropy changes it"
    print("OK  expert mixture:", parts)


def check_pauli_against_dense() -> None:
    identity = torch.eye(2, dtype=torch.complex64)
    paulis = {"x": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
              "y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
              "z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)}
    states = torch.randn(2, 16, dtype=torch.complex64)
    states = states / states.norm(dim=1, keepdim=True)
    rho = states[:, :, None] * states[:, None, :].conj()
    fast = PauliFeatures(4)(rho)

    def operator(factors: dict) -> torch.Tensor:
        matrix = torch.eye(1, dtype=torch.complex64)
        for qubit in range(4):
            matrix = torch.kron(matrix, factors.get(qubit, identity))
        return matrix

    operators = [operator({qubit: paulis[axis]}) for axis in "xyz" for qubit in range(4)]
    for first in range(4):
        for second in range(first + 1, 4):
            operators += [operator({first: paulis[a], second: paulis[b]}) for a in "xyz" for b in "xyz"]
    assert fast.shape == (2, len(operators)), fast.shape
    for index, matrix in enumerate(operators):
        reference = torch.einsum("bij,ji->b", rho, matrix).real
        assert torch.allclose(fast[:, index], reference, atol=1e-4), f"feature {index} differs"
    assert BRANCH_FEATURES == 276, BRANCH_FEATURES
    print(f"OK  {len(operators)} fast Pauli features match dense operators")


def check_front() -> None:
    """qcnn-fusion-level32-deep12p: parameter split; it starts exactly as level32 with projection 10 (same values,
    same outputs, the same random numbers drawn afterwards); the encoder gets a gradient; the classical controls
    use the same encoder layers (other random starting values)."""
    model = FusionQCNN(side_level=32, projection=10, front="deep12p")
    parts = model.count_parameters()
    assert parts == {"quantum": 4824, "front": 1646, "head": 11948, "total": 18418}, parts
    images = torch.rand(3, 1, 28, 28)
    torch.manual_seed(0)
    base = FusionQCNN(side_level=32, projection=10)
    base_next = torch.rand(5)
    torch.manual_seed(0)
    model = FusionQCNN(side_level=32, projection=10, front="deep12p")
    model_next = torch.rand(5)
    assert torch.equal(base_next, model_next), "building the front end changed the random state"
    state = model.state_dict()
    for key, value in base.state_dict().items():
        assert torch.equal(value, state[key]), f"{key} differs from level32"
    with torch.no_grad():
        difference = (model.eval()(images) - base.eval()(images)).abs().max().item()
    assert difference < 1e-5, difference
    model.train()
    torch.nn.functional.cross_entropy(model(images), torch.tensor([1, 2, 3])).backward()
    norm = model.front.net[-1].weight.grad.norm().item()
    assert norm > 0, "the encoder gets no gradient"
    # The controls use the same encoder layers; their random starting values differ (drawn at another point).
    reference = model.front.state_dict()
    for name in ("enc-cnn-deep12p", "enc-linear-deep12p"):
        control = build(name).model
        for key, value in control.front.state_dict().items():
            assert value.shape == reference[key].shape, f"{name}: front {key} has another shape"
        assert not control.front.net[-1].weight.any(), f"{name}: the last front-end conv does not start at 0"
    print(f"OK  deep12p front end: {parts}, starts as level32 (max difference {difference:.1e}), "
          f"|grad| of its last conv {norm:.2e}; controls use the same encoder layers")


def _dense_window_features(stage: QuantumConvStage, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Reference for one pixel: theta, phi (9,) of its window -> (27,) X, Y, Z of every qubit, by dense matrices."""
    identity = torch.eye(2, dtype=torch.complex128)
    paulis = [torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
              torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
              torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)]

    def kron_all(matrices):
        out = torch.eye(1, dtype=torch.complex128)
        for matrix in matrices:
            out = torch.kron(out, matrix)
        return out

    state = torch.ones(1, dtype=torch.complex128)
    for q in range(9):
        t, p = float(theta[q]), float(phi[q])
        one = torch.tensor([math.cos(math.pi * t / 2), math.sin(math.pi * t / 2) * cmath.exp(1j * math.pi * p)],
                           dtype=torch.complex128)
        state = torch.kron(state, one)
    gates = rotation_matrices(stage.angles.detach().double())
    eye = torch.eye(512, dtype=torch.complex128)
    cz = eye.clone()
    for a, b in grid_edges():
        one_a = (eye - kron_all([paulis[2] if q == a else identity for q in range(9)])) / 2
        one_b = (eye - kron_all([paulis[2] if q == b else identity for q in range(9)])) / 2
        cz = cz @ (eye - 2 * one_a @ one_b)
    for layer in range(gates.shape[0]):
        state = cz @ (kron_all([gates[layer, q] for q in range(9)]) @ state)
    values = []
    for q in range(9):
        for pauli in paulis:
            operator = kron_all([pauli if r == q else identity for r in range(9)])
            values.append((state.conj() @ operator @ state).real)
    return torch.stack(values)


def check_qconv() -> None:
    """QuantumFront (mammoth_qconv_front): the fast 9-qubit window circuit matches dense matrices (inside the image
    and at the border, where outside pixels are |0>); parameter counts; frozen angles do not learn; the full model
    starts exactly as level32 projection 10; the encoder angles get a gradient once the readout is not 0;
    checkpointing does not change output or gradients."""
    torch.manual_seed(1)
    stage = QuantumConvStage(layers=2, frozen=False)
    height = width = 4
    theta = torch.rand(2, height * width) * 1.5 - 0.2
    phi = torch.rand(2, height * width) * 2 - 1
    fast = stage.features(theta, phi, height, width)
    assert fast.shape == (2, 16, 27), fast.shape
    padded_t = torch.nn.functional.pad(theta.reshape(2, height, width), (1, 1, 1, 1))
    padded_p = torch.nn.functional.pad(phi.reshape(2, height, width), (1, 1, 1, 1))
    worst = 0.0
    for b, pixel in ((0, 0), (0, 5), (1, 10), (1, 15)):
        row, column = divmod(pixel, width)
        window_t = padded_t[b, row:row + 3, column:column + 3].flatten()
        window_p = padded_p[b, row:row + 3, column:column + 3].flatten()
        reference = _dense_window_features(stage, window_t, window_p)
        worst = max(worst, (fast[b, pixel].double() - reference).abs().max().item())
    assert worst < 1e-4, f"window circuit differs from dense matrices by {worst}"
    counts = {mode: sum(p.numel() for p in QuantumFront(mode, 16, 16).parameters())
              for mode in ("qconv2", "qconv2x2", "qconv4x3", "qconv2x2f")}
    assert counts == {"qconv2": 110, "qconv2x2": 220, "qconv4x3": 492, "qconv2x2f": 112}, counts
    frozen = QuantumFront("qconv2x2f", 16, 16)
    assert not any(isinstance(s.angles, torch.nn.Parameter) for s in frozen.stages), "frozen angles are parameters"
    assert "stages.0.angles" in frozen.state_dict(), "frozen angles are not saved"
    # The full model starts as level32 with projection 10.
    images = torch.rand(3, 1, 28, 28)
    torch.manual_seed(0)
    base = FusionQCNN(side_level=32, projection=10)
    base_next = torch.rand(5)
    torch.manual_seed(0)
    model = FusionQCNN(side_level=32, projection=10, front="qconv2x2")
    model_next = torch.rand(5)
    assert torch.equal(base_next, model_next), "building the quantum front end changed the random state"
    state = model.state_dict()
    for key, value in base.state_dict().items():
        assert torch.equal(value, state[key]), f"{key} differs from level32"
    parts = model.count_parameters()
    assert parts == {"quantum": 4824, "front": 220, "head": 11948, "total": 16992}, parts
    with torch.no_grad():
        difference = (model.eval()(images) - base.eval()(images)).abs().max().item()
    assert difference < 1e-5, difference
    # The angles get a gradient once the readouts are not 0 (at the start they are 0, like the last conv of deep12p).
    with torch.no_grad():
        for s in model.front.stages:
            s.readout.weight.normal_(0, 0.1)
    model.train()
    torch.nn.functional.cross_entropy(model(images), torch.tensor([1, 2, 3])).backward()
    norms = [s.angles.grad.norm().item() for s in model.front.stages]
    assert all(n > 0 for n in norms), norms
    # Checkpointing (training) and the plain path (eval, with grad) give the same output and gradients.
    front = model.front
    values = torch.rand(2, 256)
    front.zero_grad()
    front.train()
    out_ckpt = front(values)
    (out_ckpt[0].sum() + out_ckpt[1].square().sum()).backward()
    grads_ckpt = [p.grad.clone() for p in front.parameters()]
    front.zero_grad()
    front.eval()
    out_plain = front(values)
    (out_plain[0].sum() + out_plain[1].square().sum()).backward()
    for a, b in zip(out_ckpt, out_plain):
        assert torch.allclose(a, b, atol=1e-6), "checkpointed quantum front output differs"
    for p, g in zip(front.parameters(), grads_ckpt):
        assert torch.allclose(p.grad, g, atol=1e-5), "checkpointed quantum front gradient differs"
    for name, encoder in (("enc-linear-qconv2x2", 220), ("enc-linear-qconv4x3", 492), ("enc-linear-cnn8p", 226),
                          ("enc-linear-deep12p", 1646)):
        control = build(name).model
        assert control.channels == 2, name
        assert sum(p.numel() for p in control.front.parameters()) == encoder, name
    print(f"OK  quantum encoder: window circuit matches dense (max {worst:.1e}); {counts}; full model {parts}, "
          f"starts as level32 (max difference {difference:.1e}); |grad| of angles "
          f"{', '.join(f'{n:.2e}' for n in norms)}; checkpointing gives the same values")


def check_outputs_and_gradients() -> None:
    images = torch.rand(2, 1, 28, 28)
    labels = torch.randint(0, 10, (2,))
    for name in EXPECTED_PARAMETERS:
        model = build(name)
        logits = model(images)
        assert logits.shape == (2, 10), f"{name}: {logits.shape}"
        torch.nn.functional.cross_entropy(logits, labels).backward()
        for key, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), f"{name}: {key}"
    print("OK  outputs and gradients, including the re-upload scales")


def check_fusion_checkpointing() -> None:
    """Training mode uses gradient checkpointing; it must not change the output or the gradients."""
    model = FusionQCNN()
    images = torch.rand(2, 1, 28, 28)
    labels = torch.randint(0, 10, (2,))
    model.eval()
    plain = model(images)
    torch.nn.functional.cross_entropy(plain, labels).backward()
    expected = [parameter.grad.clone() for parameter in model.parameters()]
    model.zero_grad()
    model.train()
    checked = model(images)
    torch.nn.functional.cross_entropy(checked, labels).backward()
    assert torch.allclose(plain, checked, atol=1e-4), "checkpointed output differs"
    for parameter, reference in zip(model.parameters(), expected):
        assert torch.allclose(parameter.grad, reference, atol=1e-4), "checkpointed gradient differs"
    print("OK  gradient checkpointing gives the same output and gradients")


NEW_BACKBONES = ("cnn-5k", "cnn-12k", "cnn-17k", "cnn-19k", "cnn-30k", "cnn-50k", "cnn-85k", "mlp-19k",
                 "qcnn-fusion-level32-qconv4x3", "qcnn-fusion-level32-deep6p")  # mammoth_full_package queue


def report(device: torch.device, batch_size: int, which: str = "new") -> None:
    """Information only: time and GPU memory per training batch."""
    print(f"INFO on {device}, batch {batch_size}, Adam (data loader and evaluation not included)")
    for name in (NEW_BACKBONES if which == "new" else EXPECTED_PARAMETERS):
        try:
            model = build(name).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            images = torch.rand(batch_size, 1, 28, 28, device=device)
            labels = torch.randint(0, 10, (batch_size,), device=device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for repeat in range(4):
                if repeat == 1:
                    started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(images), labels).backward()
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = (time.perf_counter() - started) / 3
            memory = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
            print(f"INFO {name:30s} {seconds:7.4f} s/batch, {memory:5.2f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"INFO {name:30s} out of GPU memory")
        finally:
            model = optimizer = None
            if device.type == "cuda":
                torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--report", choices=("new", "all"), default="new",
                        help="time only the backbones of mammoth_qconv_front and their references, or all")
    args = parser.parse_args()
    torch.manual_seed(0)
    check_registered_and_counts()
    check_scan_orders()
    check_expert_mixture()
    check_typed_readout()
    check_side_level()
    check_lr_scale()
    check_four_trees()
    check_graph_layer()
    check_position_phase()
    check_pauli_against_dense()
    check_front()
    check_qconv()
    check_outputs_and_gradients()
    check_fusion_checkpointing()
    print("All checks passed.")
    report(torch.device(args.device), args.batch_size, args.report)


if __name__ == "__main__":
    main()
