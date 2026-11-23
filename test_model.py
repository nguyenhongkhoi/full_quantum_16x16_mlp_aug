"""Checks that need no MNIST and no training. Run this before any training step."""
from __future__ import annotations

import argparse
import math
import time

import torch

from models import EXPECTED_PARAMETERS, PRESETS, PatchConfig, build_model
from models.patch_qcnn import (Bond2Merge, GroupedAmplitudes, PatchCircuit, lattice_bonds, merge_axes,
                               patch_site_indices, product_state)
from models.unitary import LearnedUnitary2Q


def check_parameter_counts() -> None:
    for preset, expected in EXPECTED_PARAMETERS.items():
        for readout, total in expected.items():
            counts = build_model(preset, readout=readout).count_parameters()
            assert counts["total"] == total, f"{preset}/{readout}: {counts}"
    frozen = build_model("s2_p12", freeze_core=True).count_parameters()
    assert frozen["trainable"] == frozen["head"], frozen
    print("OK  parameter counts:", {name: value["mlp"] for name, value in EXPECTED_PARAMETERS.items()})


def check_patches_cover_image() -> None:
    for preset, config in PRESETS.items():
        sites = patch_site_indices(config)
        assert sites.shape == (config.grid[0] * config.grid[1], config.patch[0] * config.patch[1]), preset
        assert sites.unique().numel() == 256, f"{preset} leaves pixels uncovered"
        groups = lattice_bonds(*config.patch)
        for bonds in groups.values():
            used = [qubit for bond in bonds for qubit in bond]
            assert len(used) == len(set(used)), f"{preset}: a bond group shares a qubit"
        assert len(merge_axes(config.grid)) == int(math.log2(config.grid[0] * config.grid[1] / 4)), preset
    print("OK  patches cover all 256 pixels; bonds inside a group are disjoint")


def _dense_gate(gate: torch.Tensor, n_qubits: int, first: int, second: int) -> torch.Tensor:
    """Independent reference: build the full 2**n matrix of a two-qubit gate from basis states."""
    dim = 2 ** n_qubits
    dense = torch.zeros(dim, dim, dtype=torch.complex64)
    bit_first, bit_second = n_qubits - 1 - first, n_qubits - 1 - second
    for column in range(dim):
        in_first, in_second = (column >> bit_first) & 1, (column >> bit_second) & 1
        for out_first in range(2):
            for out_second in range(2):
                row = column & ~(1 << bit_first) & ~(1 << bit_second)
                row |= (out_first << bit_first) | (out_second << bit_second)
                dense[row, column] = gate[2 * out_first + out_second, 2 * in_first + in_second]
    return dense


def check_gate_against_dense_matrix() -> None:
    gate = LearnedUnitary2Q(init_scale=0.7)().detach()
    states = torch.randn(3, 2 ** 5, dtype=torch.complex64)
    states = states / states.norm(dim=1, keepdim=True)
    for first, second in ((0, 1), (1, 2), (3, 4), (0, 4), (4, 1)):
        amplitudes = GroupedAmplitudes(5, (first, second))
        fast = amplitudes.merge(amplitudes.split(states) @ gate.T)
        reference = states @ _dense_gate(gate, 5, first, second).T
        assert torch.allclose(fast, reference, atol=1e-5), (first, second)
    print("OK  gate application matches a dense matrix built separately")


def _single_qubit_ry(theta: float, qubit: int, n_qubits: int) -> torch.Tensor:
    cos, sin = math.cos(theta / 2), math.sin(theta / 2)
    gate = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.complex64)
    matrix = torch.eye(1, dtype=torch.complex64)
    for index in range(n_qubits):
        matrix = torch.kron(matrix, gate if index == qubit else torch.eye(2, dtype=torch.complex64))
    return matrix


def check_encoding_and_reupload() -> None:
    values = torch.tensor([[0.2, 0.7, 0.4, 0.9]])
    circuit = PatchCircuit(PatchConfig(patch=(2, 2), grid=(8, 8), layers=2), init_scale=0.0, balanced=False)
    state = product_state(values)
    reference = torch.zeros(16, dtype=torch.complex64)
    reference[0] = 1.0
    for qubit in range(4):
        reference = _single_qubit_ry(float(values[0, qubit]) * math.pi, qubit, 4) @ reference
    assert torch.allclose(state[0], reference, atol=1e-5), "encoding differs from RY gates on |0>"

    uploaded = circuit._reupload(state, values)[0]
    for qubit in range(4):
        reference = _single_qubit_ry(float(values[0, qubit]) * math.pi, qubit, 4) @ reference
    assert torch.allclose(uploaded, reference, atol=1e-5), "re-upload differs from RY gates"
    assert abs(uploaded.norm().item() - 1.0) < 1e-5, "re-upload is not unitary"
    print("OK  encoding and re-upload match explicit RY gates")


def check_kept_density_matrix() -> None:
    # Identity gates: the kept pair (centre of row 1, qubits 5 and 6 of a 3x4 patch) must be a product of two RY states.
    circuit = PatchCircuit(PatchConfig(patch=(3, 4), grid=(8, 4)), init_scale=0.0, balanced=False)
    assert circuit.kept == (5, 6), circuit.kept
    values = torch.rand(1, 12)
    rho = circuit(values)[0].detach()
    pure = lambda value: torch.tensor([math.cos(value * math.pi / 2), math.sin(value * math.pi / 2)],  # noqa: E731
                                      dtype=torch.complex64)
    expected = torch.outer(torch.kron(pure(float(values[0, 5])), pure(float(values[0, 6]))),
                           torch.kron(pure(float(values[0, 5])), pure(float(values[0, 6]))).conj())
    assert torch.allclose(rho, expected, atol=1e-5), "wrong qubits were kept"

    # Random gates: compare the partial trace with an explicit sum over all traced-out qubits.
    circuit = PatchCircuit(PatchConfig(patch=(2, 2), grid=(8, 8)), init_scale=0.5, balanced=True)
    values = torch.rand(2, 4)
    with torch.no_grad():
        rho = circuit(values)
        states = circuit._layer(product_state(values), values, 0)
    first, second = circuit.kept
    tensor = states[0].reshape(2, 2, 2, 2)
    reference = torch.zeros(4, 4, dtype=torch.complex64)
    for index in range(16):
        bits = [(index >> (3 - qubit)) & 1 for qubit in range(4)]
        for other_first in range(2):
            for other_second in range(2):
                other = list(bits)
                other[first], other[second] = other_first, other_second
                row = 2 * bits[first] + bits[second]
                column = 2 * other_first + other_second
                reference[row, column] += tensor[tuple(bits)] * tensor[tuple(other)].conj()
    assert torch.allclose(rho[0], reference, atol=1e-5), "partial trace differs from an explicit sum"
    print("OK  kept qubits and partial trace are correct")


def check_entanglement_inside_patch() -> None:
    """The kept pair must be correlated, which the earlier trees could not provide at leaf level."""
    circuit = PatchCircuit(PatchConfig(patch=(3, 4), grid=(8, 4)), init_scale=0.1, balanced=True)
    with torch.no_grad():
        rho = circuit(torch.rand(32, 12))
    trace = rho.diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(trace, torch.ones_like(trace), atol=1e-4), "trace is not 1"
    assert torch.allclose(rho, rho.conj().transpose(-1, -2), atol=1e-5), "not Hermitian"
    assert torch.linalg.eigvalsh(rho).min() > -1e-3, "not positive semidefinite"
    tensor = rho.reshape(-1, 2, 2, 2, 2)
    first = torch.einsum("nijkj->nik", tensor)
    second = torch.einsum("nijil->njl", tensor)
    product_rho = torch.einsum("nij,nkl->nikjl", first, second).reshape(-1, 4, 4)
    correlation = (rho - product_rho).abs().mean().item()
    purity = torch.einsum("nij,nji->n", rho, rho).real.mean().item()
    assert correlation > 1e-3, f"kept pair looks like a product state: {correlation}"
    print(f"OK  valid density matrices; correlation {correlation:.4f}, mean purity {purity:.3f} "
          "(0.25 means maximally mixed)")


def check_merge_against_statevector() -> None:
    merge = Bond2Merge(init_scale=0.5, balanced=True)
    left = torch.randn(4, dtype=torch.complex64)
    right = torch.randn(4, dtype=torch.complex64)
    left, right = left / left.norm(), right / right.norm()
    rho = torch.stack((torch.outer(left, left.conj()), torch.outer(right, right.conj())))[None]
    kept = merge(rho)[0, 0].detach()
    state = torch.outer(left, right).flatten()
    with torch.no_grad():
        for gate, bond in ((merge.cross(), (1, 2)), (merge.left(), (0, 1)), (merge.right(), (2, 3))):
            state = _dense_gate(gate, 4, *bond) @ state
    tensor = state.reshape(2, 2, 2, 2)
    expected = torch.einsum("aijd,akld->ijkl", tensor, tensor.conj()).reshape(4, 4)
    assert torch.allclose(kept, expected, atol=1e-4), "merge differs from statevector simulation"
    print("OK  merge matches an independent statevector simulation")


def check_pauli_measurements() -> None:
    """Compare the fast expectation values with <psi|O|psi> using dense operators."""
    circuit = PatchCircuit(PatchConfig(patch=(2, 2), grid=(8, 8)), init_scale=0.5, balanced=True)
    with torch.no_grad():
        states = circuit.evolve(torch.rand(3, 4))
        fast = circuit.measure_paulis(states)
    identity = torch.eye(2, dtype=torch.complex64)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

    def operator(factors: dict) -> torch.Tensor:
        matrix = torch.eye(1, dtype=torch.complex64)
        for qubit in range(4):
            matrix = torch.kron(matrix, factors.get(qubit, identity))
        return matrix

    operators = [operator({qubit: pauli}) for pauli in (x, y, z) for qubit in range(4)]
    operators += [operator({first: z, second: z})
                  for bonds in circuit.bond_groups.values() for first, second in bonds]
    assert fast.shape == (3, len(operators)), fast.shape
    for index, matrix in enumerate(operators):
        reference = torch.einsum("ni,ij,nj->n", states.conj(), matrix, states).real
        assert torch.allclose(fast[:, index], reference, atol=1e-4), f"feature {index} differs"
    print(f"OK  {len(operators)} Pauli features match dense operators")


def check_gradients() -> None:
    labels = torch.randint(0, 10, (4,))
    for preset in ("s1_p8", "s3_ru2", "s9_meas", "s10_meas_ru2"):
        model = build_model(preset)
        torch.nn.functional.cross_entropy(model(torch.rand(4, 1, 28, 28)), labels).backward()
        zero = []
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{preset}: {name} has no gradient"
            assert torch.isfinite(parameter.grad).all(), f"{preset}: {name} has a bad gradient"
            if parameter.grad.abs().sum() == 0:
                zero.append(name)
        print(f"OK  gradients {preset}" + (f" (WARNING, all zero: {zero})" if zero else ""))


def check_grad_checkpoint_matches() -> None:
    config = PatchConfig(patch=(2, 2), grid=(8, 8), layers=2)
    circuit = PatchCircuit(config, init_scale=0.3, balanced=True)
    values = torch.rand(4, 4)
    plain = circuit(values)
    plain.abs().sum().backward()
    expected = [parameter.grad.clone() for parameter in circuit.parameters()]
    circuit.zero_grad()
    circuit.grad_checkpoint = True
    checked = circuit(values)
    checked.abs().sum().backward()
    assert torch.allclose(plain, checked, atol=1e-5), "checkpointed forward differs"
    for parameter, reference in zip(circuit.parameters(), expected):
        assert torch.allclose(parameter.grad, reference, atol=1e-4), "checkpointed gradient differs"
    print("OK  gradient checkpointing gives the same result")


def report_presets(device: torch.device, presets: list[str]) -> None:
    """Information only: time, GPU memory and how many pixels matter at initialization."""
    print(f"INFO presets on {device}, batch 64 (data loader and validation not included)")
    for preset in presets:
        try:
            torch.manual_seed(0)
            model = build_model(preset).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            images = torch.rand(64, 1, 28, 28, device=device)
            labels = torch.randint(0, 10, (64,), device=device)
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

            model.eval()
            inputs = torch.rand(8, 1, 16, 16, device=device, requires_grad=True)
            model(inputs).abs().sum().backward()
            sensitivity = inputs.grad.abs().mean(dim=0).flatten()
            visible = (sensitivity > 0.01 * sensitivity.max()).sum().item()
            print(f"INFO {preset:11s} {seconds:7.3f} s/batch, about {seconds * 782 / 60:6.1f} min/train epoch, "
                  f"{memory:5.1f} GB, {visible}/256 pixels above 1% of the strongest")
        except torch.cuda.OutOfMemoryError:
            print(f"INFO {preset:11s} out of GPU memory")
        finally:
            model = optimizer = None
            if device.type == "cuda":
                torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--presets", nargs="+", choices=sorted(PRESETS), default=sorted(PRESETS, key=list(PRESETS).index),
                        help="Presets to time; all by default, and the largest ones take minutes")
    args = parser.parse_args()
    torch.manual_seed(0)
    check_parameter_counts()
    check_patches_cover_image()
    check_gate_against_dense_matrix()
    check_encoding_and_reupload()
    check_kept_density_matrix()
    check_entanglement_inside_patch()
    check_merge_against_statevector()
    check_pauli_measurements()
    check_gradients()
    check_grad_checkpoint_matches()
    print("All checks passed.")
    report_presets(torch.device(args.device), args.presets)


if __name__ == "__main__":
    main()
