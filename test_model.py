"""Checks that need no MNIST and no training. Run this before any training step."""
from __future__ import annotations

import argparse
import time

import torch

from models import EXPECTED_PARAMETERS, PRESETS, build_model
from models.patch_audit import AuditConfig, EncodingFeatures, PatchCircuit, ProductCircuit, patch_site_indices, product_state


def preset_init(preset: str) -> str:
    return "random" if preset == "c2_product" else "balanced"


def check_parameter_counts() -> None:
    for preset, total in EXPECTED_PARAMETERS.items():
        counts = build_model(preset, init=preset_init(preset)).count_parameters()
        assert counts["total"] == total, f"{preset}: {counts}"
    frozen = build_model("s9_meas", init="random", freeze_core=True).count_parameters()
    assert frozen["trainable"] == 2482, frozen
    print("OK  parameter counts:", EXPECTED_PARAMETERS)


def check_patches_cover_image() -> None:
    sites = patch_site_indices(AuditConfig(core="quantum"))
    assert sites.shape == (32, 12), sites.shape
    assert sites.unique().numel() == 256, "some pixels are uncovered"
    print("OK  32 patches of 12 pixels cover all 256 pixels")


def check_pauli_measurements() -> None:
    """Compare the fast expectation values with <psi|O|psi> using dense operators."""
    config = AuditConfig(core="quantum", patch=(2, 2))
    circuit = PatchCircuit(config, init_scale=0.5, balanced=True)
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


def check_product_matches_statevector() -> None:
    """c2_product must equal the full 12-qubit simulation with each gate replaced by first ⊗ second."""
    config = AuditConfig(core="product")
    circuit = PatchCircuit(config, init_scale=0.1, balanced=False)
    product = ProductCircuit(config)
    values = torch.rand(5, 12)
    with torch.no_grad():
        states = product_state(values)
        for name, amplitude_list in circuit.bond_amplitudes.items():
            gate = torch.kron(product.gates[name]["first"](), product.gates[name]["second"]())
            for amplitudes in amplitude_list:
                states = amplitudes.merge(amplitudes.split(states) @ gate.T)
        reference = circuit.measure_paulis(states)
        fast = product(values)
    assert fast.shape == reference.shape == (5, 53), (fast.shape, reference.shape)
    assert torch.allclose(fast, reference, atol=1e-4), "c2_product differs from the statevector"
    print("OK  c2_product matches the 12-qubit statevector with product gates")


def check_encoding_matches_circuit_without_gates() -> None:
    """c1_encoding must equal the Pauli features of the encoded state, leaving out Y, which is 0."""
    config = AuditConfig(core="encoding")
    circuit = PatchCircuit(config, init_scale=0.1, balanced=False)
    values = torch.rand(5, 12)
    with torch.no_grad():
        reference = circuit.measure_paulis(product_state(values))
        fast = EncodingFeatures(config)(values)
    assert reference[:, 12:24].abs().max() < 1e-5, "Y is not 0 on the encoded state"
    kept = torch.cat((reference[:, :12], reference[:, 24:]), dim=-1)
    assert fast.shape == kept.shape == (5, 41), (fast.shape, kept.shape)
    assert torch.allclose(fast, kept, atol=1e-4), "c1_encoding differs from the encoded state"
    print("OK  c1_encoding matches the 12-qubit circuit with no gates")


def _connected_zz(features: torch.Tensor, circuit) -> float:
    """Mean |<ZZ> - <Z><Z>| over all bonds; zero for any product state."""
    z = features[:, 24:36]
    bonds = [bond for group in circuit.bond_groups.values() for bond in group]
    products = torch.stack([z[:, first] * z[:, second] for first, second in bonds], dim=-1)
    return (features[:, 36:] - products).abs().mean().item()


def check_entanglement_is_only_in_s9() -> None:
    config = AuditConfig(core="quantum")
    values = torch.rand(16, 12)
    with torch.no_grad():
        balanced = PatchCircuit(config, init_scale=0.1, balanced=True)
        random = PatchCircuit(config, init_scale=1.0, balanced=False)
        product = ProductCircuit(config)
        connected = {"balanced": _connected_zz(balanced(values), balanced),
                     "random": _connected_zz(random(values), random),
                     "product": _connected_zz(product(values), product)}
    assert connected["balanced"] > 1e-3 and connected["random"] > 1e-3, connected
    assert connected["product"] < 1e-5, connected
    print(f"OK  connected ZZ: {connected}; only the two-qubit gates correlate qubits")


def check_gradients() -> None:
    labels = torch.randint(0, 10, (4,))
    for preset in PRESETS:
        model = build_model(preset, init=preset_init(preset))
        torch.nn.functional.cross_entropy(model(torch.rand(4, 1, 28, 28)), labels).backward()
        zero = []
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{preset}: {name} has no gradient"
            assert torch.isfinite(parameter.grad).all(), f"{preset}: {name} has a bad gradient"
            if parameter.grad.abs().sum() == 0:
                zero.append(name)
        print(f"OK  gradients {preset}" + (f" (WARNING, all zero: {zero})" if zero else ""))


def report_presets(device: torch.device) -> None:
    """Information only: time and GPU memory per batch."""
    print(f"INFO presets on {device}, batch 64 (data loader and validation not included)")
    for preset in PRESETS:
        try:
            torch.manual_seed(0)
            model = build_model(preset, init=preset_init(preset)).to(device)
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
            print(f"INFO {preset:13s} {seconds:7.4f} s/batch, about {seconds * 782 / 60:6.2f} min/train epoch, "
                  f"{memory:5.2f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"INFO {preset:13s} out of GPU memory")
        finally:
            model = optimizer = None
            if device.type == "cuda":
                torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(0)
    check_parameter_counts()
    check_patches_cover_image()
    check_pauli_measurements()
    check_product_matches_statevector()
    check_encoding_matches_circuit_without_gates()
    check_entanglement_is_only_in_s9()
    check_gradients()
    print("All checks passed.")
    report_presets(torch.device(args.device))


if __name__ == "__main__":
    main()
