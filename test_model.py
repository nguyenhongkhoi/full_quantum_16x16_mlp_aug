"""Checks that need no MNIST and no training. Run this before any training step."""
from __future__ import annotations

import argparse
import time
from itertools import combinations

import torch

from models import FullQuantumQCNN
from models.layers.readout import PauliFeatures, _joint_pauli_expectation
from models.reupload import add_reupload

ARCH = dict(
    target_size=(16, 16), grayscale=True, preprocess_mode="resize", final_qubits=8,
    readout_type="pauli_mlp", pair_correlators="full9", init_scale=0.01,
    merge_rounds=2, entangling_type="strongly_entangling", n_layers=4,
    readout_hidden_dim=16, readout_dropout=0.0,
)
VARIANTS = {
    "r0": (dict(spatial_orders=("row", "morton")), 12420),
    "balanced": (dict(spatial_orders=("row", "morton"), balanced_conv=True), 12420),
    "fusion2": (dict(spatial_orders=("row", "morton"), fusion=True), 12202),
    "fusion3": (dict(spatial_orders=("row", "column", "morton"), fusion=True), 18194),
}


def build(name: str) -> FullQuantumQCNN:
    options, _ = VARIANTS[name]
    model = FullQuantumQCNN(in_channels=1, num_classes=10, quantum_channels=len(options["spatial_orders"]),
                            **ARCH, **options)
    return add_reupload(model, input_sites=256, final_qubits=8, init_scale=0.01, merge_rounds=2,
                        entangling_type="strongly_entangling", n_layers=4,
                        balanced_conv=options.get("balanced_conv", False))


def check_parameter_counts() -> None:
    for name, (_, expected) in VARIANTS.items():
        total = sum(parameter.numel() for parameter in build(name).parameters())
        assert total == expected, f"{name}: {total}"
    print("OK  parameter counts:", {name: value[1] for name, value in VARIANTS.items()})


def check_r0_checkpoint_names() -> None:
    """Identity-init r0 must keep the state dict names of the run it was copied from."""
    names = set(build("r0").state_dict())
    for expected in ("tn_core.reupload_scale", "tn_core.blocks.0.conv.diagonal", "readout.linear.0.weight",
                     "extra_branches.0.tn_core.blocks.0.pool.diagonal", "extra_branches.0.readout.linear.4.bias"):
        assert expected in names, expected
    assert not any("pauli_features" in name for name in names), "Pauli buffers leaked into the state dict"
    print("OK  r0 state dict names are unchanged")


def check_pauli_features_against_dense() -> None:
    n_qubits = 8
    matrix = torch.randn(3, 2**n_qubits, 2**n_qubits, dtype=torch.complex64)
    rho = matrix @ matrix.conj().transpose(-1, -2)
    rho = rho / rho.diagonal(dim1=-2, dim2=-1).sum(-1).real[:, None, None]
    fast = PauliFeatures(n_qubits, "full9").features(rho)
    terms = [{qubit: axis} for axis in "xyz" for qubit in range(n_qubits)]
    for first, second in combinations(range(n_qubits), 2):
        terms.extend({first: a, second: b} for a in "xyz" for b in "xyz")
    assert fast.shape == (3, 276), fast.shape
    for index, term in enumerate(terms):
        reference = _joint_pauli_expectation(rho, term, n_qubits)
        assert torch.allclose(fast[:, index], reference, atol=1e-4), f"feature {index}, term {term}"
    print("OK  276 fast Pauli features match the original dense operators")


def check_balanced_conv() -> None:
    plain = build("r0").tn_core.blocks[0]
    balanced = build("balanced").tn_core.blocks[0]
    assert abs(plain.conv.raw_real_upper[3].item()) < 0.1, "r0 must keep its identity-like conv gate"
    assert abs(balanced.conv.raw_real_upper[3].item() - 0.7854) < 0.1, "conv gate is not near sqrt(iSWAP)"
    assert abs(balanced.pool.raw_real_upper[3].item()) < 0.1, "pool must stay near the identity"
    unitary = balanced.conv()
    identity = torch.eye(4, dtype=unitary.dtype)
    assert torch.allclose(unitary @ unitary.conj().T, identity, atol=1e-5), "gate is not unitary"
    print("OK  balanced init changes only the conv gates")


def check_gradients() -> None:
    labels = torch.randint(0, 10, (4,))
    for name in VARIANTS:
        model = build(name)
        torch.nn.functional.cross_entropy(model(torch.rand(4, 1, 28, 28)), labels).backward()
        zero = []
        for parameter_name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{name}: {parameter_name} has no gradient"
            assert torch.isfinite(parameter.grad).all(), f"{name}: {parameter_name} has a bad gradient"
            if parameter.grad.abs().sum() == 0:
                zero.append(parameter_name)
        print(f"OK  gradients {name}" + (f" (all zero: {len(zero)} tensors, normal for reupload at scale 0)"
                                         if zero else ""))


def report(device: torch.device) -> None:
    """Information only: time per batch, memory, and how many pixels matter at initialization."""
    print(f"INFO variants on {device}, batch 64 (data loader and validation not included)")
    for name in VARIANTS:
        torch.manual_seed(0)
        model = build(name).to(device)
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
        memory = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0

        inputs = torch.rand(8, 1, 16, 16, device=device, requires_grad=True)
        model.zero_grad(set_to_none=True)
        model(inputs).abs().sum().backward()
        sensitivity = inputs.grad.abs().mean(dim=0).flatten()
        visible = (sensitivity > 0.01 * sensitivity.max()).sum().item()
        print(f"INFO {name:9s} {seconds:7.3f} s/batch, about {seconds * 782 / 60:5.1f} min/train epoch, "
              f"{memory:5.1f} GB, {visible}/256 pixels above 1% of the strongest")
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(0)
    check_parameter_counts()
    check_r0_checkpoint_names()
    check_pauli_features_against_dense()
    check_balanced_conv()
    check_gradients()
    print("All checks passed.")
    report(torch.device(args.device))


if __name__ == "__main__":
    main()
