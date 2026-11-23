"""Quick checks that need no MNIST and no training. Run this first; it should take well under a minute."""
from __future__ import annotations

import argparse
import math
import time

import torch

from models import EXPECTED_PARAMETERS, VARIANTS, build_model
from models.bond2_tree import Bond2Core, Bond2Merge, encode_images, spatial_permutation


def qubit_density(theta: float) -> torch.Tensor:
    state = torch.tensor([math.cos(theta / 2), math.sin(theta / 2)], dtype=torch.complex64)
    return torch.outer(state, state.conj())


def check_parameter_counts() -> None:
    for variant in VARIANTS:
        total = build_model(variant).count_parameters()["total"]
        assert total == EXPECTED_PARAMETERS[variant], f"{variant}: {total}"
    assert build_model("b2", shifts=(0, 2)).count_parameters()["total"] == EXPECTED_PARAMETERS["b2"]
    print("OK  parameter counts:", EXPECTED_PARAMETERS)


def check_site_orders() -> None:
    assert spatial_permutation(16, "morton")[:4].tolist() == [0, 1, 16, 17]
    assert spatial_permutation(16, "row").tolist() == list(range(256))
    print("OK  site orders")


def check_merge_keeps_a1_b0() -> None:
    merge = Bond2Merge(init_scale=0.0, balanced=False)  # All gates are the identity.
    left = torch.kron(qubit_density(0.3), qubit_density(0.7))
    right = torch.kron(qubit_density(1.1), qubit_density(1.9))
    rho = torch.stack((left, right))[None]
    kept = merge(rho)[0, 0]
    expected = torch.kron(qubit_density(0.7), qubit_density(1.1))
    assert torch.allclose(kept, expected, atol=1e-5), "merge keeps the wrong qubits"
    print("OK  identity merge keeps qubits a1 and b0")


def _apply_gate(state: torch.Tensor, gate: torch.Tensor, axes: tuple[int, int]) -> torch.Tensor:
    moved = state.movedim(axes, (0, 1))
    updated = (gate @ moved.reshape(4, 4)).reshape(2, 2, 2, 2)
    return updated.movedim((0, 1), axes)


def check_merge_against_statevector() -> None:
    merge = Bond2Merge(init_scale=0.5, balanced=True)
    left = torch.randn(4, dtype=torch.complex64)
    right = torch.randn(4, dtype=torch.complex64)
    left, right = left / left.norm(), right / right.norm()
    rho = torch.stack((torch.outer(left, left.conj()), torch.outer(right, right.conj())))[None]
    kept = merge(rho)[0, 0].detach()

    state = torch.outer(left, right).reshape(2, 2, 2, 2)
    with torch.no_grad():
        state = _apply_gate(state, merge.cross(), (1, 2))
        state = _apply_gate(state, merge.left(), (0, 1))
        state = _apply_gate(state, merge.right(), (2, 3))
    expected = torch.einsum("aijd,akld->ijkl", state, state.conj()).reshape(4, 4)
    assert torch.allclose(kept, expected, atol=1e-4), "merge differs from statevector simulation"
    print("OK  merge matches an independent statevector simulation")


def check_density_matrices() -> None:
    core = Bond2Core(init_scale=0.3, balanced=True)
    qubits = encode_images(torch.rand(4, 1, 28, 28))
    with torch.no_grad():
        tap, final = core(qubits)
    assert tap.shape == (4, 16, 4, 4) and final.shape == (4, 4, 4, 4), (tap.shape, final.shape)
    for rho in (tap, final):
        trace = rho.diagonal(dim1=-2, dim2=-1).sum(-1)
        assert torch.allclose(trace, torch.ones_like(trace), atol=1e-4), "trace is not 1"
        assert torch.allclose(rho, rho.conj().transpose(-1, -2), atol=1e-5), "not Hermitian"
        assert torch.linalg.eigvalsh(rho).min() > -1e-3, "not positive semidefinite"

    # The 2-qubit node must hold correlations, otherwise it is no better than two separate qubits.
    batch, nodes = tap.shape[:2]
    tensor = tap.reshape(batch, nodes, 2, 2, 2, 2)
    first = torch.einsum("bnijkj->bnik", tensor)
    second = torch.einsum("bnijil->bnjl", tensor)
    product = torch.einsum("bnij,bnkl->bnikjl", first, second).reshape(batch, nodes, 4, 4)
    correlation = (tap - product).abs().mean().item()
    assert correlation > 1e-3, f"node looks like a product state: {correlation}"
    print(f"OK  valid density matrices; mean |rho - rho1 x rho2| = {correlation:.4f}")


def check_gradients() -> None:
    labels = torch.randint(0, 10, (8,))
    for variant in VARIANTS:
        model = build_model(variant)
        torch.nn.functional.cross_entropy(model(torch.rand(8, 1, 28, 28)), labels).backward()
        zero = []
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{variant}: {name} has no gradient"
            assert torch.isfinite(parameter.grad).all(), f"{variant}: {name} has a bad gradient"
            if parameter.grad.abs().sum() == 0:
                zero.append(name)
        print(f"OK  gradients {variant}" + (f" (WARNING, all zero: {zero})" if zero else ""))


def report_pixel_sensitivity() -> None:
    """Information only: how many pixels affect the output at initialization."""
    images = torch.rand(8, 1, 16, 16)
    for init in ("identity", "balanced"):
        torch.manual_seed(0)
        model = build_model("b2", init=init)
        inputs = images.clone().requires_grad_(True)
        model(inputs).abs().sum().backward()
        sensitivity = inputs.grad.abs().mean(dim=0).flatten()
        visible = (sensitivity > 0.01 * sensitivity.max()).sum().item()
        print(f"INFO init={init}: {visible}/256 pixels above 1% of the strongest pixel")


def report_speed(device: torch.device) -> None:
    """Information only: seconds per batch, before the data loader is counted."""
    for variant in VARIANTS:
        model = build_model(variant).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        images = torch.rand(64, 1, 28, 28, device=device)
        labels = torch.randint(0, 10, (64,), device=device)
        for repeat in range(6):
            if repeat == 1:  # The first step includes lazy start-up work.
                started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(images), labels).backward()
            optimizer.step()
        seconds = (time.perf_counter() - started) / 5
        print(f"INFO {variant:7s} {seconds:.4f} s/batch, about {seconds * 782:.0f} s per train epoch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(0)
    check_parameter_counts()
    check_site_orders()
    check_merge_keeps_a1_b0()
    check_merge_against_statevector()
    check_density_matrices()
    check_gradients()
    report_pixel_sensitivity()
    report_speed(torch.device(args.device))
    print("All checks passed.")


if __name__ == "__main__":
    main()
