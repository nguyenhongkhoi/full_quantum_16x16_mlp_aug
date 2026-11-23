"""Checks that need no MNIST, no checkpoint and no training. Run this before sweep_head.py."""
from __future__ import annotations

import torch

from models import FullQuantumQCNN
from models.layers.readout import PauliFeatures
from models.layers.tn_core import _bloch_vector_features
from models.reupload import add_reupload
from models.small_head import (BRANCH_FEATURES, EXPECTED_HEAD_PARAMETERS, HEADS, branch_features, build_head,
                               split_branch_features)

R0_ARCHITECTURE = dict(
    target_size=(16, 16), grayscale=True, preprocess_mode="resize", final_qubits=8,
    readout_type="pauli_mlp", pair_correlators="full9", init_scale=0.01,
    merge_rounds=2, entangling_type="strongly_entangling", n_layers=4,
    quantum_channels=2, spatial_orders=("row", "morton"),
    readout_hidden_dim=16, readout_dropout=0.0,
)


def r0_model() -> FullQuantumQCNN:
    model = FullQuantumQCNN(in_channels=1, num_classes=10, **R0_ARCHITECTURE)
    add_reupload(model, input_sites=256, final_qubits=8, init_scale=0.01, merge_rounds=2,
                 entangling_type="strongly_entangling", n_layers=4)
    return model.eval()


def check_head_parameters() -> None:
    for name, expected in EXPECTED_HEAD_PARAMETERS.items():
        count = sum(parameter.numel() for parameter in build_head(name).parameters())
        assert count == expected, f"{name}: {count}"
    model = r0_model()
    quantum = sum(tn_core.count_parameters() for tn_core, _ in model._all_branches())
    assert quantum == 1616, quantum
    assert sum(parameter.numel() for parameter in model.parameters()) == 12420
    print("OK  totals with 2 branches:", {name: 1616 + 2 * count for name, count in EXPECTED_HEAD_PARAMETERS.items()})


def check_feature_layout() -> None:
    """On a product state, every pair correlator must equal the product of the two single-qubit values."""
    states = torch.randn(3, 8, 2, dtype=torch.complex64)
    states = states / states.norm(dim=-1, keepdim=True)
    joint = states[:, 0]
    for qubit in range(1, 8):
        joint = (joint[:, :, None] * states[:, qubit, None, :]).flatten(start_dim=1)
    rho = joint[:, :, None] * joint[:, None, :].conj()
    pauli = PauliFeatures(8, "full9").features(rho)
    assert pauli.shape == (3, 276), pauli.shape

    site_rho = states[:, :, :, None] * states[:, :, None, :].conj()
    bloch = _bloch_vector_features(site_rho).reshape(3, 8, 3)
    side_input = _bloch_vector_features(torch.cat((site_rho, site_rho), dim=1))
    singles, pairs, side = split_branch_features(torch.cat((pauli, side_input), dim=-1))
    assert torch.allclose(singles, bloch, atol=1e-4), "the X, Y, Z of one qubit are not grouped together"
    assert torch.allclose(side, torch.cat((bloch, bloch), dim=1), atol=1e-4), "side nodes are not grouped"
    index = 0
    for first in range(8):
        for second in range(first + 1, 8):
            expected = bloch[:, first, :, None] * bloch[:, second, None, :]
            assert torch.allclose(pairs[:, index], expected.flatten(start_dim=1), atol=1e-4), (first, second)
            index += 1
    print("OK  qubits, pairs and side nodes are split correctly")


def check_features_reproduce_model() -> None:
    model = r0_model()
    images = torch.rand(2, 1, 28, 28)
    with torch.no_grad():
        logits = model(images)
        features = branch_features(model, images)
        rebuilt = sum(readout.linear(features[:, index])
                      for index, (_, readout) in enumerate(model._all_branches())) / 2
    assert features.shape == (2, 2, BRANCH_FEATURES), features.shape
    assert torch.allclose(logits, rebuilt, atol=1e-4), "cached features would not reproduce the model"
    print("OK  features with the R0 heads reproduce the model logits")


def check_gradients() -> None:
    features = torch.rand(4, BRANCH_FEATURES)
    labels = torch.randint(0, 10, (4,))
    for name in HEADS:
        head = build_head(name)
        torch.nn.functional.cross_entropy(head(features), labels).backward()
        for key, parameter in head.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), f"{name}: {key}"
    print("OK  gradients reach every head parameter")


def main() -> None:
    torch.manual_seed(0)
    check_head_parameters()
    check_feature_layout()
    check_features_reproduce_model()
    check_gradients()
    print("All checks passed.")


if __name__ == "__main__":
    main()
