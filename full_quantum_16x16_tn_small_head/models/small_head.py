"""Heads that share weights across the structured Pauli features of one tree branch."""
from __future__ import annotations

import torch
from torch import nn

from data.encoding import angle_ry_encode
from data.preprocessing import preprocess_full_quantum

FINAL_QUBITS = 8
SIDE_NODES = 16
PAIRS = FINAL_QUBITS * (FINAL_QUBITS - 1) // 2
SINGLE_FEATURES = 3 * FINAL_QUBITS
PAIR_FEATURES = 9 * PAIRS
SIDE_FEATURES = 3 * SIDE_NODES
BRANCH_FEATURES = SINGLE_FEATURES + PAIR_FEATURES + SIDE_FEATURES

# name -> (token size k, hidden width); k None is the R0 head.
HEADS = {
    "current": (None, 16),
    "k1_h8": (1, 8),
    "k1_h16": (1, 16),
    "k2_h16": (2, 16),
    "k4_h16": (4, 16),
}
EXPECTED_HEAD_PARAMETERS = {"current": 5402, "k1_h8": 548, "k1_h16": 1068, "k2_h16": 1918, "k4_h16": 3618}


def split_branch_features(features: torch.Tensor):
    """(batch, 324) -> qubits (batch, 8, 3), pairs (batch, 28, 9), side nodes (batch, 16, 3).

    PauliFeatures orders all X, all Y, all Z, then 9 correlators per pair; side taps are X, Y, Z per node.
    """
    batch = features.shape[0]
    singles = features[:, :SINGLE_FEATURES].reshape(batch, 3, FINAL_QUBITS).transpose(1, 2)
    pairs = features[:, SINGLE_FEATURES:SINGLE_FEATURES + PAIR_FEATURES].reshape(batch, PAIRS, 9)
    side = features[:, SINGLE_FEATURES + PAIR_FEATURES:].reshape(batch, SIDE_NODES, 3)
    return singles, pairs, side


class CurrentHead(nn.Module):
    """The R0 head of one branch."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(BRANCH_FEATURES, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, 10))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class SharedTokenHead(nn.Module):
    """One Linear shared by all pairs, one by all qubits, one by all side nodes; then a small MLP."""

    def __init__(self, tokens: int, hidden: int):
        super().__init__()
        self.qubit = nn.Linear(3, tokens)
        self.pair = nn.Linear(9, tokens)
        self.side = nn.Linear(3, tokens)
        self.mlp = nn.Sequential(nn.Linear((FINAL_QUBITS + PAIRS + SIDE_NODES) * tokens, hidden),
                                 nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, 10))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        singles, pairs, side = split_branch_features(features)
        tokens = torch.cat((self.qubit(singles), self.pair(pairs), self.side(side)), dim=1)
        return self.mlp(tokens.flatten(start_dim=1))


def build_head(name: str) -> nn.Module:
    tokens, hidden = HEADS[name]
    return CurrentHead(hidden) if tokens is None else SharedTokenHead(tokens, hidden)


def branch_features(model, images: torch.Tensor) -> torch.Tensor:
    """The features each R0 head reads: (batch, branches, 324)."""
    values = preprocess_full_quantum(images, model.target_size, grayscale=model.grayscale,
                                     mode=model.preprocess_mode)
    qubits = angle_ry_encode(values).to(images.device)
    features = []
    for index, (tn_core, readout) in enumerate(model._all_branches()):
        rho, side = tn_core.forward_with_multiscale(qubits.index_select(1, model.site_permutations[index]))
        features.append(torch.cat((readout.features(rho), side), dim=-1))
    return torch.stack(features, dim=1)
