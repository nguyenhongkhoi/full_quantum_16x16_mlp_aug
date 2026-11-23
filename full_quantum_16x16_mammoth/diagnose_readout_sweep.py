"""Offline sweep of smaller readouts on fixed circuit features. Put next to main.py and diagnose_fusion_head.py.

Same checkpoint, images and angles as diagnose_fusion_head.py. Every readout is trained on the same features,
so the numbers compare readouts only. It does not measure forgetting.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST

from backbone import get_backbone
from backbone.quantum_core.fusion_model import BRANCH_FEATURES, FINAL_QUBITS, FusionQCNN
from datasets.utils.validation import get_validation_indexes
from diagnose_fusion_head import accuracy_per_angle, extract, train_head
from utils.conf import base_path

SINGLES = 3 * FINAL_QUBITS
PAIRS = FINAL_QUBITS * (FINAL_QUBITS - 1) // 2
QUANTUM_PARAMETERS = 4824


class PauliTypeProjection(nn.Module):
    """One small Linear for every qubit (its X, Y, Z) and one for every pair (its 9 correlators), shared by all."""

    def __init__(self, size: int):
        super().__init__()
        self.single = nn.Linear(3, size)
        self.pair = nn.Linear(9, size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        singles = features[:, :SINGLES].reshape(-1, 3, FINAL_QUBITS).transpose(1, 2)
        pairs = features[:, SINGLES:].reshape(-1, PAIRS, 9)
        return torch.cat((self.single(singles).flatten(1), self.pair(pairs).flatten(1)), dim=1)


class Readout(nn.Module):
    """The two parts head_logits needs: a projection per branch, then one module on their concatenation."""

    def __init__(self, projections, fusion: nn.Module):
        super().__init__()
        self.projections = nn.ModuleList(projections)
        self.fusion = fusion


def fusion_mlp(inputs: int, hidden: int = 16, classes: int = 10) -> nn.Module:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, classes))


def build(name: str, branches: int) -> Readout:
    """current: the qcnn-fusion head. linear: one Linear on all features. typedK: PauliTypeProjection(K) shared by
    the branches, then an MLP. typedKb: the same, with one PauliTypeProjection per branch."""
    if name == "current":
        model = FusionQCNN()
        return Readout(model.projections, model.fusion)
    if name == "linear":
        return Readout([nn.Identity() for _ in range(branches)], nn.Linear(branches * BRANCH_FEATURES, 10))
    size = int(name.removeprefix("typed").removesuffix("b"))
    if name.endswith("b"):
        projections = [PauliTypeProjection(size) for _ in range(branches)]
    else:
        projections = [PauliTypeProjection(size)] * branches
    return Readout(projections, fusion_mlp(branches * (FINAL_QUBITS + PAIRS) * size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--angles", type=int, required=True)
    parser.add_argument("--angle-seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True, help="The --seed of the run, for its validation split")
    parser.add_argument("--train-per-angle", type=int, required=True)
    parser.add_argument("--val-per-angle", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--head-batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--head-seeds", type=int, required=True)
    parser.add_argument("--readouts", nargs="+", required=True, help="current, linear, typed1, typed2, ...")
    args = parser.parse_args()

    trained = get_backbone(Namespace(backbone="qcnn-fusion", num_classes=10)).model
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {key[len("net.model."):]: value for key, value in saved["model"].items() if key.startswith("net.model.")}
    trained.load_state_dict(state)
    trained.to(args.device).eval()

    mnist = MNIST(base_path() + "MNIST", train=True, download=True)
    train_all, val_all = get_validation_indexes(10, SimpleNamespace(targets=mnist.targets.numpy()), args.split_seed)
    generator = np.random.default_rng(args.angle_seed)
    angles = generator.uniform(0, 180, args.angles).tolist()
    train_indexes = [generator.choice(train_all, args.train_per_angle, replace=False) for _ in angles]
    val_indexes = [generator.choice(val_all, args.val_per_angle, replace=False) for _ in angles]

    print("Features of the checkpoint circuit...", flush=True)
    train_sets = extract(trained, mnist.data, mnist.targets, train_indexes, angles, args.batch_size, args.device)
    val_sets = extract(trained, mnist.data, mnist.targets, val_indexes, angles, args.batch_size, args.device)
    print(f"A  checkpoint head, must match the earlier A: {np.mean(accuracy_per_angle(trained, val_sets)):.2f}", flush=True)

    rows = []
    for name in args.readouts:
        per_seed = []
        for seed in range(args.head_seeds):
            torch.manual_seed(seed)
            readout = build(name, len(trained.cores)).to(args.device)
            train_head(readout, train_sets, args)
            per_seed.append(np.mean(accuracy_per_angle(readout, val_sets)))
        count = sum(parameter.numel() for parameter in readout.parameters())
        share = 100 * QUANTUM_PARAMETERS / (QUANTUM_PARAMETERS + count)
        rows.append((name, count, share, np.mean(per_seed)))
        print(f"{name:8s} head {count:6d}  quantum share {share:5.1f}%  mean {np.mean(per_seed):6.2f}  seeds "
              + " ".join(f"{value:.2f}" for value in per_seed), flush=True)

    print("Readout    head  quantum%   accuracy")
    for name, count, share, accuracy in rows:
        print(f"{name:8s} {count:6d}  {share:7.1f}  {accuracy:9.2f}")


if __name__ == "__main__":
    main()
