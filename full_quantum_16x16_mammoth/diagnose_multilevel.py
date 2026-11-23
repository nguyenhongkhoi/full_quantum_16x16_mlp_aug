"""Which measurements of the circuit hold the information? Put next to main.py and diagnose_fusion_head.py.

Same checkpoint, images and angles as diagnose_fusion_head.py. The circuit stays frozen; only a new head is trained,
with the same layout as the qcnn-fusion head, on different sets of measurements:

    pre24    X, Y, Z of the 8 qubits that enter the 8-qubit core (after re-upload): what the tree passes on
    final276 the 276 Pauli values that qcnn-fusion reads now (equals B of diagnose_fusion_head.py)
    level16  X, Y, Z of the 16 qubits after tree level 4 (48 values)
    level32  X, Y, Z of the 32 qubits after tree level 3 (96 values)

A readout is several blocks joined by "+", for example final276+level16. A suffix @K projects each branch to K values
instead of 16, to keep the head size: final276+level16+level32@11 has about the qcnn-fusion head size. Inside the tree every qubit is independent
of the others, so X, Y, Z of a level describe that level exactly. It does not measure forgetting.
"""
from __future__ import annotations

import argparse
import time
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST

from backbone import get_backbone
from backbone.quantum_core.encoding import angle_ry_encode
from backbone.quantum_core.fusion_model import TARGET_SIZE
from backbone.quantum_core.preprocessing import preprocess_full_quantum
from backbone.quantum_core.tn_core import _pure_to_density
from datasets.utils.validation import get_validation_indexes
from diagnose_fusion_head import circuit_features, rotated_images
from utils.conf import base_path

QUANTUM_PARAMETERS = 4824


def bloch(rho: torch.Tensor) -> torch.Tensor:
    """(batch, qubits, 2, 2) -> (batch, 3 * qubits): all X, all Y, all Z. rho = (I + xX + yY + zZ) / 2."""
    x = 2 * rho[..., 0, 1].real
    y = -2 * rho[..., 0, 1].imag
    z = (rho[..., 0, 0] - rho[..., 1, 1]).real
    return torch.cat((x, y, z), dim=1)


@torch.no_grad()
def measurement_blocks(model, images: torch.Tensor) -> dict:
    """Every block per branch, shape (batch, branches, size). Repeats ReuploadQuantumTNCore.forward step by step."""
    values = preprocess_full_quantum(images, TARGET_SIZE, grayscale=True, mode="resize")
    qubits = angle_ry_encode(values).to(images.device)
    blocks = {"pre24": [], "final276": [], "level16": [], "level32": []}
    for index, core in enumerate(model.cores):
        branch = qubits.index_select(1, model.site_permutations[index])
        rho = _pure_to_density(branch)
        for block in core.blocks:
            rho = block(rho)
            if rho.shape[1] == 32:
                blocks["level32"].append(bloch(rho))
            if rho.shape[1] == 16:
                blocks["level16"].append(bloch(rho))
        angles = 2 * torch.atan2(branch[..., 1].abs(), branch[..., 0].abs())
        angles = angles.reshape(angles.shape[0], core.final_qubits, -1).mean(dim=2) * core.reupload_scale[None, :]
        c, s = torch.cos(angles / 2), torch.sin(angles / 2)
        gate = torch.stack((torch.stack((c, -s), -1), torch.stack((s, c), -1)), -2).to(rho.dtype)
        rho = gate @ rho @ gate.conj().transpose(-1, -2)
        blocks["pre24"].append(bloch(rho))
        blocks["final276"].append(model.pauli(core.merge_stage(rho)))
    return {name: torch.stack(parts, dim=1) for name, parts in blocks.items()}


def extract(model, data, labels, indexes_per_angle, angles, batch_size, device):
    """One (blocks, labels) entry per angle."""
    results = []
    for degrees, indexes in zip(angles, indexes_per_angle):
        images = rotated_images(data, indexes, degrees)
        parts = [measurement_blocks(model, images[start:start + batch_size].to(device))
                 for start in range(0, len(images), batch_size)]
        blocks = {name: torch.cat([part[name] for part in parts]) for name in parts[0]}
        results.append((blocks, labels[indexes].to(device)))
    return results


class Head(nn.Module):
    """The qcnn-fusion head with a different input size: per branch Linear-LN-GELU to `projection` (16 in
    qcnn-fusion), then the shared MLP with 16 hidden units."""

    def __init__(self, inputs: int, branches: int, projection: int = 16, hidden: int = 16, classes: int = 10):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(inputs, projection), nn.LayerNorm(projection), nn.GELU()) for _ in range(branches)
        ])
        self.fusion = nn.Sequential(nn.Linear(branches * projection, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                    nn.Linear(hidden, classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([projection(features[:, index])
                                      for index, projection in enumerate(self.projections)], dim=1))


def joined(sets, names):
    return [(torch.cat([blocks[name] for name in names], dim=2), target) for blocks, target in sets]


def train_and_score(train_sets, val_sets, args, seed, projection):
    torch.manual_seed(seed)
    features = torch.cat([item[0] for item in train_sets])
    target = torch.cat([item[1] for item in train_sets])
    head = Head(features.shape[2], features.shape[1], projection).to(args.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    for _ in range(args.epochs):
        for chunk in torch.randperm(len(target), device=target.device).split(args.head_batch_size):
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(head(features[chunk]), target[chunk]).backward()
            optimizer.step()
    with torch.no_grad():
        per_angle = [100 * (head(f).argmax(1) == t).float().mean().item() for f, t in val_sets]
    return float(np.mean(per_angle)), sum(parameter.numel() for parameter in head.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--readouts", nargs="+", required=True, help="e.g. pre24 final276 final276+level16")
    args = parser.parse_args()
    started = time.perf_counter()

    model = get_backbone(Namespace(backbone="qcnn-fusion", num_classes=10)).model
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict({key[len("net.model."):]: value for key, value in state.items()
                           if key.startswith("net.model.")})
    model.to(args.device).eval()

    mnist = MNIST(base_path() + "MNIST", train=True, download=True)
    train_all, val_all = get_validation_indexes(10, SimpleNamespace(targets=mnist.targets.numpy()), args.split_seed)
    generator = np.random.default_rng(args.angle_seed)
    angles = generator.uniform(0, 180, args.angles).tolist()
    train_indexes = [generator.choice(train_all, args.train_per_angle, replace=False) for _ in angles]
    val_indexes = [generator.choice(val_all, args.val_per_angle, replace=False) for _ in angles]
    print("Angles:", " ".join(f"{degrees:.1f}" for degrees in angles), flush=True)

    # The step-by-step path must give the model's own features, and Z of pre24 must be a real Bloch vector.
    check = rotated_images(mnist.data, val_indexes[0][:8], angles[0]).to(args.device)
    blocks = measurement_blocks(model, check)
    difference = (blocks["final276"] - circuit_features(model, check)).abs().max().item()
    assert difference < 1e-4, f"Step-by-step features differ from the model: {difference}"
    for name in ("pre24", "level16", "level32"):
        size = blocks[name].shape[2] // 3
        length = blocks[name].reshape(*blocks[name].shape[:2], 3, size).norm(dim=2)
        assert length.max().item() <= 1 + 1e-4, f"{name}: Bloch vector longer than 1"

    print("Measurements of the checkpoint circuit...", flush=True)
    train_sets = extract(model, mnist.data, mnist.targets, train_indexes, angles, args.batch_size, args.device)
    val_sets = extract(model, mnist.data, mnist.targets, val_indexes, angles, args.batch_size, args.device)

    rows = []
    for readout in args.readouts:
        blocks_part, _, projection = readout.partition("@")
        names = blocks_part.split("+")
        train_joined, val_joined = joined(train_sets, names), joined(val_sets, names)
        scores = []
        for seed in range(args.head_seeds):
            score, count = train_and_score(train_joined, val_joined, args, seed, int(projection or 16))
            scores.append(score)
        size = train_joined[0][0].shape[2]
        share = 100 * QUANTUM_PARAMETERS / (QUANTUM_PARAMETERS + count)
        rows.append((readout, size, count, share, float(np.mean(scores))))
        print(f"{readout:28s} values/branch {size:4d}  head {count:6d}  quantum {share:5.1f}%  mean {np.mean(scores):6.2f}"
              "  seeds " + " ".join(f"{value:.2f}" for value in scores), flush=True)

    print("Readout                      values   head  quantum%  accuracy")
    for readout, size, count, share, accuracy in rows:
        print(f"{readout:28s} {size:6d} {count:6d}  {share:7.1f}  {accuracy:8.2f}")
    print(f"Done in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
