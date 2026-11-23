"""Does a task checkpoint forget in the circuit or in the head? Copy next to Mammoth's main.py and run it from there.

Freezes the circuit and retrains only the head on rotated MNIST. No test images are used: the split is Mammoth's
validation split. The angles are new ones, because Mammoth does not log the angles of its tasks.
"""
from __future__ import annotations

import argparse
import time
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn
from torchvision.datasets import MNIST

from backbone import get_backbone
from backbone.quantum_core.encoding import angle_ry_encode
from backbone.quantum_core.fusion_model import SCAN_ANGLE_ORDERS, SPATIAL_ORDERS, TARGET_SIZE, FusionQCNN
from backbone.quantum_core.preprocessing import preprocess_full_quantum
from datasets.utils.validation import get_validation_indexes
from utils.conf import base_path

ORDERS = {"qcnn-fusion": SPATIAL_ORDERS, "qcnn-fusion-scan": SCAN_ANGLE_ORDERS}


def rotated_images(data: torch.Tensor, indexes: np.ndarray, degrees: float) -> torch.Tensor:
    """Same operations as Mammoth's Rotation and ToTensor."""
    return torch.stack([TF.to_tensor(TF.rotate(Image.fromarray(data[i].numpy()), degrees)) for i in indexes])


@torch.no_grad()
def circuit_features(model: FusionQCNN, images: torch.Tensor) -> torch.Tensor:
    """The same steps as FusionQCNN.forward up to the head. Shape: (batch, branches, features)."""
    values = preprocess_full_quantum(images, TARGET_SIZE, grayscale=True, mode="resize")
    qubits = angle_ry_encode(values).to(images.device)
    return torch.stack([
        model.pauli(core(qubits.index_select(1, model.site_permutations[index])))
        for index, core in enumerate(model.cores)
    ], dim=1)


def head_logits(model: FusionQCNN, features: torch.Tensor) -> torch.Tensor:
    branches = [projection(features[:, index]) for index, projection in enumerate(model.projections)]
    return model.fusion(torch.cat(branches, dim=1))


def extract(model: FusionQCNN, data, labels, indexes_per_angle, angles, batch_size, device):
    """Features and labels, one entry per angle."""
    model.eval()
    results = []
    for degrees, indexes in zip(angles, indexes_per_angle):
        images = rotated_images(data, indexes, degrees)
        features = [circuit_features(model, images[start:start + batch_size].to(device))
                    for start in range(0, len(images), batch_size)]
        results.append((torch.cat(features), labels[indexes].to(device)))
    return results


@torch.no_grad()
def accuracy_per_angle(model: FusionQCNN, sets) -> list:
    return [100 * (head_logits(model, features).argmax(1) == target).float().mean().item() for features, target in sets]


def train_head(model: FusionQCNN, train_sets, args) -> None:
    features = torch.cat([item[0] for item in train_sets])
    target = torch.cat([item[1] for item in train_sets])
    parameters = list(model.projections.parameters()) + list(model.fusion.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    for _ in range(args.epochs):
        for chunk in torch.randperm(len(target), device=target.device).split(args.head_batch_size):
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(head_logits(model, features[chunk]), target[chunk]).backward()
            optimizer.step()


def retrained_accuracy(train_sets, val_sets, args) -> list:
    """Mean over the head seeds, per angle. Only a new head is trained, on fixed circuit features."""
    per_seed = []
    for seed in range(args.head_seeds):
        torch.manual_seed(seed)
        model = FusionQCNN(orders=ORDERS[args.backbone]).to(args.device)
        train_head(model, train_sets, args)
        per_seed.append(accuracy_per_angle(model, val_sets))
        print(f"      head seed {seed}: {np.mean(per_seed[-1]):.2f}", flush=True)
    return np.mean(per_seed, axis=0).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", required=True)
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
    args = parser.parse_args()
    started = time.perf_counter()

    trained = get_backbone(Namespace(backbone=args.backbone, num_classes=10)).model
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    state = {key[len("net.model."):]: value for key, value in state.items() if key.startswith("net.model.")}
    trained.load_state_dict(state)
    trained.to(args.device).eval()

    mnist = MNIST(base_path() + "MNIST", train=True, download=True)
    train_all, val_all = get_validation_indexes(10, SimpleNamespace(targets=mnist.targets.numpy()), args.split_seed)
    generator = np.random.default_rng(args.angle_seed)
    angles = generator.uniform(0, 180, args.angles).tolist()
    train_indexes = [generator.choice(train_all, args.train_per_angle, replace=False) for _ in angles]
    val_indexes = [generator.choice(val_all, args.val_per_angle, replace=False) for _ in angles]
    print("Angles:", " ".join(f"{degrees:.1f}" for degrees in angles), flush=True)

    # The separate feature path must give what the model itself gives.
    check = rotated_images(mnist.data, val_indexes[0][:8], angles[0]).to(args.device)
    with torch.no_grad():
        difference = (trained(check) - head_logits(trained, circuit_features(trained, check))).abs().max().item()
    assert difference < 1e-3, f"Feature path differs from the model: {difference}"

    print("Features of the checkpoint circuit...", flush=True)
    train_sets = extract(trained, mnist.data, mnist.targets, train_indexes, angles, args.batch_size, args.device)
    val_sets = extract(trained, mnist.data, mnist.targets, val_indexes, angles, args.batch_size, args.device)
    kept_head = accuracy_per_angle(trained, val_sets)
    print(f"A  checkpoint circuit + checkpoint head: {np.mean(kept_head):.2f}", flush=True)

    print("B  checkpoint circuit + new head:", flush=True)
    new_head = retrained_accuracy(train_sets, val_sets, args)
    print(f"B  mean: {np.mean(new_head):.2f}", flush=True)

    print("Features of a newly initialised circuit...", flush=True)
    torch.manual_seed(0)
    fresh = FusionQCNN(orders=ORDERS[args.backbone]).to(args.device)
    fresh_train = extract(fresh, mnist.data, mnist.targets, train_indexes, angles, args.batch_size, args.device)
    fresh_val = extract(fresh, mnist.data, mnist.targets, val_indexes, angles, args.batch_size, args.device)
    print("C  untrained circuit + new head:", flush=True)
    fresh_head = retrained_accuracy(fresh_train, fresh_val, args)
    print(f"C  mean: {np.mean(fresh_head):.2f}", flush=True)

    print("Angle      A      B      C")
    for row in zip(angles, kept_head, new_head, fresh_head):
        print("{:5.1f} {:6.2f} {:6.2f} {:6.2f}".format(*row))
    print(f"B - A: {np.mean(new_head) - np.mean(kept_head):.2f}   B - C: {np.mean(new_head) - np.mean(fresh_head):.2f}")
    print(f"Done in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
