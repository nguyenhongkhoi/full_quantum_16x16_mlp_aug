"""Offline check of the expert head on fixed circuit features. Put next to main.py and diagnose_fusion_head.py.

Same checkpoint, images and angles as diagnose_fusion_head.py, whose B line is the current head on the same data.
It shows what the head can represent when every angle is available. It does not measure forgetting.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.datasets import MNIST

from backbone import get_backbone
from backbone.quantum_core.fusion_model import FusionQCNN
from datasets.utils.validation import get_validation_indexes
from diagnose_fusion_head import accuracy_per_angle, extract, train_head
from utils.conf import base_path


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
    parser.add_argument("--experts", type=int, required=True)
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

    per_seed = []
    for seed in range(args.head_seeds):
        torch.manual_seed(seed)
        model = FusionQCNN(experts=args.experts).to(args.device)
        train_head(model, train_sets, args)
        per_seed.append(accuracy_per_angle(model, val_sets))
        print(f"      head seed {seed}: {np.mean(per_seed[-1]):.2f}", flush=True)
    mean_per_angle = np.mean(per_seed, axis=0)
    print(f"E  checkpoint circuit + expert head: {np.mean(mean_per_angle):.2f}", flush=True)

    print("Angle      E  mean expert weights (last seed)")
    with torch.no_grad():
        for degrees, accuracy, (features, _) in zip(angles, mean_per_angle, val_sets):
            branches = [projection(features[:, index]) for index, projection in enumerate(model.projections)]
            weights = model.fusion.weights(torch.cat(branches, dim=1)).mean(dim=0).tolist()
            print(f"{degrees:5.1f} {accuracy:6.2f}  " + " ".join(f"{weight:.2f}" for weight in weights))


if __name__ == "__main__":
    main()
