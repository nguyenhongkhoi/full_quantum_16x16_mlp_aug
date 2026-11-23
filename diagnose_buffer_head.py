"""How far does retraining only the head on the replay buffer go? Put next to main.py and diagnose_fusion_head.py.

Uses the buffer stored in the checkpoint and the same validation images and angles as diagnose_fusion_head.py.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.datasets import MNIST

from backbone import get_backbone
from datasets.utils.validation import get_validation_indexes
from diagnose_fusion_head import accuracy_per_angle, circuit_features, extract, retrained_accuracy
from utils.conf import base_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--angles", type=int, required=True)
    parser.add_argument("--angle-seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True, help="The --seed of the run, for its validation split")
    parser.add_argument("--train-per-angle", type=int, required=True, help="Same as diagnose_fusion_head.py, for the same images")
    parser.add_argument("--val-per-angle", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--head-batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--head-seeds", type=int, required=True)
    args = parser.parse_args()

    trained = get_backbone(Namespace(backbone=args.backbone, num_classes=10)).model
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {key[len("net.model."):]: value for key, value in saved["model"].items() if key.startswith("net.model.")}
    trained.load_state_dict(state)
    trained.to(args.device).eval()

    mnist = MNIST(base_path() + "MNIST", train=True, download=True)
    train_all, val_all = get_validation_indexes(10, SimpleNamespace(targets=mnist.targets.numpy()), args.split_seed)
    generator = np.random.default_rng(args.angle_seed)
    angles = generator.uniform(0, 180, args.angles).tolist()
    for _ in angles:  # Same random draws as diagnose_fusion_head.py, so the validation images match.
        generator.choice(train_all, args.train_per_angle, replace=False)
    val_indexes = [generator.choice(val_all, args.val_per_angle, replace=False) for _ in angles]

    val_sets = extract(trained, mnist.data, mnist.targets, val_indexes, angles, args.batch_size, args.device)
    kept_head = accuracy_per_angle(trained, val_sets)
    print(f"A  checkpoint circuit + checkpoint head: {np.mean(kept_head):.2f}", flush=True)

    images, labels = saved["buffer"]["examples"], saved["buffer"]["labels"].to(args.device)
    features = torch.cat([circuit_features(trained, images[start:start + args.batch_size].to(args.device))
                          for start in range(0, len(images), args.batch_size)])
    print(f"Buffer: {len(images)} images, per class {torch.bincount(labels, minlength=10).tolist()}", flush=True)

    results = {}
    epoch_list = args.epochs
    for epochs in epoch_list:
        args.epochs = epochs
        print(f"D  checkpoint circuit + new head on the buffer, {epochs} epochs:", flush=True)
        results[epochs] = retrained_accuracy([(features, labels)], val_sets, args)
        print(f"D  mean: {np.mean(results[epochs]):.2f}", flush=True)

    print("Angle      A " + " ".join(f"D{epochs:<5d}" for epochs in epoch_list))
    for index, degrees in enumerate(angles):
        print(f"{degrees:5.1f} {kept_head[index]:6.2f} " + " ".join(f"{results[e][index]:6.2f}" for e in epoch_list))


if __name__ == "__main__":
    main()
