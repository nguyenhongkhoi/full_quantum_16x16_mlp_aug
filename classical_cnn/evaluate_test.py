"""Evaluate ONE selected checkpoint on the official MNIST test set (10,000 images). Run once, after choosing the
model on validation only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default=None,
                        help="Default: the dataset used for training (from the checkpoint)")
    parser.add_argument("--data-root", default="./mnist_data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"]["args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    print(f"Loaded {path} (epoch {checkpoint['epoch']}), parameters {model.count_parameters()['total']}")

    dataset = args.dataset or checkpoint["config"]["args"].get("dataset", "mnist")
    print(f"Test set: {dataset} (official test split)")
    dataset_cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}[dataset]
    test_set = dataset_cls(root=args.data_root, train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu()
            confusion += torch.bincount(labels * 10 + preds, minlength=100).reshape(10, 10)
    accuracy = confusion.diagonal().sum().item() / confusion.sum().item()
    print(f"Official test accuracy: {accuracy * 100:.2f}%")
    result = {"checkpoint": str(path), "test_accuracy": accuracy, "confusion_matrix": confusion.tolist(),
              "param_counts": model.count_parameters()}
    out = path.with_name(path.stem + "_test.json")
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
