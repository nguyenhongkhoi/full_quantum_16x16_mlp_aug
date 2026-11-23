"""Evaluate one selected checkpoint on official MNIST test data. Run once, after choosing by validation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=str(ROOT.parent / "mnist_data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA selected but PyTorch cannot access CUDA")

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_args = checkpoint["config"]["args"]
    model = build_model(run_args["preset"], readout=run_args["readout"], init=run_args["init"],
                        init_scale=run_args["init_scale"], freeze_core=run_args["freeze_core"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset = datasets.MNIST(root=args.data_root, train=False, download=True,
                             transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    errors: list[dict] = []
    total = 0

    with torch.inference_mode():
        for start, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            confidences, predictions = model(images).softmax(dim=1).max(dim=1)
            flat = labels * 10 + predictions
            confusion += torch.bincount(flat.cpu(), minlength=100).reshape(10, 10)
            for offset in predictions.ne(labels).nonzero(as_tuple=False).flatten().tolist():
                errors.append({
                    "test_index": start * args.batch_size + offset,
                    "true_label": int(labels[offset].item()),
                    "predicted_label": int(predictions[offset].item()),
                    "confidence": float(confidences[offset].item()),
                })
            total += labels.numel()

    correct = int(confusion.diagonal().sum().item())
    per_class = {}
    for class_id in range(10):
        class_total = int(confusion[class_id].sum().item())
        class_correct = int(confusion[class_id, class_id].item())
        per_class[str(class_id)] = {
            "correct": class_correct,
            "total": class_total,
            "accuracy": class_correct / class_total if class_total else 0.0,
        }

    output_path = checkpoint_path.with_name(f"official_test_{run_args['preset']}.json")
    output_path.write_text(json.dumps({
        "checkpoint": str(checkpoint_path),
        "preset": run_args["preset"],
        "readout": run_args["readout"],
        "test_size": total,
        "correct": correct,
        "accuracy": correct / total,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "misclassified_count": len(errors),
    }, indent=2), encoding="utf-8")
    errors_path = output_path.with_name(output_path.stem + "_errors.csv")
    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_index", "true_label", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(errors)
    print(f"Official MNIST test accuracy: {correct / total:.4%} ({correct}/{total})")
    print(f"Results: {output_path}")
    print(f"Misclassified examples: {errors_path}")


if __name__ == "__main__":
    main()
