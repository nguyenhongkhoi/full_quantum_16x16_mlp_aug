"""Evaluate a selected FullQuantum Level32 (+ front end) checkpoint on official MNIST test data.
Run this ONCE after selecting the best model on validation.
"""
from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--data-root", default="./mnist_data", help="Root directory for MNIST dataset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default=None, help="Output JSON path (defaults next to checkpoint)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device(args.device)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reconstruct model from stored config
    cfg_args = checkpoint["config"]["args"]
    model = build_model(cfg_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    print(f"Model successfully loaded. Parameter counts: {model.count_parameters()['total']}")

    # Official MNIST test dataset (no augmentation)
    test_dataset = datasets.MNIST(
        root=args.data_root, train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    confusion = torch.zeros((10, 10), dtype=torch.int64)
    errors: list[dict] = []
    total = 0

    print("Evaluating on official MNIST test set (10,000 images)...")
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

            logits = model(images)
            probabilities = logits.softmax(dim=1)
            confidences, predictions = probabilities.max(dim=1)

            flat = labels * 10 + predictions
            confusion += torch.bincount(flat.cpu(), minlength=100).reshape(10, 10)

            wrong = predictions.ne(labels).nonzero(as_tuple=False).flatten()
            for offset in wrong.tolist():
                errors.append({
                    "test_index": batch_idx * args.batch_size + offset,
                    "true_label": int(labels[offset].item()),
                    "predicted_label": int(predictions[offset].item()),
                    "confidence": float(confidences[offset].item()),
                })
            total += labels.numel()

    correct = int(confusion.diagonal().sum().item())
    accuracy = correct / total

    per_class = {}
    for class_id in range(10):
        class_total = int(confusion[class_id].sum().item())
        class_correct = int(confusion[class_id, class_id].item())
        per_class[str(class_id)] = {
            "correct": class_correct,
            "total": class_total,
            "accuracy": class_correct / class_total if class_total else 0.0,
        }

    output_path = (
        Path(args.output).resolve()
        if args.output
        else checkpoint_path.with_name("official_test_result.json")
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "test_size": total,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "accuracy_pct": round(accuracy * 100, 2),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "misclassified_count": len(errors),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Save misclassified details
    errors_csv_path = output_path.with_name("official_test_errors.csv")
    with errors_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_index", "true_label", "predicted_label", "confidence"]
        )
        writer.writeheader()
        writer.writerows(errors)

    print("=" * 60)
    print("OFFICIAL TEST EVALUATION RESULT:")
    print(f"Total test samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Official Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    print(f"Misclassified: {len(errors)} samples")
    print(f"Result JSON saved to: {output_path}")
    print(f"Errors CSV saved to: {errors_csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
