"""Evaluate a selected two-branch multiscale MLP-16 checkpoint on official MNIST test data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import FullQuantumQCNN
from models.reupload import add_reupload


ROOT = Path(__file__).resolve().parent
ARCH = dict(
    target_size=(16, 16), grayscale=True, preprocess_mode="resize", final_qubits=8,
    readout_type="pauli_mlp", pair_correlators="full9", init_scale=0.01,
    merge_rounds=2, entangling_type="strongly_entangling", n_layers=4,
    quantum_channels=2, spatial_orders=("row", "morton"),
    readout_hidden_dim=16, readout_dropout=0.0,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=str(ROOT.parent / "mnist_data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=None,
                        help="JSON path; defaults next to the checkpoint")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA selected but PyTorch cannot access CUDA")

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FullQuantumQCNN(in_channels=1, num_classes=10, **ARCH)
    add_reupload(
        model, input_sites=16 * 16, final_qubits=8, init_scale=0.01,
        merge_rounds=2, entangling_type="strongly_entangling", n_layers=4,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset = datasets.MNIST(
        root=args.data_root, train=False, download=True, transform=transforms.ToTensor()
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    errors: list[dict] = []
    total = 0

    with torch.inference_mode():
        for start, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            probabilities = model(images).softmax(dim=1)
            confidences, predictions = probabilities.max(dim=1)
            flat = labels * 10 + predictions
            confusion += torch.bincount(flat.cpu(), minlength=100).reshape(10, 10)
            wrong = predictions.ne(labels).nonzero(as_tuple=False).flatten()
            for offset in wrong.tolist():
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

    output_path = Path(args.output).resolve() if args.output else checkpoint_path.with_name(
        "official_test_2branch_multiscale_mlp16.json"
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "test_size": total,
        "correct": correct,
        "accuracy": correct / total,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "misclassified_count": len(errors),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    errors_path = output_path.with_name(output_path.stem + "_errors.csv")
    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "test_index", "true_label", "predicted_label", "confidence"
        ])
        writer.writeheader()
        writer.writerows(errors)
    print(f"Official MNIST test accuracy: {correct / total:.4%} ({correct}/{total})")
    print(f"Results: {output_path}")
    print(f"Misclassified examples: {errors_path}")


if __name__ == "__main__":
    main()
