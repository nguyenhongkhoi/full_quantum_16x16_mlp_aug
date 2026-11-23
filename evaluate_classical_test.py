from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import ClassicalCNN  # noqa: E402

TTA_VIEWS = (
    ("original", 0.0, (0, 0)),
    ("rotate_plus_2_5", 2.5, (0, 0)),
    ("rotate_minus_2_5", -2.5, (0, 0)),
    ("shift_right_1px", 0.0, (1, 0)),
    ("shift_left_1px", 0.0, (-1, 0)),
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classical CNN checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("Expected run_convergence checkpoint with model_state_dict.")
    model = ClassicalCNN(in_channels=1, num_classes=10)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint


def affine_view(images: torch.Tensor, angle: float, translate: tuple[int, int]) -> torch.Tensor:
    if angle == 0.0 and translate == (0, 0):
        return images
    return TF.affine(images, angle=angle, translate=list(translate), scale=1.0,
                     shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, *, use_tta: bool, phase: str):
    seen = correct = 0
    loss_sum = 0.0
    started = time.perf_counter()
    total_batches = len(loader)
    views = TTA_VIEWS if use_tta else TTA_VIEWS[:1]
    print(f"[{phase}] {len(loader.dataset):,} ảnh, {total_batches} batch, {len(views)} view.", flush=True)
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        probabilities = []
        for view_idx, (name, angle, shift) in enumerate(views, start=1):
            logits = model(affine_view(images, angle, shift))
            probabilities.append(torch.softmax(logits, dim=1))
            print(f"\r[{phase}] batch {batch_idx}/{total_batches} | view {view_idx}/{len(views)} "
                  f"({name}) | {time.perf_counter() - started:.0f}s", end="", flush=True)
        probs = torch.stack(probabilities).mean(dim=0)
        predictions = probs.argmax(dim=1)
        loss_sum += float(F.nll_loss(probs.clamp_min(1e-12).log(), labels, reduction="sum"))
        correct += int((predictions == labels).sum())
        seen += labels.numel()
    print(flush=True)
    return {"examples": seen, "correct": correct, "incorrect": seen - correct,
            "accuracy": correct / seen, "loss": loss_sum / seen,
            "elapsed_seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained ClassicalCNN on official MNIST test.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to classical_best.pt produced by run_convergence.py")
    parser.add_argument("--data-root", default=str(ROOT.parent / "mnist_data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default=str(ROOT / "classical_mnist_test_results.json"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA selected but PyTorch cannot access CUDA")

    seed_all(args.seed)
    device = torch.device(args.device)
    model, checkpoint = load_model(Path(args.checkpoint), device)
    test_set = datasets.MNIST(root=args.data_root, train=False, download=True,
                              transform=transforms.ToTensor())
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    single = evaluate(model, loader, device, use_tta=False, phase="classical_single_view")
    tta = evaluate(model, loader, device, use_tta=True, phase="classical_tta_5_views")
    result = {
        "model": "ClassicalCNN(1,10), widths=(32,64)",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch", (checkpoint.get("row") or {}).get("epoch")),
        "checkpoint_row": checkpoint.get("row"),
        "dataset": "MNIST official test split",
        "examples_evaluated": len(test_set),
        "seed": args.seed,
        "single_view": single,
        "tta_mean_softmax": tta,
        "accuracy_delta_tta_minus_single": tta["accuracy"] - single["accuracy"],
        "tta_views": [{"name": name, "angle_degrees": angle, "translate_pixels_xy": list(shift)}
                      for name, angle, shift in TTA_VIEWS],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
