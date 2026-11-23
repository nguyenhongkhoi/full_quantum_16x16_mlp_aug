from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import FullQuantumQCNN  # noqa: E402
from models.reupload import add_reupload  # noqa: E402

FQ_ARCH = dict(
    target_size=(32, 32), grayscale=True, preprocess_mode="pad", final_qubits=8,
    readout_type="pauli_mlp", pair_correlators="full9", init_scale=0.01,
    merge_rounds=2, entangling_type="strongly_entangling", n_layers=4,
    quantum_channels=3,
)

# Five fixed views from DE_XUAT_Y_TUONG_TOI_UU_ACCURACY_99.txt.
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
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Pass the 32x32-padded phase-3 checkpoint with --checkpoint."
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a runner checkpoint containing model_state_dict.")

    model = FullQuantumQCNN(in_channels=1, num_classes=10, **FQ_ARCH)
    input_sites = FQ_ARCH["target_size"][0] * FQ_ARCH["target_size"][1]
    add_reupload(model, input_sites=input_sites, final_qubits=FQ_ARCH["final_qubits"],
                 init_scale=FQ_ARCH["init_scale"], merge_rounds=FQ_ARCH["merge_rounds"],
                 entangling_type=FQ_ARCH["entangling_type"], n_layers=FQ_ARCH["n_layers"])
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights do not match the 32x32-padded, 3-branch MLP + reupload architecture. "
            "Use fq_mlp_aug_phase3_reupload_best.pt or full_quantum_mlp_aug_best.pt from the same run."
        ) from exc
    model.to(device).eval()
    return model, checkpoint


def affine_view(images: torch.Tensor, angle: float, translate: tuple[int, int]) -> torch.Tensor:
    if angle == 0.0 and translate == (0, 0):
        return images
    return TF.affine(
        images, angle=angle, translate=list(translate), scale=1.0, shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=0,
    )


PROGRESS_COLUMNS = [
    "timestamp", "phase", "batch", "total_batches", "view", "view_index",
    "views_total", "progress_percent", "completed_examples", "total_examples",
    "accuracy_so_far", "loss_so_far", "elapsed_seconds",
]


def write_progress(progress_writer, progress_handle, *, phase, batch_index, total_batches,
                   view, view_index, views_total, completed_examples, total_examples,
                   correct, loss_sum, started):
    progress_fraction = ((batch_index - 1) * views_total + view_index) / (total_batches * views_total)
    progress_writer.writerow({
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "phase": phase,
        "batch": batch_index,
        "total_batches": total_batches,
        "view": view,
        "view_index": view_index,
        "views_total": views_total,
        "progress_percent": round(progress_fraction * 100, 2),
        "completed_examples": completed_examples,
        "total_examples": total_examples,
        "accuracy_so_far": round(correct / completed_examples, 6) if completed_examples else "",
        "loss_so_far": round(loss_sum / completed_examples, 6) if completed_examples else "",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    })
    progress_handle.flush()


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, *, phase: str, use_tta: bool,
             progress_writer, progress_handle):
    seen = correct = 0
    loss_sum = 0.0
    started = time.perf_counter()
    total_batches = len(loader)
    total_examples = len(loader.dataset)
    views_total = len(TTA_VIEWS) if use_tta else 1
    print(f"[{phase}] bắt đầu: {total_examples:,} ảnh, {total_batches} batch, "
          f"{views_total} lượt/view mỗi batch.", flush=True)

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        if use_tta:
            probs = []
            for view_index, (view_name, angle, translate) in enumerate(TTA_VIEWS, start=1):
                logits = model(affine_view(images, angle, translate))
                probs.append(torch.softmax(logits, dim=1))
                view_fraction = ((batch_index - 1) * views_total + view_index) / (total_batches * views_total)
                print(f"\r[{phase}] batch {batch_index}/{total_batches} | view "
                      f"{view_index}/{views_total} ({view_name}) | tổng {view_fraction:.1%} | "
                      f"{time.perf_counter() - started:.0f}s", end="", flush=True)
                write_progress(
                    progress_writer, progress_handle, phase=phase, batch_index=batch_index,
                    total_batches=total_batches, view=view_name, view_index=view_index,
                    views_total=views_total, completed_examples=seen, total_examples=total_examples,
                    correct=correct, loss_sum=loss_sum, started=started,
                )
            probabilities = torch.stack(probs, dim=0).mean(dim=0)
            predictions = probabilities.argmax(dim=1)
            # Report NLL of the averaged probability distribution.
            loss_sum += float(F.nll_loss(probabilities.clamp_min(1e-12).log(), labels, reduction="sum"))
        else:
            logits = model(images)
            predictions = logits.argmax(dim=1)
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((predictions == labels).sum())
        seen += labels.numel()
        print(f"\r[{phase}] batch {batch_index}/{total_batches} hoàn tất | "
              f"{seen}/{total_examples} ảnh | acc {correct / seen:.2%} | "
              f"{time.perf_counter() - started:.0f}s", end="", flush=True)
        write_progress(
            progress_writer, progress_handle, phase=phase, batch_index=batch_index,
            total_batches=total_batches,
            view="batch_complete", view_index=views_total, views_total=views_total,
            completed_examples=seen, total_examples=total_examples,
            correct=correct, loss_sum=loss_sum, started=started,
        )
    print(flush=True)
    print(f"[{phase}] xong: accuracy={correct / seen:.4%}, loss={loss_sum / seen:.6f}, "
          f"thời gian={time.perf_counter() - started:.1f}s", flush=True)
    return {"examples": seen, "correct": correct, "incorrect": seen - correct,
            "accuracy": correct / seen, "loss": loss_sum / seen}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single-view and fixed 5-view TTA validation.")
    parser.add_argument("--checkpoint", default=str(
        ROOT / "runs" / "mlp_aug_32pad_3phase" / "full_quantum_mlp_aug_best.pt"))
    parser.add_argument("--data-root", default=str(ROOT.parent / "mnist_data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260911)
    parser.add_argument("--split", choices=("validation", "test"), default="validation",
                        help="validation = held-out subset of MNIST train; test = official MNIST test set.")
    parser.add_argument("--validation-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=str(ROOT / "tta_validation_32pad_results.json"))
    parser.add_argument("--progress-csv", default=str(ROOT / "tta_32pad_progress.csv"),
                        help="CSV được cập nhật sau từng view/batch để theo dõi tiến độ.")
    args = parser.parse_args()
    if not 1 <= args.validation_size <= 10_000:
        parser.error("validation-size must be 1..10000")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA selected but PyTorch cannot access CUDA")

    seed_all(args.seed)
    device = torch.device(args.device)
    print(f"Đang nạp checkpoint: {Path(args.checkpoint).resolve()}", flush=True)
    model, checkpoint = load_model(Path(args.checkpoint), device)
    print(f"Đã nạp checkpoint trên {device}.", flush=True)

    if args.split == "validation":
        # Match run_mlp_aug.py: first validation_size indices of randperm(60000, split_seed).
        split_order = torch.randperm(
            60_000, generator=torch.Generator().manual_seed(args.split_seed)
        ).tolist()
        validation_indices = split_order[:args.validation_size]
        dataset = datasets.MNIST(
            root=args.data_root, train=True, download=True, transform=transforms.ToTensor()
        )
        evaluated_dataset = Subset(dataset, validation_indices)
        dataset_description = "MNIST official train split, held-out validation subset"
    else:
        dataset = datasets.MNIST(
            root=args.data_root, train=False, download=True, transform=transforms.ToTensor()
        )
        evaluated_dataset = dataset
        dataset_description = "MNIST official test split"
    loader = DataLoader(evaluated_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    progress_path = Path(args.progress_csv)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("w", newline="", encoding="utf-8-sig") as progress_file:
        progress_writer = csv.DictWriter(progress_file, fieldnames=PROGRESS_COLUMNS)
        progress_writer.writeheader()
        progress_file.flush()
        single = evaluate(model, loader, device, phase="single_view", use_tta=False,
                          progress_writer=progress_writer, progress_handle=progress_file)
        tta = evaluate(model, loader, device, phase="tta_5_views", use_tta=True,
                       progress_writer=progress_writer, progress_handle=progress_file)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch", (checkpoint.get("row") or {}).get("epoch")),
        "checkpoint_row": checkpoint.get("row"),
        "checkpoint_phase": checkpoint.get("phase"),
        "dataset": dataset_description,
        "split": args.split,
        "examples_evaluated": len(evaluated_dataset),
        "validation_size": len(evaluated_dataset) if args.split == "validation" else None,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "tta_views": [{"name": name, "angle_degrees": angle, "translate_pixels_xy": list(shift)}
                      for name, angle, shift in TTA_VIEWS],
        "single_view": single,
        "tta_mean_softmax": tta,
        "accuracy_delta_tta_minus_single": tta["accuracy"] - single["accuracy"],
        "test_set_used": args.split == "test",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Results saved to: {output_path.resolve()}")
    print(f"Progress CSV: {progress_path.resolve()}")


if __name__ == "__main__":
    main()
