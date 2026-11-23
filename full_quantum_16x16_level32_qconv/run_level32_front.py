"""Train FullQuantum Level32 with an optional small classical front end on Static MNIST.

Same as full_quantum_16x16_level32_aug/run_level32_aug.py, plus --front (models/front.py):
  none (level32), conv1 (+10 parameters), cnn<C> (+19*C+1, e.g. cnn4 +77), deep<C>, deep<C>p (deep12p +1,646).
  The front end starts as the identity.
--model enc-linear / enc-cnn (models/classical.py): classical controls with the same front end and no circuit.

Architecture:
  - 3 branches (row, column, morton scan orders)
  - 16x16 input (256 sites) -> optional classical front end (shared) -> Ry angle encoding
  - Optional 8-neighbour quantum graph layer (GraphZZLayer)
  - 5-block quantum tree conv/pool with re-upload
  - Multi-level side tap at 32-qubit stage (reads Bloch X, Y, Z -> 96 features/branch)
  - Fast Pauli measurement (276 Pauli features/branch)
  - Total 372 features/branch -> Linear projection to 12 -> Dropout -> Concatenate (36)
    -> MLP (16 -> 10) with Dropout. Dropout adds no trainable parameters.
  - Total parameters: 19,118 (19,148 with graph layer)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import MODELS, build_model


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


AUG_INTERPOLATION = {
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "nearest": transforms.InterpolationMode.NEAREST,
}


def make_loaders(args: argparse.Namespace, device: torch.device):
    # Deterministic 50,000 train / 10,000 validation split
    split_order = torch.randperm(
        60_000, generator=torch.Generator().manual_seed(args.split_seed)
    ).tolist()
    validation_indices = split_order[:10_000]
    train_indices = split_order[10_000:]

    has_aug = args.aug_degrees > 0 or args.aug_translate > 0
    if has_aug:
        train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=args.aug_degrees,
                translate=(args.aug_translate, args.aug_translate),
                interpolation=AUG_INTERPOLATION[args.aug_interpolation],
                fill=0,
            ),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.ToTensor()

    validation_transform = transforms.ToTensor()

    train_base = datasets.MNIST(
        root=args.data_root, train=True, download=True, transform=train_transform
    )
    validation_base = datasets.MNIST(
        root=args.data_root, train=True, download=True, transform=validation_transform
    )

    pin = device.type == "cuda"
    train_generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        Subset(train_base, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    validation_loader = DataLoader(
        Subset(validation_base, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    return train_loader, validation_loader


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 1,
    total_epochs: int = 50,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)
    tag = "Train" if is_train else "Val  "

    context = torch.enable_grad() if is_train else torch.no_grad()
    t_start = time.time()
    with context:
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            t_batch_0 = time.time()
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            # In tiến độ chi tiết mỗi 10 batch để không bị cảm giác treo màn hình
            if batch_idx % 10 == 0 or batch_idx == num_batches:
                elapsed = time.time() - t_start
                it_per_sec = batch_idx / max(elapsed, 1e-4)
                sec_per_it = 1.0 / it_per_sec
                eta = (num_batches - batch_idx) * sec_per_it
                eta_m, eta_s = divmod(int(eta), 60)
                cur_acc = (correct / total) * 100
                cur_loss = total_loss / total
                print(
                    f"\r  [{tag} Ep {epoch}/{total_epochs}] {batch_idx:3d}/{num_batches} ({batch_idx/num_batches*100:5.1f}%) "
                    f"| Loss: {cur_loss:7.4f} | Acc: {cur_acc:5.1f}% | {sec_per_it:4.2f}s/it | ETA: {eta_m:02d}:{eta_s:02d} ",
                    end="",
                    flush=True,
                )

    print()  # Xuống dòng khi kết thúc loader
    return total_loss / max(total, 1), correct / max(total, 1)


def save_plot(history: list[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    train_accs = [h["train_acc"] * 100 for h in history]
    val_accs = [h["val_acc"] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss", color="royalblue")
    ax1.plot(epochs, val_losses, label="Val Loss", color="crimson")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("Loss Convergence")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Acc (%)", color="royalblue")
    ax2.plot(epochs, val_accs, label="Val Acc (%)", color="crimson")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Data & System
    parser.add_argument("--data-root", default="./mnist_data", help="Root directory for MNIST dataset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260911)

    # Augmentation
    parser.add_argument("--aug-degrees", type=float, default=5.0, help="Rotation range in degrees (+/-)")
    parser.add_argument("--aug-translate", type=float, default=0.05, help="Translation range (+/-)")
    parser.add_argument("--aug-interpolation", choices=list(AUG_INTERPOLATION.keys()), default="bilinear")

    # Architecture
    parser.add_argument("--side-level", type=int, default=32, help="Tree level to tap Bloch vectors from")
    parser.add_argument("--projection", type=int, default=12, help="Per-tree projection dimension")
    parser.add_argument("--hidden", type=int, default=16, help="Fusion hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability after GELU in the classical head (0 disables it)")
    parser.add_argument("--graph-layer", action="store_true", help="Enable 8-neighbor ZZ graph layer")
    parser.add_argument("--circuit-lr-scale", type=float, default=1.0, help="Learning rate multiplier for quantum circuits under Adam")
    parser.add_argument("--orders", nargs="+", default=["row", "column", "morton"], help="Scan orders")
    parser.add_argument("--front", default="none",
                        help="Front end before the circuit: none, conv1, cnn<C>[p], deep<C>[p], "
                             "or the quantum encoder qconv<L>[x<S>][h<H>][f] (e.g. qconv2x2, qconv4x3, qconv2x2h16)")
    parser.add_argument("--model", choices=MODELS, default="qcnn",
                        help="qcnn (default); enc-linear / enc-cnn: the front end with a classical readout, no tree circuit")
    parser.add_argument("--cnn-widths", type=int, nargs=2, default=[20, 32], help="enc-cnn conv widths")
    parser.add_argument("--cnn-hidden", type=int, default=20, help="enc-cnn hidden size")
    parser.add_argument("--linear-pool", type=int, default=4,
                        help="enc-linear: AvgPool size before Linear (4: 4x4 cells, 330 for 2 channels; 2: 8x8, 1,290)")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on val loss")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau decay factor")
    parser.add_argument("--lr-patience", type=int, default=3, help="ReduceLROnPlateau patience")
    parser.add_argument("--min-lr", type=float, default=1e-6)

    # Output
    parser.add_argument("--out-dir", default="./runs/level32_front_run1", help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    device = torch.device(args.device)

    print("=" * 70)
    print("FullQuantum Level32 + classical front end, Static Training with Augmentation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Front end: {args.front}")
    print(f"Augmentation: +/-{args.aug_degrees} deg, +/-{args.aug_translate} translate, {args.aug_interpolation}")
    print(f"Architecture: side_level={args.side_level}, projection={args.projection}, graph_layer={args.graph_layer}")
    print(f"Head dropout: p={args.dropout}")
    print(f"Quantum LR scale: x{args.circuit_lr_scale}")
    print(f"Output directory: {out_dir}")

    # Build Model
    model = build_model(vars(args)).to(device)

    param_counts = model.count_parameters()
    print("-" * 70)
    print(f"Parameter counts: Quantum={param_counts['quantum']}, Front={param_counts['front']}, "
          f"(circuit angles {param_counts['front_quantum']}), "
          f"Head={param_counts['head']}, Total={param_counts['total']}, "
          f"Quantum share={param_counts['quantum_share'] * 100:.1f}%")
    print("-" * 70)

    # Save Config
    config = {
        "args": vars(args),
        "param_counts": param_counts,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Load Data
    train_loader, val_loader = make_loaders(args, device)
    print(f"Loaded: 50,000 train images, 10,000 val images (batch size {args.batch_size})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
    )

    history: list[dict] = []
    history_csv_path = out_dir / "history.csv"

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_acc_epoch = -1
    best_loss_epoch = -1
    patience_counter = 0

    print("\nStarting training loop...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'LR':>9} | {'Time':>7}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n>>> Epoch {epoch}/{args.epochs} [LR: {optimizer.param_groups[0]['lr']:.2e}]")
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, epoch=epoch, total_epochs=args.epochs)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None, epoch=epoch, total_epochs=args.epochs)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Record history
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": current_lr,
            "epoch_sec": round(elapsed, 2),
        }
        history.append(record)

        # Write to CSV
        with history_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            writer.writeheader()
            writer.writerows(history)

        print(
            f"{epoch:5d} | {train_loss:10.6f} | {train_acc * 100:8.2f}% | "
            f"{val_loss:10.6f} | {val_acc * 100:8.2f}% | {current_lr:9.2e} | {elapsed:6.1f}s"
        )

        # Check Best Val Acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_acc_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": config,
                },
                out_dir / "model_best_val_acc.pt",
            )

        # Check Best Val Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": config,
                },
                out_dir / "model_best_val_loss.pt",
            )
        else:
            patience_counter += 1

        # Always save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config": config,
            },
            out_dir / "model_last.pt",
        )

        # Save curve plot
        save_plot(history, out_dir / "convergence.png")

        # Early Stopping
        if patience_counter >= args.patience:
            print(f"\n[Early Stopping] No improvement in val loss for {args.patience} epochs. Stopping at epoch {epoch}.")
            break

    print("=" * 75)
    print("Training Completed!")
    print(f"Best Val Acc:  {best_val_acc * 100:.2f}% (Epoch {best_acc_epoch}) -> saved to model_best_val_acc.pt")
    print(f"Best Val Loss: {best_val_loss:.6f} (Epoch {best_loss_epoch}) -> saved to model_best_val_loss.pt")
    print("=" * 75)

    summary = {
        "best_val_acc": best_val_acc,
        "best_acc_epoch": best_acc_epoch,
        "best_val_loss": best_val_loss,
        "best_loss_epoch": best_loss_epoch,
        "total_epochs": len(history),
        "total_time_sec": sum(h["epoch_sec"] for h in history),
        "param_counts": param_counts,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
