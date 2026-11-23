"""Train QCNN có encoder quanvolution học được (qconv4x3) trên MNIST tĩnh 16x16.

Chép từ full_quantum_16x16_level32_qconv/run_level32_front.py: cùng cách chia dữ liệu, augmentation, Adam,
ReduceLROnPlateau, dừng sớm, checkpoint và file đầu ra (history.csv, config.json, summary.json, convergence.png,
model_best_val_acc.pt, model_best_val_loss.pt, model_last.pt, thêm train_log.txt). Chỉ có code model được chuyển
sang package qcnn/ (data, encoder_quanvolution, core, layers). Mặc định là cấu hình run 2c
(--front qconv4x3 --projection 10 --hidden 16 --dropout 0 --head mlp): 17.264 tham số.
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

from qcnn import build_model


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
    # Chia cố định 50.000 train / 10.000 validation
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

            # Tiến độ trên MỘT dòng mỗi epoch (ghi đè bằng \r), chỉ khi chạy trên console; file log và đầu ra bị
            # pipe chỉ nhận dòng kết quả của mỗi epoch do main() in.
            if PROGRESS and (batch_idx % 10 == 0 or batch_idx == num_batches):
                elapsed = time.time() - t_start
                sec_per_it = elapsed / batch_idx
                eta_m, eta_s = divmod(int((num_batches - batch_idx) * sec_per_it), 60)
                status(f"Ep {epoch}/{total_epochs} {tag} {batch_idx:4d}/{num_batches} "
                       f"({batch_idx / num_batches * 100:5.1f}%) | loss {total_loss / total:.4f} | "
                       f"acc {correct / total * 100:5.2f}% | {sec_per_it:4.2f}s/it | ETA {eta_m:02d}:{eta_s:02d}")

    return total_loss / max(total, 1), correct / max(total, 1)


# ---- in ra console và ghi log ----
PROGRESS = sys.stdout.isatty()
LINE_WIDTH = 118
_LOG = None


def status(text: str) -> None:
    """Ghi đè dòng hiện tại trên console (tiến độ); không ghi vào log."""
    sys.stdout.write("\r" + text[:LINE_WIDTH].ljust(LINE_WIDTH))
    sys.stdout.flush()


def log(text: str = "") -> None:
    """Một dòng đầy đủ: thay dòng tiến độ trên console, và được ghi thêm vào file log."""
    if PROGRESS:
        sys.stdout.write("\r" + " " * LINE_WIDTH + "\r")
    print(text, flush=True)
    if _LOG is not None:
        _LOG.write(text + "\n")
        _LOG.flush()


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
    # Dữ liệu và hệ thống
    parser.add_argument("--data-root", default="./mnist_data", help="Root directory for MNIST dataset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260911)

    # Tăng cường dữ liệu
    parser.add_argument("--aug-degrees", type=float, default=5.0, help="Rotation range in degrees (+/-)")
    parser.add_argument("--aug-translate", type=float, default=0.05, help="Translation range (+/-)")
    parser.add_argument("--aug-interpolation", choices=list(AUG_INTERPOLATION.keys()), default="bilinear")

    # Kiến trúc
    parser.add_argument("--side-level", type=int, default=32, help="Tree level to tap Bloch vectors from")
    parser.add_argument("--projection", type=int, default=10, help="Per-tree projection dimension (head mlp)")
    parser.add_argument("--hidden", type=int, default=16, help="Fusion hidden dimension (head mlp)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout probability after GELU in the head mlp (0 disables it)")
    parser.add_argument("--head", choices=["mlp", "linear3", "linear-shared"], default="mlp",
                        help="mlp: projection + fusion MLP (run 2c); linear3: one Linear(372, 10) per tree, mean of "
                             "the logits; linear-shared: one Linear for the three trees")
    parser.add_argument("--orders", nargs="+", default=["row", "column", "morton"], help="Scan orders")
    parser.add_argument("--front", default="qconv4x3",
                        help="Quanvolution encoder qconv<L>[x<S>][h<H>][f] (e.g. qconv4x3, qconv2x2), or none")
    parser.add_argument("--model", choices=["qcnn"], default="qcnn", help="Only qcnn here")

    # Siêu tham số train
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on val loss")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau decay factor")
    parser.add_argument("--lr-patience", type=int, default=3, help="ReduceLROnPlateau patience")
    parser.add_argument("--min-lr", type=float, default=1e-6)

    # Đầu ra
    parser.add_argument("--out-dir", default="./runs/front_qconv4x3_proj10", help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # File log thay cho Tee-Object (pipe qua Tee-Object làm tiến độ một dòng bị nhảy thành nhiều dòng).
    global _LOG
    _LOG = (out_dir / "train_log.txt").open("a", encoding="utf-8")
    log(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} train.py {' '.join(sys.argv[1:])}")

    seed_all(args.seed)
    device = torch.device(args.device)

    log("=" * 70)
    log("QCNN level32 + quanvolution encoder, Static Training with Augmentation")
    log("=" * 70)
    log(f"Device: {device}")
    log(f"Model: {args.model}")
    log(f"Front end: {args.front}")
    log(f"Augmentation: +/-{args.aug_degrees} deg, +/-{args.aug_translate} translate, {args.aug_interpolation}")
    log(f"Architecture: side_level={args.side_level}, projection={args.projection}, hidden={args.hidden}, head={args.head}")
    log(f"Head dropout: p={args.dropout}")
    log(f"Output directory: {out_dir}")
    log(f"Log file: {out_dir / 'train_log.txt'}")

    # Dựng model
    model = build_model(vars(args)).to(device)

    param_counts = model.count_parameters()
    log("-" * 70)
    log(f"Parameter counts: Quantum={param_counts['quantum']}, Front={param_counts['front']}, "
        f"(circuit angles {param_counts['front_quantum']}), "
        f"Head={param_counts['head']}, Total={param_counts['total']}, "
        f"Quantum share={param_counts['quantum_share'] * 100:.1f}%")
    log("-" * 70)

    # Lưu cấu hình
    config = {
        "args": vars(args),
        "param_counts": param_counts,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Nạp dữ liệu
    train_loader, val_loader = make_loaders(args, device)
    log(f"Loaded: 50,000 train images, 10,000 val images (batch size {args.batch_size})")

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

    log("")
    log("Starting training loop... (one line per epoch; *best val acc -> model_best_val_acc.pt saved, "
        "*best val loss -> model_best_val_loss.pt saved)")
    log(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'LR':>9} | {'Time':>7}")
    log("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, epoch=epoch, total_epochs=args.epochs)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None, epoch=epoch, total_epochs=args.epochs)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Ghi lịch sử
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

        # Ghi ra CSV
        with history_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            writer.writeheader()
            writer.writerows(history)

        marks = (("  *best val acc" if val_acc > best_val_acc else "")
                 + ("  *best val loss" if val_loss < best_val_loss else ""))
        log(f"{epoch:5d} | {train_loss:10.6f} | {train_acc * 100:8.2f}% | "
            f"{val_loss:10.6f} | {val_acc * 100:8.2f}% | {current_lr:9.2e} | {elapsed:6.1f}s{marks}")

        # Kiểm val acc tốt nhất (MAX)
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

        # Kiểm val loss tốt nhất (CK)
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

        # Luôn lưu checkpoint cuối
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

        # Lưu biểu đồ hội tụ
        save_plot(history, out_dir / "convergence.png")

        # Dừng sớm
        if patience_counter >= args.patience:
            log(f"[Early Stopping] No improvement in val loss for {args.patience} epochs. Stopping at epoch {epoch}.")
            break

    ck_acc = next(h["val_acc"] for h in history if h["epoch"] == best_loss_epoch)
    log("=" * 75)
    log("Training Completed!")
    log(f"Best Val Acc:  {best_val_acc * 100:.2f}% (Epoch {best_acc_epoch}) -> saved to model_best_val_acc.pt")
    log(f"Best Val Loss: {best_val_loss:.6f} (Epoch {best_loss_epoch}), val acc there (CK) {ck_acc * 100:.2f}% "
        f"-> saved to model_best_val_loss.pt")
    log("=" * 75)

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
