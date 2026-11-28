"""Train a classical CNN baseline (models.py) on static MNIST or Fashion-MNIST 16x16 (--dataset).

Data, split, augmentation, optimizer, LR schedule, early stopping, checkpoints and outputs (history.csv,
config.json, summary.json, convergence.png, model_best_val_acc.pt, model_best_val_loss.pt, model_last.pt) are
the same as full_quantum_16x16_level32_qconv/run_level32_front.py, so the numbers compare directly with the QCNN
runs. Only the model differs.
"""
from __future__ import annotations

import argparse
import csv
import json
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

from models import ARCHS, PRESETS, STANDARD_PRESETS, build_model, resolve_hidden, resolve_widths


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


DATASETS = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}
DATASET_NAMES = {"mnist": "MNIST", "fashion": "Fashion-MNIST"}


def make_loaders(args: argparse.Namespace, device: torch.device):
    # Deterministic 50,000 train / 10,000 validation split (same as the QCNN runner)
    split_order = torch.randperm(60_000, generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    validation_indices = split_order[:10_000]
    train_indices = split_order[10_000:]
    limit = getattr(args, "limit_samples", 0) or 0
    if limit > 0:  # smoke test: chỉ dùng `limit` ảnh đầu của train và của validation
        train_indices, validation_indices = train_indices[:limit], validation_indices[:limit]

    if args.aug_degrees > 0 or args.aug_translate > 0:
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

    dataset_cls = DATASETS[getattr(args, "dataset", "mnist")]
    train_base = dataset_cls(root=args.data_root, train=True, download=True, transform=train_transform)
    validation_base = dataset_cls(root=args.data_root, train=True, download=True, transform=validation_transform)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        Subset(train_base, train_indices), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.num_workers, pin_memory=pin,
    )
    validation_loader = DataLoader(
        Subset(validation_base, validation_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )
    return train_loader, validation_loader


def run_epoch(model, loader, device, optimizer=None, epoch=1, total_epochs=50):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    num_batches = len(loader)
    tag = "Train" if is_train else "Val  "
    context = torch.enable_grad() if is_train else torch.no_grad()
    t_start = time.time()
    with context:
        for batch_idx, (images, labels) in enumerate(loader, start=1):
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
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size
            if batch_idx % 100 == 0 or batch_idx == num_batches:
                elapsed = time.time() - t_start
                print(f"\r  [{tag} Ep {epoch}/{total_epochs}] {batch_idx:4d}/{num_batches} "
                      f"| Loss: {total_loss / total:7.4f} | Acc: {correct / total * 100:5.1f}% | {elapsed:6.1f}s ",
                      end="", flush=True)
    print()
    return total_loss / max(total, 1), correct / max(total, 1)


def save_plot(history, out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="Train Loss", color="royalblue")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="Val Loss", color="crimson")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    ax2.plot(epochs, [h["train_acc"] * 100 for h in history], label="Train Acc (%)", color="royalblue")
    ax2.plot(epochs, [h["val_acc"] * 100 for h in history], label="Val Acc (%)", color="crimson")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_checkpoint(path: Path, model, epoch, val_acc, val_loss, config) -> None:
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc, "val_loss": val_loss,
                "config": config}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Data & system
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist", help="mnist or fashion")
    parser.add_argument("--data-root", default="./mnist_data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260911)
    parser.add_argument("--limit-samples", type=int, default=0,
                        help="Smoke test: chỉ dùng N ảnh train và N ảnh validation (0 = đủ 50.000 / 10.000)")
    # Augmentation
    parser.add_argument("--aug-degrees", type=float, default=5.0)
    parser.add_argument("--aug-translate", type=float, default=0.05)
    parser.add_argument("--aug-interpolation", choices=list(AUG_INTERPOLATION), default="bilinear")
    # Model
    parser.add_argument("--arch", choices=ARCHS, default="small",
                        help="small: SmallCNN (2 conv, same-size baselines); standard: VGG-style StandardCNN")
    parser.add_argument("--preset", choices=list(PRESETS) + list(STANDARD_PRESETS), default=None,
                        help="Size preset (see models.py); default cnn5k (small) / std85k (standard)")
    parser.add_argument("--widths", type=int, nargs=2, default=None, help="Conv widths a b; overrides --preset")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden size of the head (default: b)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout after the hidden ReLU (0: none)")
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on val loss")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--lr-patience", type=int, default=3, help="ReduceLROnPlateau patience")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--out-dir", default=None, help="Default: runs/<preset or a_b>_lr<lr>_seed<seed>")
    args = parser.parse_args()

    if args.preset is None:
        args.preset = "std85k" if args.arch == "standard" else "cnn5k"
    if (args.preset in STANDARD_PRESETS) != (args.arch == "standard"):
        parser.error(f"--preset {args.preset} does not match --arch {args.arch}")
    widths = resolve_widths(vars(args))
    hidden = resolve_hidden(vars(args))
    name = args.preset if args.widths is None else f"w{widths[0]}_{widths[1]}"
    if args.arch == "standard" and args.widths is not None:
        name = f"std_{name}"
    if args.dataset != "mnist":
        name = f"{args.dataset}_{name}"
    out_dir = Path(args.out_dir or ROOT / "runs" / f"{name}_lr{args.lr:g}_seed{args.seed}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    device = torch.device(args.device)
    model = build_model(vars(args)).to(device)
    param_counts = model.count_parameters()

    print("=" * 70)
    print(f"Classical CNN baseline, static {DATASET_NAMES[args.dataset]} 16x16, augmentation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {type(model).__name__} widths={widths}, hidden={hidden}, dropout={args.dropout} ({name})")
    print(f"Parameter counts: Conv={param_counts['conv']}, Head={param_counts['head']}, Total={param_counts['total']}")
    print(f"LR: {args.lr:g} (ReduceLROnPlateau x{args.lr_factor}, patience {args.lr_patience}), "
          f"weight decay {args.weight_decay:g}, early stop {args.patience}")
    print(f"Augmentation: +/-{args.aug_degrees} deg, +/-{args.aug_translate} translate")
    print(f"Output directory: {out_dir}")

    config = {"args": vars(args), "arch": args.arch, "widths": list(widths), "hidden": hidden,
              "param_counts": param_counts,
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    train_loader, val_loader = make_loaders(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)

    history = []
    best_val_acc, best_val_loss = 0.0, float("inf")
    best_acc_epoch = best_loss_epoch = -1
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, epoch, args.epochs)
        val_loss, val_acc = run_epoch(model, val_loader, device, None, epoch, args.epochs)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        record = {"epoch": epoch, "train_loss": round(train_loss, 6), "train_acc": round(train_acc, 6),
                  "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 6), "lr": current_lr,
                  "epoch_sec": round(elapsed, 2)}
        history.append(record)
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        print(f"{epoch:5d} | train {train_loss:.6f} {train_acc * 100:6.2f}% | val {val_loss:.6f} "
              f"{val_acc * 100:6.2f}% | lr {current_lr:.2e} | {elapsed:5.1f}s")

        if val_acc > best_val_acc:
            best_val_acc, best_acc_epoch = val_acc, epoch
            save_checkpoint(out_dir / "model_best_val_acc.pt", model, epoch, val_acc, val_loss, config)
        if val_loss < best_val_loss:
            best_val_loss, best_loss_epoch = val_loss, epoch
            patience_counter = 0
            save_checkpoint(out_dir / "model_best_val_loss.pt", model, epoch, val_acc, val_loss, config)
        else:
            patience_counter += 1
        save_checkpoint(out_dir / "model_last.pt", model, epoch, val_acc, val_loss, config)
        save_plot(history, out_dir / "convergence.png")
        if patience_counter >= args.patience:
            print(f"[Early Stopping] No improvement in val loss for {args.patience} epochs. Stopping at epoch {epoch}.")
            break

    ck_acc = next(h["val_acc"] for h in history if h["epoch"] == best_loss_epoch)
    print("=" * 70)
    print(f"Best Val Acc:  {best_val_acc * 100:.2f}% (Epoch {best_acc_epoch})")
    print(f"Best Val Loss: {best_val_loss:.6f} (Epoch {best_loss_epoch}), val acc there (CK) {ck_acc * 100:.2f}%")
    print("=" * 70)
    summary = {"name": name, "arch": args.arch, "widths": list(widths), "hidden": hidden, "lr": args.lr, "seed": args.seed,
               "best_val_acc": best_val_acc, "best_acc_epoch": best_acc_epoch,
               "best_val_loss": best_val_loss, "best_loss_epoch": best_loss_epoch, "ck_val_acc": ck_acc,
               "total_epochs": len(history), "total_time_sec": round(sum(h["epoch_sec"] for h in history), 1),
               "param_counts": param_counts}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
