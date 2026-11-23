"""Freeze the quantum core of a trained checkpoint, cache its features, and train small heads on the cache."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import FullQuantumQCNN
from models.reupload import add_reupload
from models.small_head import BRANCH_FEATURES, HEADS, branch_features, build_head


def load_checkpoint_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint["config"]["architecture"]
    if architecture.get("fusion"):
        raise ValueError("The sweep needs a checkpoint with one head per branch, such as R0")
    model = FullQuantumQCNN(in_channels=1, num_classes=10, **architecture)
    add_reupload(model, input_sites=architecture["target_size"][0] * architecture["target_size"][1],
                 final_qubits=architecture["final_qubits"], init_scale=architecture["init_scale"],
                 merge_rounds=architecture["merge_rounds"], entangling_type=architecture["entangling_type"],
                 n_layers=architecture["n_layers"], balanced_conv=architecture.get("balanced_conv", False))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


@torch.inference_mode()
def extract(model, dataset, device, batch_size: int, label: str):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    features, labels = [], []
    for index, (images, targets) in enumerate(loader, start=1):
        features.append(branch_features(model, images.to(device)).float().cpu())
        labels.append(targets)
        print(f"\r[extract {label}] batch {index}/{len(loader)}", end="", flush=True)
    print(flush=True)
    return torch.cat(features), torch.cat(labels)


def make_cache(args, model, device, out_dir: Path) -> dict:
    cache_path = out_dir / "features_cache.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache["aug_views"] == args.aug_views and cache["checkpoint"] == str(Path(args.checkpoint).resolve()):
            print(f"Using cached features: {cache_path}", flush=True)
            return cache
        raise RuntimeError("The cache was made with other settings; choose a new --out-dir.")
    order = torch.randperm(60_000, generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    validation_indices, train_indices = order[:10_000], order[10_000:]
    plain = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transforms.ToTensor())
    augmented = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transforms.Compose([
        transforms.RandomAffine(degrees=args.aug_degrees, translate=(args.aug_translate, args.aug_translate),
                                interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
        transforms.ToTensor(),
    ]))
    validation_x, validation_y = extract(model, Subset(plain, validation_indices), device,
                                         args.extract_batch_size, "validation")
    views, train_y = [], None
    for view in range(args.aug_views):
        torch.manual_seed(args.seed + 1000 * view)
        features, train_y = extract(model, Subset(augmented, train_indices), device,
                                    args.extract_batch_size, f"train view {view + 1}/{args.aug_views}")
        views.append(features)
    cache = {"aug_views": args.aug_views, "checkpoint": str(Path(args.checkpoint).resolve()),
             "train_x": torch.stack(views), "train_y": train_y,
             "validation_x": validation_x, "validation_y": validation_y}
    torch.save(cache, cache_path)
    return cache


@torch.inference_mode()
def evaluate(heads, features: torch.Tensor, labels: torch.Tensor, batch_size: int = 2048):
    loss_sum, correct = 0.0, 0
    for start in range(0, labels.numel(), batch_size):
        x, y = features[start:start + batch_size], labels[start:start + batch_size]
        logits = sum(head(x[:, index]) for index, head in enumerate(heads)) / len(heads)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(dim=1) == y).sum())
    return loss_sum / labels.numel(), correct / labels.numel()


def check_checkpoint(model, checkpoint, cache, device) -> float:
    """The cached features with the checkpoint's own heads must reproduce the logged accuracy."""
    heads = [readout.linear for _, readout in model._all_branches()]
    _, accuracy = evaluate(heads, cache["validation_x"].to(device), cache["validation_y"].to(device))
    logged = checkpoint["row"]["validation_accuracy"]
    print(f"Checkpoint check: val acc from cached features {accuracy:.2%}, logged {logged:.2%}", flush=True)
    if abs(accuracy - logged) > 0.0005:
        raise RuntimeError("The cached features do not reproduce the checkpoint; do not trust the sweep.")
    return accuracy


def train_head(name: str, seed: int, args, train_x, train_y, validation_x, validation_y, device) -> dict:
    torch.manual_seed(seed)
    views, count, branches = train_x.shape[:3]
    heads = nn.ModuleList([build_head(name) for _ in range(branches)]).to(device)
    optimizer = torch.optim.Adam(heads.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_loss, best_accuracy, best_epoch, max_accuracy, max_epoch = float("inf"), 0.0, 0, 0.0, 0
    for epoch in range(1, args.epochs + 1):
        heads.train()
        order = torch.randperm(count, device=device)
        for start in range(0, count, args.batch_size):
            batch = order[start:start + args.batch_size]
            view = torch.randint(views, (batch.numel(),), device=device)
            x = train_x[view, batch]
            logits = sum(head(x[:, index]) for index, head in enumerate(heads)) / branches
            loss = F.cross_entropy(logits, train_y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        heads.eval()
        val_loss, val_accuracy = evaluate(heads, validation_x, validation_y)
        if val_loss < best_loss:
            best_loss, best_accuracy, best_epoch = val_loss, val_accuracy, epoch
        if val_accuracy > max_accuracy:
            max_accuracy, max_epoch = val_accuracy, epoch
    return {"head": name, "seed": seed, "head_parameters_all_branches": sum(p.numel() for p in heads.parameters()),
            "best_val_loss": best_loss, "ck_accuracy": best_accuracy, "ck_epoch": best_epoch,
            "max_accuracy": max_accuracy, "max_epoch": max_epoch}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--heads", nargs="+", choices=sorted(HEADS), default=list(HEADS))
    parser.add_argument("--head-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=0, help="Augmentation stream of the cached views")
    parser.add_argument("--split-seed", type=int, default=20260911)
    parser.add_argument("--aug-views", type=int, default=4, help="Augmented copies of the train set in the cache")
    parser.add_argument("--aug-degrees", type=float, default=5.0)
    parser.add_argument("--aug-translate", type=float, default=0.05)
    parser.add_argument("--extract-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if min(args.aug_views, args.extract_batch_size, args.batch_size, args.epochs) < 1 or args.lr <= 0:
        parser.error("Counts and learning rate must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA selected but PyTorch cannot access CUDA")

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint_model(Path(args.checkpoint), device)
    cache = make_cache(args, model, device, out_dir)
    checkpoint_accuracy = check_checkpoint(model, checkpoint, cache, device)
    quantum = sum(tn_core.count_parameters() for tn_core, _ in model._all_branches())
    train_x, train_y = cache["train_x"].to(device), cache["train_y"].to(device)
    validation_x, validation_y = cache["validation_x"].to(device), cache["validation_y"].to(device)
    if train_x.shape[-1] != BRANCH_FEATURES:
        raise RuntimeError(f"Expected {BRANCH_FEATURES} features per branch, got {train_x.shape[-1]}")
    del model

    rows = []
    for name in args.heads:
        for seed in args.head_seeds:
            row = train_head(name, seed, args, train_x, train_y, validation_x, validation_y, device)
            row["total_parameters"] = row["head_parameters_all_branches"] + quantum
            rows.append(row)
            print(f"[{name} seed {seed}] total {row['total_parameters']} | CK {row['ck_accuracy']:.2%} "
                  f"(epoch {row['ck_epoch']}, val loss {row['best_val_loss']:.4f}) | "
                  f"MAX {row['max_accuracy']:.2%} (epoch {row['max_epoch']})", flush=True)
            with (out_dir / "head_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)

    summary = []
    print(f"RESULT quantum {quantum}, checkpoint val acc {checkpoint_accuracy:.2%}, frozen core, val only", flush=True)
    for name in args.heads:
        chosen = [row for row in rows if row["head"] == name]
        ck = torch.tensor([row["ck_accuracy"] for row in chosen], dtype=torch.float64)
        spread = ck.std().item() if ck.numel() > 1 else 0.0
        summary.append({"head": name, "total_parameters": chosen[0]["total_parameters"],
                        "ck_mean": ck.mean().item(), "ck_std": spread,
                        "max_mean": sum(row["max_accuracy"] for row in chosen) / len(chosen),
                        "val_loss_mean": sum(row["best_val_loss"] for row in chosen) / len(chosen)})
        item = summary[-1]
        print(f"RESULT {name:8s} total {item['total_parameters']:6d} | CK {item['ck_mean']:.2%} +- {spread:.2%} | "
              f"MAX {item['max_mean']:.2%} | val loss {item['val_loss_mean']:.4f}", flush=True)
    (out_dir / "head_sweep_summary.json").write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_validation_accuracy": checkpoint_accuracy,
        "quantum_parameters": quantum, "aug_views": args.aug_views, "epochs": args.epochs,
        "test_set_used": False, "heads": summary,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
