"""Train the R1-R4 variants with the same split, augmentation and three-phase LR schedule as the 2-branch run."""
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
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from models import DEFAULT_INIT, EXPECTED_PARAMETERS, VARIANTS, build_model, is_head_parameter  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


AUG_INTERPOLATION = {"bilinear": transforms.InterpolationMode.BILINEAR,
                     "nearest": transforms.InterpolationMode.NEAREST}


def augmentation_enabled(args: argparse.Namespace) -> bool:
    return args.aug_degrees > 0 or args.aug_translate > 0


def make_loaders(args: argparse.Namespace, device: torch.device):
    # Same deterministic 50k/10k split as the earlier 16x16 runs.
    split_order = torch.randperm(
        60_000, generator=torch.Generator().manual_seed(args.split_seed)
    ).tolist()
    validation_indices = split_order[:10_000]
    train_indices = split_order[10_000:]

    # Separate dataset objects are essential: augmentation must never touch validation.
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=args.aug_degrees,
                                translate=(args.aug_translate, args.aug_translate),
                                interpolation=AUG_INTERPOLATION[args.aug_interpolation],
                                fill=0),
        transforms.ToTensor(),
    ]) if augmentation_enabled(args) else transforms.ToTensor()
    train_base = datasets.MNIST(root=args.data_root, train=True, download=True,
                                transform=train_transform)
    validation_base = datasets.MNIST(root=args.data_root, train=True, download=True,
                                     transform=transforms.ToTensor())
    pin = device.type == "cuda"
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(train_base, train_indices), batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=args.num_workers, pin_memory=pin,
    )
    validation_loader = DataLoader(
        Subset(validation_base, validation_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=pin,
    )
    return train_loader, validation_loader, generator, train_indices, validation_indices


def phase_model(args: argparse.Namespace, device: torch.device, phase_number: int, state: dict | None = None):
    seed_all(args.seed)
    model = build_model(args.variant, shifts=args.shifts, init=args.init)
    if state is not None:
        model.load_state_dict(state)
    # Phases 2/3 get their own augmentation stream instead of replaying phase 1's.
    if phase_number > 1:
        seed_all(args.seed + 10_000 * phase_number)
    return model.to(device)


def run_epoch(model, loader, device, optimizer=None, progress_label: str | None = None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, seen = 0.0, 0, 0
    started = time.perf_counter()
    total_batches = len(loader)
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss khong huu han")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach()) * labels.numel()
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            seen += labels.numel()
            if progress_label:
                elapsed = time.perf_counter() - started
                print(f"\r[{progress_label}] batch {batch_index:>3}/{total_batches} "
                      f"loss={loss_sum / seen:.4f} acc={correct / seen:.2%} elapsed={elapsed:.0f}s",
                      end="", flush=True)
    if progress_label:
        print(flush=True)
    return loss_sum / seen, correct / seen


# Same columns and file names as the 2-branch run, so the logs can be compared directly.
HISTORY_COLUMNS = ["epoch", "phase", "train_loss", "train_accuracy", "validation_loss",
                   "validation_accuracy", "lr_core", "lr_readout", "epoch_seconds"]


def write_history(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(rows: list[dict], out_dir: Path, title: str) -> None:
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in rows], label="Validation")
    axes[0].set(title="Cross entropy", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=.25)
    axes[0].legend()
    axes[1].plot(epochs, [100 * row["train_accuracy"] for row in rows], label="Train")
    axes[1].plot(epochs, [100 * row["validation_accuracy"] for row in rows], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(80, 100))
    axes[1].grid(alpha=.25)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_mlp_aug.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _replace_with_retry(temporary: Path, path: Path) -> None:
    for attempt in range(60):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 59:
                raise
            time.sleep(0.5)


def save_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    _replace_with_retry(temporary, path)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    _replace_with_retry(temporary, path)


def _normalized_config(config: dict) -> dict:
    normalized = json.loads(json.dumps(config, sort_keys=True, default=str))
    normalized.pop("output_directory", None)
    run_args = normalized.get("args", {})
    for key in ("out_dir", "no_resume", "num_workers"):
        run_args.pop(key, None)
    return normalized


def validate_run_directory(out_dir: Path, config: dict, *, resume: bool) -> None:
    config_path = out_dir / "config_mlp_aug.json"
    artifact_patterns = ("fq_mlp_aug_*_best.pt", "fq_mlp_aug_*_last.pt", "fq_mlp_aug_*_done.json",
                         "*_history.csv", "history_full_quantum_mlp_aug.csv", "summary_mlp_aug.json",
                         "full_quantum_mlp_aug_best.pt")
    artifacts = [path for pattern in artifact_patterns for path in out_dir.glob(pattern)]
    if artifacts and not config_path.exists():
        raise RuntimeError(f"Run artifacts exist without {config_path.name}; choose a new --out-dir.")
    if artifacts and not resume:
        raise RuntimeError(f"Run artifacts already exist in {out_dir}; choose a new --out-dir.")
    if artifacts:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if _normalized_config(previous) != _normalized_config(config):
            raise RuntimeError("Saved run configuration differs from the requested settings. "
                               "Keep the original arguments to resume, or choose a new --out-dir.")


def train_phase(*, tag, phase, model, optimizer, scheduler, train_loader, validation_loader,
                generator, epochs, select, patience, offset, out_dir, device, rows, resume=True):
    done_path = out_dir / f"{tag}_done.json"
    last_path = out_dir / f"{tag}_last.pt"
    best_path = out_dir / f"{tag}_best.pt"
    history_path = out_dir / "history_full_quantum_mlp_aug.csv"
    if resume and done_path.exists() and best_path.exists():
        phase_rows = json.loads(done_path.read_text(encoding="utf-8"))["rows"]
        rows.extend(phase_rows)
        write_history(rows, history_path)
        print(f"[{tag}] already complete ({len(phase_rows)} epochs) - skipping", flush=True)
        return phase_rows, torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"]

    phase_started = time.perf_counter()
    sign = 1.0 if select == "val_loss" else -1.0
    best, stale, phase_rows, start_epoch = float("inf"), 0, [], 1
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best, stale, phase_rows = checkpoint["best"], checkpoint["stale"], checkpoint["rows"]
        start_epoch = checkpoint["epoch"] + 1
        generator.set_state(checkpoint["shuffle_rng"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        rows.extend(phase_rows)
        print(f"[{tag}] resumed after epoch {checkpoint['epoch']}", flush=True)

    reason = "epoch_budget"
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer,
                                          progress_label=f"{tag} epoch {epoch}/{epochs} train")
        val_loss, val_acc = run_epoch(model, validation_loader, device,
                                      progress_label=f"{tag} epoch {epoch}/{epochs} validation")
        lr_core, lr_readout = optimizer.param_groups[0]["lr"], optimizer.param_groups[-1]["lr"]
        row = {"epoch": offset + epoch, "phase": phase, "train_loss": train_loss,
               "train_accuracy": train_acc, "validation_loss": val_loss,
               "validation_accuracy": val_acc, "lr_core": lr_core,
               "lr_readout": lr_readout, "epoch_seconds": time.perf_counter() - started}
        rows.append(row)
        phase_rows.append(row)
        signed_metric = sign * (val_acc if select == "val_accuracy" else val_loss)
        improved = signed_metric < best - 1e-7
        if improved:
            best, stale = signed_metric, 0
            save_atomic({"model_state_dict": model.state_dict(), "epoch": epoch,
                         "row": row, "phase": phase}, best_path)
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(val_loss)
        save_atomic({"model_state_dict": model.state_dict(), "epoch": epoch, "row": row,
                     "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                     "best": best, "stale": stale, "rows": phase_rows,
                     "shuffle_rng": generator.get_state(),
                     "torch_rng_state": torch.get_rng_state(),
                     "cuda_rng_state_all": torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None}, last_path)
        write_history(rows, history_path)
        print(f"[{tag}] epoch {epoch:3d}/{epochs} train {train_loss:.4f}/{train_acc:.4f} "
              f"val {val_loss:.4f}/{val_acc:.4f} lr {lr_core:.2e}/{lr_readout:.2e} "
              f"{row['epoch_seconds']:.0f}s{'  *best' if improved else ''}", flush=True)
        if patience is not None and stale >= patience:
            reason = "early_stop_patience"
            print(f"[{tag}] early stop after {patience} epochs without improvement.", flush=True)
            break
    phase_seconds = time.perf_counter() - phase_started
    print(f"[{tag}] finished {len(phase_rows)} epoch(s) in {phase_seconds:.1f}s reason={reason}", flush=True)
    write_json(done_path, {"rows": phase_rows, "stop_reason": reason, "best_metric": sign * best,
                           "select": select, "phase_seconds": phase_seconds})
    return phase_rows, torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--shifts", type=int, nargs="+", default=[0],
                        help="Pixel shifts sharing all parameters, e.g. --shifts 0 2 (b0/b1/b2 only)")
    parser.add_argument("--init", choices=["balanced", "identity"], default=None,
                        help="Default: balanced for b0/b1/b2, identity for tree1q")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "mnist_data"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260911)
    parser.add_argument("--phase1-epochs", type=int, default=20)
    parser.add_argument("--phase2-epochs", type=int, default=5)
    parser.add_argument("--phase3-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--phase1-lr", type=float, default=2e-3)
    parser.add_argument("--phase2-lr", type=float, default=2e-4)
    parser.add_argument("--core-lr", type=float, default=1e-4)
    parser.add_argument("--readout-lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--aug-degrees", type=float, default=5.0,
                        help="0 together with --aug-translate 0 disables augmentation")
    parser.add_argument("--aug-translate", type=float, default=0.05)
    parser.add_argument("--aug-interpolation", choices=sorted(AUG_INTERPOLATION), default="bilinear")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Data loader workers; try 2-4 if the GPU waits for data")
    parser.add_argument("--no-resume", action="store_true",
                        help="Refuse existing run artifacts; use a new output directory for a fresh run.")
    args = parser.parse_args()
    if min(args.phase1_epochs, args.phase2_epochs, args.phase3_epochs, args.batch_size, args.patience) < 1:
        parser.error("Epoch counts, batch size, and patience must be positive")
    if min(args.phase1_lr, args.phase2_lr, args.core_lr, args.readout_lr) <= 0:
        parser.error("Learning rates must be positive")
    if not 0 <= args.aug_translate < 0.5 or args.aug_degrees < 0:
        parser.error("Invalid augmentation settings")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA selected but PyTorch cannot access CUDA")
    is_cnn = args.variant.startswith("cnn")
    if (is_cnn or args.variant == "tree1q") and args.shifts != [0]:
        parser.error("--shifts is only supported by b0, b1 and b2")
    if is_cnn and args.init is not None:
        parser.error("--init does not apply to CNN variants")

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / args.variant
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    resume = not args.no_resume
    seed_all(args.seed)
    train_loader, validation_loader, generator, train_indices, validation_indices = make_loaders(args, device)
    model = phase_model(args, device, 1)
    counts = model.count_parameters()
    if counts["total"] != EXPECTED_PARAMETERS[args.variant]:
        raise RuntimeError(f"Expected {EXPECTED_PARAMETERS[args.variant]} parameters, got {counts['total']}")
    print(f"Variant {args.variant} | shifts {args.shifts} | parameters {counts} | device {device} | "
          f"outputs {out_dir}", flush=True)
    print(f"Training on {len(train_indices)} examples "
          f"{'with RandomAffine' if augmentation_enabled(args) else 'without augmentation'}; "
          f"validation uses {len(validation_indices)} unaugmented examples.", flush=True)

    config = {
        "model": f"{args.variant}, 16x16 input",
        "parameters": counts,
        "init": args.init or DEFAULT_INIT.get(args.variant),
        "data": {"dataset": "MNIST official train split", "train_size": 50000, "validation_size": 10000,
                 "split_seed": args.split_seed, "official_test_used": False},
        "seed": args.seed, "device": str(device), "output_directory": str(out_dir.resolve()),
        "args": vars(args),
    }
    validate_run_directory(out_dir, config, resume=resume)
    write_json(out_dir / "config_mlp_aug.json", config)
    with (out_dir / "split_indices.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "dataset_index"])
        writer.writerows(("train", index) for index in train_indices)
        writer.writerows(("validation", index) for index in validation_indices)

    rows, offset = [], 0
    started = time.perf_counter()

    # Phase 1: fresh model, select by validation accuracy.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.phase1_lr)
    phase_rows, best_state = train_phase(
        tag="fq_mlp_aug_phase1_pretrain", phase="pretrain", model=model, optimizer=optimizer,
        scheduler=None, train_loader=train_loader, validation_loader=validation_loader, generator=generator,
        epochs=args.phase1_epochs, select="val_accuracy", patience=None, offset=offset,
        out_dir=out_dir, device=device, rows=rows, resume=resume)
    offset += len(phase_rows)

    # Phase 2: continue from phase-1 best with a fresh Adam optimizer.
    train_loader, validation_loader, generator, _, _ = make_loaders(args, device)
    model = phase_model(args, device, 2, best_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.phase2_lr)
    phase_rows, best_state = train_phase(
        tag="fq_mlp_aug_phase2_refine", phase="refine", model=model, optimizer=optimizer,
        scheduler=None, train_loader=train_loader, validation_loader=validation_loader, generator=generator,
        epochs=args.phase2_epochs, select="val_accuracy", patience=None, offset=offset,
        out_dir=out_dir, device=device, rows=rows, resume=resume)
    offset += len(phase_rows)

    # Phase 3: separate core/head LRs and plateau decay; unlike the 2-branch run there is no re-upload.
    train_loader, validation_loader, generator, _, _ = make_loaders(args, device)
    model = phase_model(args, device, 3, best_state)
    core_params, head_params = [], []
    for name, parameter in model.named_parameters():
        (head_params if is_head_parameter(name) else core_params).append(parameter)
    optimizer = torch.optim.Adam([{"params": core_params, "lr": args.core_lr},
                                  {"params": head_params, "lr": args.readout_lr}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-7)
    phase_rows, best_state = train_phase(
        tag="fq_mlp_aug_phase3_final", phase="final", model=model, optimizer=optimizer,
        scheduler=scheduler, train_loader=train_loader, validation_loader=validation_loader, generator=generator,
        epochs=args.phase3_epochs, select="val_loss", patience=args.patience, offset=offset,
        out_dir=out_dir, device=device, rows=rows, resume=resume)

    # The saved checkpoint is the phase-3 best-val-loss state.
    best_row = min(phase_rows, key=lambda item: item["validation_loss"])
    best_accuracy_row = max(rows, key=lambda item: item["validation_accuracy"])
    best_path = out_dir / "full_quantum_mlp_aug_best.pt"
    save_atomic({"model_state_dict": best_state, "row": best_row, "config": config}, best_path)
    summary = {
        "variant": args.variant,
        "shifts": args.shifts,
        "parameters": counts,
        "epochs_completed": len(rows),
        "checkpoint_rule": "final phase, lowest validation loss",
        "best_validation_loss": best_row["validation_loss"],
        "validation_accuracy_at_best_loss": best_row["validation_accuracy"],
        "best_loss_epoch": best_row["epoch"],
        "max_validation_accuracy_any_epoch": best_accuracy_row["validation_accuracy"],
        "max_validation_accuracy_epoch": best_accuracy_row["epoch"],
        "elapsed_seconds": time.perf_counter() - started,
        "best_checkpoint": str(best_path.resolve()),
        "test_set_used": False,
    }
    write_json(out_dir / "summary_mlp_aug.json", summary)
    save_plots(rows, out_dir, f"16x16 {args.variant}, shifts {args.shifts}, {counts['total']} parameters")
    print(f"Best checkpoint: {best_path}", flush=True)
    print(f"Checkpoint (epoch {best_row['epoch']}): val loss {best_row['validation_loss']:.6f}, "
          f"val acc {best_row['validation_accuracy']:.2%}", flush=True)
    print(f"Max validation accuracy over all epochs (not the saved checkpoint): "
          f"{best_accuracy_row['validation_accuracy']:.2%} (epoch {best_accuracy_row['epoch']})", flush=True)


if __name__ == "__main__":
    main()
