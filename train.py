"""Train encoder qconv4x3 + router (mixture of experts) + các cây level32 + head tuyến tính, MNIST tĩnh 16x16.

Cùng điều kiện train với run 2c / L1 (chép từ qcnn_qconv4x3/train.py): split 50.000 / 10.000 theo --split-seed, aug
xoay +/-5 độ dịch +/-0,05, Adam, ReduceLROnPlateau theo val loss, dừng sớm theo val loss, 50 epoch.
Loss train = cross entropy (đầu ra soft) + loss phụ của router (--balance-weight, --entropy-weight).
Val loss / val acc / chọn checkpoint dùng đầu ra soft. Mỗi epoch còn ghi: acc top1, uniform, từng cây, oracle
(ít nhất một cây đúng), trung bình g và tỷ lệ được chọn top1 của từng cây, entropy của g.

Đầu ra trong --out-dir: history.csv, config.json, summary.json, convergence.png, train_log.txt,
model_best_val_acc.pt (MAX), model_best_val_loss.pt (CK), model_last.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import AUG_INTERPOLATION, collect, make_loaders, seed_all, summarize
from models import ANSATZE, DEFAULT_OBS, EXPERT_HEADS, FRONT_READOUTS, MIXES, ROUTER_SOURCES, ROUTERS, build_model

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


def train_epoch(model, loader: DataLoader, device: torch.device, optimizer, args, epoch: int) -> dict:
    model.train()
    total_ce = total_aux = 0.0
    correct = total = 0
    num_batches = len(loader)
    t_start = time.time()
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad()
        output = model(images, mode="soft")
        ce = F.cross_entropy(output, labels)
        aux = model.aux_loss(args.balance_weight, args.entropy_weight)
        (ce + aux).backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_ce += ce.item() * batch_size
        total_aux += float(aux) * batch_size
        correct += (output.argmax(dim=1) == labels).sum().item()
        total += batch_size
        if PROGRESS and (batch_idx % 10 == 0 or batch_idx == num_batches):
            sec_per_it = (time.time() - t_start) / batch_idx
            eta_m, eta_s = divmod(int((num_batches - batch_idx) * sec_per_it), 60)
            status(f"Ep {epoch}/{args.epochs} Train {batch_idx:4d}/{num_batches} "
                   f"({batch_idx / num_batches * 100:5.1f}%) | ce {total_ce / total:.4f} | aux {total_aux / total:.4f} | "
                   f"acc {correct / total * 100:5.2f}% | {sec_per_it:4.2f}s/it | ETA {eta_m:02d}:{eta_s:02d}")
    return {"train_loss": total_ce / max(total, 1), "train_aux": total_aux / max(total, 1),
            "train_acc": correct / max(total, 1)}


def save_plot(history: list[dict], experts: int, out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="Train CE", color="royalblue")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="Val Loss (soft)", color="crimson")
    ax1.set_xlabel("Epoch"), ax1.set_title("Loss"), ax1.grid(True, linestyle="--", alpha=0.5), ax1.legend()
    for key, label in (("val_acc", "soft"), ("val_acc_top1", "top1"), ("val_acc_uniform", "uniform"),
                       ("val_acc_oracle", "oracle")):
        ax2.plot(epochs, [h[key] * 100 for h in history], label=label)
    for index in range(experts):
        ax2.plot(epochs, [h[f"val_acc_tree{index}"] * 100 for h in history], linestyle=":", label=f"tree{index}")
    ax2.set_xlabel("Epoch"), ax2.set_title("Validation accuracy (%)"), ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    for index in range(experts):
        ax3.plot(epochs, [h[f"gate_mean{index}"] for h in history], label=f"mean g{index}")
        ax3.plot(epochs, [h[f"top1_share{index}"] for h in history], linestyle="--", label=f"top1 share {index}")
    ax3.plot(epochs, [h["gate_entropy"] for h in history], color="black", label="entropy / log E")
    ax3.set_xlabel("Epoch"), ax3.set_title("Router (validation)"), ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--front", default="qconv4x3",
                        help="Encoder: qconv<L>[x<S>][h<H>][f] (quantum, e.g. qconv4x3), cnn<C>[p] / deep<C>[p] "
                             "(classical), or none")
    parser.add_argument("--front-readout", choices=FRONT_READOUTS, default="linear",
                        help="linear: Linear(27, 2) per stage as in front.py (qconv4x3: 492); direct: no classical "
                             "readout, the centre qubit itself goes on (qconv4x3: 324 circuit angles)")
    parser.add_argument("--orders", nargs="+", default=["row", "column", "morton"],
                        help="Scan order of each tree = one expert each (row, column, morton, morton_shift<N>, "
                             "angle<D>); e.g. row row row for the same-scan ablation")
    parser.add_argument("--side-level", type=int, default=32, help="Tree level whose Bloch vectors are read (0: none)")
    parser.add_argument("--expert-head", choices=EXPERT_HEADS, default="linear",
                        help="linear: Linear(372, 10) per tree (11,190); shared: one Linear for all trees (3,730); "
                             "mlp: Linear(372, H) GELU Dropout Linear(H, 10) per tree; bloch8: Linear(24, 10) on X, Y, "
                             "Z of the 8 final qubits (750); obs10: 10 fixed observables, scale + bias (60); "
                             "fusion: the 2c MLP head (needs --router none; 11,948 with projection 10); nocorr: "
                             "Linear(120, 10) (3,630); nocorr-z: Linear(56, 10), only <Z> of the side level (1,710); "
                             "nocorr-shared: one Linear(120, 10) (1,210); nocorr-lowrank: rank --head-rank (r=4: 1,590); "
                             "samebasis: Linear(204, 10), same-axis pairs XX/YY/ZZ only (6,150); zz: Linear(148, 10), "
                             "ZZ pairs only (4,470)")
    parser.add_argument("--head-rank", type=int, default=4, help="Rank r of --expert-head nocorr-lowrank")
    parser.add_argument("--share-final", action="store_true",
                        help="All trees share one final 8-qubit block (4 strongly-entangling layers); conv-pool "
                             "stays per tree. Saves 2 x 640 free parameters with 3 trees")
    parser.add_argument("--share-gates", action="store_true",
                        help="Final 8-qubit block: each layer uses ONE 1-qubit unitary for all 8 qubits and ONE "
                             "2-qubit unitary for all 8 ring pairs (20 instead of 160 free parameters per layer)")
    parser.add_argument("--expert-hidden", type=int, default=16, help="H of --expert-head mlp")
    parser.add_argument("--projection", type=int, default=10, help="Per-tree projection of --expert-head fusion")
    parser.add_argument("--hidden", type=int, default=16, help="Fusion hidden size of --expert-head fusion")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout of --expert-head mlp / fusion")
    parser.add_argument("--obs-terms", nargs="+", default=list(DEFAULT_OBS),
                        help="--expert-head obs10: 10 Pauli observables of the 8 final qubits, e.g. Z0 ... Z7 X0 X4 "
                             "or pairs like Z0Z1")
    parser.add_argument("--router", choices=ROUTERS, default="quantum",
                        help="none: uniform mean (= linhead L1); quantum: circuit; linear / mlp: classical")
    parser.add_argument("--router-source", choices=ROUTER_SOURCES, default="encoder",
                        help="encoder: router reads the encoder's theta, phi; pixels: the resized image, phi = 0")
    parser.add_argument("--router-detach", action="store_true",
                        help="Router gradients do not flow back into the encoder")
    parser.add_argument("--router-grid", type=int, nargs=2, default=[4, 4], help="Pooling grid gh gw (cells)")
    parser.add_argument("--router-qubits", type=int, default=8, help="Quantum router: qubits (cells %% Q == 0)")
    parser.add_argument("--router-layers", type=int, default=1, help="Quantum router: trainable layers per upload")
    parser.add_argument("--router-ansatz", choices=ANSATZE, default="se",
                        help="se: learned 1- and 2-qubit unitaries (40Q per layer); rot: RZ RY RZ + CZ ring (3Q)")
    parser.add_argument("--router-init-scale", type=float, default=0.01, help="Quantum router: initial gate scale")
    parser.add_argument("--router-hidden", type=int, default=18, help="Router mlp hidden size (18: 651 parameters)")
    parser.add_argument("--mix", choices=MIXES, default="logits",
                        help="logits: sum g_e * logits_e; probs: log sum g_e softmax(logits_e)")
    parser.add_argument("--gate-noise", type=float, default=0.0, help="Std of noise on router logits (training)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="./mnist_data", help="Root directory for MNIST dataset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260911)
    parser.add_argument("--aug-degrees", type=float, default=5.0, help="Rotation range in degrees (+/-)")
    parser.add_argument("--aug-translate", type=float, default=0.05, help="Translation range (+/-)")
    parser.add_argument("--aug-interpolation", choices=list(AUG_INTERPOLATION.keys()), default="bilinear")
    add_model_arguments(parser)
    parser.add_argument("--balance-weight", type=float, default=0.01, help="Load-balancing loss weight")
    parser.add_argument("--entropy-weight", type=float, default=0.0,
                        help="Per-image gate entropy weight (> 0: sharper routing, helps top1)")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam")
    parser.add_argument("--router-lr", type=float, default=None, help="Separate learning rate for the router")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on val loss")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau decay factor")
    parser.add_argument("--lr-patience", type=int, default=3, help="ReduceLROnPlateau patience")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--out-dir", default="./runs/moe_quantum", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    global _LOG
    _LOG = (out_dir / "train_log.txt").open("a", encoding="utf-8")
    log(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} train.py {' '.join(sys.argv[1:])}")

    seed_all(args.seed)
    device = torch.device(args.device)
    log("=" * 70)
    log("QCNN level32 + quanvolution encoder + mixture-of-experts router, Static MNIST with augmentation")
    log("=" * 70)
    log(f"Device: {device} | Front end: {args.front} (readout {args.front_readout}) | Trees (experts): {args.orders} "
        f"| side_level={args.side_level}")
    head_detail = {"mlp": f" (hidden {args.expert_hidden}, dropout {args.dropout})",
                   "fusion": f" (projection {args.projection}, hidden {args.hidden}, dropout {args.dropout})",
                   "obs10": f" ({' '.join(args.obs_terms)})",
                   "nocorr-lowrank": f" (rank {args.head_rank})"}.get(args.expert_head, "")
    if args.share_final:
        log("Trees share ONE final 8-qubit block (--share-final)")
    if args.share_gates:
        log("Final 8-qubit block: one shared 1-qubit and one shared 2-qubit unitary per layer (--share-gates)")
    log(f"Expert head: {args.expert_head}{head_detail}"
        + (" (side level not measured)" if args.expert_head in ("bloch8", "obs10") else ""))
    log(f"Router: {args.router} | source={args.router_source} detach={args.router_detach} grid={args.router_grid}"
        + (f" qubits={args.router_qubits} layers/upload={args.router_layers} ansatz={args.router_ansatz} "
           f"init={args.router_init_scale}" if args.router == "quantum" else "")
        + (f" hidden={args.router_hidden}" if args.router == "mlp" else ""))
    log(f"Mix: {args.mix} | gate noise {args.gate_noise} | balance weight {args.balance_weight} | "
        f"entropy weight {args.entropy_weight} | router lr {args.router_lr or args.lr}")
    log(f"Augmentation: +/-{args.aug_degrees} deg, +/-{args.aug_translate} translate, {args.aug_interpolation}")
    log(f"Output directory: {out_dir}")

    model = build_model(vars(args)).to(device)
    experts = len(args.orders)
    counts = model.count_parameters()
    log("-" * 70)
    log(f"Parameter counts: Trees={counts['quantum']}, Front={counts['front']} (circuit angles "
        f"{counts['front_quantum']}), Router={counts['router']} (circuit {counts['router_quantum']}), "
        f"Head={counts['head']}, Total={counts['total']}, Quantum share={counts['quantum_share'] * 100:.1f}%")
    log(f"FREE parameters (learned unitaries counted by real degrees of freedom, 16 / 4): Trees="
        f"{counts['quantum_free']}, Router circuit={counts['router_quantum_free']}, Total={counts['total_free']}, "
        f"Quantum share={counts['quantum_share_free'] * 100:.1f}% | without global phase (15 / 3): Total="
        f"{counts['total_free_nophase']}, Quantum share={counts['quantum_share_free_nophase'] * 100:.1f}%")
    log("-" * 70)

    config = {"args": vars(args), "param_counts": counts, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    train_loader, val_loader = make_loaders(args, device)
    log(f"Loaded: 50,000 train images, 10,000 val images (batch size {args.batch_size})")

    if args.router_lr is not None and model.router is not None:
        router_ids = {id(parameter) for parameter in model.router.parameters()}
        groups = [{"params": [p for p in model.parameters() if id(p) not in router_ids]},
                  {"params": list(model.router.parameters()), "lr": args.router_lr}]
    else:
        groups = model.parameters()
    optimizer = torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)

    history: list[dict] = []
    best_val_acc, best_val_loss = 0.0, float("inf")
    best_acc_epoch = best_loss_epoch = -1
    patience_counter = 0

    log("")
    log("One line per epoch. soft = trained output; top1 = only the tree with the largest gate; uni = uniform mean; "
        "orc = some tree right; g = mean gate per tree; H = gate entropy / log E.")
    log("*best val acc -> model_best_val_acc.pt (MAX); *best val loss -> model_best_val_loss.pt (CK)")
    log("-" * 110)

    def val_progress(index: int, total: int) -> None:
        if PROGRESS and (index % 10 == 0 or index == total):
            status(f"Ep {epoch}/{args.epochs} Val {index:4d}/{total}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train = train_epoch(model, train_loader, device, optimizer, args, epoch)
        val = summarize(model, collect(model, val_loader, device, val_progress))
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        val_loss, val_acc = val["loss"], val["acc_soft"]
        scheduler.step(val_loss)

        record = {"epoch": epoch, **{k: round(v, 6) for k, v in train.items()},
                  "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 6),
                  "val_acc_top1": round(val["acc_top1"], 6), "val_acc_uniform": round(val["acc_uniform"], 6),
                  "val_acc_oracle": round(val["acc_oracle"], 6)}
        for index in range(experts):
            record[f"val_acc_tree{index}"] = round(val[f"acc_tree{index}"], 6)
        for index in range(experts):
            record[f"gate_mean{index}"] = round(val[f"gate_mean{index}"], 6)
            record[f"top1_share{index}"] = round(val[f"top1_share{index}"], 6)
        record["gate_entropy"] = round(val["gate_entropy"], 6)
        record["lr"] = current_lr
        record["epoch_sec"] = round(elapsed, 2)
        history.append(record)
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            writer.writeheader()
            writer.writerows(history)

        marks = (("  *best val acc" if val_acc > best_val_acc else "")
                 + ("  *best val loss" if val_loss < best_val_loss else ""))
        trees = " ".join(f"{val[f'acc_tree{i}'] * 100:.2f}" for i in range(experts))
        gates = " ".join(f"{val[f'gate_mean{i}']:.2f}" for i in range(experts))
        log(f"Ep {epoch:3d} | train ce {train['train_loss']:.4f} aux {train['train_aux']:.4f} acc "
            f"{train['train_acc'] * 100:.2f} | val loss {val_loss:.6f} soft {val_acc * 100:.2f} top1 "
            f"{val['acc_top1'] * 100:.2f} uni {val['acc_uniform'] * 100:.2f} orc {val['acc_oracle'] * 100:.2f} "
            f"trees {trees} | g {gates} H {val['gate_entropy']:.2f} | lr {current_lr:.2e} | {elapsed:.0f}s{marks}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc, "val_loss": val_loss,
                 "config": config}
        if val_acc > best_val_acc:
            best_val_acc, best_acc_epoch = val_acc, epoch
            torch.save(state, out_dir / "model_best_val_acc.pt")
        if val_loss < best_val_loss:
            best_val_loss, best_loss_epoch = val_loss, epoch
            patience_counter = 0
            torch.save(state, out_dir / "model_best_val_loss.pt")
        else:
            patience_counter += 1
        torch.save(state, out_dir / "model_last.pt")
        save_plot(history, experts, out_dir / "convergence.png")
        if patience_counter >= args.patience:
            log(f"[Early Stopping] No improvement in val loss for {args.patience} epochs. Stopping at epoch {epoch}.")
            break

    at = {h["epoch"]: h for h in history}
    log("=" * 75)
    log("Training Completed!")
    log(f"Best Val Acc (soft):  {best_val_acc * 100:.2f}% (Epoch {best_acc_epoch}); top1 there "
        f"{at[best_acc_epoch]['val_acc_top1'] * 100:.2f}% -> model_best_val_acc.pt")
    log(f"Best Val Loss: {best_val_loss:.6f} (Epoch {best_loss_epoch}), val acc there (CK) "
        f"{at[best_loss_epoch]['val_acc'] * 100:.2f}%, top1 {at[best_loss_epoch]['val_acc_top1'] * 100:.2f}% "
        f"-> model_best_val_loss.pt")
    log(f"Max val acc top1 over epochs: {max(h['val_acc_top1'] for h in history) * 100:.2f}% (reference only)")
    log("=" * 75)

    summary = {"best_val_acc": best_val_acc, "best_acc_epoch": best_acc_epoch, "best_val_loss": best_val_loss,
               "best_loss_epoch": best_loss_epoch, "ck_val_acc": at[best_loss_epoch]["val_acc"],
               "top1_at_best_acc": at[best_acc_epoch]["val_acc_top1"],
               "top1_at_best_loss": at[best_loss_epoch]["val_acc_top1"],
               "total_epochs": len(history), "total_time_sec": sum(h["epoch_sec"] for h in history),
               "param_counts": counts}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
