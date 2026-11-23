"""Phân tích một checkpoint trên tập VALIDATION (cùng split với lúc train; không dùng test).

In và lưu:
  - accuracy các chế độ: soft, top1, uniform, từng cây, oracle (ít nhất một cây đúng = trần của mọi router chọn 1 cây);
  - theo từng lớp (0-9): accuracy của từng cây, soft, top1; trung bình g và tỷ lệ top1 chọn mỗi cây;
  - từng cặp cây: tỷ lệ ảnh hai cây đoán khác nhau, tỷ lệ chỉ một trong hai đúng.
Dùng được cho checkpoint của folder này và checkpoint L1 / L2 của full_quantum_16x16_level32_qconv_linhead
(--head linear3 / linear-shared, đọc như router none): chạy trên L1 trước khi train MoE để xem router có chỗ để giúp
không (oracle - uniform lớn thì có; nhỏ thì không).

Ví dụ:
  python .\\analyze_routing.py --checkpoint .\\runs\\moe_quantum\\model_best_val_acc.pt --data-root .\\mnist_data
Đầu ra: <tên checkpoint>_routing.json và _routing_per_class.csv cạnh checkpoint (hoặc --out-prefix).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import collect, make_loaders, predictions, summarize
from models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-root", default="./mnist_data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-prefix", default=None, help="Output path prefix (default: next to the checkpoint)")
    args = parser.parse_args()

    path = Path(args.checkpoint).resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(state["config"]["args"])
    model = build_model(cfg)
    model.load_state_dict(state["model_state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    print(f"Checkpoint {path} (epoch {state.get('epoch')}), router={getattr(model, 'router_kind', '?')}, "
          f"trees={list(model.orders)}, mix={model.mix}")

    data_args = SimpleNamespace(split_seed=cfg.get("split_seed", 20260911), seed=cfg.get("seed", 0),
                                aug_degrees=0.0, aug_translate=0.0, aug_interpolation="bilinear",
                                data_root=args.data_root, batch_size=args.batch_size, num_workers=0,
                                dataset=cfg.get("dataset", "mnist"))
    _, val_loader = make_loaders(data_args, device)
    data = collect(model, val_loader, device)
    overall = summarize(model, data)
    preds = predictions(model, data)
    labels, gates = data["labels"], data["gates"]
    experts = gates.shape[1] if data["expert_logits"] is not None else 0  # head fusion: không có logit từng cây
    chosen = gates.argmax(dim=1)
    if experts == 0:
        print()
        print(f"Head fusion (no per-tree logits): val loss {overall['loss']:.6f}, acc {overall['acc_soft'] * 100:.2f}%")
        for digit in range(10):
            mask = labels == digit
            print(f"  {digit}: {(preds['soft'][mask] == digit).float().mean().item() * 100:6.2f}%")
        return

    print("\nValidation (10,000 images):")
    print(f"  loss (soft) {overall['loss']:.6f}")
    for mode in ["soft", "top1", "uniform", *[f"tree{i}" for i in range(experts)], "oracle"]:
        print(f"  acc {mode:8s} {overall[f'acc_{mode}'] * 100:6.2f}%")
    print(f"  oracle - uniform = {(overall['acc_oracle'] - overall['acc_uniform']) * 100:+.2f} points "
          "(room a per-image choice of one tree could use at most)")
    print("  gates: " + "  ".join(f"tree{i} mean {overall[f'gate_mean{i}']:.3f} top1 {overall[f'top1_share{i}']:.3f}"
                                   for i in range(experts)) + f"  entropy/logE {overall['gate_entropy']:.3f}")

    rows = []
    print("\nPer class: acc of each tree | soft | top1 || mean g per tree | top1 share per tree")
    for digit in range(10):
        mask = labels == digit
        row = {"class": digit, "count": int(mask.sum())}
        for i in range(experts):
            row[f"acc_tree{i}"] = (preds[f"tree{i}"][mask] == digit).float().mean().item()
        row["acc_soft"] = (preds["soft"][mask] == digit).float().mean().item()
        row["acc_top1"] = (preds["top1"][mask] == digit).float().mean().item()
        for i in range(experts):
            row[f"gate_mean{i}"] = gates[mask, i].mean().item()
            row[f"top1_share{i}"] = (chosen[mask] == i).float().mean().item()
        rows.append(row)
        trees = " ".join(f"{row[f'acc_tree{i}'] * 100:6.2f}" for i in range(experts))
        means = " ".join(f"{row[f'gate_mean{i}']:.2f}" for i in range(experts))
        shares = " ".join(f"{row[f'top1_share{i}']:.2f}" for i in range(experts))
        print(f"  {digit} (n={row['count']:4d}): {trees} | {row['acc_soft'] * 100:6.2f} | {row['acc_top1'] * 100:6.2f}"
              f" || {means} | {shares}")

    pairs = {}
    print("\nPairs of trees: disagree = predictions differ; one_right = exactly one of the two is right")
    for i in range(experts):
        for j in range(i + 1, experts):
            a, b = preds[f"tree{i}"], preds[f"tree{j}"]
            disagree = (a != b).float().mean().item()
            one_right = ((a == labels) ^ (b == labels)).float().mean().item()
            pairs[f"tree{i}-tree{j}"] = {"disagree": disagree, "one_right": one_right}
            print(f"  tree{i} ({model.orders[i]}) vs tree{j} ({model.orders[j]}): disagree {disagree * 100:.2f}%, "
                  f"one right {one_right * 100:.2f}%")

    prefix = Path(args.out_prefix) if args.out_prefix else path.with_suffix("")
    with open(f"{prefix}_routing.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(path), "epoch": state.get("epoch"), "orders": list(model.orders),
                   "overall": overall, "pairs": pairs, "per_class": rows}, f, indent=2)
    with open(f"{prefix}_routing_per_class.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {prefix}_routing.json and {prefix}_routing_per_class.csv")


if __name__ == "__main__":
    main()
