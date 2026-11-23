"""Đánh giá checkpoint đã chọn trên tập TEST chính thức của MNIST. Chỉ chạy MỘT lần, cho bản chốt cuối.

--mode chọn đầu ra (mặc định soft, đầu ra đã train). Chỉ chế độ này được tính; chọn chế độ trước khi chạy.
Ghi thêm tỷ lệ top1 chọn mỗi cây theo từng lớp (không phải accuracy).

  python .\\evaluate_test.py --checkpoint .\\runs\\moe_quantum\\model_best_val_acc.pt --mode soft --data-root .\\mnist_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import DATASET_NAMES, DATASETS
from models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", default="soft", help="soft, top1, uniform or tree<i>")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default=None,
                        help="Mặc định: dataset lúc train (đọc từ checkpoint)")
    parser.add_argument("--data-root", default="./mnist_data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=None, help="Output JSON (default next to the checkpoint)")
    args = parser.parse_args()

    path = Path(args.checkpoint).resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(dict(state["config"]["args"]))
    model.load_state_dict(state["model_state_dict"])
    if model.fusion is not None and args.mode != "soft":
        raise SystemExit("--expert-head fusion has only the soft output; use --mode soft")
    device = torch.device(args.device)
    model.to(device).eval()
    print(f"Loaded {path} (epoch {state.get('epoch')}); parameters {model.count_parameters()['total']}; "
          f"mode {args.mode}")

    dataset = args.dataset or state["config"]["args"].get("dataset", "mnist")
    print(f"Test set: {DATASET_NAMES[dataset]} (official test, 10.000 ảnh)")
    test = DATASETS[dataset](root=args.data_root, train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    experts = len(model.orders)
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    routed = torch.zeros(10, experts, dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            output = model(images, mode=args.mode)
            predicted = output.argmax(dim=1).cpu()
            confusion += torch.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)
            chosen = model.last["gates"].argmax(dim=1).cpu()
            routed += torch.bincount(labels * experts + chosen, minlength=10 * experts).reshape(10, experts)
    correct = int(confusion.diagonal().sum())
    total = int(confusion.sum())
    result = {"checkpoint": str(path), "mode": args.mode, "test_size": total, "correct": correct,
              "accuracy_pct": round(correct / total * 100, 2),
              "per_class_accuracy": {str(d): confusion[d, d].item() / max(confusion[d].sum().item(), 1)
                                     for d in range(10)},
              "confusion_matrix": confusion.tolist(), "top1_route_counts_per_class": routed.tolist()}
    output = Path(args.output) if args.output else path.with_name(f"official_test_{args.mode}.json")
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("=" * 60)
    print(f"OFFICIAL TEST ({args.mode}): {correct / total * 100:.2f}% ({correct}/{total})")
    print(f"Saved {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
