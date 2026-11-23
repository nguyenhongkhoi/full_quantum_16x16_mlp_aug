"""Vẽ toàn bộ biểu đồ so sánh từ các thư mục run (history.csv + config.json), mỗi hình lưu cả PNG và PDF (vector).

Một lệnh (đứng ở thư mục gốc repo):
    python .\\plot_all.py
Mặc định quét .\\runs và .\\classical_cnn\\runs, ghi vào .\\figures. Xem plot.md để biết đủ tùy chọn.

Hình cho mỗi dataset (mnist / fashion) có run:
    convergence_<ds>          val acc, val loss, train loss, train acc theo epoch (QCNN nét liền, CNN nét đứt)
    convergence_zoom_<ds>     val acc phóng vào vùng cao (so sát các model)
    acc_vs_params_<ds>        MAX val acc theo số tham số (QCNN: FREE), trục x log
    best_acc_<ds>             cột MAX val acc và CK (val acc ở epoch val loss nhỏ nhất)
    lr_<ds>                   lịch learning rate
    time_<ds>                 thời gian trung bình mỗi epoch
Cho mỗi run QCNN:
    router_<run>              soft / top1 / uniform / oracle / từng cây; g trung bình; tỷ lệ top1; entropy cổng
Bảng: summary_<ds>.csv và summary_<ds>.md (MAX, epoch MAX, CK, val loss nhỏ nhất, tham số, số epoch).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({
    "pdf.fonttype": 42,  # chữ trong PDF giữ dạng text (sửa được trong Illustrator / Inkscape)
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 100,
})
DATASET_NAMES = {"mnist": "MNIST", "fashion": "Fashion-MNIST"}


# ----------------------------------------------------------------------------------------------- đọc run
def read_history(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None) else math.nan) for k, v in row.items()})
    return rows


def load_run(run_dir: Path, label: str | None = None) -> dict | None:
    hist_path = run_dir / "history.csv"
    if not hist_path.is_file():
        return None
    history = read_history(hist_path)
    if not history:
        return None
    config = {}
    if (run_dir / "config.json").is_file():
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    args = config.get("args", {})
    counts = config.get("param_counts", {})
    kind = "qcnn" if "val_acc_top1" in history[0] else "cnn"
    params = counts.get("total_free", counts.get("total")) if counts else None
    return {"dir": run_dir, "label": label or run_dir.name, "kind": kind, "history": history,
            "dataset": args.get("dataset", "mnist"), "params": params, "lr": args.get("lr"),
            "widths": config.get("widths"), "finished": (run_dir / "summary.json").is_file()}


def stats(run: dict) -> dict:
    h = run["history"]
    best = max(h, key=lambda r: r["val_acc"])
    low = min(h, key=lambda r: r["val_loss"])
    return {"max_acc": best["val_acc"] * 100, "max_epoch": int(best["epoch"]), "ck_acc": low["val_acc"] * 100,
            "ck_epoch": int(low["epoch"]), "min_loss": low["val_loss"], "epochs": len(h),
            "sec_per_epoch": sum(r.get("epoch_sec", math.nan) for r in h) / len(h)}


def keep_best_lr(cnn_runs: list[dict]) -> list[dict]:
    """Với CNN cùng cỡ (cùng widths / tham số) chạy nhiều lr, chỉ giữ lr có MAX val cao nhất."""
    groups: dict = {}
    for run in cnn_runs:
        key = (run["dataset"], tuple(run["widths"]) if run["widths"] else run["params"])
        if key not in groups or stats(run)["max_acc"] > stats(groups[key])["max_acc"]:
            groups[key] = run
    return list(groups.values())


def cnn_label(run: dict) -> str:
    w = run["widths"]
    size = f"{run['params']:,}".replace(",", ".") if run["params"] else "?"
    return f"CNN {w[0]}-{w[1]} ({size})" if w else run["label"]


# ----------------------------------------------------------------------------------------------- lưu hình
def save(fig, out_dir: Path, name: str, formats: list[str]) -> None:
    fig.tight_layout()
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=200 if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.{{{','.join(formats)}}}")


def style(run: dict, index: int) -> dict:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {"color": colors[index % len(colors)], "linestyle": "-" if run["kind"] == "qcnn" else "--",
            "linewidth": 1.6 if run["kind"] == "qcnn" else 1.2}


def series(run: dict, key: str, scale: float = 1.0):
    h = run["history"]
    return [r["epoch"] for r in h if key in r], [r[key] * scale for r in h if key in r]


# ----------------------------------------------------------------------------------------------- các hình
def plot_convergence(runs, ds, out_dir, formats):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    panels = [("val_acc", 100, "Validation accuracy (%)"), ("val_loss", 1, "Validation loss"),
              ("train_loss", 1, "Train loss (cross-entropy)"), ("train_acc", 100, "Train accuracy (%)")]
    for ax, (key, scale, title) in zip(axes.flat, panels):
        for i, run in enumerate(runs):
            x, y = series(run, key, scale)
            if y:
                ax.plot(x, y, label=run["label"], **style(run, i))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        if "loss" in key:
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Convergence - {DATASET_NAMES.get(ds, ds)} 16x16 (QCNN solid, classical CNN dashed)")
    save(fig, out_dir, f"convergence_{ds}", formats)


def plot_convergence_zoom(runs, ds, out_dir, formats, low=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    best = max(stats(r)["max_acc"] for r in runs)
    for i, run in enumerate(runs):
        x, y = series(run, "val_acc", 100)
        ax.plot(x, y, label=f"{run['label']}  (max {stats(run)['max_acc']:.2f})", **style(run, i))
    ax.set_ylim(low if low is not None else math.floor(best - 2.0), math.ceil(best + 0.3))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"Validation accuracy (zoom) - {DATASET_NAMES.get(ds, ds)}")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, out_dir, f"convergence_zoom_{ds}", formats)


def plot_acc_vs_params(runs, ds, out_dir, formats):
    runs = [r for r in runs if r["params"]]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for kind, marker, color, name in [("qcnn", "o", "tab:blue", "QCNN (FREE params)"),
                                      ("cnn", "s", "tab:orange", "Classical CNN")]:
        sel = [r for r in runs if r["kind"] == kind]
        if not sel:
            continue
        ax.scatter([r["params"] for r in sel], [stats(r)["max_acc"] for r in sel], marker=marker, s=55,
                   color=color, label=name, zorder=3)
        for r in sel:
            ax.annotate(r["label"], (r["params"], stats(r)["max_acc"]), textcoords="offset points", xytext=(5, 4),
                        fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters (log scale)")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title(f"Accuracy vs parameters - {DATASET_NAMES.get(ds, ds)}")
    ax.legend(fontsize=8)
    save(fig, out_dir, f"acc_vs_params_{ds}", formats)


def plot_best_bar(runs, ds, out_dir, formats):
    runs = sorted(runs, key=lambda r: stats(r)["max_acc"])
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(runs) + 1.5))
    y = range(len(runs))
    ax.barh([i + 0.2 for i in y], [stats(r)["max_acc"] for r in runs], height=0.4, label="MAX val acc",
            color=["tab:blue" if r["kind"] == "qcnn" else "tab:orange" for r in runs])
    ax.barh([i - 0.2 for i in y], [stats(r)["ck_acc"] for r in runs], height=0.4, label="CK (at min val loss)",
            color="lightgray")
    for i, r in enumerate(runs):
        ax.text(stats(r)["max_acc"] + 0.02, i + 0.2, f"{stats(r)['max_acc']:.2f}", va="center", fontsize=7)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["label"] for r in runs], fontsize=8)
    low = min(stats(r)["ck_acc"] for r in runs)
    ax.set_xlim(math.floor(low - 1), 100)
    ax.set_xlabel("Validation accuracy (%)")
    ax.set_title(f"Best validation accuracy - {DATASET_NAMES.get(ds, ds)} (blue QCNN, orange CNN)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, out_dir, f"best_acc_{ds}", formats)


def plot_lr_and_time(runs, ds, out_dir, formats):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, run in enumerate(runs):
        x, y = series(run, "lr")
        if y:
            ax.plot(x, y, label=run["label"], **style(run, i))
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title(f"Learning-rate schedule (ReduceLROnPlateau) - {DATASET_NAMES.get(ds, ds)}")
    ax.legend(fontsize=7)
    save(fig, out_dir, f"lr_{ds}", formats)

    timed = [r for r in runs if not math.isnan(stats(r)["sec_per_epoch"])]
    if timed:
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(timed) + 1.5))
        ax.barh(range(len(timed)), [stats(r)["sec_per_epoch"] / 60 for r in timed],
                color=["tab:blue" if r["kind"] == "qcnn" else "tab:orange" for r in timed])
        ax.set_yticks(range(len(timed)))
        ax.set_yticklabels([r["label"] for r in timed], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Minutes per epoch (log scale; depends on GPU and machine load)")
        ax.set_title(f"Training time - {DATASET_NAMES.get(ds, ds)}")
        save(fig, out_dir, f"time_{ds}", formats)


def plot_router(run, out_dir, formats):
    h = run["history"]
    trees = sorted(int(k[len("val_acc_tree"):]) for k in h[0] if k.startswith("val_acc_tree"))
    names = ["row", "column", "morton"] + [f"tree{i}" for i in range(3, 10)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    for key, label, ls in [("val_acc", "soft (model)", "-"), ("val_acc_top1", "top1", "--"),
                           ("val_acc_uniform", "uniform", "-."), ("val_acc_oracle", "oracle", ":")]:
        x, y = series(run, key, 100)
        ax.plot(x, y, ls, label=label, linewidth=1.6)
    for t in trees:
        x, y = series(run, f"val_acc_tree{t}", 100)
        ax.plot(x, y, linewidth=0.9, alpha=0.7, label=f"only {names[t]}")
    ax.set_title("Validation accuracy by inference mode")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("%")
    ax.legend(fontsize=7)
    ax = axes[1]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for t in trees:
        x, y = series(run, f"gate_mean{t}")
        ax.plot(x, y, color=colors[t % len(colors)], label=f"mean g ({names[t]})")
        x, y = series(run, f"top1_share{t}")
        ax.plot(x, y, "--", color=colors[t % len(colors)], label=f"top1 share ({names[t]})")
    ax.set_title("Router: mean gate and top-1 share per tree")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)
    ax = axes[2]
    x, y = series(run, "gate_entropy")
    ax.plot(x, y, color="purple", label="gate entropy")
    if trees:
        ax.axhline(math.log(len(trees)), color="gray", linestyle=":", label=f"ln {len(trees)} (uniform)")
    ax.set_title("Router: mean per-image gate entropy")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)
    fig.suptitle(f"Router analysis - {run['label']} ({DATASET_NAMES.get(run['dataset'], run['dataset'])})")
    save(fig, out_dir, f"router_{run['label']}".replace(" ", "_"), formats)


def write_summary(runs, ds, out_dir):
    rows = []
    for r in sorted(runs, key=lambda r: -stats(r)["max_acc"]):
        s = stats(r)
        rows.append({"run": r["label"], "kind": r["kind"], "params": r["params"] or "", "max_val_acc": f"{s['max_acc']:.2f}",
                     "max_epoch": s["max_epoch"], "ck_val_acc": f"{s['ck_acc']:.2f}", "min_val_loss": f"{s['min_loss']:.6f}",
                     "ck_epoch": s["ck_epoch"], "epochs": s["epochs"], "finished": r["finished"], "dir": str(r["dir"])})
    with (out_dir / f"summary_{ds}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {DATASET_NAMES.get(ds, ds)} - validation (seed 0)", "",
             "| Run | Loại | Tham số | MAX val | epoch | CK | val loss min | epoch | số epoch | xong |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['kind']} | {r['params']} | {r['max_val_acc']} | {r['max_epoch']} | "
                     f"{r['ck_val_acc']} | {r['min_val_loss']} | {r['ck_epoch']} | {r['epochs']} | {r['finished']} |")
    (out_dir / f"summary_{ds}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  summary_{ds}.csv / .md")


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=[str(ROOT / "runs"), str(ROOT / "classical_cnn" / "runs")],
                        help="Thư mục chứa các run (tìm mọi history.csv bên dưới)")
    parser.add_argument("--add", nargs="*", default=[], metavar="NHAN=THU_MUC_RUN",
                        help="Thêm run lẻ với nhãn riêng, vd S1=D:\\old\\runs\\moe_quantum_samebasis")
    parser.add_argument("--labels", nargs="*", default=[], metavar="TEN_THU_MUC=NHAN",
                        help="Đổi nhãn hiển thị, vd mnist_S1_samebasis=S1")
    parser.add_argument("--include", nargs="*", default=None, help="Chỉ vẽ run có tên chứa một trong các chuỗi này")
    parser.add_argument("--exclude", nargs="*", default=[], help="Bỏ run có tên chứa một trong các chuỗi này")
    parser.add_argument("--dataset", choices=["mnist", "fashion", "all"], default="all")
    parser.add_argument("--all-cnn-lr", action="store_true",
                        help="Vẽ mọi lr của CNN (mặc định mỗi cỡ CNN chỉ giữ lr có MAX val cao nhất)")
    parser.add_argument("--zoom-low", type=float, default=None, help="Cận dưới trục y của hình zoom (mặc định tự chọn)")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="png pdf svg ...")
    parser.add_argument("--out-dir", default=str(ROOT / "figures"))
    args = parser.parse_args()

    rename = dict(item.split("=", 1) for item in args.labels)
    runs = []
    for base in args.runs:
        base = Path(base)
        if not base.is_dir():
            print(f"[bỏ qua] không có thư mục {base}")
            continue
        for hist in sorted(base.rglob("history.csv")):
            run = load_run(hist.parent, rename.get(hist.parent.name))
            if run:
                runs.append(run)
    for item in args.add:
        label, path = item.split("=", 1)
        run = load_run(Path(path), label)
        if run:
            runs.append(run)
        else:
            print(f"[bỏ qua] {path}: không có history.csv")
    if args.include:
        runs = [r for r in runs if any(s in r["dir"].name or s in r["label"] for s in args.include)]
    runs = [r for r in runs if not any(s in r["dir"].name or s in r["label"] for s in args.exclude)]
    if not args.all_cnn_lr:
        runs = [r for r in runs if r["kind"] == "qcnn"] + keep_best_lr([r for r in runs if r["kind"] == "cnn"])
    for r in runs:
        if r["kind"] == "cnn" and r["label"] == r["dir"].name and r["dir"].name not in rename:
            r["label"] = cnn_label(r) + ("" if not args.all_cnn_lr else f" lr{r['lr']:g}")
    if not runs:
        raise SystemExit("Không tìm thấy run nào (cần history.csv). Xem plot.md.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted({r["dataset"] for r in runs}) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        sel = sorted([r for r in runs if r["dataset"] == ds], key=lambda r: (r["kind"] != "qcnn", r["label"]))
        if not sel:
            continue
        print(f"{DATASET_NAMES.get(ds, ds)}: {len(sel)} run -> {out_dir}")
        for r in sel:
            s = stats(r)
            print(f"    {r['kind']:4s} {r['label']:40s} params {r['params'] or '?':>7} | MAX {s['max_acc']:.2f} "
                  f"(ep {s['max_epoch']}) | CK {s['ck_acc']:.2f} | {s['epochs']} epoch")
        plot_convergence(sel, ds, out_dir, args.formats)
        plot_convergence_zoom(sel, ds, out_dir, args.formats, args.zoom_low)
        plot_acc_vs_params(sel, ds, out_dir, args.formats)
        plot_best_bar(sel, ds, out_dir, args.formats)
        plot_lr_and_time(sel, ds, out_dir, args.formats)
        for r in sel:
            if r["kind"] == "qcnn":
                plot_router(r, out_dir, args.formats)
        write_summary(sel, ds, out_dir)
    print("Xong.")


if __name__ == "__main__":
    main()
