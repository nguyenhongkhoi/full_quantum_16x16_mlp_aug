"""Charts for the old static-MNIST runs (3 phases, 5 or 8 degree aug, one validation split, seed 0).

Reads one history_full_quantum_mlp_aug.csv per run, named <run_id>.csv, and DANH_MUC_RUN.csv for groups and
parameter counts. Writes bieu_do/tong_hop_cu.csv and three PNG charts. Validation accuracy only, no test numbers.

    python plot_ket_qua_cu.py --csv-dir D:\\data_system\\Download\\ket_qua_csv --out-dir bieu_do
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Fixed categorical order, one hue per group, checked with the dataviz validator (light surface).
GROUPS = {
    "CNN": ("#2a78d6", "o"),
    "QCNN cây": ("#eb6834", "s"),
    "Cây bond-2": ("#1baf7a", "^"),
    "Ô vướng víu": ("#eda100", "D"),
    "Quanvolution": ("#e87ba4", "v"),
    "Đối chứng cổ điển": ("#008300", "P"),
    "Đối chứng không vướng víu": ("#6250d6", "X"),
}
PHASE_ENDS = (20, 25)  # pretrain ends at epoch 20, refine at 25, for every runner here
# Panels of the curve chart. The s9 controls share a panel with s9, since they are compared with it.
PANELS = {
    "CNN": ["CNN"],
    "QCNN cây": ["QCNN cây"],
    "Cây bond-2 và ô vướng víu": ["Cây bond-2", "Ô vướng víu"],
    "s9_meas và các đối chứng": ["Quanvolution", "Đối chứng cổ điển", "Đối chứng không vướng víu"],
}
# The tree panel holds close curves, so each run there gets its own hue, in the fixed categorical order.
RUN_COLORS = {"R0": "#2a78d6", "balanced": "#eb6834", "fusion2": "#1baf7a", "fusion3": "#eda100", "tree1q": "#e87ba4"}
# Label offsets in points, only where two labels would overlap.
LABEL_OFFSETS = {"R0": (6, -12), "balanced": (6, 5), "c1_encoding": (-6, -12), "b2": (6, -12),
                 "s9_meas": (6, -12), "c2_product": (6, 5)}


def read_history(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: (value if key == "phase" else float(value)) for key, value in row.items()}
                for row in csv.DictReader(handle)]


def summarize(rows: list[dict]) -> dict:
    """CK: the epoch with the lowest validation loss. MAX: the highest validation accuracy."""
    best = min(rows, key=lambda row: row["validation_loss"])
    top = max(rows, key=lambda row: row["validation_accuracy"])
    phase = lambda end: max(row["validation_accuracy"] for row in rows if row["epoch"] <= end)  # noqa: E731
    return {"epochs": int(rows[-1]["epoch"]), "ck": 100 * best["validation_accuracy"], "ck_epoch": int(best["epoch"]),
            "max": 100 * top["validation_accuracy"], "max_epoch": int(top["epoch"]),
            "phase1": 100 * phase(PHASE_ENDS[0]), "phase2": 100 * phase(PHASE_ENDS[1]),
            "seconds": sum(row["epoch_seconds"] for row in rows) / len(rows)}


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK_2)
    axis.tick_params(colors=INK_2, labelsize=9)


def scatter_params(runs: dict, out: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6.5), facecolor=SURFACE)
    style(axis)
    for group, (color, marker) in GROUPS.items():
        members = [run for run in runs.values() if run["group"] == group and run["params"]]
        if not members:
            continue
        axis.scatter([run["params"] for run in members], [run["ck"] for run in members], s=70, marker=marker,
                     color=color, edgecolors=SURFACE, linewidths=2, label=group, zorder=3)
        for run in members:
            dx, dy = LABEL_OFFSETS.get(run["name"], (6, 4))
            axis.annotate(run["name"], (run["params"], run["ck"]), xytext=(dx, dy), textcoords="offset points",
                          fontsize=8, color=INK_2, ha="right" if dx < 0 else "left")
    axis.set_xscale("log")
    axis.set_xlabel("Tổng số tham số (thang log)", color=INK)
    axis.set_ylabel("Validation accuracy tại CK (%)", color=INK)
    axis.set_title("Accuracy theo số tham số, MNIST 16×16, một seed", color=INK, loc="left")
    axis.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    figure.text(0.01, 0.01, "CK: epoch có validation loss thấp nhất. Chênh dưới khoảng 0,3 điểm chưa đủ kết luận. "
                "Run chưa xong vẫn được vẽ tại epoch cuối có trong CSV.", fontsize=8, color=INK_2)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def spread(values: list[float], gap: float) -> list[float]:
    """Move end labels apart so that no two are closer than `gap`, keeping their order."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    placed = list(values)
    for previous, current in zip(order, order[1:]):
        placed[current] = max(placed[current], placed[previous] + gap)
    return placed


def curves(runs: dict, histories: dict, out: Path) -> None:
    panels = {title: groups for title, groups in PANELS.items()
              if any(run["group"] in groups for run in runs.values())}
    figure, axes = plt.subplots(len(panels), 1, figsize=(10, 3.6 * len(panels)), facecolor=SURFACE, sharex=True)
    for axis, (title, groups) in zip(axes, panels.items()):
        style(axis)
        members = sorted((run for run in runs.values() if run["group"] in groups), key=lambda run: -run["ck"])
        # Runs of one group share its color, so line style tells them apart, with a label at the line end.
        styles = {}
        for run in members:
            line_styles = ("-", "--", ":", "-.", (0, (6, 2, 1, 2, 1, 2)))
            same_group = sum(1 for name in styles if runs[name]["group"] == run["group"])
            styles[run["name"]] = line_styles[same_group % len(line_styles)]
        ends = [100 * histories[run["name"]][-1]["validation_accuracy"] for run in members]
        for run, label_y in zip(members, spread(ends, 0.28)):
            rows = histories[run["name"]]
            own_color = run["name"] in RUN_COLORS
            axis.plot([row["epoch"] for row in rows], [100 * row["validation_accuracy"] for row in rows],
                      color=RUN_COLORS.get(run["name"], GROUPS[run["group"]][0]), linewidth=2,
                      linestyle="-" if own_color else styles[run["name"]], label=run["name"])
            axis.annotate(run["name"], (rows[-1]["epoch"], label_y), xytext=(4, 0), textcoords="offset points",
                          fontsize=8, color=INK_2, va="center")
        for end in PHASE_ENDS:
            axis.axvline(end, color=INK_2, linewidth=0.8, linestyle=(0, (2, 3)))
        axis.set_ylim(92, 99.6)
        axis.set_ylabel("Val accuracy (%)", color=INK, fontsize=9)
        axis.set_title(title, color=INK, loc="left", fontsize=10)
        axis.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right", ncol=2)
    axes[-1].set_xlabel("Epoch (vạch đứt: hết phase 1 ở epoch 20, hết phase 2 ở epoch 25)", color=INK)
    figure.tight_layout()
    figure.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(figure)


def control_bars(runs: dict, out: Path) -> None:
    """The s9 controls are compared at the end of phase 2, the only point every one of them reached.

    Dots, not bars: the axis does not start at 0, and a bar's length would then mislead.
    """
    names = ["c0_pixel", "c1_encoding", "c2_product", "c4_local_mlp", "s9_meas"]
    members = [runs[name] for name in names if name in runs]
    members.sort(key=lambda run: run["phase2"])
    figure, axis = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    style(axis)
    axis.grid(True, axis="x", color=GRID)
    axis.grid(False, axis="y")
    positions = range(len(members))
    for position, run in zip(positions, members):
        color, marker = GROUPS[run["group"]]
        axis.scatter(run["phase2"], position, s=110, marker=marker, color=color, edgecolors=SURFACE, linewidths=2,
                     zorder=3)
        axis.text(run["phase2"] + 0.08, position, f"{run['phase2']:.2f}%  ({run['params']:,} tham số, {run['group']})"
                  .replace(",", ".", 1), va="center", fontsize=9, color=INK)
    axis.set_yticks(list(positions), [run["name"] for run in members], color=INK)
    axis.set_xlim(95, 99.6)
    axis.set_ylim(-0.6, len(members) - 0.4)
    axis.set_xlabel("Val accuracy cao nhất đến hết phase 2 (%)", color=INK)
    axis.set_title("Bỏ dần phần quantum của s9_meas, cùng ô ảnh và cùng head", color=INK, loc="left")
    figure.tight_layout()
    figure.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).with_name("DANH_MUC_RUN.csv"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.catalog.open(newline="", encoding="utf-8") as handle:
        catalog = {row["run_id"]: row for row in csv.DictReader(handle)}
    runs, histories = {}, {}
    for path in sorted(args.csv_dir.glob("*.csv")):
        name = path.stem
        if name not in catalog or catalog[name]["giao_thuc"] != "cu_3phase_aug":
            print(f"skip {path.name}: not an old static-MNIST run in DANH_MUC_RUN.csv")
            continue
        histories[name] = read_history(path)
        params = catalog[name]["tong_tham_so"]
        runs[name] = {"name": name, "group": catalog[name]["nhom"], "params": int(params) if params else 0,
                      **summarize(histories[name])}

    with (args.out_dir / "tong_hop_cu.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "nhom", "tham_so", "epoch_cuoi", "ck", "epoch_ck", "max", "epoch_max",
                         "max_het_phase1", "max_het_phase2", "giay_moi_epoch"])
        for run in sorted(runs.values(), key=lambda run: -run["ck"]):
            writer.writerow([run["name"], run["group"], run["params"], run["epochs"], f"{run['ck']:.2f}",
                             run["ck_epoch"], f"{run['max']:.2f}", run["max_epoch"], f"{run['phase1']:.2f}",
                             f"{run['phase2']:.2f}", f"{run['seconds']:.1f}"])
    scatter_params(runs, args.out_dir / "1_accuracy_theo_tham_so.png")
    curves(runs, histories, args.out_dir / "2_duong_cong_val.png")
    control_bars(runs, args.out_dir / "3_doi_chung_s9.png")
    print(f"{len(runs)} runs -> {args.out_dir}")


if __name__ == "__main__":
    main()
