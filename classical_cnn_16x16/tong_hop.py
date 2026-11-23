"""Collect runs/*/summary.json into ket_qua_cnn.csv and print one table, plus the best learning rate per size
(chosen by MAX val acc; validation only). Runs still training (history.csv, no summary.json) are listed as such.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    rows = []
    for run in sorted((ROOT / "runs").glob("*")):
        summary_path, history_path = run / "summary.json", run / "history.csv"
        if summary_path.is_file():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append({"run": run.name, "params": s["param_counts"]["total"], "lr": s["lr"], "seed": s["seed"],
                         "max_val": round(s["best_val_acc"] * 100, 2), "max_epoch": s["best_acc_epoch"],
                         "ck_val": round(s["ck_val_acc"] * 100, 2), "ck_epoch": s["best_loss_epoch"],
                         "epochs": s["total_epochs"], "minutes": round(s["total_time_sec"] / 60, 1),
                         "status": "done"})
        elif history_path.is_file():
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            with history_path.open(encoding="utf-8") as f:
                history = list(csv.DictReader(f))
            if not history:
                continue
            best = max(history, key=lambda h: float(h["val_acc"]))
            ck = min(history, key=lambda h: float(h["val_loss"]))
            rows.append({"run": run.name, "params": config["param_counts"]["total"], "lr": config["args"]["lr"],
                         "seed": config["args"]["seed"], "max_val": round(float(best["val_acc"]) * 100, 2),
                         "max_epoch": best["epoch"], "ck_val": round(float(ck["val_acc"]) * 100, 2),
                         "ck_epoch": ck["epoch"], "epochs": len(history), "minutes": "",
                         "status": "running/stopped"})
    if not rows:
        print("No runs yet.")
        return
    rows.sort(key=lambda r: (r["params"], -float(r["lr"])))
    with (ROOT / "ket_qua_cnn.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'run':28s} {'params':>7s} {'lr':>7s} {'MAX':>6s} {'ep':>3s} {'CK':>6s} {'ep':>3s} {'n_ep':>4s} status")
    for r in rows:
        print(f"{r['run']:28s} {r['params']:7d} {float(r['lr']):7.0e} {r['max_val']:6.2f} {r['max_epoch']:>3} "
              f"{r['ck_val']:6.2f} {r['ck_epoch']:>3} {r['epochs']:4d} {r['status']}")
    print("\nBest lr per size (MAX val, finished runs only):")
    done = [r for r in rows if r["status"] == "done"]
    for params in sorted({r["params"] for r in done}):
        best = max((r for r in done if r["params"] == params), key=lambda r: r["max_val"])
        print(f"  {params:7d} params: lr {float(best['lr']):.0e} -> MAX {best['max_val']:.2f} (CK {best['ck_val']:.2f})")
    print(f"\nSaved {ROOT / 'ket_qua_cnn.csv'}")


if __name__ == "__main__":
    main()
