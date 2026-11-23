"""Tách khoảng cách accuracy sau task 20 so với cnn-19k thành: học kém ngay trong task (đường chéo) và quên (BWT).

mean_t a(20,t) = mean_t a(t,t) + (1/20) * sum_{t<20} [a(20,t) - a(t,t)], nên khoảng cách = Δ học + Δ quên, cộng đúng.
Chỉ đọc CSV, không train. Chạy: python chan_doan_khoang_cach_cnn.py
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

PATH = Path(__file__).with_name("mammoth_tasks_20261108.csv")
RUNS = [
    ("cnn-19k", "cnn19k_er_buf5120_seed0"),
    ("cnn-12k", "cnn12k_er_buf5120_seed0"),
    ("level32", "qcnnfusionlevel32_er_buf5120_seed0_full"),
    ("mlp-19k", "mlp19k_er_buf5120_seed0"),
    ("qcnn-fusion", "qcnnfusion_er_buf5120_seed0"),
    ("typed4", "qcnnfusiontyped4_er_buf5120_seed0"),
    ("mlp-12k", "mlp12k_er_buf5120_seed0"),
]
T = 20
acc = defaultdict(dict)
for row in csv.DictReader(open(PATH, encoding="utf-8")):
    if row["after_task"] != "final":
        acc[row["run"]][(int(row["after_task"]), int(row["task"]))] = float(row["accuracy"])

stats = {}
for name, run in RUNS:
    a = acc[run]
    diag = st.mean(a[(t, t)] for t in range(1, T + 1))
    last = st.mean(a[(T, t)] for t in range(1, T + 1))
    drop = sum(a[(T, t)] - a[(t, t)] for t in range(1, T)) / T
    forget = st.mean(max(a[(k, t)] for k in range(t, T)) - a[(T, t)] for t in range(1, T))
    bwt = st.mean(a[(T, t)] - a[(t, t)] for t in range(1, T))
    first = a[(1, 1)]
    worst = min(a[(T, t)] for t in range(1, T + 1))
    same = all(a[(k, 6)] == a[(k, 11)] for k in range(11, T + 1))
    stats[name] = (diag, drop, last)
    print(f"{name:12s} hoc_ngay {diag:6.2f}  sau20 {last:6.2f}  phan_quen/20 {drop:6.2f}  BWT {bwt:6.2f}  "
          f"Forgetting {forget:5.2f}  task1 {first:6.2f}  kem_nhat {worst:6.2f}  task6=task11 {same}")

print("\nKhoảng cách so với cnn-19k (âm = thấp hơn): tổng = do học + do quên")
cd, cq, cl = stats["cnn-19k"]
for name, (d, q, l) in stats.items():
    print(f"{name:12s} tổng {l - cl:6.2f} = học {d - cd:6.2f} + quên {q - cq:6.2f}")
print("\nSo với MLP cùng cỡ:")
for a_, b_ in (("level32", "mlp-19k"), ("typed4", "mlp-12k"), ("qcnn-fusion", "mlp-19k")):
    (d1, q1, l1), (d2, q2, l2) = stats[a_], stats[b_]
    print(f"{a_} - {b_}: tổng {l1 - l2:+.2f} = học {d1 - d2:+.2f} + quên {q1 - q2:+.2f}")
