import csv
import statistics as st
from collections import defaultdict

PATH = r"D:\data_system\full_quantum_16x16_mlp_aug\ket_qua_mammoth\mammoth_tasks.csv"
ANGLES = [98.79, 128.52, 115.73, 16.83, 33.38, 148.17, 75.44, 76.98, 142.60, 147.03, 174.34, 179.79, 138.01, 33.84,
          35.59, 139.71, 93.57, 11.85, 105.51, 73.72]
acc = defaultdict(dict)
final = defaultdict(dict)
for row in csv.DictReader(open(PATH, encoding="utf-8")):
    if row["after_task"] == "final":
        final[row["run"]][int(row["task"])] = float(row["accuracy"])
    else:
        acc[row["run"]][(int(row["after_task"]), int(row["task"]))] = float(row["accuracy"])

T = 20
for run, a in acc.items():
    diag = [a[(t, t)] for t in range(1, T + 1)]
    last = [a[(T, t)] for t in range(1, T + 1)]
    forget = [max(a[(k, t)] for k in range(t, T)) - a[(T, t)] for t in range(1, T)]
    bwt = [a[(T, t)] - a[(t, t)] for t in range(1, T)]
    # spread of old tasks at every mark (tasks 1..k), averaged over marks 5..20
    spreads = [st.pstdev([a[(k, t)] for t in range(1, k + 1)]) for k in range(5, T + 1)]
    mins = [min(a[(k, t)] for t in range(1, k + 1)) for k in range(5, T + 1)]
    gaps = [st.mean([a[(k, t)] for t in range(1, k + 1)]) - min(a[(k, t)] for t in range(1, k + 1)) for k in range(5, T + 1)]
    print(f"\n{run}")
    print(f"  plasticity (diag mean) {st.mean(diag):.2f}  min {min(diag):.2f}")
    print(f"  after 20: mean {st.mean(last):.2f} min {min(last):.2f} max {max(last):.2f} std {st.pstdev(last):.2f}"
          f" cv {100*st.pstdev(last)/st.mean(last):.2f}%  range {max(last)-min(last):.2f}")
    print(f"  forgetting {st.mean(forget):.2f}  bwt {st.mean(bwt):.2f}")
    print(f"  marks 5-20: mean std {st.mean(spreads):.2f}  mean(min) {st.mean(mins):.2f}  mean(mean-min) {st.mean(gaps):.2f}")
    f = final[run]
    fl = [f[t] for t in range(1, T + 1)]
    print(f"  test final: mean {st.mean(fl):.2f} min {min(fl):.2f} std {st.pstdev(fl):.2f} range {max(fl)-min(fl):.2f}")
    # errors relative: error rate ratio
    print("  after20 per task:", " ".join(f"{v:.1f}" for v in last))

runs = list(acc)
print("\nerror-rate ratio vs cnn at task 20 per task (fusion err / cnn err):")
fu, cn, ml = (acc[r] for r in ("qcnnfusion_er_buf5120_seed0", "cnn19k_er_buf5120_seed0", "mlp19k_er_buf5120_seed0"))
for t in range(1, T + 1):
    print(t, ANGLES[t - 1], f"fusion {fu[(T,t)]:.2f} mlp {ml[(T,t)]:.2f} cnn {cn[(T,t)]:.2f}  f-m {fu[(T,t)]-ml[(T,t)]:+.2f}")
import math
for name, a in (("fusion", fu), ("mlp", ml), ("cnn", cn)):
    last = [a[(T, t)] for t in range(1, T + 1)]
    err = [100 - v for v in last]
    print(name, "error mean", f"{st.mean(err):.2f}", "error std/mean", f"{st.pstdev(err)/st.mean(err):.3f}")
# correlation of per-task final accuracy between models
def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / math.sqrt(sum((a-mx)**2 for a in x) * sum((b-my)**2 for b in y))
L = lambda a: [a[(T, t)] for t in range(1, T + 1)]
print("corr fusion-mlp", f"{corr(L(fu), L(ml)):.3f}", "fusion-cnn", f"{corr(L(fu), L(cn)):.3f}", "mlp-cnn", f"{corr(L(ml), L(cn)):.3f}")
# duplicates
for run, a in acc.items():
    dup = [(t, u) for t in range(1, T + 1) for u in range(t + 1, T + 1) if all(a[(k, t)] == a[(k, u)] for k in range(u, T + 1))]
    print(run, "identical task pairs:", dup)
