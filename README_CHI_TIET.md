# classical_cnn_16x16

Thang CNN cổ điển trên MNIST tĩnh 16×16, làm đối chứng cho các bản QCNN (2026-11-15).

- **Cùng điều kiện với runner QCNN level32** (`full_quantum_16x16_level32_qconv/run_level32_front.py`): cùng split
  50.000/10.000 (`--split-seed 20260911`), aug xoay ±5° và dịch ±0,05, resize song tuyến về 16×16 rồi kẹp [0, 1],
  Adam, ReduceLROnPlateau (×0,5, chờ 3 epoch), dừng sớm 10 epoch, 50 epoch, batch 64; cùng file kết quả
  (`history.csv`, `summary.json`, checkpoint MAX / CK).
- **Bố cục SmallCNN**, giống cnn1k–cnn5k cũ và cnn-3k/5k/12k/19k trên Mammoth:
  `Conv(1→a) BN ReLU MaxPool → Conv(a→b) BN ReLU MaxPool → Linear(16b, b) ReLU → Linear(b, 10)`.
  Số tham số `12a + 9ab + 16b² + 14b + 10` (tính cả Linear ra 10 lớp; BatchNorm running stats không tính).

| Preset | (a, b) | Tham số | Ghi chú |
|---|---|---:|---|
| cnn1k | (4, 6) | 934 | = cnn1k cũ |
| cnn2k | (6, 9) | 1.990 | = cnn2k cũ |
| cnn3k | (8, 11) | 2.988 | = cnn3k cũ |
| cnn5k | (8, 16) | 5.578 | = cnn5k cũ (cũ train bằng lịch 3 pha khác) |
| cnn12k | (11, 24) | 12.070 | = Mammoth cnn-12k |
| cnn17k | (16, 28) | 17.170 | cỡ QCNN + qconv4x3 (17.264) |
| cnn19k | (15, 30) | 19.060 | = Mammoth cnn-19k |
| cnn30k | (20, 38) | 30.726 | |
| cnn50k | (24, 49) | 49.984 | |
| cnn85k | (32, 64) | 85.258 | |

Mỗi cỡ quét lr 3e-3, 1e-3, 3e-4 (`run_sweep.ps1`), chọn lr theo MAX val (`tong_hop.py`). Có thể dùng
`--widths a b`, `--hidden`, `--dropout`, `--weight-decay` cho cấu hình khác.

File: `models.py`, `run_cnn.py`, `run_sweep.ps1`, `tong_hop.py`, `evaluate_test.py`, `test_model.py`, `command.txt`.
