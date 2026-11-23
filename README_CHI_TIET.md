# qcnn_qconv4x3

Bản quantum có encoder quanvolution **qconv4x3** + mạch cây level32 (run 2c, MAX val 99,03 trên MNIST tĩnh 16×16,
17.264 tham số, phần quantum 29,8%).

- cùng seed cho ra đúng trọng số và kết quả như code gốc (`test_model.py` kiểm khi folder gốc nằm cạnh);
- nạp được checkpoint của run 2c (`evaluate_test.py`).

## Cấu trúc

```
qcnn_qconv4x3/
├─ requirements.txt          torch bản CUDA 12.6 (cố định +cu126, không lấy nhầm bản CPU)
├─ train.py                  train (dữ liệu, aug, Adam, giảm lr, dừng sớm, checkpoint) – như run_level32_front.py
├─ evaluate_test.py          official test, 1 lần
├─ test_model.py             kiểm tra trước khi train
├─ command.txt               lệnh đầy đủ
└─ qcnn/
   ├─ model.py               QCNNQconv: ghép các phần, luồng dữ liệu
   ├─ data/
   │  └─ preprocessing.py    resize 28×28 → 16×16, kẹp [0,1], làm phẳng
   ├─ encoder_quanvolution/
   │  ├─ quanvolution.py     qconv<L>x<S>: mạch 9 qubit trên ô 3×3, RZ·RY·RZ + CZ, đo X/Y/Z, Linear(27,2) cộng dư
   │  └─ angle_encoding.py   mã hóa 256 qubit lá: cos(πθ/2)|0⟩ + e^{iπφ} sin(πθ/2)|1⟩
   ├─ core/                  mạch cây tensor network
   │  ├─ unitary.py          unitary học được U = exp(iH)
   │  ├─ tn_core.py          khối conv-pool, strongly-entangling, QuantumTNCore
   │  ├─ reupload.py         re-upload ở 8 qubit cuối, lấy trạng thái một tầng (tầng 32)
   │  ├─ pauli.py            276 đặc trưng Pauli của 8 qubit cuối
   │  ├─ measurement.py      vector Bloch của một tầng (96 số ở tầng 32)
   │  └─ scan_orders.py      thứ tự lá: hàng, cột, Morton
   └─ layers/
      └─ readout.py          head: mỗi cây Linear(372,10)+LN+GELU+Dropout → MLP gộp (hoặc head tuyến tính)
```

## Số tham số (qconv4x3, projection 10, hidden 16)

| Phần                      | Folder                 |    Tham số | Loại          |
| ------------------------- | ---------------------- | ---------: | ------------- |
| Encoder: góc mạch         | `encoder_quanvolution` |        324 | quantum       |
| Encoder: Linear(27,2) × 3 | `encoder_quanvolution` |        168 | cổ điển       |
| 3 cây                     | `core`                 |      4.824 | quantum       |
| Head                      | `layers`               |     11.948 | cổ điển       |
| **Tổng**                  |                        | **17.264** | quantum 29,8% |

`--head linear3`: 16.506; `--head linear-shared`: 9.046.

## Chạy

Theo `command.txt`: tạo `.venv`, `pip install -r requirements.txt` (kiểm `torch.cuda.is_available()` là True),
`python .\test_model.py`, rồi lệnh train. Giải thích chi tiết kiến trúc: `../full_quantum_16x16_level32_qconv/KIEN_TRUC_front_qconv4x3_proj10.txt`.
