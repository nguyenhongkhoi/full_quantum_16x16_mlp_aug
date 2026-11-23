# full_quantum_16x16_level32_qconv_linhead

Bản chép của `full_quantum_16x16_level32_qconv` (2026-11-15). Encoder lượng tử `qconv4x3` và mạch level32 (3 cây,
đo tầng 32) giữ nguyên; chỉ đổi head cổ điển sau 3 cây.

| `--head` | Head | Tổng (qconv4x3) | Quantum |
|---|---:|---:|---:|
| `linear3` (mặc định) | 3 × Linear(372, 10), logit = trung bình | 16.506 | 31,2% |
| `linear-shared` | 1 × Linear(372, 10) dùng chung | 9.046 | 56,9% |
| `mlp` | projection 10 + MLP gộp (như level32_qconv) | 17.264 | 29,8% |

Với head tuyến tính: không projection, LayerNorm, GELU, dropout hay MLP gộp; `--projection`, `--hidden`,
`--dropout` không có tác dụng. Trung bình logit của 3 Linear bằng một Linear(1116, 10) trên đặc trưng nối lại.

File sửa so với level32_qconv: `models/fusion_model.py` (tham số `head`), `models/__init__.py` (truyền `head`; config
cũ không có `head` thì dùng `mlp`), `run_level32_front.py` (`--head`, mặc định `linear3`), `test_model.py`
(`test_linear_heads`, tốc độ 3 kiểu head). `models/front.py` giống hệt level32_qconv và mammoth_qconv_front.

Lệnh và ngưỡng: `command.txt`. Kiến trúc chi tiết (bản head MLP): `../full_quantum_16x16_level32_qconv/KIEN_TRUC_front_qconv4x3_proj10.txt`.

## Đối chứng cổ điển cùng readout: `--model cnn3`

Thay encoder + 3 cây (5.316 tham số) bằng 3 CNN nhỏ độc lập (`Conv(1→8) BN ReLU MaxPool → Conv(8→23) BN ReLU MaxPool
→ Flatten 368`, mỗi nhánh 1.821), giữ nguyên head theo `--head`: mlp 17.291 (qcnn 17.264), linear3 16.533 (16.506),
linear-shared 9.153 (9.046). Lệnh C1, C2 và ngưỡng trong `command.txt`.
