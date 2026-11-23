# mammoth_level32_front

Đưa encoder cổ điển `deep12p` (bản đạt MAX val 99,14 trên MNIST tĩnh, folder `full_quantum_16x16_level32_front`) vào backbone level32 của Mammoth, bài rot-mnist (20 task, 1 epoch mỗi task, không augmentation).

Chép từ `full_quantum_16x16_mammoth` (folder gốc không bị sửa). `backbone/` ở đây là bản mở rộng: mọi backbone cũ vẫn có và tính giống hệt trước, chỉ thêm 3 backbone mới.

## Backbone mới

| Tên | Tham số | Mô tả |
|---|---:|---|
| `qcnn-fusion-level32-deep12p` | 18.418 | level32, thêm encoder deep12p trước mạch (3 conv 3×3 12 kênh, ra θ và φ; mỗi qubit RZ(πφ)·RY(πθ)\|0⟩). Mỗi cây nén về 10 (head 11.948), như bản tĩnh. Quantum 4.824 (26,2%), encoder 1.646 |
| `enc-cnn-deep12p` | 18.392 | Đối chứng cổ điển: cùng encoder, mạch + head thay bằng CNN (Conv 20, Conv 32, Linear 20) |
| `enc-linear-deep12p` | 1.976 | Chỉ encoder + AvgPool 4×4 + Linear(32,10) |

- Không có dropout, như mọi backbone Mammoth khác (bản tĩnh có dropout 0,2).
- Encoder khởi đầu là đồng nhất (conv cuối bằng 0), tạo trong `fork_rng`, nên lúc khởi đầu model giống hệt level32 với projection 10. Encoder của 2 bản đối chứng cùng lớp, cùng cách khởi tạo, nhưng giá trị ngẫu nhiên ban đầu khác.
- Khi có encoder, giá trị không bị kẹp về [0, 1] trước RY.

## File thay đổi so với `full_quantum_16x16_mammoth`
- `backbone/quantum_core/front.py`: mới, chép từ `full_quantum_16x16_level32_front/models/front.py`.
- `backbone/quantum_core/encoding.py`: thêm tùy chọn `clamp`.
- `backbone/quantum_core/fusion_model.py`: tham số `front`; `count_parameters` thêm `front` khi có encoder.
- `backbone/QuantumCL.py`: `EncoderControl` và 3 backbone mới.
- `test_quantum_backbones.py`: thêm `check_front`.
- `run_mammoth.py`, `export_mammoth_logs.py`: không đổi.

## Chạy
Theo `command.txt`: chép code vào Mammoth, chạy test, chạy thử, sàng lọc 6 task, đối chứng, rồi 20 task nếu đạt. Ngưỡng chốt trước ở cuối `command.txt`.

Trạng thái: mới kiểm cú pháp (`ast.parse`), chưa chạy test.
