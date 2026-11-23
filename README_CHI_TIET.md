# mammoth_qconv_front

Thử một **mạch quantum nhỏ làm encoder** thay cho encoder CNN `deep12p`, trên Mammoth rot-mnist (20 task, 1 epoch mỗi task, không augmentation, er buffer 5120).

Folder này chép từ `mammoth_level32_front`; folder đó không bị sửa. `backbone/` ở đây là bản mở rộng: mọi backbone cũ vẫn có và tính giống hệt trước.

## Encoder quantum `qconv<L>x<S>` (`backbone/quantum_core/front.py`, `QuantumFront`)
Mỗi pixel có một mạch 9 qubit trên ô 3×3 quanh nó. Cổng dùng chung cho mọi pixel, giống kernel của conv.
1. **Nạp dữ liệu:** mỗi qubit là RZ(πφ)·RY(πθ)|0⟩ của pixel tương ứng; ngoài biên ảnh là |0⟩. Ở tầng đầu θ là giá trị pixel và φ = 0.
2. **L lớp mạch,** mỗi lớp gồm:
   - một cổng xoay RZ·RY·RZ trên mỗi qubit, có tham số riêng;
   - CZ trên 12 cạnh của lưới 3×3.
3. **Đo:** lấy X, Y, Z của cả 9 qubit (27 giá trị, tính chính xác bằng statevector 512 biên độ).
4. **Readout:** một `Linear(27, 2)` theo từng pixel (56 tham số, tuyến tính). Đầu ra cộng vào θ và φ. Lớp này khởi tạo bằng 0, nên lúc đầu encoder là đồng nhất.
5. **Xếp S tầng:** đầu ra (θ, φ) của tầng trước là đầu vào của tầng sau. Cửa sổ nhìn được là (2S+1)×(2S+1).
6. **Hậu tố `f`:** góc của mạch bị đóng băng ở giá trị ngẫu nhiên ban đầu, chỉ readout học. Dùng để kiểm xem việc train mạch có ích không.

| Encoder | Tham số | Cửa sổ | Ghi chú |
|---|---:|---|---|
| `qconv2x2` | 220 (góc 108, readout 112) | 5×5 | bản chính |
| `qconv4x3` | 492 (góc 324, readout 168) | 7×7 | cùng cửa sổ với deep12p |
| `qconv2x2f` | 112 học được (góc 108 đóng băng) | 5×5 | đối chứng mạch đóng băng |
| `cnn8p` (cổ điển) | 226 | 5×5 | CNN cùng cỡ với `qconv2x2` (Conv 1→8, GELU, Conv 8→2) |
| `deep12p` (cổ điển) | 1.646 | 7×7 | encoder CNN đang dùng |

## Backbone mới

**Thí nghiệm 1, khả năng encode:** encoder + cùng một readout `AvgPool 4×4 → Linear(32, 10)` (330 tham số), không có mạch cây.

| Tên | Tham số | Vai trò |
|---|---:|---|
| `enc-linear-qconv2x2` | 550 | quantum + linear |
| `enc-linear-deep12p` | 1.976 | CNN nhỏ 1k9 (đã có từ mammoth_level32_front) |
| `enc-linear-cnn8p` | 556 | CNN cùng cỡ với qconv2x2 |
| `enc-linear-qconv2x2f` | 442 | mạch đóng băng |
| `enc-linear-qconv4x3` | 822 | quantum lớn hơn, cửa sổ 7×7 |

**Thí nghiệm 2, bản đầy đủ:** encoder quantum đặt trước mạch level32. Mạch level32 giữ nguyên (4.824 tham số); mỗi cây nén về 10 như deep12p (head 11.948).

| Tên | Tham số | Vai trò |
|---|---:|---|
| `qcnn-fusion-level32-qconv2x2` | 16.992 | bản chính, toàn bộ phần trước head là quantum (trừ 2 Linear(27, 2)) |
| `qcnn-fusion-level32-qconv4x3` | 17.264 | encoder lớn hơn |
| `qcnn-fusion-level32-cnn8p` | 16.998 | đối chứng: cùng mạch và head, encoder CNN cùng cỡ |
| `qcnn-fusion-level32-deep12p` | 18.418 | đã chạy: trung bình 6 mốc 94,28 |

- Mọi encoder khởi đầu là đồng nhất và được tạo trong `fork_rng`, nên lúc đầu model giống hệt level32 với projection 10.
- Không dropout, như mọi backbone Mammoth.
- Khi train, mạch 9 qubit được tính lại trong lượt backward (`checkpoint`) để tiết kiệm bộ nhớ; kết quả không đổi.

## File thay đổi so với `mammoth_level32_front`
- `backbone/quantum_core/front.py`: thêm `QuantumFront`, `QuantumConvStage`, `parse_qconv`, `make_front`, `front_has_phase`; `cnn<C>p` (thêm kênh φ cho `cnn<C>`).
- `backbone/quantum_core/fusion_model.py`: dùng `make_front`.
- `backbone/QuantumCL.py`: `EncoderControl` nhận encoder quantum; thêm 7 backbone mới.
- `test_quantum_backbones.py`: thêm `check_qconv`, gồm:
  - so mạch nhanh với ma trận đầy đủ 512×512, cả trong ảnh và ở biên;
  - số tham số;
  - góc đóng băng không học;
  - khởi đầu giống level32;
  - gradient tới các góc;
  - `checkpoint` cho cùng kết quả.

  Phần đo thời gian mặc định chỉ chạy các backbone mới (`--report all` để chạy tất cả).
- `run_mammoth.py`, `export_mammoth_logs.py`: không đổi.

## Chạy
Theo `command.txt`. Ngưỡng chốt trước ở cuối file đó.

Trạng thái: mới kiểm cú pháp (`ast.parse`), chưa chạy test.
