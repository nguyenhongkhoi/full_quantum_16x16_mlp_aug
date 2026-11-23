# full_quantum_16x16_level32_qconv

MNIST tĩnh 16×16: thử **mạch quantum nhỏ làm encoder** thay cho encoder CNN `deep12p`, trước khi thử trên Mammoth (`mammoth_qconv_front`, dùng cùng `front.py`).

Folder này chép từ `full_quantum_16x16_level32_front`; folder đó không bị sửa. Mọi `--front` và `--model` cũ vẫn chạy được ở đây và tính giống hệt trước.

## Encoder quantum `--front qconv<L>x<S>[f]` (`models/front.py`, `QuantumFront`)
Mỗi pixel có một mạch 9 qubit trên ô 3×3 quanh nó. Cổng dùng chung cho mọi pixel, giống kernel của conv.
1. **Nạp dữ liệu:** mỗi qubit là RZ(πφ)·RY(πθ)|0⟩ của pixel tương ứng; ngoài biên ảnh là |0⟩.
2. **L lớp mạch,** mỗi lớp gồm:
   - một cổng xoay RZ·RY·RZ trên mỗi qubit;
   - CZ trên 12 cạnh của lưới 3×3.
3. **Đo:** lấy X, Y, Z của 9 qubit (27 giá trị, tính chính xác).
4. **Readout:** một `Linear(27, 2)` theo từng pixel. Đầu ra cộng vào θ và φ. Lớp này khởi tạo bằng 0, nên lúc đầu encoder là đồng nhất.
5. **Xếp S tầng:** cửa sổ nhìn được là (2S+1)×(2S+1).
6. **Hậu tố `f`:** góc của mạch bị đóng băng, chỉ readout học.

| `--front` | Tham số | Cửa sổ |
|---|---:|---|
| `qconv2x2` | 220 (góc 108, readout 112) | 5×5 |
| `qconv4x3` | 492 (góc 324, readout 168) | 7×7 |
| `qconv2x2f` | 112 học được (góc 108 đóng băng) | 5×5 |
| `qconv2x2h16` (2026-11-14) | 1.072 (góc 108; readout `Linear(27,16)→GELU→Linear(16,2)` mỗi tầng) | 5×5 |
| `cnn8p` (cổ điển, mới) | 226 | 5×5 |
| `cnn38p` (cổ điển, đối chứng của h16) | 1.066 | 5×5 |
| `deep6p` (cổ điển, đối chứng của qconv4x3) | 500 | 7×7 |
| `deep12p` (cổ điển) | 1.646 | 7×7 |

Hậu tố `h<H>` đổi readout mỗi tầng thành `Linear(27,H) → GELU → Linear(H,2)` (lớp cuối khởi tạo 0). Không có `h` thì module và các số ngẫu nhiên giống hệt trước.

`--linear-pool P` (chỉ cho `--model enc-linear`): `AvgPool P×P → Linear`. Mặc định 4 như cũ (330 tham số với 2 kênh); 2 cho 1.290.

`evaluate_test.py --front ... --linear-pool ...` ghi đè config đã lưu, để evaluate các run làm bằng code sửa tay.

## Hai thí nghiệm (`command.txt`)

**Thí nghiệm 1, khả năng encode:** `--model enc-linear`. Encoder đi qua `AvgPool 4×4 → Linear(32, 10)` (330 tham số), không có mạch cây.
- **1a** qconv2x2: 550 tham số
- **1b** deep12p: 1.976 tham số. Đây là "CNN nhỏ 1k9", đã chạy ở level32_front 2f.
- **1c** cnn8p: 556 tham số
- **1d** qconv2x2f: 442 tham số
- **1e** qconv4x3: 822 tham số

**Thí nghiệm 2, bản đầy đủ:** `--model qcnn --front qconv2x2 --projection 10`, tức encoder quantum đặt trước mạch level32 giữ nguyên. Tổng 16.992 tham số. Đối chứng là `--front cnn8p` (16.998 tham số).

## File thay đổi so với `full_quantum_16x16_level32_front`
- `models/front.py`: chép từ `mammoth_qconv_front`. Thêm `QuantumFront`, `QuantumConvStage`, `make_front`, `front_has_phase`, `front_circuit_parameters` và `cnn<C>p`.
- `models/fusion_model.py`: dùng `make_front`. `count_parameters` thêm `front_quantum` (số góc mạch của encoder) và tính chúng vào `quantum_share`.
- `models/classical.py`: `enc-linear` và `enc-cnn` nhận được encoder quantum; `count_parameters` thêm `front_quantum`.
- `run_level32_front.py`: chỉ đổi phần help và dòng in số tham số.
- `test_model.py`: thêm `test_quantum_encoder`, gồm:
  - so mạch nhanh với ma trận 512×512;
  - số tham số;
  - góc đóng băng;
  - khởi đầu giống level32;
  - gradient tới các góc;
  - `checkpoint` cho cùng kết quả;
  - số tham số của các bản enc-linear.

  Thêm `test_speed` để đo thời gian mỗi batch 64, chỉ để tham khảo.

Thí nghiệm 2c (qconv4x3) và đối chứng 2d (deep6p) ở `command.txt`, mục cập nhật 2026-11-14.

Trạng thái: mới kiểm cú pháp (`ast.parse`), chưa chạy test sau lần sửa 2026-11-14.
