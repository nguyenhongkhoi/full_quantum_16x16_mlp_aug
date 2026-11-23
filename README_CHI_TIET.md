# full_quantum_16x16_multilevel

MNIST tĩnh 16×16: level32 **không có encoder**, đọc thêm các tầng khác của cây. Mục tiêu là tăng accuracy; người dùng chấp nhận head lớn hơn và tỷ lệ quantum thấp hơn.

Folder này chép từ `full_quantum_16x16_level32_aug`; folder đó không bị sửa. `--side-level 32` ở đây cho model giống hệt level32_aug.

## Ý tưởng
- Mỗi cây gộp 256 qubit → 128 → 64 → 32 → 16 → 8.
- Level32 đọc 276 giá trị Pauli của 8 qubit cuối, cộng X, Y, Z của 32 qubit ở tầng 32.
- Trong cây các qubit độc lập (tích các trạng thái 1 qubit). Vì vậy X, Y, Z của từng qubit là toàn bộ thông tin của một tầng, và tính chính xác. Không có tương quan nào để đo thêm trong cùng tầng.
- `--side-level` nay nhận nhiều tầng, ví dụ `--side-level 32 64`. Tầng N qubit thêm 3N giá trị mỗi cây.
- Tầng 64 giữ chi tiết cục bộ hơn (mỗi qubit gom 4 pixel), tầng 16 thô hơn.

## Số tham số
Quantum luôn là 4.824. Head = 3P(F+3) + 48P + 218, với F = 276 + 3·(tổng số qubit các tầng đọc) và P = `--projection`.

| `--side-level` | `--projection` | Head | Tổng | Quantum |
|---|---:|---:|---:|---:|
| 32 (level32) | 12 | 14.294 | 19.118 | 25,2% |
| 32 64 | 12 | 21.206 | 26.030 | 18,5% |
| 16 32 64 | 12 | 22.934 | 27.758 | 17,4% |
| 16 32 | 12 | 16.022 | 20.846 | 23,1% |
| 32 (đối chứng cỡ head) | 18 | 21.332 | 26.156 | 18,4% |

## File thay đổi so với `full_quantum_16x16_level32_aug`
- `models/fusion_model.py`: `side_level` nhận một số hoặc một danh sách tầng.
- `run_level32_aug.py`: `--side-level` nhận nhiều số; in thêm tỷ lệ quantum.
- `evaluate_test.py`: không đổi (đọc `side_level` từ config, số hay danh sách đều được).
- `test_model.py`: số tham số, `[32]` giống hệt level32, đặc trưng từng tầng, forward/backward, thời gian mỗi batch.

## Chạy
Theo `command.txt`. Ngưỡng chốt trước ở cuối file đó.

Trạng thái: mới kiểm cú pháp (`ast.parse`), chưa chạy test.
