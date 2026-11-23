# R0 cải tiến: QCNN chuẩn với fusion và phép đo nhanh hơn

Folder này giữ kiến trúc QCNN của R0 (`full_quantum_16x16_2branch_multiscale_mlp16`): cây 1 qubit, side tap, stage 8 qubit và re-upload ở phase 3. Folder gốc không bị sửa.

**Trạng thái:** code mới chỉ được kiểm tra cú pháp, chưa chạy lần nào và chưa có accuracy.

## Ba thay đổi

| Thay đổi | Cờ | Mô tả |
|---|---|---|
| **Phép đo nhanh hơn** | Luôn bật | Không dựng 276 ma trận 256×256 mỗi batch. Mỗi toán tử Pauli chỉ có 1 phần tử khác 0 mỗi cột, nên chỉ cần đọc 256 phần tử của `rho`. Kết quả không đổi về toán học |
| **Khởi tạo cân bằng** | `--balanced-conv` | Gate `conv` bắt đầu gần sqrt(iSWAP), để nút giữ lại phụ thuộc vào cả hai con ngay từ đầu. `pool` vẫn gần identity, vì hai sqrt(iSWAP) liên tiếp sẽ thành iSWAP và chỉ giữ con bên phải |
| **Fusion** | `--fusion` | Một MLP chung đọc đặc trưng của tất cả nhánh, thay vì mỗi nhánh một MLP rồi lấy trung bình logits |

Bản fusion ở đây được viết từ code R0, **không dựa trên code trong `noread/`**. Nó có thêm side tap, nên có thể khác bản fusion cũ.

## Các run

| Run | Nhánh | Fusion | Khởi tạo | Tham số (phase 3) | So với |
|---|---|---|---|---:|---|
| `fusion3` | row, column, morton | Có | Như R0 | 18.194 | Fusion cũ, khoảng 98,7% |
| `fusion2` | row, morton | Có | Như R0 | 12.202 | R0, cao nhất 98,15% khi đang chạy |
| `balanced` | row, morton | Không | Cân bằng | 12.420 | R0 |
| `fusion3_balanced` | row, column, morton | Có | Cân bằng | 18.194 | `fusion3` |

Công thức head fusion: `Linear(324 × số nhánh, 16) → LayerNorm(16) → GELU → Linear(16, 10)`. Mỗi nhánh có 276 đặc trưng Pauli và 48 đặc trưng side tap.

## Cấu trúc code

Sao chép từ folder R0, chỉ sửa các phần liên quan.

| File | Thay đổi so với R0 |
|---|---|
| `models/layers/readout.py` | Thêm `PauliFeatures`; `PauliObservableReadout` dùng nó |
| `models/layers/unitary.py` | `LearnedUnitary2Q` có thêm `balanced` |
| `models/layers/tn_core.py` | `QuantumConvPoolBlock`, `QuantumTNCore` có thêm `balanced_conv` |
| `models/full_quantum_qcnn.py` | Thêm `fusion`, `balanced_conv`, `fusion_head` |
| `run_r0.py` | Thêm `--spatial-orders`, `--balanced-conv`, `--fusion` |
| `evaluate_test.py` | Đọc kiến trúc từ checkpoint |
| `test_model.py` | File mới, kiểm tra không cần MNIST |
| `command.txt` | Toàn bộ lệnh |

## Chạy

Chạy từ thư mục gốc dự án, theo thứ tự trong `command.txt`, bắt đầu bằng `test_model.py`. Lịch train, split, augmentation và tên file đầu ra đều giống R0, chỉ khác `--out-dir`. Chạy lại cùng lệnh sẽ resume.

Gửi lại `summary_mlp_aug.json` hoặc các dòng `Checkpoint` và `Max validation` cuối log. Chênh lệch dưới khoảng 0,3 điểm chưa đủ để kết luận với một seed.

## Khi so sánh

- Bản fusion cũ có thể khác aug, seed hoặc kiến trúc head, nên chỉ so tham khảo cho đến khi xác nhận được cấu hình.
- `fusion2` và `balanced` so trực tiếp được với R0, vì cùng protocol.
- Bản này vẫn có head cổ điển chiếm phần lớn tham số (khoảng 87%), và vẫn mô phỏng được trên máy cổ điển.
