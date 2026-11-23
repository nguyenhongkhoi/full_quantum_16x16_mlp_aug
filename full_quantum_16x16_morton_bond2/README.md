# Morton bond-2 tree: phân loại MNIST 16×16

Folder này chứa code cho các run R1–R4 trong `DANH_GIA_KIEN_TRUC_VA_DE_XUAT.txt`. Nó độc lập với `full_quantum_16x16_2branch_multiscale_mlp16`; run 2 nhánh đó là mốc đối chứng R0.

**Trạng thái:** code mới chỉ được kiểm tra cú pháp, chưa chạy lần nào và chưa có accuracy. Hãy chạy `test_model.py` trước.

## Ý tưởng

Trong cây hiện tại, mỗi node chỉ giữ 1 qubit, tức tối đa 3 số thực. Tám qubit cuối vì thế chỉ mang khoảng 24 số, và stage 8 qubit không thêm được thông tin mới về ảnh.

Bản này cho mỗi node giữ 2 qubit (15 số thực) và bỏ stage 8 qubit.

```text
Ảnh 28×28 → resize 16×16 → mã hóa RY 256 qubit → sắp xếp Morton
  Level A      256 qubit → 128 node 1×2     1 gate, không trace
  Level B      → 64 node 2×2                3 gate, trace 2 qubit
  Level C      → 32 node 2×4                3 gate, trace 2 qubit
  Level D      → 16 node 4×4  ← lấy tap     3 gate, trace 2 qubit
  Level E      → 8 node 4×8                 3 gate, trace 2 qubit
  Level F      → 4 node 8×8   ← lấy cuối    3 gate, trace 2 qubit
  Mỗi node đo 15 toán tử Pauli 2 qubit → head → 10 logits
```

Gate được dùng chung cho mọi node trong cùng một level. Một lần ghép ở level B–F gồm hai node con `[a0 a1]` và `[b0 b1]`:

1. Gate `(a1, b0)` tạo tương quan giữa hai node con.
2. Gate `(a0, a1)` và `(b0, b1)` kéo thông tin từ `a0`, `b1` về hai qubit được giữ.
3. Trace `a0`, `b1`, giữ `(a1, b0)`.

Thứ tự này là bắt buộc. Nếu gate `(a1, b0)` chạy sau cùng, nó chỉ tác động lên tích của 2 qubit riêng lẻ, tức chỉ có 6 số thực. Cây vẫn được mô phỏng chính xác, vì các cây con dùng các pixel khác nhau nên thực sự độc lập.

## Các biến thể

| `--variant` | Run | Mô tả | Core | Head | Tổng |
|---|---|---|---:|---:|---:|
| `b0` | R3 | Bond-2, head `Linear(60,10)` | 256 | 610 | 866 |
| `b1` | R3 | Bond-2 + tap, `Linear(15,6)` dùng chung + `Linear(156,10)` | 256 | 1.666 | 1.922 |
| `b2` | R3 | Như `b1`, nhưng head `156→16→10` có LayerNorm và GELU | 256 | 2.810 | 3.066 |
| `tree1q` | R2 | Cây 1 qubit 2 nhánh row + Morton, không có stage 8 qubit | 320 | 2.740 | 3.060 |
| `cnn5k` | R1 | CNN cổ điển nhỏ, đầu vào 16×16, độ rộng (8, 16) | 1.296 | 4.282 | 5.578 |
| `cnn3k` | So với `b2` | Cùng bố cục, độ rộng (8, 11) | 921 | 2.067 | 2.988 |
| `cnn2k` | So với `b1` | Cùng bố cục, độ rộng (6, 9) | 585 | 1.405 | 1.990 |
| `cnn1k` | So với `b0` | Cùng bố cục, độ rộng (4, 6) | 282 | 652 | 934 |

`--shifts 0 2` (R4, chỉ cho `b0`/`b1`/`b2`) chạy thêm một lượt trên ảnh dịch vòng 2 pixel, với cùng gate và cùng head, rồi lấy trung bình logits. Số tham số không đổi, nhưng thời gian chạy gấp đôi. Lưu ý dịch vòng là phép xử lý cố định bên trong model, không phải augmentation.

## Cấu trúc code

| File | Vai trò |
|---|---|
| `models/unitary.py` | Gate 2 qubit `exp(iH)`, 16 tham số, có khởi tạo gần identity hoặc gần sqrt(iSWAP). |
| `models/bond2_tree.py` | Mã hóa ảnh, thứ tự Morton, `Bond2Leaf`, `Bond2Merge`, `Bond2Core`, `Bond2Head`, `Bond2TreeClassifier`, và `OneQubitTreeClassifier` cho R2. |
| `models/baselines.py` | `SmallCNN` cho R1. |
| `models/__init__.py` | `build_model`, danh sách biến thể và số tham số kỳ vọng. |
| `run_bond2.py` | Train 3 phase, resume, log, checkpoint, biểu đồ. Dừng ngay nếu số tham số khác kỳ vọng. |
| `test_model.py` | Kiểm tra nhanh, không cần MNIST. |
| `evaluate_test.py` | Đánh giá một checkpoint trên official test. |
| `command.txt` | Toàn bộ lệnh chạy theo thứ tự. |

## `test_model.py` kiểm tra gì

- Số tham số của từng biến thể và thứ tự Morton.
- `Bond2Merge` giữ đúng qubit `a1`, `b0`, và cho kết quả khớp với một mô phỏng statevector viết riêng.
- Ma trận mật độ hợp lệ (vết bằng 1, Hermitian, không âm) và node 2 qubit thực sự có tương quan.
- Gradient tồn tại và hữu hạn ở mọi tham số.
- Hai dòng `INFO` chỉ để tham khảo: số pixel có ảnh hưởng lúc khởi tạo và thời gian mỗi batch.

## Train

Giữ giống R0: split 50.000/10.000 (`split-seed=20260911`), `RandomAffine` 5° và dịch 0,05, batch 64, seed 0.

| Phase | Tối ưu | Epoch | Chọn checkpoint theo |
|---|---|---:|---|
| 1 | Adam, LR `2e-3` | 20 | Val accuracy |
| 2 | Adam mới, LR `2e-4` | 5 | Val accuracy |
| 3 | Adam, core `1e-4`, head `1e-3`, ReduceLROnPlateau | Tối đa 60, dừng sau 10 epoch không cải thiện | Val loss |

## Khác biệt so với R0, cần nhớ khi so sánh

- Không có re-upload ở phase 3. Vì vậy `tree1q` khác R0 ở hai yếu tố: bỏ stage 8 qubit và bỏ re-upload.
- `b0`/`b1`/`b2` khác R0 ở nhiều yếu tố cùng lúc: bond 2 qubit, 1 nhánh, gate mới, khởi tạo sqrt(iSWAP), chuẩn hóa đặc trưng và head mới. Kết quả chỉ trả lời "kiến trúc này tốt hơn hay kém hơn", không cho biết yếu tố nào tạo ra khác biệt. Muốn tách khởi tạo thì chạy thêm với `--init identity`.
- `tree1q` giữ khởi tạo gần identity như R0.

## Chạy và đầu ra

Chạy theo thứ tự trong `command.txt` từ thư mục gốc dự án, bắt đầu bằng `test_model.py`. Máy chạy cần PyTorch và torchvision có sẵn; `requirements.txt` chỉ liệt kê NumPy và Matplotlib để không thay bản PyTorch CUDA đang dùng.

Mỗi run ghi vào `--out-dir`, và chạy lại cùng lệnh sẽ resume. Dùng thư mục mới nếu đổi cấu hình.

```text
config_mlp_aug.json, summary_mlp_aug.json, split_indices.csv
history_full_quantum_mlp_aug.csv, convergence_mlp_aug.png
fq_mlp_aug_phase*_best.pt / _last.pt / _done.json
full_quantum_mlp_aug_best.pt      checkpoint cuối, chọn theo val loss
```

Nếu GPU phải chờ dữ liệu, thử `--num-workers 2`. Mô hình mới nhẹ hơn nhiều nên data loader có thể trở thành phần chậm nhất.

Sau mỗi run, hãy gửi `summary_mlp_aug.json`, `history_full_quantum_mlp_aug.csv`, `config_mlp_aug.json`, thời gian mỗi epoch, và traceback nếu có lỗi. Chỉ chạy `evaluate_test.py` một lần cho cấu hình đã chốt bằng validation.
