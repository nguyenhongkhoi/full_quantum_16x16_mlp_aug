# Patch QCNN vướng víu: phân loại MNIST 16×16

Thử nghiệm giả thuyết: các model đã chạy gần như không có vướng víu giữa các pixel ở tầng thấp, vì mỗi cặp bị trace xuống ngay. Bản này mô phỏng chính xác nhiều qubit trong mỗi ô ảnh, nên các gate xen kẽ tạo được vướng víu thật ngay từ đầu.

**Trạng thái:** code mới chỉ được kiểm tra cú pháp, chưa chạy lần nào và chưa có accuracy. Đây là giả thuyết. Morton bond-2 từng được kỳ vọng tương tự và không đạt, nên đừng coi việc đạt 99% là chắc chắn.

## Kiến trúc

```text
Ảnh 28×28 → resize 16×16
  → Chia thành các ô (patch), mỗi pixel là 1 qubit, mã hóa RY
  → Trong mỗi ô, statevector đầy đủ:
       [gate lưới 2D xen kẽ chẵn/lẻ] → [mã hóa lại pixel → gate] × (layers - 1)
  → Trace, giữ 2 qubit giữa ô (ma trận 4×4)
  → Cây gộp bond-2 (Bond2Merge) trên lưới ô, tới 4 node
  → Đo 15 Pauli 2 qubit ở tầng 16 node và tầng 4 node
  → Readout MLP hoặc Linear → 10 logits
```

- Gate được dùng chung cho mọi ô, theo từng nhóm liên kết (ngang chẵn, ngang lẻ, dọc chẵn, dọc lẻ) và từng lớp.
- Ô chồng lấn vẫn mô phỏng chính xác: mỗi ô được coi là chuẩn bị từ bản sao pixel riêng, nên các ô độc lập với nhau.
- `channels` là số bản copy độc lập của toàn bộ core, tương tự số kênh của CNN.

## Các bước

Mỗi bước chỉ đổi một yếu tố so với bước được so sánh.

| Preset | Thay đổi | So với | Ô | Lưới | Tổng tham số (MLP) | Bản Linear |
|---|---|---|---|---|---:|---:|
| `s1_p8` | Nền | `b2` (97,10%) | 2×4 | 8×4 | 2.522 | 1.594 |
| `s2_p12` | Ô lớn hơn | `s1_p8` | 3×4 | 8×4 | 2.538 | 1.610 |
| `s3_ru2` | Re-upload 2 vòng | `s2_p12` | 3×4 | 8×4 | 2.602 | 1.674 |
| `s4_ru3` | Re-upload 3 vòng | `s3_ru2` | 3×4 | 8×4 | 2.666 | 1.738 |
| `s5_overlap` | Ô chồng lấn | `s3_ru2` | 3×4 | 8×8 | 2.650 | 1.722 |
| `s6_k4` | 4 kênh | `s5_overlap` | 3×4 | 8×8 | 4.150 | 3.222 |
| `s7_p16` | Ô 16 qubit | `s2_p12` | 4×4 | 4×4 | 2.490 | 1.562 |
| `s8_xl` | 8 kênh, head lớn hơn | `s6_k4` | 3×4 | 8×8 | 10.042 | 6.106 |

Hai preset đo nhiều Pauli, không trace xuống 2 qubit và không có cây gộp:

| Preset | Thay đổi | So với | Ô | Lưới | Tổng tham số (MLP) | Bản Linear |
|---|---|---|---|---|---:|---:|
| `s9_meas` | Đo 53 Pauli mỗi ô: X, Y, Z trên 12 qubit và ZZ trên 17 liên kết | `s2_p12` | 3×4 | 8×4 | 2.546 | 1.570 |
| `s10_meas_ru2` | Như trên, re-upload 2 vòng | `s9_meas`, `s3_ru2` | 3×4 | 8×4 | 2.610 | 1.634 |

Một `Linear(53,4)` dùng chung cho 32 ô, rồi tới readout. Phần lượng tử chỉ còn mạch trong ô, nên bản này gần với quanvolution hơn, và khoảng 95–98% tham số là cổ điển. Nếu thấy cải thiện, cần kiểm tra thêm bằng `--init random --freeze-core`.

**Mốc so sánh (val):** `cnn2k` 98,09% (đến epoch 52), `cnn3k` 98,64%, `b2` 97,10%, R0 cao nhất 97,97% khi đang chạy. Các số này do người dùng gửi, một seed.

**Cách đi qua các bước:**

1. Chạy `test_model.py`, gửi toàn bộ output để biết thời gian và bộ nhớ thực tế.
2. Chạy `s1_p8` và `s2_p12`. Nếu cả hai không hơn `b2` rõ ràng, hãy dừng và xem lại giả thuyết vướng víu trước khi chạy tiếp.
3. Chạy `s3_ru2`, rồi `s5_overlap`. Bước nào không tăng thì giữ cấu hình trước đó.
4. Chỉ chạy `s6_k4`, `s7_p16`, `s8_xl` khi các bước trên có tín hiệu tốt.
5. Với cấu hình tốt nhất, chạy thêm hai phép đối chứng dưới đây.

**Chênh lệch dưới khoảng 0,3 điểm chưa đủ để kết luận** với một seed.

## Hai phép đối chứng

| Lệnh | Câu hỏi |
|---|---|
| `--readout linear` | Bỏ MLP thì mất bao nhiêu? Mất ít nghĩa là core đang làm phần lớn việc. |
| `--init random --freeze-core` | Nếu gate không học, chỉ head học, thì sao? Bản học gate phải cao hơn rõ thì phần lượng tử mới thực sự đóng góp. |

## Cấu trúc code

| File | Vai trò |
|---|---|
| `models/unitary.py` | Gate 2 qubit `exp(iH)`, 16 tham số. |
| `models/patch_qcnn.py` | Chia ô, statevector, gate lưới, re-upload, partial trace, cây gộp, head. |
| `models/__init__.py` | `PRESETS`, số tham số kỳ vọng, `build_model`. |
| `run_patch.py` | Train 3 phase, resume, log, checkpoint, biểu đồ. Dừng ngay nếu số tham số khác kỳ vọng. |
| `test_model.py` | Kiểm tra không cần MNIST, kèm đo thời gian và bộ nhớ từng preset. |
| `evaluate_test.py` | Đánh giá official test, chỉ chạy một lần cuối. |
| `command.txt` | Toàn bộ lệnh đầy đủ tham số, theo thứ tự. |

## `test_model.py` kiểm tra gì

- Số tham số của mọi preset, và mỗi preset phủ đủ 256 pixel.
- Áp gate khớp với ma trận đầy đủ dựng riêng, kể cả qubit không liền kề.
- Mã hóa và re-upload khớp với các gate RY tường minh.
- Giữ đúng 2 qubit, và partial trace khớp với phép cộng tường minh.
- Ma trận mật độ hợp lệ và cặp giữ lại thực sự có tương quan.
- `Bond2Merge` khớp với mô phỏng statevector riêng.
- Gradient hữu hạn, và gradient checkpointing cho kết quả giống hệt.
- Dòng `INFO` cho từng preset: thời gian mỗi batch, bộ nhớ GPU, số pixel có ảnh hưởng lúc khởi tạo.

Lần đầu nên chú ý cột số pixel: nếu thấp hơn nhiều so với 256 thì nhiều pixel chưa được dùng do mạch quá nông, và cần tăng số lớp.

## Train

Giữ giống các run trước: split 50.000/10.000 (`split-seed=20260911`), `RandomAffine` 5° và dịch 0,05, batch 64, seed 0.

| Phase | Tối ưu | Epoch | Chọn checkpoint theo |
|---|---|---:|---|
| 1 | Adam, LR `2e-3` | 20 | Val accuracy |
| 2 | Adam mới, LR `2e-4` | 5 | Val accuracy |
| 3 | Adam, core `1e-4`, head `1e-3`, ReduceLROnPlateau | Tối đa 60, dừng sau 10 epoch không cải thiện | Val loss |

## Đầu ra

Mỗi run ghi vào `--out-dir`, tên file giống các run trước. Chạy lại cùng lệnh sẽ resume; đổi GPU hoặc `--num-workers` vẫn resume được.

```text
config_mlp_aug.json, summary_mlp_aug.json, split_indices.csv
history_full_quantum_mlp_aug.csv, convergence_mlp_aug.png
fq_mlp_aug_phase*_best.pt / _last.pt / _done.json
full_quantum_mlp_aug_best.pt      checkpoint cuối, chọn theo val loss
```

Sau mỗi bước, gửi dòng `Checkpoint` và `Max validation` cuối log, hoặc `summary_mlp_aug.json`. Máy chạy cần PyTorch và torchvision có sẵn.

## Giới hạn cần nhớ

- Đo Pauli ở các tầng là phép đo riêng cho từng cây con, nên trên phần cứng thật cần chạy nhiều mạch.
- Mô phỏng ở đây là lý tưởng: không có nhiễu và không tính số shot hữu hạn.
- Toàn bộ vẫn mô phỏng được trên máy cổ điển, nên kết quả không phải bằng chứng về lợi thế lượng tử.
