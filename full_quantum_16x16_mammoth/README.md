# Model quantum trong Mammoth

Áp model quantum vào continual learning bằng [Mammoth](https://github.com/aimagelab/mammoth), commit `e75a491`, bài Rotated MNIST: 20 task, mỗi task xoay ảnh một góc cố định trong 0-180 độ, 1 epoch mỗi task, không có augmentation.

**Trạng thái:** `test_quantum_backbones.py` đạt trên máy 4090. Đã xong `cnn-19k` er và sgd, `qcnn-fusion` er đang chạy. Số và nguồn ở `SO_NGHIEN_CUU_QCNN.txt`.

## Backbone

| Tên | Tham số | Mô tả |
|---|---:|---|
| `qcnn-fusion` | 19.202 | Fusion 3 nhánh của `noread/full_quantum_16x16_3scan_fusion_mlp16`; 16.802 nếu bỏ 2.400 tham số dư |
| `qcnn-fusion-scan` | 19.202 | Như `qcnn-fusion`, nhưng 3 cây quét theo góc 0, 60, 120 độ. Ảnh đầu vào giữ nguyên, chỉ đổi thứ tự pixel |
| `qcnn-fusion-moe` | 13.900 | Như `qcnn-fusion`, nhưng mỗi cây nén về 8 giá trị rồi qua 4 MLP chuyên gia và một lớp tự chọn, không dùng ID task. Quantum 4.824, head 9.076 |
| `qcnn-fusion-phase1`, `-phase2`, `-phase4` | 19.202 | Như `qcnn-fusion`, thêm pha theo tọa độ vào mã hóa, 1, 2 hoặc 4 chu kỳ trên ảnh, hướng 0, 60, 120 độ theo nhánh. Giá trị pixel giữ nguyên; pixel đúng 0 hoặc 1 không bị ảnh hưởng |
| `qcnn-fusion-typed4` | 12.010 | Như `qcnn-fusion`, head nhỏ: một `Linear(3,4)` cho mọi qubit và một `Linear(9,4)` cho mọi cặp, dùng chung cho 3 cây, rồi MLP 16. Quantum 4.824, head 7.186 (40,2%) |
| `qcnn-fusion-level32` | 19.118 | Như `qcnn-fusion`, đọc thêm X, Y, Z của 32 qubit sau tầng 3 của mỗi cây; mỗi cây nén về 12 thay vì 16. Quantum 4.824, head 14.294 |
| `qcnn-fusion-level32-lr4` | 19.118 | Như `qcnn-fusion-level32`, nhưng mọi tham số quantum học nhanh gấp 4 dưới Adam (giá trị = 4 × tham số lưu). Cùng tham số, cùng điểm xuất phát |
| `qcnn-fusion-level32-4tree` | 19.162 | Như `qcnn-fusion-level32`, thêm cây thứ 4 quét Morton dịch 4 pixel (khối 4×4 giữ nguyên, khối 8×8 giữa ảnh thành một nhánh); mỗi cây nén về 8. Quantum 6.432 (33,6%), head 12.730 |
| `qcnn-fusion-level32-graph` | 19.148 | Như `qcnn-fusion-level32`, thêm trước mỗi cây một lớp đồ thị quantum: mỗi pixel nối 8 láng giềng bằng exp(−iγ ZZ) (γ riêng cho cạnh thẳng và chéo, giống nhau theo mọi hướng), rồi một cổng 1 qubit chung. Lấy trạng thái 1 qubit chính xác của mỗi pixel, bỏ tương quan giữa pixel. 30 tham số, khởi đầu là phép đồng nhất |
| `qcnn-s9` | 2.546 | s9_meas, quanvolution + MLP, không phải tensor network |
| `c4-local-mlp` | 2.542 | Đối chứng cổ điển của `qcnn-s9` |
| `mlp-19k` | 18.858 | Đối chứng cổ điển của `qcnn-fusion`, cùng đầu vào 16x16 |
| `cnn-3k` | 2.988 | CNN cùng kiểu cnn3k, đối chứng chính của `qcnn-s9` |
| `cnn-5k` | 5.578 | CNN cùng kiểu cnn5k, mốc tốt nhất của các run trước |
| `cnn-19k` | 19.060 | CNN cùng kiểu, đối chứng chính của `qcnn-fusion` |
| `mlp-12k` | 11.970 | Như `mlp-19k` (256-42-22-10), cùng cỡ `qcnn-fusion-typed4` |
| `cnn-12k` | 12.070 | Như `cnn-19k` (kênh 11, 24), đối chứng chính của `qcnn-fusion-typed4` |
| `mnistmlp` | 89.610 | Mốc mặc định của Mammoth, đầu vào 28x28 |

## Khác bản fusion gốc

- Có re-upload ngay từ đầu, vì Mammoth không có 3 phase. Hệ số re-upload bắt đầu từ 0.
- Dùng cách đo Pauli nhanh, không đổi kết quả về toán học; `test_quantum_backbones.py` so với toán tử đầy đủ.
- Dùng gradient checkpointing theo từng nhánh khi train; test xác nhận output và gradient không đổi. Ở batch 64: 18,26 GB xuống 6,23 GB, 0,91 lên 1,22 s/batch.
- Chỉ học 1 epoch mỗi task. Bản gốc cần nhiều chục epoch để đạt 98,70%, ở ep 1 chỉ đạt 93,84%, và số đó có aug. Không so trực tiếp với 98,70%.

## Cấu hình chung

Adam, lr 1e-3, batch 32, replay 32, buffer 5120 (trước 2026-11-06 là 500), cho mọi backbone. Khác mặc định của Mammoth (SGD, batch 128), vì `qcnn-fusion` không đủ VRAM ở batch lớn.

`--validation 10 --validation_mode complete` để chọn cấu hình trên validation. Tập test chỉ dùng một lần cho cấu hình đã chốt.

Dòng `Accuracy for final task(s)` **không phải validation**: khi run chạy hết, Mammoth tự đánh giá trên test thật với các góc xoay mới rút. Backward Transfer và Forgetting vẫn tính trên validation. Mọi so sánh dùng dòng `Accuracy for N task(s)`. `--stop_after N` dừng sau N task, nhưng vẫn chạy bước test này; dừng bằng tay để tránh.

## Sửa trên máy train

- Đổi tên `datasets/seq_8vision.py`, `seq_dtd.py`, `seq_gtrsrb.py`, `seq_sun397.py`, `seq_svhn.py`, `models/tak.py` thành `.disabled`, vì cần `open_clip`.
- Sửa `utils/loggers.py` dòng 387: với `rot-mnist`, chỉ số Task-IL là `None` nên `--enable_other_metrics 1` làm run lỗi ngay trước khi ghi `logs.pyd`.
- Cài `pandas` vào venv.
- Lệnh trong `command.txt` đã có `--num_workers 0` cho Windows.
- Trên máy cp949, chạy `$env:PYTHONIOENCODING = "utf-8"` trước khi dùng `Tee-Object`.

## Chạy

`run_mammoth.py` chạy `main.py` với cùng tham số, thêm `--name` (log vào `logs/log_<name>.txt`, gồm cả stderr), `--end_at N` (dừng sau task N, không chạy test cuối). `--resume` đã tắt (2026-11-07): khi chạy tiếp, Mammoth dựng lại các task cũ với góc khác góc đã train, và lấy `stop_after` lưu trong checkpoint nên dừng ngay rồi chạy test cuối. Muốn có đủ 20 task thì chạy lại từ đầu.

Theo thứ tự trong `command.txt`. Mỗi bước phải xong rồi mới sang bước sau.
