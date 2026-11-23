# mammoth_full_package

Gói chạy Mammoth rot-mnist trên **máy Windows mới, chưa cài gì** (chỉ cần GPU NVIDIA có driver và internet).
Gồm thang CNN cổ điển và bản quantum tốt nhất hiện tại (`qcnn-fusion-level32-qconv4x3`, bản đạt 99,03 val trên MNIST tĩnh).

## Chạy

Giải nén cả folder vào một chỗ ổ đĩa còn trống (đường dẫn nên ngắn, ví dụ `D:\mammoth_full_package`), rồi **một trong hai**:

- Nhấp đúp `CHAY_TAT_CA.bat`, hoặc
- Mở PowerShell trong folder này và chạy:
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\run_all.ps1
  ```

Lần đầu, `run_all.ps1` tự gọi `setup.ps1`:
1. Kiểm GPU (`nvidia-smi`).
2. Tìm Python 3.11; nếu chưa có thì cài bằng `winget` (nếu máy không có `winget`, script dừng và chỉ link tải Python).
3. Tạo môi trường `.venv`, cài PyTorch bản CUDA (cu124).
4. Tải Mammoth đúng commit `e75a491c69fd729edeb01431afb753d9157d9a81`, cài requirements của Mammoth.
5. Chép backbone và script của dự án vào `mammoth\`.
6. Chạy `test_quantum_backbones.py` (phải có "All checks passed.").

Sau đó chạy thử vài bước model quantum (tải MNIST lần đầu), rồi chạy lần lượt cả hàng đợi.

## Hàng đợi (đều 20 task, cấu hình như mọi run trước của dự án)

ER, buffer 5120, Adam lr 1e-3, batch 32, replay 32, 1 epoch mỗi task, seed 0, validation 10% (complete), không augmentation.

| Backbone | Tham số | Vai trò |
|---|---:|---|
| `cnn-5k` | 5.578 | CNN cổ điển (bố cục SmallCNN) |
| `cnn-12k` | 12.070 | |
| `cnn-17k` | 17.170 | **CNN cùng cỡ với model quantum** |
| `cnn-19k` | 19.060 | mốc cũ: 95,43 sau task 20 |
| `cnn-30k` | 30.726 | |
| `cnn-50k` | 49.984 | |
| `cnn-85k` | 85.258 | |
| `mlp-19k` | 18.858 | MLP cổ điển (mốc cũ 91,24) |
| `qcnn-fusion-level32-qconv4x3` | 17.264 | **model quantum chính**: encoder quanvolution qconv4x3 + 3 cây level32 + head (projection 10) |
| `qcnn-fusion-level32-deep6p` | 17.272 | đối chứng: cùng cây và head, encoder CNN 7×7 cùng cỡ (500) |

- Run đã xong (log có dòng "Accuracy for final task") sẽ được bỏ qua: tắt máy hay Ctrl+C giữa chừng rồi chạy lại lệnh là tiếp tục. Run đang dở sẽ chạy lại từ task 1 (log cũ được đổi tên, không xóa).
- Chạy một phần: `powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Only cnn-17k,qcnn-fusion-level32-qconv4x3`
- Thời gian (ước, chưa đo trên máy mới): mỗi CNN khoảng 30 phút đến 2 giờ; model quantum lâu nhất, có thể 1 đến 2 ngày.

## Kết quả

Sau mỗi run, `ket_qua\` có `mammoth_mean.csv` (accuracy sau mỗi task), `mammoth_tasks.csv` (từng task) và log
(`log_<tên>.txt`; trong log có Backward Transfer, Forgetting). Gửi cả folder `ket_qua\`.

Lưu ý khi đọc: "Accuracy for N task(s)" là **validation**; dòng "Accuracy for final task(s)" ở cuối là **test** với các
góc mới, chỉ để báo cáo, không dùng để chọn model.

## Ngưỡng chốt trước (val sau task 20, seed 0)

- Quantum (`qconv4x3`) so `cnn-17k`: hơn nếu ≥ +0,30; ngang trong ±0,30; kém nếu ≤ −0,30.
- Encoder quantum có ích trong bản đầy đủ nếu `qconv4x3` ≥ `deep6p` + 0,30.
- Mốc cũ cùng cấu hình: cnn-19k 95,43; deep12p (encoder CNN 1.646) 94,58; level32 không encoder 91,50.

## Nội dung

- `setup.ps1`, `run_all.ps1`, `CHAY_TAT_CA.bat`, `LENH_CHAY_LE.txt` (lệnh chạy tay từng run)
- `patch_mammoth.ps1` (2026-11-16, `setup.ps1` và `run_all.ps1` tự gọi): đổi tên các file `datasets\`/`models\` cần
  `open_clip` thành `.py.disabled` (nếu không, mọi lệnh lỗi `No module named 'open_clip'`), sửa `utils\loggers.py`
  dòng 387 (Task-IL là `None` với rot-mnist), cài `six` (`main.py` cần để tải MNIST), `requests`, `pandas`.
  Chạy được nhiều lần.
- `mammoth_files\`: `backbone\` (QuantumCL.py + quantum_core), `run_mammoth.py` (log, dừng sớm), `export_mammoth_logs.py`
  (log → CSV), `test_quantum_backbones.py`. Chép từ `mammoth_qconv_front`, thêm `cnn-17k/30k/50k/85k` và
  `qcnn-fusion-level32-deep6p`; `front.py` giống hệt bản MNIST tĩnh.
