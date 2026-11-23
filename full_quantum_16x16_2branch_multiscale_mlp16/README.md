# FullQuantum 16×16: hai nhánh đa tỉ lệ, MLP-16

Folder này là một thí nghiệm độc lập dựa trên runner 3scan MLP-16. Mô hình giữ ảnh MNIST ở 16×16, dùng hai thứ tự quét `row` và `morton`, đọc thêm đặc trưng ở tầng pooling trung gian, và tham số hóa trực tiếp generator Hermitian của các unitary để bỏ các chiều tham số dư.

Đây là giả thuyết kiến trúc cần đo trên validation; chưa có bảo đảm sẽ vượt 99%. Không dùng checkpoint của folder 3scan vì cấu trúc nhánh và cách lưu tham số unitary đã đổi. Script train từ đầu và giữ nguyên lịch ba phase, cách chia dữ liệu, augmentation, checkpoint, đồ thị và log batch.

## Luồng kiến trúc

```text
MNIST 28×28
  → bilinear resize 16×16
  → angle encoding thành 256 site
  → hai core độc lập: row và Morton
      → 5 block conv/pool, từ 256 site còn 8 qubit
      → side tap sau block 4: 16 site × (X, Y, Z) = 48 đặc trưng
      → reupload góc ảnh ở 8 qubit cuối trong phase 3
      → strongly-entangling stage trên 8 qubit
      → 276 Pauli feature cuối + 48 feature side tap = 324
      → MLP riêng mỗi nhánh: 324 → 16 → 10
  → trung bình logits hai nhánh
```

Side tap lấy các Bloch vector của 16 trạng thái một qubit sau block thứ tư, trước lần pooling cuối. Nó bổ sung một mức biểu diễn ở giữa core vào readout; không phải ghép logits hoặc ghép đặc trưng cuối của các head.

## Khác với bản 3scan MLP-16

| Thành phần | Bản cũ 3scan | Bản mới |
|---|---|---|
| Kích thước ảnh | 16×16 | 16×16 |
| Số core / thứ tự quét | 3: row, column, Morton | 2: row, Morton |
| Pooling | 256 → 8 qua 5 block | Giữ nguyên |
| Đặc trưng readout | 276 Pauli ở cuối core | 276 Pauli cuối + 48 Bloch giữa core |
| Readout | MLP riêng từng nhánh, hidden=16 | MLP riêng từng nhánh, hidden=16 |
| Reupload phase 3 | 8 scale/core ở cuối pooling | Giữ nguyên |
| Unitary | Lưu ma trận real/imag đầy đủ rồi Hermitian hóa | Lưu trực tiếp các hệ số Hermitian độc lập |
| Số tham số phase 3 | khoảng 18.726 | khoảng 12.420 |

Ước tính mới dùng đúng cấu hình MLP-16, `full9`, 8 qubit cuối và hai nhánh:

- Mỗi readout nhận 324 đặc trưng: `Linear(324,16)` + `LayerNorm(16)` + `Linear(16,10)` = 5.402 tham số.
- Mỗi core còn khoảng 800 tham số unitary sau khi bỏ các chiều dư trong generator.
- Hai core + hai readout + 16 scale reupload = `2×800 + 2×5.402 + 16 = 12.420` tham số phase 3.
- Bản 3scan cũ khoảng 18.726 tham số. Mức giảm ước tính là 6.306 tham số, khoảng 33,7%.

Trong unitary cũ, `H = 0.5 * (A + A†)` chỉ phụ thuộc vào phần Hermitian của `A`; các chiều còn lại không tác động đến phép forward. Generator mới lưu trực tiếp diagonal thực và tam giác trên phức, rồi dựng phần đối xứng Hermitian. Vì thế nó giữ cùng họ phép biến đổi `exp(iH)` nhưng có ít tham số lưu trữ hơn. Cách khởi tạo được chọn để giữ gần phân phối generator ban đầu.

## Train và log

Mở PowerShell tại `D:\data_system\full_quantum_16x16_mlp_aug` rồi chạy lệnh trong `command.txt`. Lệnh mặc định dùng CUDA, seed 0, split seed 20260911, batch 64, augmentation xoay ±5° và dịch 0,05.

Trong mỗi batch, terminal tiếp tục hiển thị loss và accuracy tích lũy đến batch hiện tại cùng thời gian đã chạy; cuối mỗi epoch ghi train/validation loss, accuracy, learning rate và số giây. Tỷ lệ progress cập nhật ngay trên cùng một dòng.

Run lưu vào `runs/run_2branch_multiscale/`:

- `fq_mlp_aug_phase*_best.pt`, `*_last.pt`: checkpoint tốt nhất và trạng thái để resume của từng phase.
- `full_quantum_mlp_aug_best.pt`: checkpoint phase 3 có validation loss thấp nhất.
- `history_full_quantum_mlp_aug.csv` và các `*_history.csv`: log từng epoch.
- `convergence_mlp_aug.png`: biểu đồ train/validation loss và accuracy.
- `summary_mlp_aug.json`, `config_mlp_aug.json`, `split_indices.csv`: kết quả, cấu hình và split.

Script chọn checkpoint phase 3 theo validation loss; summary cũng ghi epoch có validation accuracy cao nhất. MNIST official test không dùng để chọn checkpoint. Sau khi chốt cấu hình bằng validation, chạy lệnh đánh giá test thứ hai trong `command.txt`; nó ghi accuracy, accuracy từng lớp, confusion matrix và danh sách ảnh phân loại sai.

## Cấu trúc code

- `run_mlp_aug_16x16_multiscale_mlp16.py`: train ba phase, lưu checkpoint/CSV/JSON và vẽ biểu đồ.
- `models/full_quantum_qcnn.py`: hai nhánh quét, side-feature readout và trung bình logits.
- `models/layers/tn_core.py`: pooling core và trích Bloch vector ở tầng trung gian.
- `models/layers/readout.py`: Pauli readout nhận thêm 48 đặc trưng side tap.
- `models/layers/unitary.py`: unitary với tham số Hermitian gọn.
- `models/reupload.py`: reupload phase 3, đồng thời giữ side tap.
- `evaluate_test.py`: đánh giá checkpoint đã chọn trên test chính thức.

Cài các dependency trong `requirements.txt` trong môi trường PyTorch/torchvision có CUDA phù hợp với máy train.
