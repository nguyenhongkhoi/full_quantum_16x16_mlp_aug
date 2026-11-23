# FullQuantum 16x16 Level32 — Static MNIST với Data Augmentation

Thư mục này hiện thực hóa mô hình **FullQuantum QCNN Level32** (kiến trúc đã chứng minh hiệu quả vượt trội trên framework Mammoth) để quay trở lại giải quyết bài toán **Phân loại MNIST tĩnh** có hỗ trợ **Data Augmentation** ($\pm 5^\circ$ xoay, $\pm 0.05$ dịch).

---

## 1. Mục tiêu & Động lực thiết kế

Ở các phiên bản tĩnh trước đây (`full_quantum_16x16_mlp_aug`, `3scan_fusion_mlp16`):
- Toàn bộ 256 site pixel bị co cụm liên tiếp qua 5 tầng pooling xuống còn đúng **8 qubit** rồi mới thực hiện đo Pauli. Quá trình trace-out liên tục đã xóa sổ các chi tiết nét chữ ở độ phân giải trung gian, khiến validation accuracy bị chặn ở trần **98.70%**.
- Thuật toán đo Pauli cũ phải sinh 276 ma trận kích thước $256 \times 256$ trong không gian Hilbert, gây tốn bộ nhớ VRAM và kéo dài thời gian huấn luyện (~17 giờ cho 60 epoch).
- Quy trình huấn luyện bị xé nhỏ thành 3 phase thủ công (Phase 1 pretrain $\rightarrow$ Phase 2 refine $\rightarrow$ Phase 3 vá re-upload).

**Bản `full_quantum_16x16_level32_aug` khắc phục toàn bộ các hạn chế trên:**
1. **Multi-level Side-Tap (Tầng 32-qubit):** Trích xuất thêm 96 đặc trưng vector Bloch $(X, Y, Z)$ từ 32 qubit ở tầng giữa, bảo toàn trọn vẹn thông tin nét chữ đa tỉ lệ.
2. **Fast Pauli Engine (`PauliFeatures`):** Tận dụng tính chất thưa của chuỗi Pauli, chỉ tra cứu $2^n$ phần tử thay vì nhân ma trận $256 \times 256$. Tiêu tốn bộ nhớ gần như bằng 0, tốc độ tăng vọt.
3. **Gradient Checkpointing:** Tiết kiệm hơn 60% VRAM (chỉ còn ~6 GB ở batch 64), loại bỏ hoàn toàn nguy cơ tràn bộ nhớ CUDA.
4. **Re-upload Tích hợp Sẵn:** Tham số scale re-upload được khởi tạo bằng 0 và học ngay từ epoch đầu, cho phép huấn luyện **End-to-End một mạch duy nhất** (Single Phase) với bộ điều chỉnh Learning Rate tự động.
5. **Tùy chọn Graph Layer 8 láng giềng:** Tích hợp tương tác Ising ZZ giữa 8 pixel lân cận, củng cố tính liên tục không gian 2D của nét chữ số MNIST.

---

## 2. Thông số Kiến trúc & Phân bổ Tham số

| Thành phần | Chi tiết | Số tham số |
| :--- | :--- | :---: |
| **Input & Mã hóa** | Ảnh $16 \times 16$ (256 sites) $\rightarrow$ Angle $R_y$ Encoding | 0 |
| **Tầng Đồ thị (Tùy chọn)** | `GraphZZLayer` (Ising ZZ 8 láng giềng, 10 param $\times$ 3 nhánh) | 0 hoặc 30 |
| **Quantum Tree Core** | 3 nhánh (`row`, `column`, `morton`), mỗi nhánh 5 block conv/pool + 4 layer Strongly Entangling + Reupload scale | 4.824 |
| **Trích xuất Đặc trưng** | - 96 đặc trưng Bloch từ 32 qubit tầng giữa<br>- 276 đặc trưng Pauli từ 8 qubit cuối<br>$\rightarrow$ **372 đặc trưng / nhánh** | 0 |
| **Per-tree Projection** | Mỗi nhánh: `Linear(372, 12) -> LayerNorm(12) -> GELU -> Dropout(0.2)` | $3 \times 4.476 = 13.428$ |
| **Fusion MLP Head** | Concat 3 nhánh ($36$) $\rightarrow$ `Linear(36, 16) -> LayerNorm(16) -> GELU -> Dropout(0.2) -> Linear(16, 10)` | 866 |
| **TỔNG CỘNG** | **Bản chuẩn (Standard Level32)** | **19.118** |
| **TỔNG CỘNG** | **Bản kèm Graph Layer (`--graph-layer`)** | **19.148** |

Dropout mặc định `p=0.2` chỉ được bật khi train và tự tắt ở eval; nó không có tham số học được. Các nhánh quantum, scan order, kích thước Linear/LayerNorm và tổng tham số giữ nguyên. Có thể đặt `--dropout 0` để tắt.

---

## 3. Cấu trúc Thư mục

```text
full_quantum_16x16_level32_aug/
├── models/
│   ├── __init__.py           # Export FusionQCNN, SPATIAL_ORDERS, SCAN_ANGLE_ORDERS
│   ├── fusion_model.py       # Kiến trúc chính FusionQCNN (hỗ trợ side_level=32, projection=12)
│   ├── graph_layer.py        # Tầng đồ thị lượng tử 8 láng giềng GraphZZLayer
│   ├── pauli.py              # Bộ đo Fast Pauli Features (tra cứu chỉ số thưa)
│   ├── reupload.py           # Core Re-upload lượng tử
│   ├── tn_core.py            # Cây Tensor Network Conv/Pool & Strongly Entangling
│   ├── unitary.py            # Cổng Unitary 1Q và 2Q tham số hóa ma trận mũ
│   ├── encoding.py           # Mã hóa góc Ry
│   └── preprocessing.py      # Tiền xử lý ảnh (Resize 16x16, grayscale)
├── run_level32_aug.py        # Script huấn luyện chính (Data Augmentation, Early Stopping, Logging, Plotting)
├── evaluate_test.py          # Script đánh giá trên tập official MNIST test (10.000 ảnh)
├── test_model.py             # Script kiểm tra số tham số và forward/backward pass
├── command.txt               # Tập hợp các câu lệnh sẵn sàng chạy
├── requirements.txt          # Danh sách thư viện phụ thuộc
└── README.md                 # Tài liệu này
```

---

## 4. Hướng dẫn Sử dụng

### Bước 1: Kiểm tra mô hình
Chạy test kiểm tra số lượng tham số và tính tương thích của đồ thị tính toán:
```powershell
python .\test_model.py
```

### Bước 2: Huấn luyện trên tập tĩnh (Train)
Huấn luyện bản Level32 chuẩn (19.118 tham số) với độ lệch góc xoay $\pm 5^\circ$ và tịnh tiến $\pm 0.05$:
```powershell
python .\run_level32_aug.py --side-level 32 --projection 12 --dropout 0.2 --aug-degrees 5 --aug-translate 0.05 --batch-size 64 --epochs 50 --lr 1e-3 --patience 10 --device cuda --data-root .\mnist_data --out-dir .\runs\level32_std_aug
```

Nếu muốn thử nghiệm bản có **Graph Layer 8 láng giềng** (19.148 tham số):
```powershell
python .\run_level32_aug.py --side-level 32 --projection 12 --dropout 0.2 --graph-layer --aug-degrees 5 --aug-translate 0.05 --batch-size 64 --epochs 50 --lr 1e-3 --patience 10 --device cuda --data-root .\mnist_data --out-dir .\runs\level32_graph_aug
```

Kết quả huấn luyện sẽ tự động lưu vào thư mục `out-dir`:
- `model_best_val_acc.pt`: Checkpoint tại epoch có validation accuracy cao nhất.
- `model_best_val_loss.pt`: Checkpoint tại epoch có validation loss thấp nhất.
- `history.csv`: Bảng ghi lại loss, accuracy, lr theo từng epoch.
- `convergence.png`: Biểu đồ đường cong hội tụ loss và accuracy.
- `summary.json` & `config.json`: Tóm tắt kết quả và cấu hình huấn luyện.

### Bước 3: Đánh giá trên tập Official Test (10.000 ảnh)
Sau khi kết thúc huấn luyện và chọn được checkpoint tốt nhất theo validation accuracy, chạy đánh giá test chính thức đúng 1 lần:
```powershell
python .\evaluate_test.py --checkpoint .\runs\level32_std_aug\model_best_val_acc.pt --data-root .\mnist_data --device cuda
```
Script sẽ xuất ra:
- `official_test_result.json`: Accuracy tổng, accuracy từng class (0–9), và ma trận nhầm lẫn (confusion matrix).
- `official_test_errors.csv`: Danh sách chi tiết các mẫu ảnh bị đoán sai kèm độ tin cậy (confidence).
