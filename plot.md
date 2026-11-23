# Vẽ biểu đồ so sánh

Script `plot_all.py` đọc mọi run đã train và vẽ hết các biểu đồ so sánh. Mỗi hình được lưu hai bản: **PNG** để xem nhanh và **PDF vector** để đưa vào bài báo (chữ trong PDF vẫn là text, sửa được). Script không cần GPU và chỉ đọc file `history.csv` / `config.json`, nên chạy được cả khi run chưa xong.

## 1. Một lệnh vẽ đủ

Đứng ở thư mục gốc repo (cùng chỗ với `train.py`):

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
python .\plot_all.py
```

- Script tìm mọi `history.csv` trong `.\runs` (QCNN) và `.\classical_cnn\runs` (CNN).
- Kết quả nằm trong `.\figures`.
- MNIST và Fashion-MNIST được tách riêng. Dataset đọc từ `config.json` của từng run.
- Mỗi cỡ CNN có thể đã chạy 3 lr; mặc định script chỉ giữ lr có MAX val cao nhất.

Lệnh này in ra bảng tóm tắt từng run, sau đó là danh sách các file đã ghi.

## 2. Các hình được tạo

| File (`.png` và `.pdf`) | Nội dung |
|---|---|
| `convergence_<ds>` | Hình 2×2 theo epoch: val acc, val loss (trục log), train loss (trục log), train acc. QCNN vẽ nét liền, CNN nét đứt. |
| `convergence_zoom_<ds>` | Val acc phóng vào vùng cao, chú thích có MAX của từng run. Dùng để so sát S1, G1 với CNN. |
| `acc_vs_params_<ds>` | MAX val acc theo số tham số, trục x log. QCNN dùng số FREE. |
| `best_acc_<ds>` | Cột MAX val acc và CK (val acc ở epoch có val loss nhỏ nhất). Xanh là QCNN, cam là CNN. |
| `lr_<ds>` | Lịch learning rate (ReduceLROnPlateau). |
| `time_<ds>` | Số phút trung bình mỗi epoch. Số này phụ thuộc máy và tải GPU. |
| `router_<run>` | Mỗi run QCNN một hình: (1) val acc theo chế độ soft / top1 / uniform / oracle / từng cây; (2) g trung bình và tỷ lệ top1 của từng cây; (3) entropy của cổng router. |
| `summary_<ds>.csv`, `summary_<ds>.md` | Bảng: MAX val, epoch đạt MAX, CK, val loss nhỏ nhất, số tham số, số epoch, run đã xong chưa. |

`<ds>` là `mnist` hoặc `fashion`.

## 3. Lệnh thường dùng

Đặt nhãn ngắn cho hình. Theo mặc định nhãn là tên thư mục run; CNN tự có nhãn dạng `CNN 14-20 (9.378)`:

```powershell
python .\plot_all.py --labels mnist_S1_samebasis=S1 mnist_G1_sharegates=G1 mnist_G2_sharegates_sharefinal=G2 fashion_S1_samebasis=S1 
```

Chỉ vẽ một dataset:

```powershell
python .\plot_all.py --dataset mnist
python .\plot_all.py --dataset fashion
```

Chỉ vẽ vài run (lọc theo chuỗi có trong tên thư mục hoặc nhãn), hoặc bỏ bớt run:

```powershell
python .\plot_all.py --include S1 G1 w14_20 w13_18
python .\plot_all.py --exclude G2
```

Thêm run nằm ở thư mục khác, ví dụ run cũ trên máy train:

```powershell
python .\plot_all.py --add "S1 cu=D:\tensornetwork_mnist\tensornetwork_mnist\fullquantum_16x16_frontQuan4x3_Moe\runs\moe_quantum_samebasis"
```

Quét thêm thư mục run khác (thay hoàn toàn danh sách mặc định):

```powershell
python .\plot_all.py --runs .\runs .\classical_cnn\runs D:\runs_khac
```

Vẽ đủ mọi lr của CNN để so lr:

```powershell
python .\plot_all.py --all-cnn-lr --include w14_20 --out-dir .\figures_cnn_lr
```

Chọn định dạng (thêm SVG) và thư mục ra:

```powershell
python .\plot_all.py --formats png pdf svg --out-dir .\figures_paper
```

Tự đặt cận dưới trục y cho hình zoom (mặc định là MAX cao nhất trừ 2):

```powershell
python .\plot_all.py --zoom-low 98
```

## 4. Lệnh đầy đủ cho bài báo (MNIST + Fashion, nhãn gọn, PDF vector)

```powershell
python .\plot_all.py --dataset all --formats png pdf --out-dir .\figures_paper --labels mnist_S1_samebasis=S1 mnist_G1_sharegates=G1 mnist_G2_sharegates_sharefinal=G2 fashion_S1_samebasis=S1 fashion_G1_sharegates=G1
```

Các file cần đưa vào bài nằm ở `.\figures_paper\*.pdf`, chèn bằng `\includegraphics{...pdf}`.

## 5. Ghi chú

- Run đang chạy vẫn vẽ được vì `history.csv` được ghi lại sau mỗi epoch. Cột `xong` trong bảng tóm tắt bằng `False` khi run chưa có `summary.json`.
- Mỗi run vẫn có hình riêng `convergence.png` trong thư mục run do `train.py` / `run_cnn.py` tự vẽ. `plot_all.py` dùng để so nhiều run với nhau.
- Mọi số là **validation** (chọn model). Số test chính thức lấy từ `evaluate_test.py`, chỉ chạy một lần cho bản chốt, không vẽ ở đây.
- `figures/` không đưa lên git. Muốn lưu hình thì chép ra ngoài hoặc commit riêng.
