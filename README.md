# Quantum-Routed Mixture of Scan-Order QCNN Experts (S1)

Phân loại ảnh 16×16 (MNIST, Fashion-MNIST) bằng QCNN tensor network với router lượng tử. Repo có kèm CNN cổ điển cùng cỡ để so sánh.

**Kiến trúc S1:**
1. Encoder là quanvolution `qconv4x3`: 3 tầng, cửa sổ 3×3 trên 9 qubit, 4 lớp RZRYRZ + CZ. Encoder sinh góc mã hóa θ, φ cho từng pixel.
2. 3 cây QCNN level32 quét ảnh theo 3 thứ tự: hàng, cột, Morton. Mỗi cây gồm 5 tầng conv-pool dùng chung cổng và khối 8 qubit ở gốc.
3. Mỗi cây có head `Linear(204,10)`, nhận 24 giá trị Bloch, 84 tương quan cùng trục và 96 giá trị đo ở tầng 32.
4. Router lượng tử 8 qubit tính trọng số softmax g, rồi trộn logit các cây: Σ gₑ·ℓₑ.

| Model | Tham số FREE | MNIST MAX val | Fashion-MNIST |
|---|---:|---:|---|
| **S1** (bản chính) | 9.389 | 99,10 | chưa chạy |
| G1 = S1 + `--share-gates` | 7.709 | 98,88 | chưa chạy |
| CNN cùng cỡ (`--widths 14 20`) | 9.378 | chưa chạy | chưa chạy |

Mọi run dùng seed 0, aug (xoay ±5°, dịch ±0,05) và dropout 0. Số tham số đếm theo cách FREE: unitary 2 qubit có 16 bậc tự do, 1 qubit có 4; cách đếm chi tiết ở `CACH_DEM_THAM_SO.txt`.

**Các file:**
- `requirements.txt`: cài torch CUDA 12.6.
- `command.txt`: lệnh cài đặt và toàn bộ lệnh chạy (S1, G1/G2, CNN, Fashion-MNIST, test).
- `train.py`, `models/`, `common.py`: code QCNN S1, có đủ tùy chọn (`python train.py --help`).
- `test_model.py`: kiểm tra số tham số và so mạch với mô phỏng ma trận đầy đủ.
- `evaluate_test.py`: chạy test chính thức (chỉ một lần cho bản chốt).
- `analyze_routing.py`: phân tích router trên tập validation.
- `classical_cnn/`: CNN cổ điển (`run_cnn.py`, `run_sweep.ps1`, `tong_hop.py`, `evaluate_test.py`).
- `PSEUDOCODE_S1.txt`: pseudocode toàn bộ pipeline (tiếng Anh).

Các phiên bản cũ và toàn bộ thí nghiệm nằm ở các branch khác: `archive-all` chứa toàn bộ workspace, `archive-main-s1-v1` là main cũ.
