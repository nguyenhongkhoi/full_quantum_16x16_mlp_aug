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

Mọi run dùng seed 0, aug (xoay ±5°, dịch ±0,05) và dropout 0. Số tham số đếm theo cách FREE: unitary 2 qubit có 16 bậc tự do, 1 qubit có 4; cách đếm chi tiết ở `docs/CACH_DEM_THAM_SO.txt`.

**Cấu trúc:**
```
train.py            train QCNN (python train.py --help: mọi tùy chọn, --dataset mnist|fashion)
evaluate_test.py    test chính thức (chỉ một lần cho bản chốt)
analyze_routing.py  phân tích router trên validation
test_model.py       kiểm tra số tham số, so mạch với mô phỏng ma trận đầy đủ
plot_all.py         vẽ mọi biểu đồ so sánh (PNG + PDF vector) -> xem plot.md
smoke_test.ps1      smoke test vài phút (--limit-samples) trước khi train thật
command.txt         lệnh cài đặt và toàn bộ lệnh chạy
requirements.txt    torch CUDA 12.6
qcnn/
  layers/     unitary.py (unitary học được), gates.py (cổng statevector), pauli.py (đặc trưng Pauli),
              density.py (Bloch <-> ma trận mật độ), counting.py (đếm tham số FREE)
  encoding/   preprocessing.py (28x28 -> 16x16), angle.py (mã hóa góc), quanvolution.py (encoder qconv),
              quanvolution_direct.py (encoder không readout cổ điển)
  core/       tn_core.py (cây QCNN: conv-pool, khối 8 qubit strongly entangling), reupload.py (re-upload),
              scan.py (thứ tự quét row / column / morton)
  moe/        heads.py (head mỗi cây: samebasis, nocorr, ...), router.py (router lượng tử / cổ điển),
              model.py (MoEQCNN: ghép encoder + cây + head + router)
  data.py     MNIST / Fashion-MNIST, split 50k/10k, augmentation
  evaluation.py  gom logit mọi chế độ suy luận, tóm tắt
classical_cnn/  CNN cổ điển cùng cỡ (run_cnn.py, run_sweep.ps1, tong_hop.py, evaluate_test.py)
docs/           CACH_DEM_THAM_SO.txt (cách đếm tham số), PSEUDOCODE_S1.txt (pseudocode, tiếng Anh)
```

Các phiên bản cũ và toàn bộ thí nghiệm nằm ở các branch khác: `archive-all` chứa toàn bộ workspace, `archive-main-s1-v1` là main cũ.
