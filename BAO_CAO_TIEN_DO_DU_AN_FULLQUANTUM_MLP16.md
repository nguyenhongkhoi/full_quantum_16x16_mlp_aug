# Báo cáo tiến độ dự án FullQuantum MNIST 16×16 MLP-16

**Cập nhật:** 03/11/2026  
**Vị trí file:** thư mục gốc `full_quantum_16x16_mlp_aug` — nằm ngoài folder thí nghiệm `full_quantum_16x16_2branch_multiscale_mlp16`.

## 1. Mục tiêu dự án

Dự án nghiên cứu mô hình FullQuantumQCNN/tensor-network cho phân loại MNIST, với hai tiêu chí được ưu tiên đồng thời:

1. Độ chính xác cao, hướng tới mốc 99%.
2. Số tham số học được thấp.

Mốc 99% là mục tiêu thử nghiệm, chưa phải kết quả được đảm bảo. Cần tách rõ accuracy trên validation và official test: validation dùng để chọn kiến trúc/checkpoint; test chỉ dùng báo cáo đánh giá cuối cho cấu hình đã chốt.

## 2. Dự án đang ở giai đoạn nào?

### Đã hoàn thành

- Đã thử và so sánh nhiều hướng kiến trúc trong quá trình làm dự án: nhiều quantum branch, thay thứ tự quét ảnh, thay kích thước đầu vào lên 32×32, ghép/fusion readout, và các biến thể MLP-16.
- Đã tạo thí nghiệm mới tại `full_quantum_16x16_2branch_multiscale_mlp16/` để ưu tiên giảm tham số nhưng vẫn giữ thêm thông tin nhiều tỉ lệ.
- Folder mới có runner train ba phase, model hai nhánh, các layer cho side tap đa tỉ lệ và generator Hermitian gọn, evaluator test, README, requirements và command.
- Lỗi `NameError: architecture is not defined` xuất hiện ở lần chạy đầu đã được sửa: runner hiện khởi tạo `architecture = fq_arch(args)` trước khi tính số tham số phase 3.

### Chưa hoàn thành

- Chưa có epoch train nào được ghi nhận cho kiến trúc hai nhánh đa tỉ lệ trong folder run này.
- Hiện chưa có checkpoint, history CSV, summary hay biểu đồ hội tụ của thí nghiệm mới.
- Vì vậy chưa biết kiến trúc mới tăng/giảm validation accuracy bao nhiêu, chưa xác nhận được mức tiết kiệm tham số bằng một run thực tế, và chưa có kết quả official test.
- Lần chạy được báo qua ảnh đã tải MNIST rồi dừng ở lỗi trước epoch đầu tiên. Sau khi sửa lỗi, cần chạy lại command; dữ liệu đã tải chỉ được tái sử dụng nếu còn ở đúng `--data-root` trên máy chạy.

**Kết luận trạng thái:** phần thiết kế và code thí nghiệm mới đã sẵn sàng để chạy; bước quyết định tiếp theo là train và đọc kết quả validation. Chưa thể kết luận nó tốt hơn các bản trước hay đạt 99%.

## 3. Các mốc kết quả cũ cần hiểu đúng

Các con số dưới đây là kết quả đã được người dùng cung cấp hoặc ghi trong báo cáo cũ; chúng không phải kết quả của mô hình hai nhánh mới:

| Thí nghiệm cũ | Kết quả được ghi nhận | Giới hạn khi diễn giải |
|---|---:|---|
| Ba nhánh, MLP-16, dropout | Validation khoảng 98,52% ở checkpoint được báo cáo | Cần đối chiếu cấu hình/run cụ thể nếu dùng làm mốc so sánh chính thức. |
| Ba nhánh, MLP-16, tắt dropout | Ảnh log giữa run cho thấy khoảng 98,63% | Đây là số giữa quá trình train, chưa đủ để gọi là kết quả cuối. |
| Bản đầu vào 32×32 | Người dùng ghi nhận khoảng 98,6% | Khác độ phân giải và số phép pooling; không phải so sánh tham số/công bằng trực tiếp với 16×16. |
| KD/fusion và các thử nghiệm thay đổi nhánh | Chưa cho thấy cải thiện đủ ổn định để trở thành hướng chính | Giữ lại làm lịch sử thử nghiệm, không xem là bằng chứng rằng mọi cách fusion đều thất bại. |

Các khác biệt nhỏ cỡ vài phần mười điểm phần trăm cần được xác nhận bằng cùng split, seed, augmentation và quy tắc chọn checkpoint. Trên validation 10.000 ảnh, 0,1 điểm phần trăm chỉ tương đương khoảng 10 ảnh.

## 4. Kiến trúc thí nghiệm mới: hai nhánh đa tỉ lệ

Luồng xử lý dự kiến:

```text
MNIST 28×28 grayscale
  → resize bilinear về 16×16
  → angle encoding 256 site
  → hai core lượng tử độc lập theo thứ tự row và Morton
      → 5 block convolution/pooling, 256 site giảm còn 8 qubit
      → trích side tap sau block thứ tư: 16 site × (X,Y,Z) = 48 đặc trưng
      → phase 3 re-upload góc ảnh lên 8 qubit cuối
      → strongly-entangling stage: 4 layer
      → 276 đặc trưng Pauli cuối + 48 đặc trưng side tap
      → readout riêng mỗi nhánh: 324 → 16 → 10
  → trung bình logits của hai nhánh
```

### Điều gì thay đổi so với mô hình ba nhánh 16×16?

- Dùng hai thứ tự quét khác nhau (`row`, `morton`) thay vì ba nhánh.
- Mỗi nhánh vẫn có core và MLP riêng; không chia sẻ readout.
- Mỗi readout vẫn có hidden size 16.
- Đọc thêm 48 đặc trưng Bloch ở tầng giữa core để bổ sung thông tin đa tỉ lệ cho readout cuối.
- Unitary được tham số hóa bằng các hệ số Hermitian độc lập thay vì lưu ma trận real/imag dư thừa rồi mới Hermitian hóa.
- Giữ input 16×16 và lịch train ba phase để có thể đánh giá một thay đổi kiến trúc có kiểm soát.

Side tap không tạo một classifier/head mới. Nó nối thông tin tầng giữa vào vector đặc trưng của readout cuối mỗi nhánh.

### Số tham số

README của thí nghiệm ước tính phase 3 có khoảng **12.420 tham số**, so với khoảng **18.726** ở mô hình ba nhánh MLP-16 cũ: giảm xấp xỉ **6.306 tham số (33,7%)**. Ước tính mới gồm hai core khoảng 800 tham số/core, hai readout 5.402 tham số/readout và 16 scale re-upload.

Đây là số theo cấu hình dự kiến trong README; cần ghi lại số thực tế mà runner in ra khi bắt đầu train để đối chiếu. Ít tham số hơn không tự động đồng nghĩa accuracy cao hơn hoặc thời gian train ngắn hơn: phép tính quantum/tensor-network vẫn có thể chiếm phần lớn thời gian.

## 5. Dữ liệu và cấu hình train hiện tại

- Dataset: MNIST.
- Ảnh xám 28×28 được resize bilinear về 16×16.
- Chia tập train/validation cố định: 50.000/10.000, `split-seed=20260911`.
- Validation không augmentation; chỉ chuyển ảnh thành tensor.
- Train augmentation: RandomAffine, xoay ±5°, tịnh tiến tối đa 0,05, nội suy bilinear.
- Batch size: 64; seed train mặc định 0.
- Phase 1: Adam, LR `2e-3`, tối đa 20 epoch; chọn checkpoint theo validation accuracy.
- Phase 2: tiếp tục từ checkpoint tốt nhất phase 1, Adam mới, LR `2e-4`, tối đa 5 epoch; chọn theo validation accuracy.
- Phase 3: thêm re-upload, LR core `1e-4`, LR readout `1e-3`, ReduceLROnPlateau; tối đa 60 epoch, dừng sớm sau 10 epoch không cải thiện validation loss; checkpoint phase 3 chọn theo validation loss.
- Official test không tham gia train hoặc chọn checkpoint.

Vì phase 3 chọn checkpoint theo validation loss, epoch có validation accuracy cao nhất có thể khác epoch được lưu làm checkpoint cuối. Runner báo cả hai thông tin trong summary.

## 6. Các file chính của thí nghiệm mới

| File | Vai trò |
|---|---|
| `run_mlp_aug_16x16_multiscale_mlp16.py` | Tạo split, train ba phase, resume, lưu log/checkpoint và vẽ biểu đồ. |
| `models/full_quantum_qcnn.py` | Hai nhánh row/Morton, readout riêng, trung bình logits. |
| `models/layers/tn_core.py` | Các block pooling và trích side tap tầng giữa. |
| `models/layers/readout.py` | Kết hợp Pauli features cuối với 48 features từ side tap. |
| `models/layers/unitary.py` | Generator Hermitian tham số hóa gọn. |
| `models/reupload.py` | Bổ sung re-upload phase 3 và chuyển tiếp side tap. |
| `evaluate_test.py` | Đánh giá checkpoint trên MNIST official test, gồm accuracy, theo lớp, confusion matrix và ảnh dự đoán sai. |
| `command.txt` | Lệnh train và lệnh test chạy từ thư mục gốc dự án. |
| `README.md` | Mô tả thiết kế, so sánh mô hình cũ và mới, file đầu ra. |

## 7. Lệnh chạy và đầu ra

Mở PowerShell tại thư mục gốc `D:\data_system\full_quantum_16x16_mlp_aug`, rồi chạy lệnh train trong `full_quantum_16x16_2branch_multiscale_mlp16\command.txt`. Lệnh đang cấu hình CUDA và ghi kết quả vào:

```text
full_quantum_16x16_2branch_multiscale_mlp16/runs/run_2branch_multiscale/
```

Runner dự kiến lưu:

- checkpoint `best` và `last` cho từng phase để khôi phục/resume;
- `full_quantum_mlp_aug_best.pt` cho checkpoint phase 3 được chọn theo validation loss;
- CSV log theo epoch và theo phase;
- `config_mlp_aug.json`, `summary_mlp_aug.json`, `split_indices.csv`;
- biểu đồ `convergence_mlp_aug.png`.

Sau khi hoàn tất train và chốt checkpoint bằng validation, chạy lệnh evaluator thứ hai trong `command.txt` đúng một lần cho đánh giá official test. Nếu cần đổi hyperparameter hoặc kiến trúc, tạo `--out-dir` mới để không lẫn artifacts giữa các lần chạy.

## 8. Kế hoạch đánh giá kết quả

1. Chạy lại đúng command đã lưu, giữ nguyên seed, split, augmentation và hyperparameter.
2. Xác nhận runner bắt đầu được, in số tham số, rồi ghi epoch 1; nếu lỗi thì lưu toàn bộ traceback để sửa đúng nguyên nhân.
3. Khi train kết thúc, đọc `summary_mlp_aug.json` và CSV: ghi riêng validation accuracy tại checkpoint theo loss, validation accuracy tối đa, epoch tương ứng và thời gian train.
4. So sánh với baseline chỉ khi cấu hình dữ liệu/split tương đương; báo cả số tham số và độ chính xác, không chỉ accuracy.
5. Chỉ sau khi chọn cấu hình bằng validation mới chạy `evaluate_test.py`; báo test accuracy cùng confusion matrix và per-class accuracy.
6. Không điều chỉnh chính sách augmentation/TTA hoặc chọn epoch dựa trên official test.

## 9. Tiêu chí kết luận

- **Thành công về nén:** số tham số thực tế thấp hơn baseline và model vẫn giữ được validation accuracy gần baseline.
- **Thành công về accuracy:** validation tăng ổn định theo cùng protocol; nếu chênh lệch nhỏ, nên xác nhận thêm qua seed khác trước khi khẳng định cải thiện.
- **Đạt mục tiêu tổng thể:** accuracy test cạnh tranh đồng thời với mức tham số thấp hơn đáng kể. Chỉ được kết luận sau khi có log train hoàn chỉnh và test đánh giá đúng quy trình.

Hiện tại, thí nghiệm hai nhánh đa tỉ lệ là một hướng hợp lý để kiểm tra trade-off giữa accuracy và số tham số, nhưng kết quả thực nghiệm vẫn đang chờ.
