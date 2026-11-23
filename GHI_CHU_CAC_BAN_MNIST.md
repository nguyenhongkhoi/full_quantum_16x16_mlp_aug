# Ghi chú các bản đã thử trên MNIST tĩnh

Cập nhật 2026-11-06. Chỉ gồm các run MNIST tĩnh trước khi chuyển sang Mammoth. Số lấy từ `bieu_do/tong_hop_cu.csv`
(18 CSV người dùng gửi) và `SO_NGHIEN_CUU_QCNN.txt`. Mọi số là **validation**, một seed, chưa dùng official test.

- **CK:** accuracy tại epoch có validation loss thấp nhất. **MAX:** accuracy cao nhất qua các epoch.
- Chênh dưới khoảng 0,3 điểm chưa đủ để kết luận.

## 1. Phần chung của mọi bản

| Thành phần | Cách làm |
|---|---|
| Dữ liệu | MNIST, 10.000 ảnh validation tách cố định (split-seed 20260911) |
| Đầu vào | Ảnh 28×28 resize về 16×16 (256 pixel) |
| Augmentation khi train | Xoay ±5 độ, dịch tối đa 0,05. Riêng `mlp256_3nhanh` xoay 8 độ |
| Mã hóa quantum | Mỗi pixel `x` thành 1 qubit `cos(πx/2)\|0⟩ + sin(πx/2)\|1⟩` |
| Lịch train | Phase 1: 20 epoch, lr 2e-3. Phase 2: 5 epoch, lr 2e-4. Phase 3: lr mạch 1e-4, lr head 1e-3, giảm lr khi chững, dừng sau 10 epoch không cải thiện |
| Mô phỏng | Chính xác trên máy cổ điển, không nhiễu, không shot. Không phải bằng chứng lợi thế lượng tử |

## 2. Cây các ý tưởng

```text
QCNN cây 1 qubit
├─ tree1q ──► R0 ──┬─ balanced   (đổi khởi tạo)
│                  ├─ fusion2    (đổi cách gộp nhánh)
│                  └─ fusion3    (3 nhánh + gộp)
└─ fusion_cu  (3 nhánh, gộp, không side tap; các bản MLP-16 3 nhánh là anh em của nó)

Cây bond-2 (giữ 2 qubit mỗi nút):  b0 ──► b1 ──► b2 ──► b2_shift02

Ô vướng víu:  s1_p8 ──► s2_p12 ──┬─ s3_ru2
                                 └─ s9_meas ──┬─ s9_meas_frozen, s9_meas_random
                                              └─ đối chứng c0, c1, c2, c4

CNN cổ điển:  cnn1k, cnn2k, cnn3k, cnn5k
```

## 3. Họ QCNN cây 1 qubit

**Ý tưởng chung:** 256 qubit xếp theo một thứ tự quét. Mỗi tầng ghép từng cặp kề nhau, áp 2 gate (`conv`, `pool`)
rồi bỏ một qubit (partial trace): 256 → 128 → … → 8 qubit. Mọi cặp trong cùng tầng dùng chung gate. Sau đó 8 qubit được
vướng víu với nhau, đo các giá trị Pauli, rồi đưa vào head cổ điển.

| Bản | Khác bản trước ở đâu | Tham số | CK | MAX | Ghi chú |
|---|---|---:|---:|---:|---|
| `tree1q` | Cây 1 qubit, 2 nhánh, head nhỏ | 3.060 | 96,85 | 96,91 | Mốc của họ cây |
| `R0` | Thêm side tap (đo ở tầng giữa), stage vướng víu 8 qubit, re-upload ở phase 3; 2 nhánh `row` và `morton`, mỗi nhánh một MLP-16, lấy **trung bình logits** | 12.420 | 98,32 | 98,34 | Quantum 1.616 |
| `balanced` | Như R0, chỉ đổi khởi tạo gate `conv` gần `sqrt(iSWAP)` để mọi pixel ảnh hưởng ngay từ đầu | 12.420 | 98,33 | 98,33 | Đến ep 41. Ngang R0, ep 1 còn thấp hơn (93,55 so với 94,38) |
| `fusion2` | Như R0, nhưng **một MLP chung** đọc đặc trưng cả 2 nhánh, thay vì trung bình logits | 12.202 | 98,06 | 98,08 | Đến ep 46. Thấp hơn R0 |
| `fusion3` | 3 nhánh (`row`, `column`, `morton`), MLP chung, **có side tap**; nối thẳng 972 → 16 | 18.194 | 98,33 | 98,43 | Đến ep 55. Ngang R0 dù nhiều tham số hơn |
| `fusion_cu` | 3 nhánh, **không side tap**; mỗi nhánh nén 276 → 16 trước, rồi MLP chung 48 → 16 → 10 | 19.202 | 98,70 | 98,72 | QCNN tốt nhất; CSV chưa có trong folder |
| `mlp16_3nhanh` | Như `fusion_cu` nhưng mỗi nhánh tự phân loại rồi **trung bình logits** | 18.726 | – | ~98,63 | Số do báo cáo cũ, chưa có file |
| `mlp256_3nhanh` | 3 nhánh, head MLP-256, aug 8 độ | 226.806 | 98,80 | 98,84 | Head rất lớn |

**Rút ra:** tăng tham số và số nhánh giúp ít; các bản mạnh nhất đều ở khoảng 98,3-98,7. `fusion_cu` hơn `fusion3`
0,37 điểm dù không có side tap, nên side tap chưa chắc có ích. Khoảng 75-87% tham số của họ này nằm ở head cổ điển.

## 4. Cây bond-2

**Ý tưởng chung:** giữ **2 qubit** ở mỗi nút cây thay vì 1, để giảm cổ chai. Thứ tự Morton (gom pixel theo ô 2D),
1 nhánh. Mỗi nút đo 15 giá trị Pauli 2 qubit. Phần mạch chỉ có 256 tham số.

| Bản | Khác bản trước ở đâu | Tham số | CK | MAX |
|---|---|---:|---:|---:|
| `b0` | Head chỉ là `Linear(60,10)` trên 4 nút cuối | 866 | – | – |
| `b1` | Đọc thêm tầng 16 nút qua `Linear(15,6)` dùng chung, rồi `Linear(156,10)` | 1.922 | – | – |
| `b2` | Head thành 156 → 16 → 10 có LayerNorm, GELU | 3.066 | 97,10 | 97,15 |
| `b2_shift02` | Như `b2`, chạy thêm một lượt trên ảnh dịch vòng 2 pixel, cùng gate và head, trung bình logits | 3.066 | 97,47* | 97,54* |

\* Số của bản 80 epoch. File trong folder chỉ có 50 epoch (CK 97,51), cần thay.

**Rút ra:** giữ 2 qubit chỉ hơn `tree1q` khoảng 0,25 điểm, nên cổ chai 1 qubit không phải giới hạn chính.
Hai lượt dịch thêm 0,37 điểm, dưới ngưỡng 0,5 đã đặt.

## 5. Ô vướng víu và `s9`

**Ý tưởng chung:** chia ảnh thành ô, mô phỏng **đầy đủ** mọi qubit trong ô (statevector), để có vướng víu giữa các pixel
ngay ở tầng đầu. Mọi ô dùng chung mạch, giống bộ lọc CNN.

| Bản | Khác bản trước ở đâu | Tham số | CK | MAX |
|---|---|---:|---:|---:|
| `s1_p8` | Ô 2×4 = 8 qubit, giữ 2 qubit mỗi ô, gộp tiếp bằng cây bond-2 | 2.522 | 96,47 | 96,55 |
| `s2_p12` | Ô lớn hơn: 3×4 = 12 qubit | 2.538 | 96,78 | 96,78 |
| `s3_ru2` | Như `s2_p12`, mã hóa lại pixel 2 vòng (re-upload) | 2.602 | 97,07 | ≥97,12 |
| `s9_meas` | **Bỏ cây**: đo 53 giá trị mỗi ô (X, Y, Z trên 12 qubit và ZZ trên 17 cặp), `Linear(53,4)` dùng chung, rồi MLP. Thực chất là quanvolution | 2.546 | 98,09 | 98,10 |
| `s9_meas_frozen` | Như `s9_meas` nhưng gate không học | 2.546 | 95,77 | 95,77 |
| `s9_meas_random` | Như `s9_meas` nhưng khởi tạo random thay `balanced` | 2.546 | – | 97,37 (hết phase 2) |

**Rút ra:** `s9_meas` là bản quantum nhỏ tốt nhất, nhưng gate chỉ có 64 tham số (2,5%); phần còn lại là cổ điển.

## 6. Đối chứng của `s9`

Cùng ô ảnh, cùng head; mỗi bản bỏ đi một phần. So ở **hết phase 2** (epoch 25), vì mọi bản đều chạy đến đó.

| Bản | Còn lại gì | Tham số | Hết phase 2 |
|---|---|---:|---:|
| `c0_pixel` | Chỉ pixel thô, không mã hóa, không gate | 2.318 | 95,65 |
| `c1_encoding` | Đặc trưng của mã hóa sin, cos (tính bằng công thức), không gate | 2.434 | 97,13 |
| `c2_product` | Gate 1 qubit, **không vướng víu**, khởi tạo random | 2.514 | 97,22 |
| `c4_local_mlp` | MLP nhỏ cổ điển cho mỗi ô **thay hẳn mạch** | 2.542 | 98,03 |
| `s9_meas` | Mạch đầy đủ, khởi tạo `balanced` | 2.546 | 98,09 |
| `s9_meas_random` | Mạch đầy đủ, khởi tạo random | 2.546 | 97,37 |

**Rút ra:**
- Riêng mã hóa sin, cos đã thêm 1,48 điểm (c0 → c1).
- Bỏ vướng víu khi cùng khởi tạo random gần như không mất gì: `c2_product` 97,22 so với `s9_meas_random` 97,37.
- MLP cổ điển cùng cỡ gần bằng mạch quantum: `c4_local_mlp` 98,03 so với `s9_meas` 98,09, mà nhanh hơn khoảng 18 lần.
- Chưa tách được vai trò của khởi tạo `balanced`, vì không có bản không vướng víu tương ứng.

## 7. CNN cổ điển

Hai lớp conv (BatchNorm, ReLU, max pool), rồi head nhỏ, trên cùng ảnh 16×16, cùng lịch train.

| Bản | Tham số | CK | MAX |
|---|---:|---:|---:|
| `cnn1k` | 934 | 96,53 | 96,55 |
| `cnn2k` | 1.990 | 98,09 | 98,12 |
| `cnn3k` | 2.988 | 98,64 | 98,64 |
| `cnn5k` | 5.578 | 99,04 | 99,11 |

**Rút ra:** ở mọi cỡ, CNN cao hơn bản quantum cùng cỡ. `cnn3k` với 2.988 tham số đã cao hơn mọi QCNN cây trên 12.000 tham số.

## 8. Kết luận chung từ giai đoạn này

1. QCNN tốt nhất là `fusion_cu` (98,70), nhưng CNN nhỏ hơn 3 lần (`cnn5k`, 99,04) vẫn hơn.
2. Họ cây chạm trần khoảng 98,3-98,7; đổi khởi tạo, cách gộp nhánh hay số qubit mỗi nút đều tăng ít.
3. Phần lớn tham số và một phần đáng kể accuracy đến từ head cổ điển.
4. Ở `s9`, đối chứng cổ điển cùng cỡ đạt gần bằng, và bỏ vướng víu chưa thấy mất rõ.
5. Tất cả là một seed. Muốn viết báo cần 3-5 seed và official test một lần cho bản chốt.
