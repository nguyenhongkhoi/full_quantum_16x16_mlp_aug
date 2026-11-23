# fullquantum_16x16_frontQuan4x3_Moe

Encoder lượng tử `qconv4x3` → **router** (mixture of experts) → 3 cây level32 → mỗi cây một `Linear(372, 10)` →
trộn logit theo trọng số router. MNIST tĩnh 16×16. Tạo 2026-11-17; mới kiểm cú pháp, chưa chạy test.

## Tóm tắt: ba nhóm tùy chọn, ghép tùy ý

| Nhóm | Tùy chọn | Giá trị |
|---|---|---|
| MoE | `--router` | `none` (không MoE, trung bình đều) / `quantum` / `mlp` / `linear` |
| Head mỗi cây | `--expert-head` | `linear` (L1) / `shared` / `mlp` / **`bloch8`** / **`obs10`** / **`nocorr`** / `fusion` (head 2c, chỉ với router none) |
| Readout encoder | `--front-readout` | `linear` (như cũ) / **`direct`** (bỏ `Linear(27,2)`, qubit giữa vào thẳng cây) |

Số tham số với router none (thêm router lượng tử: +643):

| Cấu hình | Tổng | Quantum |
|---|---:|---:|
| L1 (linear) | 16.506 | 31,2% |
| bloch8 | 6.066 | 84,9% |
| obs10 | 5.376 | 95,8% |
| **obs10 + direct** | **5.208** | **98,8%** |
| obs10 + direct + router lượng tử | 5.851 | 98,9% |

### Lưu ý
- `--front-readout direct` **không bắt đầu như ánh xạ đồng nhất**: mọi run trước (và readout linear) lúc khởi tạo
  giống hệt bản không encoder; run direct thì không. Nhớ điều này khi so.
- `obs10` dùng 10 quan sát mặc định `Z0 … Z7 X0 X4` (chọn tạm); đổi bằng `--obs-terms`, cặp viết như `Z0Z1`.
- `nocorr` (thêm 2026-11-18): Linear(120,10) trên 24 Bloch + 96 tầng 32, chỉ bỏ 252 tương quan cặp; router quantum 9.589 (quantum ~60,4%), router none 8.946. Lệnh N1 trong command.txt.
- Head nhỏ hơn nocorr (thêm 2026-11-19, chạy nếu N1 >= 98,73; lệnh N2-N5 trong command.txt; số với router quantum): `nocorr-z` Linear(56,10), chỉ <Z> tầng 32 (7.669, quantum 75,5%); `nocorr-lowrank --head-rank 4` (7.549, 76,7%), `--head-rank 2` (6.769, 85,5%); `nocorr-shared` một Linear(120,10) chung (7.169, 80,7%).
- `--share-final` (thêm 2026-11-20, lệnh S3): 3 cây dùng chung khối 8 qubit cuối; S1 + share-final: FREE 8.109 (bỏ pha 7.983), quantum 22,0%. CNN cùng cỡ `--widths 11 19` (8.065).
- **Đếm đúng tham số (thêm 2026-11-20).** Unitary học được lưu ma trận phức A (32 số cho 2 qubit, 8 cho 1 qubit) nhưng chỉ dùng phần Hermite của A, nên bậc tự do thật là 16 / 4 (bỏ pha toàn cục: 15 / 3). Các số Total ở trên là đếm cũ (thừa). Log và `count_parameters()` nay in thêm dòng FREE: mỗi cây 808 (734), mạch router 320 (288), M1 14.429 (quantum 21,3%), S1 9.389 (32,7%), N1 6.869 (44,7%), router none 14.106. Việc train không đổi.
- Giữ tương quan cùng trục (thêm 2026-11-19, lệnh S1, S2): `samebasis` Linear(204,10) = 24 Bloch + 84 XX/YY/ZZ + 96 tầng 32, đúng các số lấy được từ 3 cấu hình đo X, Y, Z (12.109, quantum 47,8%); `zz` Linear(148,10) chỉ giữ ZZ (10.429, 55,5%). Head riêng mỗi cây.
- `bloch8`, `obs10` bỏ 252 tương quan cặp và tầng 32 (rẻ hơn trên máy thật, nhưng mất thông tin: accuracy có thể tụt).
- Mọi run MNIST tĩnh chạy với `--dropout 0`.

### Thứ tự chạy (lệnh đầy đủ trong `command.txt`)
1. `python .\test_model.py`: phải in "ALL TESTS PASSED"; gửi toàn bộ output.
2. Head nhỏ, router none (tách tác dụng của head, so L1): **H1** bloch8, **H2** obs10, **H3** obs10 + direct, H4 linear +
   direct (tách tác dụng của direct). F: fusion (kiểm tra folder tái tạo 2c).
3. Router: D0 `analyze_routing.py` trên checkpoint L1 (chẩn đoán), **M1** quantum, **M2** mlp18 (đối chứng bắt buộc),
   **M4** `row row row` (ablation thứ tự quét); M0 none chỉ khi chưa có L1; tùy chọn M3, M5, M6.
4. Nếu head nhỏ giữ được accuracy: ghép head nhỏ với router lượng tử.

### Ngưỡng chốt trước (MAX val soft, seed 0) và dự đoán
- Head nhỏ (so L1): đáng viết hướng "head siêu nhỏ" nếu H1 hoặc H2 ≥ 98,5. ≥ L1 − 0,30: không mất gì; L1 − 1 đến
  L1 − 0,30: mất vừa; < L1 − 1: head lớn gánh nhiều. Direct có hại nếu H3 ≤ H2 − 0,30.
- Router: có ích nếu M1 ≥ L1 + 0,30; lượng tử hơn cổ điển nếu M1 ≥ M2 + 0,30; top1 dùng được nếu tụt ≤ 0,30 so soft;
  thứ tự quét khác giúp nếu M1 ≥ M4 + 0,30; D0: oracle − uniform < 0,30 thì router khó giúp.
- Dự đoán Claude: H1 97–98,5; H2 khoảng 94–97; H3 thấp hơn H2; M1 ngang L1 và ngang M2; top1 tụt 0,3–1,0.

### Còn thiếu (chưa code)
Đối chứng cổ điển dùng cùng head nhỏ: cnn3 + bloch8 / obs10 và hồi quy logistic trên pixel 16×16. Cần có nếu viết
bài theo hướng head siêu nhỏ.

```
ảnh 16×16 → ENCODER qconv4x3 (492) → θ, φ mỗi pixel
                │                        │
                │                 ROUTER: gộp θ, φ theo lưới 4×4 → mạch 8 qubit → ⟨Z⟩ 3 qubit → softmax → g1 g2 g3
                ├→ cây row    → Linear(372,10) → l1
                ├→ cây column → Linear(372,10) → l2
                └→ cây morton → Linear(372,10) → l3
                        logit = g1·l1 + g2·l2 + g3·l3        (router none: (l1+l2+l3)/3 = linhead L1)
```

Router không chặn đường dữ liệu: cả 3 cây đều nhận đầu ra encoder; router chỉ cho trọng số. Mọi router bắt đầu với
g đều (lớp/hệ số cuối = 0), nên lúc khởi tạo model giống hệt L1; router được tạo sau cùng trong trạng thái ngẫu
nhiên tách riêng, nên phần còn lại khởi tạo giống hệt L1 cùng seed.

## Số tham số (qconv4x3, 3 cây, đo tầng 32, head linear)

| `--router` | Router | Tổng | Quantum |
|---|---:|---:|---:|
| `none` (= L1) | 0 | 16.506 | 31,2% |
| `quantum` (mặc định: lưới 4×4, 8 qubit, 1 lớp `se` mỗi vòng) | 643 (mạch 640 + 3 hệ số) | 17.149 | 33,8% |
| `quantum --router-ansatz rot` | 51 (mạch 48 + 3) | 16.557 | 31,4% |
| `mlp --router-hidden 18` (đối chứng cùng cỡ) | 651 | 17.157 | 30,0% |
| `linear` | 99 | 16.605 | 31,0% |

## Head mỗi cây (`--expert-head`) và readout encoder (`--front-readout`)

Số với `--router none` (cộng router nếu dùng: quantum +643, mlp18 +651, linear +99).

| Head | Head (3 cây) | Tổng, readout linear | Quantum | Tổng, readout direct | Quantum |
|---|---:|---:|---:|---:|---:|
| `linear` (L1) | 11.190 | 16.506 | 31,2% | 16.338 | 31,5% |
| `shared` (L2) | 3.730 | 9.046 | 56,9% | 8.878 | 58,0% |
| `mlp` (H 16) | 18.414 | 23.730 | 21,7% | | |
| `fusion` (head MLP của 2c; chỉ với router none) | 11.948 | 17.264 | 29,8% | | |
| **`bloch8`**: Linear(24,10) trên X, Y, Z của 8 qubit cuối | 750 | 6.066 | 84,9% | 5.898 | 87,3% |
| **`obs10`**: 10 quan sát cố định × (hệ số + bias) | 60 | 5.376 | 95,8% | 5.208 | 98,8% |

`bloch8`, `obs10` chỉ đọc 8 qubit cuối (bỏ 252 tương quan cặp và tầng 32: rẻ hơn trên máy thật). Quan sát của `obs10`
chọn bằng `--obs-terms` (mặc định `Z0 … Z7 X0 X4`; cặp viết như `Z0Z1`).

`--front-readout direct` (`models/front_direct.py`): encoder qconv bỏ Linear(27, 2); lấy thẳng qubit giữa của mạch 3×3
(giữa các tầng: hướng Bloch thành góc mới; sau tầng cuối: ma trận mật độ của qubit giữa vào thẳng cây). qconv4x3:
324 góc, 0 tham số cổ điển. KHÔNG bắt đầu như ánh xạ đồng nhất (khác bản linear). `models/front.py` không đổi.

Với router quantum: bloch8 6.709 (86,3%), obs10 6.019 (96,2%), obs10 + direct 5.851 (98,9%).

## Router lượng tử (`models/router.py`)

1. θ, φ của 256 pixel lấy trung bình theo lưới `--router-grid gh gw` (mặc định 4×4 = 16 ô, mỗi ô 4×4 pixel).
2. `Q = --router-qubits` qubit (mặc định 8), bắt đầu |0…0⟩. Lặp `R = ô / Q` vòng (mặc định 2): qubit q nhận ô
   `r·Q + q` bằng RZ(πφ)RY(πθ) (cùng cách encoder mã hóa pixel), rồi `--router-layers` lớp học được.
   - `se`: mỗi qubit một unitary 1 qubit học được (8 số), rồi Q unitary 2 qubit học được trên vòng (i, i+r) như khối
     strongly-entangling cuối của cây (32 số mỗi cái): 40Q mỗi lớp.
   - `rot`: RZ RY RZ mỗi qubit (3 góc), rồi CZ vòng: 3Q mỗi lớp.
3. Đo ⟨Z⟩ của 3 qubit cách đều (0, 2, 5 khi Q = 8), nhân hệ số học được (khởi tạo 0), softmax → g.
Mô phỏng statevector chính xác (256 biên độ), một mạch mỗi ảnh. Trên máy thật cần ước lượng ⟨Z⟩ bằng số lần đo hữu hạn.

## Tùy chọn (`python .\train.py --help`)

| Nhóm | Tùy chọn |
|---|---|
| Encoder, cây | `--front` (qconv…, cnn…, deep…, none), `--orders` (mỗi thứ tự quét một cây; vd `row row row`), `--side-level` (0/16/32/64/128) |
| Encoder | `--front-readout linear / direct` |
| Head mỗi cây | `--expert-head linear / shared / mlp / bloch8 / obs10 / fusion`, `--expert-hidden`, `--obs-terms`, `--projection`, `--hidden`, `--dropout` (mlp, fusion) |
| Router | `--router none / quantum / linear / mlp`, `--router-source encoder / pixels`, `--router-detach`, `--router-grid`, `--router-qubits`, `--router-layers`, `--router-ansatz se / rot`, `--router-init-scale`, `--router-hidden` |
| Trộn | `--mix logits / probs`, `--gate-noise` |
| Loss phụ | `--balance-weight` (cân bằng tải, mặc định 0,01), `--entropy-weight` (> 0: mỗi ảnh chọn rõ một cây) |
| Train | `--lr`, `--router-lr`, `--epochs`, `--patience`, `--lr-factor`, `--lr-patience`, aug, seed, split-seed |

Chế độ suy luận: `soft` (đầu ra đã train), `top1` (chỉ cây có g lớn nhất), `uniform`, `tree<i>`. Mỗi epoch
`history.csv` ghi accuracy mọi chế độ, oracle (ít nhất một cây đúng), trung bình g và tỷ lệ top1 từng cây, entropy.

## File

- `train.py`: train; `common.py`: dữ liệu và đánh giá mọi chế độ.
- `analyze_routing.py`: phân tích một checkpoint trên validation (từng cây, theo lớp, oracle, cặp cây); đọc được cả
  checkpoint L1 / L2 của `full_quantum_16x16_level32_qconv_linhead`.
- `evaluate_test.py`: official test, chỉ một lần, chọn `--mode` trước.
- `test_model.py`: số tham số, khởi tạo giống router none, so router none với linhead L1 (nếu folder đó nằm cạnh),
  router lượng tử so mô phỏng ma trận đầy đủ, chế độ trộn, loss phụ, gradient, tốc độ.
- `models/`: `front.py`, `tn_core.py`, `unitary.py`, `pauli.py`, `preprocessing.py`, `encoding.py`, `reupload.py`
  chép nguyên từ linhead (`front.py` giống hệt level32_qconv và mammoth_qconv_front); mới: `router.py`, `moe_model.py`,
  `front_direct.py`, `__init__.py`.

Lệnh, ngưỡng chốt trước và dự đoán: `command.txt`.

## Liên quan (đã có trong tài liệu)

QMoE (Nguyen và cộng sự, IEEE QCE 2025): router lượng tử điều khiển trực tiếp các expert, gộp bằng giao thoa trong một
mạch. Ở đây router được đo rồi trộn logit cổ điển (theo QMoE mô tả, giống kiểu Tognini, Banchi, De Palma, arXiv 2505.14789; chưa đọc bài này), expert là
các cây QCNN khác nhau ở thứ tự quét, và có chế độ top1.
