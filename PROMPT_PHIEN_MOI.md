# Prompt mở phiên mới (dự án QCNN, cập nhật 2026-11-21 01:48)

Dán toàn bộ phần dưới vào phiên mới.

---

Dự án: QCNN / tensor network cho MNIST 16×16. Hai mảng: (A) continual learning bằng Mammoth (commit e75a491), bài Rotated MNIST (domain-IL, 20 task, 1 epoch mỗi task); (B) MNIST tĩnh, encoder nhỏ (cổ điển hoặc quantum) trước mạch level32, và đọc thêm tầng của cây. Ý tưởng do anh Nguyễn Tùng đề xuất. Trả lời bằng tiếng Việt, ngắn gọn.

## Đọc trước, theo thứ tự
0. **Đọc trước tiên** mục "TRẠNG THÁI HIỆN TẠI" ngay dưới đây, rồi các mục SO từ 2026-11-18 đến hết (M1 cuối, N1, S1, đính chính đếm tham số, share-final, folder _sharedgate, G1). Sau đó đọc **`fullquantum_16x16_frontQuan4x3_Moe/`**: `CACH_DEM_THAM_SO.txt`, `PSEUDOCODE_S1.txt`, `README.md`, `command.txt`, `models/moe_model.py`; và **`fullquantum_16x16_frontQuan4x3_Moe_sharedgate/`**: `README.md`, `command.txt`.
1. `AI_COLLAB_CLAUDE_CODEX.txt` (nhật ký, chỉ ghi nối thêm): các mục từ "Tạo full_quantum_16x16_level32_front" ngày 2026-11-12 14:12 đến hết; quan trọng nhất là các mục ngày 2026-11-14 đến 2026-11-17.
2. `SO_NGHIEN_CUU_QCNN.txt`: các mục từ "KẾT QUẢ CUỐI qcnn-fusion-level32" đến hết (đứng trước dòng "Mục dưới đây là kế hoạch trước khi đổi hướng."). Quan trọng nhất là các mục từ "Kết quả đầu tiên full_quantum_16x16_level32_qconv" (2026-11-14) trở xuống, nhất là các mục 2026-11-15/16/17 (2c 99,03, enc-linear qconv4x3, linhead, cnn3, thang CNN, tra tài liệu, hướng paper, gói Mammoth, **Mammoth qconv4x3 93,92, router MoE, QMoE, folder Moe, head siêu nhỏ**; 2026-11-18: **M1 98,78 giữa run, H2 ~95, bỏ riêng tương quan cặp / nocorr**).
2b. **`fullquantum_16x16_frontQuan4x3_Moe/`** (code 2026-11-17, thêm nocorr 2026-11-18; người dùng đã chạy M1, H2, H1+router trên máy train; kết quả lưu `ket_qua/`): `README.md`, `command.txt`, `models/router.py`, `models/moe_model.py`.
3. **`full_quantum_16x16_level32_qconv/`** (2c đang chạy): `KIEN_TRUC_front_qconv4x3_proj10.txt` (giải thích kiến trúc, đọc trước), `README.md`, `command.txt` (mục "CẬP NHẬT 2026-11-14"), `models/front.py`, `models/fusion_model.py`, `ket_qua/`.
4. **`full_quantum_16x16_level32_qconv_linhead/`**, **`classical_cnn_16x16/`**, **`mammoth_full_package/`** (code xong 2026-11-15/16, chưa chạy): `README.md`, `command.txt` (gói Mammoth: `README.md`, `LENH_CHAY_LE.txt`, `run_all.ps1`, `setup.ps1`).
5. `full_quantum_16x16_multilevel/` (code xong, chưa chạy): `README.md`, `command.txt`.
6. `mammoth_qconv_front/README.md`, `command.txt`; `mammoth_level32_front/command.txt`.
7. `TONG_HOP_ACCURACY_CAC_Y_TUONG.txt`: phần Mammoth (khoảng dòng 102 trở đi) và phần level32_front.

Sau khi đọc: xác nhận ngắn gọn trạng thái và đang chờ gì. Không chạy gì.

## TRẠNG THÁI HIỆN TẠI (2026-11-21) — đọc trước, thay cho các mục cũ khi mâu thuẫn

### Quyết định của người dùng (2026-11-19/20)
- **Chỉ so với CNN cổ điển cùng cỡ** (cùng runner, cùng split/aug). KHÔNG làm thêm đối chứng (logistic pixel, mạch đóng băng, cnn3 cùng head nhỏ...). Không đề xuất lại.
- **Tỷ lệ quantum không quan trọng nữa; ưu tiên giảm tổng tham số** để so được với CNN nhỏ hơn mà vẫn ~99%.
- **S1 là bản chính** (99,10). Nếu G1 ≥ 98,80 thì G1 có thể thay S1 (ít tham số hơn).
- Hướng bài: "cải thiện con đường quantum ngày càng tốt", không tuyên bố hơn cổ điển hay lợi thế lượng tử. Tiêu đề Claude gợi ý: "Quantum-Routed Mixture of Scan-Order QCNN Experts for Image Classification".
- **Fashion-MNIST: làm SAU khi chốt model**; chưa code, phải hỏi trước khi code (`--dataset fashion` cho folder Moe/_sharedgate và classical_cnn_16x16).

### [ĐÍNH CHÍNH QUAN TRỌNG] Cách đếm tham số (2026-11-20)
- `models/unitary.py` LearnedUnitary1Q/2Q lưu ma trận phức A nhưng U = exp(i(A+A†)/2) → **một nửa tham số thừa** (gradient 0). Bậc tự do thật: unitary 2 qubit **16**, 1 qubit **4** (bỏ pha toàn cục 15/3). Huấn luyện không sai, chỉ đếm sai.
- **Mọi số tham số / tỷ lệ quantum ghi trước 2026-11-20 (17.149, 12.109, 9.589, 2c 17.264 / 29,8%, deep12p, Mammoth...) là cách đếm CŨ.** Từ nay báo số **FREE** (dòng "FREE parameters" trong log; `count_parameters` có `total_free`, `quantum_free`, `_free_nophase`). Giải thích chi tiết: `fullquantum_16x16_frontQuan4x3_Moe/CACH_DEM_THAM_SO.txt`.
- Theo FREE: encoder 492 (góc 324 + Linear 168); mỗi cây 808 (bỏ pha 734) = 5 tầng CP 160 + re-upload 8 + khối 8 qubit 640; router 323 (mạch 320).

### Kiến trúc chính S1 (folder `fullquantum_16x16_frontQuan4x3_Moe/`)
encoder quanvolution qconv4x3 (3 tầng, cửa sổ 3×3 9 qubit, 4 lớp RZRYRZ + CZ, ⟨XYZ⟩ → Linear(27,2), cập nhật dư θ, φ) → 3 cây level32 (quét row / column / Morton; 5 tầng conv-pool dùng chung U_c, U_p; đo Bloch tầng 32 = 96; re-upload RY(sᵢ·ᾱᵢ); khối 8 qubit 4 lớp strongly entangling, mỗi lớp 8 U1 rồi 8 U2 trên vòng (i, i+r), r = 1..4) → head mỗi cây **samebasis** Linear(204,10) (24 Bloch + 84 tương quan cùng trục XX/YY/ZZ + 96 tầng 32) → router lượng tử (đọc **đầu ra encoder**, lưới 4×4 → 8 qubit, 2 vòng re-upload + lớp se, đo Z₀,Z₂,Z₅ × sₑ → softmax) → Σ gₑℓₑ. Cây = TTN, vướng víu chỉ trong nút và khối 8 qubit gốc; mô phỏng cổ điển được.
Tổng S1: 492 + 2.424 + 323 + 6.150 = **9.389 FREE** (bỏ pha 9.135; quantum 32,7%).

### Bảng kết quả MNIST tĩnh mới (MAX val soft, seed 0, dropout 0, aug; số FREE)
| Run | Lệnh chính | FREE (bỏ pha) | MAX val | Ghi chú |
|---|---|---:|---:|---|
| L1 linhead (router none) | head Linear(372,10) | 14.106 | ~98 | [NGƯỜI DÙNG BÁO], chưa có CSV |
| **M1** router quantum | `--expert-head linear` | 14.429 (14.175) | **99,03** (ep ~38) | CSV ep 1–41; router thay được head MLP |
| N1 nocorr | `--expert-head nocorr` (Linear(120,10)) | 6.869 (6.615) | 98,42 (ep 28) | chưa có số cuối / history đủ |
| **S1 samebasis (BẢN CHÍNH)** | `--expert-head samebasis` | **9.389** (9.135) | **99,10** (ep 46) = CK | CSV ep 1–46; top1 79,97, uniform 97,63, oracle 96,23; Morton 74% |
| **G1** S1 + `--share-gates` | folder `_sharedgate` | **7.709** (7.623) | 98,67 (ep 23) = CK, **đang chạy** | CSV ep 1–24; lr chưa giảm; bám sát S1 cùng epoch (−0,12..+0,11) |
| G2 G1 + `--share-final` | folder `_sharedgate` | 7.549 (7.479) | chưa chạy | |
| S3 S1 + `--share-final` | folder Moe | 8.109 (7.983) | chưa chạy, có thể bỏ | |
| H1 bloch8 + router / H2 obs10 | | | ~95 | head siêu nhỏ: thất bại |
- Kết luận: bỏ toàn bộ tương quan cặp mất ~0,6 điểm (N1); chỉ giữ tương quan cùng trục thì không mất (S1). Router thay được head MLP (L1 ~98 → M1 99,03). Top1 tụt mạnh (~80) vì các cây không phải bộ phân loại độc lập.
- Ngưỡng chốt trước: G1 ≥ 98,80 (S1 − 0,30) thì dùng chung cổng không mất gì; G2 so G1 ±0,30. Dự đoán Claude G1 98,6–98,9, nay nghiêng ~98,9–99,0 sau khi giảm lr.
- `--share-gates`: khối 8 qubit mỗi lớp dùng chung 1 U1 + 1 U2 (20 thay vì 160 tham số/lớp; khối 640 → 80; cây 808 → 248).
- Người dùng chạy các run mới bằng lệnh trong `command.txt` của từng folder; tốc độ G1 ~20 phút/epoch.

### CNN cổ điển cùng cỡ (`classical_cnn_16x16/command.txt`, mỗi cỡ 3 lr, CHƯA CÓ SỐ)
- S1 9.389 ↔ `--widths 11 21` (9.571) hoặc `--widths 10 21`; S3 8.109 ↔ `--widths 11 19` (8.065); G1 7.709 ↔ `--widths 13 18` (7.708); G2 7.549 ↔ `--widths 12 18` (7.534); M1 14.429 ↔ chưa có cỡ đúng.

### File tài liệu đã viết (folder Moe)
- `CACH_DEM_THAM_SO.txt` (UTF-8 BOM, CRLF): cách đếm từng khối, bảng tổng, cách báo trong bài, mục share-final.
- `PSEUDOCODE_S1.txt` (tiếng Anh): toàn pipeline S1, để gửi người khác.
- `hinh_kien_truc_M1.tex` (2 trang, tiếng Anh), `hinh_kien_truc_M1_gon.tex` (1 trang); `hinh_mot_cay.tex` (vntex; trang 1 cây tổng quát gọn, trang 2 khối 8 qubit). Biên dịch trên Overleaf.
- `ket_qua/`: CSV M1, N1 (ep 2–18), S1 (ep 29, 46); `_sharedgate/ket_qua/history_G1_sharegates_ep24.csv`.

### Tài liệu so sánh đã đọc (2026-11-19)
- Terra Quantum (Senokosov và cộng sự, MLST 2024, 99,21): mạch chỉ ~0,4% tham số, bảng có mâu thuẫn; so không công bằng.
- Jyoti, Babu, Patel 2607.07754 (preprint): quantum-inspired MoE, không có router lượng tử, MNIST 97,73.
- Chưa thấy QCNN thuần đạt ~99% test MNIST 10 lớp; cần đọc toàn văn 2504.14995 và Delhibabu 2603.11131 khi người dùng gửi.

## Quy tắc bắt buộc
- Máy này chỉ đọc và viết code. Không train, không chạy test, kể cả `test_model.py`, `test_quantum_backbones.py`. Chỉ kiểm cú pháp bằng `ast.parse`. Người dùng chạy trên máy train (4090, qua AnyDesk) rồi gửi ảnh hoặc log. Đọc CSV, đọc nguồn và tính số trên máy này thì được.
- `noread/`: chỉ được đọc `noread/full_quantum_16x16_3scan_fusion_mlp16/`. Phần khác phải hỏi.
- Không xóa output (kể cả `checkpoints\paused\` trên máy train). Không sửa `full_quantum_16x16_2branch_multiscale_mlp16/` (R0), `full_quantum_16x16_level32_aug/` (bản gốc của level32_front và multilevel), `full_quantum_16x16_mammoth/`, `full_quantum_16x16_level32_front/`, `mammoth_level32_front/` (đã có kết quả). `models/front.py` của level32_qconv và `backbone/quantum_core/front.py` của mammoth_qconv_front phải giống hệt nhau: sửa một thì chép sang cái kia, kiểm bằng `cmp`, tạo lại cả hai zip.
- Có số mới, kể cả giữa run: cập nhật ngay `SO_NGHIEN_CUU_QCNN.txt` (UTF-8 có BOM; chèn trước dòng "Mục dưới đây là kế hoạch trước khi đổi hướng.", sửa dòng 2 "Cập nhật: ...") và ghi nhật ký vào `AI_COLLAB_CLAUDE_CODEX.txt`. Có kết quả cuối thì cập nhật thêm `TONG_HOP_ACCURACY_CAC_Y_TUONG.txt` (UTF-8 có BOM). Tách CK và MAX nếu có, ghi nguồn từng số ([ẢNH], [CSV], [LOG], [NGƯỜI DÙNG BÁO], ...).
- AI_COLLAB: ghi [ĐANG LÀM] trước khi sửa code, [DONE] sau khi xong, [ĐÍNH CHÍNH] khi sai. Người dùng dừng giữa chừng thì ghi [DONE] ... HỦY.
- Giờ theo Asia/Saigon: Bash `TZ=UTC-7 date`. Không dùng `TZ=Asia/Saigon` trong Bash. **Mỗi mục ghi sổ/nhật ký phải lấy giờ từ lệnh này ngay lúc ghi, không đoán** (2026-11-17 đã ghi sai giờ 3 mục, phải đính chính).
- Lệnh chạy phải ghi đủ tham số. Official test chỉ dùng một lần cuối, không dùng để chọn model.
- Một seed: chênh dưới khoảng 0,3 điểm chưa đủ kết luận. Nói rõ khi dự đoán sai.
- Mammoth: không dùng augmentation. MNIST tĩnh: giữ aug xoay ±5°, dịch ±0,05 như bản gốc.
- **Mammoth: phân biệt các đại lượng.** "TB 6 mốc" = trung bình các dòng "Accuracy for N task(s)" N=1..6; "sau task N" = dòng "Accuracy for N task(s)" (validation). Dòng "Accuracy for final task(s)" in sau các dòng [ROTATION] góc mới là **TEST**, không dùng để chọn. Luôn so cùng đại lượng.
- Ảnh mờ: số đọc từ ảnh phải kiểm lại bằng trung bình hoặc tổng; lệch thì không ghi số từng phần.
- Tạo file có dấu `\` bằng Python trong Bash dễ hỏng: dùng `chr(92)`, chuỗi raw, hoặc công cụ Write/Edit, rồi kiểm tra lại. Mỗi chuỗi thay thế phải xuất hiện đúng 1 lần.
- Máy cp949: chạy `$env:PYTHONIOENCODING = "utf-8"` trước khi in hay dùng `Tee-Object`.
- Folder Mammoth trên máy train: `C:\Users\Public\full_quantum_16x16_mlp_aug_5deg\full_quantum_16x16_mlp_aug\mammoth`. level32_qconv trên máy train: `D:\tensornetwork_mnist\tensornetwork_mnist\full_quantum_16x16_level32_qconv`.
- Người dùng hay tự sửa code trên máy train; khi họ muốn tự sửa thì đưa đoạn code/file đầy đủ để dán, kèm cách kiểm (số tham số in ra).
- **Người dùng chạy MỌI run MNIST tĩnh với `--dropout 0`** (chỉ tăng khi thấy overfit). Lệnh mới luôn ghi `--dropout 0`.
- Lệnh có `| Tee-Object -FilePath .\logs\...` cần folder `logs` có sẵn (`New-Item -ItemType Directory -Force -Path .\logs`), nếu không sẽ lỗi DirectoryNotFound.
- Folder mới hoặc code sửa: tạo lại zip cùng tên (người dùng chép zip sang máy train). Script `.ps1` viết chữ ASCII (PowerShell 5.1 đọc UTF-8 không BOM sai); kiểm bằng `[System.Management.Automation.Language.Parser]::ParseFile`.
- **Số tham số: luôn báo số FREE** (unitary 16/4, kèm bản bỏ pha 15/3), không dùng số "Total" cũ trong log.
- Folder `_sharedgate` là bản sao code Moe + `--share-gates`; sửa code chung thì sửa cả hai folder và tạo lại cả hai zip.
- Người dùng muốn giải thích dễ hiểu, có ví dụ cụ thể; hay hỏi "có mới không": trả lời thẳng, tách rõ phần đã có trong tài liệu và phần có thể mới.

## Mục tiêu và nguyên tắc chọn (cũ, xem quyết định 2026-11-19/20 ở trên khi mâu thuẫn; số tham số ở các mục dưới là cách đếm CŨ)
- Model "càng quantum càng tốt", theo hướng tensor network. Ưu tiên của anh Tùng: phần quantum đóng góp nhiều nhất, đồng thời accuracy đủ tốt để viết báo. **Không tuyên bố lợi thế lượng tử**: mô hình mô phỏng cổ điển hiệu quả (trong cây các qubit độc lập giữa các tầng; encoder qconv 9 qubit). Đã giải thích cho người dùng: head cổ điển không phải đồ giả lập; claim được là "khối mạch N tham số hơn khối cổ điển cùng N tham số khi giữ nguyên phần còn lại" (cần ≥ +0,30, lý tưởng 3 seed).
- Người dùng chấp nhận encoder 1–2k tham số trước mạch (chưa hỏi anh Tùng). Với hướng đo thêm tầng: người dùng nói không cần giữ tỷ lệ quantum, ưu tiên tăng accuracy trước.
- Đối chứng chính là CNN cùng cỡ, MLP cùng cỡ là tham khảo. Báo số tham số và tỷ lệ quantum. Mỗi thử nghiệm phải chốt ngưỡng trước khi chạy; chạy ngoài ngưỡng thì ghi là ngoại lệ.

## (A) Mammoth rot-mnist (er buffer 5120, seed 0, validation)
Cấu hình: `--dataset rot-mnist --model er --optimizer adam --lr 1e-3 --batch_size 32 --n_epochs 1 --seed 0 --device cuda:0 --validation 10 --validation_mode complete --enable_other_metrics 1 --num_workers 0 --buffer_size 5120 --minibatch_size 32`.

| Model | Tham số | Sau task 20 (val) | Học ngay | BWT | Forgetting | Test dòng final |
|---|---:|---:|---:|---:|---:|---:|
| cnn-19k | 19.060 | 95,43 | 97,48 | −2,16 | 2,45 | 94,94 |
| **qcnn-fusion-level32-deep12p** | 18.418 | **94,58** | ? | −2,19 | 2,59 | 94,60 |
| cnn-12k | 12.070 | 94,38 | 96,88 | −2,64 | 2,89 | |
| **qcnn-fusion-level32-qconv4x3** (gói mới) | 17.264 | **93,92** | ? | ? | ? | 93,89 |
| qcnn-fusion-level32 | 19.118 | 91,50 | 94,72 | −3,39 | 3,85 | 90,65 |
| mlp-19k | 18.858 | 91,24 | 93,93 | −2,83 | 3,50 | 90,32 |
| qcnn-fusion-typed4 | 12.010 | 89,65 | 93,39 | −3,94 | 4,42 | |
| qcnn-fusion | 19.202 | 89,19 | 93,59 | −4,63 | 5,21 | |
| mlp-12k | 11.970 | 88,54 | 92,14 | −3,79 | 4,36 | |

- deep12p 20 task [ẢNH 2026-11-14]: hơn level32 +3,08 (đạt ngưỡng 91,80); không hơn cnn-19k (−0,85; ngưỡng 95,73); ngang cnn-12k (+0,20). Chưa có TB 20 mốc, học ngay (cần `mammoth_tasks.csv`). 6 task: TB 6 mốc 94,28, sau task 6 93,25.
- **Còn chờ quan trọng nhất: 3b `enc-cnn-deep12p` 20 task** (mammoth_level32_front). Phần quantum có ích nếu 94,58 ≥ enc-cnn + 0,30 (enc-cnn ≤ 94,28). Dự đoán Claude: enc-cnn gần cnn-19k → mạch nhiều khả năng không hơn.
- [CÓ THỂ ĐÍNH CHÍNH] sàng lọc graph 90,23 / lr4 89,05 là TB 6 mốc hay sau task 6: chờ người dùng xác nhận. level32 seed 1 20 task: chưa rõ đã chạy.
- `--resume` hỏng trên rot-mnist.
- `mammoth_qconv_front` (encoder quantum trên Mammoth): code xong, chưa chạy; chờ kết quả tĩnh trước.
- **qconv4x3 Mammoth 20 task** [CSV mammoth_mean.csv, 2026-11-17; lưu `mammoth_full_package/ket_qua/mammoth_mean_qcnnfusionlevel32qconv4x3_seed0.csv`]: val sau task 20 **93,92**; TB 20 mốc 93,14; TB 6 mốc 94,41; sau task 6 92,95; thấp nhất 90,53 sau task 8; final (test) 93,89. So (máy/torch có thể khác, tham khảo): deep12p −0,66; level32 +2,42 (encoder quantum có ích rõ); cnn-12k −0,46; cnn-19k −1,51. Ngưỡng chốt (cnn-17k, deep6p cùng máy) CHƯA so được. Dự đoán Claude: kém cnn-17k (cần cnn-17k ≤ 93,62 mới hơn). Chưa có BWT/Forgetting (cần log hoặc `mammoth_tasks.csv`). Run này chạy trên máy `D:\tensornetwork_mnist` bằng gói mới.
- **`mammoth_full_package/`** (2026-11-16, đã chạy qconv4x3, còn lại chưa rõ; người dùng sẽ chạy trên MÁY MỚI chưa cài gì): `run_all.ps1` / `CHAY_TAT_CA.bat` tự gọi `setup.ps1` (GPU, Python 3.11 qua winget, `.venv`, torch cu124, Mammoth zip commit đầy đủ e75a491c69fd729edeb01431afb753d9157d9a81, requirements bỏ dòng torch, chép `mammoth_files\`, chạy test), smoke test `--debug_mode 1`, rồi hàng đợi 20 task: cnn-5k, cnn-12k, **cnn-17k** (17.170, mới), cnn-19k, cnn-30k (30.726), cnn-50k (49.984), cnn-85k (85.258), mlp-19k, **qcnn-fusion-level32-qconv4x3** (17.264), **qcnn-fusion-level32-deep6p** (17.272, mới). Bỏ qua run có dòng "Accuracy for final task"; log dở được đổi tên; CSV vào `ket_qua\` sau mỗi run; `-Only a,b` để chạy lẻ; `LENH_CHAY_LE.txt` có lệnh tay. Ngưỡng chốt trước (val sau task 20): qconv4x3 hơn cnn-17k nếu ≥ +0,30; encoder quantum có ích nếu qconv4x3 ≥ deep6p + 0,30. Máy mới có thể khác phiên bản torch: so các model chạy trên cùng máy.

## (B) MNIST tĩnh (validation, seed 0, split-seed 20260911, aug, 50 epoch, patience 10)

| Model | Tổng tham số | MAX val | CK | Nguồn |
|---|---:|---:|---:|---|
| none (level32_aug, projection 12) | 19.118 | **98,84** | ? | [NGƯỜI DÙNG BÁO 2026-11-14]; trước đó báo ~98,5; cần history.csv |
| conv1 | 19.128 | ~98,6 ở ep 30 | ? | [NGƯỜI DÙNG BÁO] |
| cnn4 | 19.195 | 98,82 | ? | [ẢNH] |
| **deep12p, projection 10** | 18.418 | **99,14** | 99,09 | [ẢNH, CSV] |
| qconv2x2 + level32 (2a, dropout 0,2) | 16.992 | ~98,8 | ? | [NGƯỜI DÙNG BÁO], Claude hiểu là 2a, chưa xác nhận |
| qconv2x2h16 + level32 (readout to, dropout 0) | 17.844 | 98,86 đến ep 23 (giữa run) | tạm 98,83 (ep 18) | [ẢNH history.csv] |
| **qconv4x3 + level32 (2c, `front_qconv4x3_proj10`, dropout 0)** | 17.264 (quantum 29,8%) | **99,03** (ep 29, giữa run ep 32) | 99,03 (ep 29) | [CSV `full_quantum_16x16_level32_qconv/ket_qua/history_front_qconv4x3_proj10_ep32.csv`] |
| enc-linear deep12p (2f) | 1.976 | ~98 | ? | [NGƯỜI DÙNG BÁO] |
| enc-linear qconv4x3 (1e, bỏ cây) | 822 nếu pool 4 (chưa rõ) | 95,77 (ep 13, giữa run) | 95,77 | [ẢNH history.csv 2026-11-15]; cần config.json để biết pool |
| enc-cnn (2g) | 18.392 | chưa chạy | | |
| cnn5k cổ điển | 5.578 | 99,11 | 99,04 | TONG_HOP |

- qconv2x2 so none: −0,04, **ngang** (không làm yếu; ít hơn 2.126 tham số).
- **Người dùng chạy MỌI run với dropout 0** (chỉ tăng khi thấy overfit, chưa thấy); command.txt ghi 0.2 là sai thực tế (2b, 2d đã sửa thành 0). Chưa kiểm dropout của deep12p/cnn4/none.
- **2c qconv4x3 MAX 99,03 = CK (ep 29), giữa run ep 32**: đạt ngưỡng 98,84 và 99,00; so deep12p 99,14: −0,11 ngang, ít hơn 1.154 tham số; so 2a (~98,8, cùng dropout 0): ~+0,23, chưa đủ +0,30 (cần MAX chính xác của 2a); so qconv2x2h16 dropout 0 (98,86 giữa run): +0,17 ngang; so cnn5k 99,11: −0,08. Dừng sớm ở ep 39 nếu val loss không < 0,036316. 2d deep6p chạy `--dropout 0` (command.txt đã sửa). Dự đoán Claude 98,9–99,1: đúng. Chưa có 2d deep6p → chưa kết luận encoder quantum hơn CNN cùng cỡ. Chưa ghi TONG_HOP (chờ số cuối).
- TN1 enc-linear (pool 4) [NGƯỜI DÙNG BÁO, chưa có số]: qconv2x2 (550) **hơn** cnn8p (556), **thua** deep12p (1.976). Dự đoán "1a ngang 1c" có thể sai; cần số để so ngưỡng ±0,30. Chưa có 1d (đóng băng).
- Run readout to: out-dir `front_qconv2x2_proj10_nodro_bigfront_readout` **trùng** out-dir run Front=220 dropout 0 trước đó (có thể ghi đè). Patience 10: nếu val loss không < 0,034895 thì dừng ở ep 28. Dự đoán Claude: MAX cuối 98,85–99,0, khó đạt 99. Đối chứng cùng cỡ: `cnn38p` (2e, dropout 0).
- Một run chưa rõ tên in "Best Val Loss: 0.184262 (Epoch 28)" (Claude đoán enc-linear): chờ người dùng báo.

### Code level32_qconv (cập nhật 2026-11-14, mới ast.parse; zip tạo lại)
- `--front qconv<L>x<S>[h<H>][f]`: `h<H>` = readout `Linear(27,H)→GELU→Linear(H,2)` (lớp cuối 0); không `h` thì giống hệt bản cũ. qconv2x2 220; qconv4x3 492 (góc 324); qconv2x2h16 1.072; qconv2x2f 112.
- `--linear-pool P` cho `--model enc-linear` (mặc định 4: readout 330; 2: 1.290). Người dùng đã từng sửa tay pool 2 (run enc-linear pool 2 không so với pool 4).
- `evaluate_test.py --front ... --linear-pool ...` ghi đè config cho run làm bằng code sửa tay.
- Máy train: người dùng đã dán `front.py`, `classical.py` mới, kiểm `QuantumFront` qconv4x3 = 492, qconv2x2h16 = 1.072 [ẢNH]. **Chưa xác nhận** đã sửa `models/__init__.py` (truyền `linear_pool`) và `run_level32_front.py` (`--linear-pool`).
- **Đang chạy 2c** `--front qconv4x3` (17.264, thực tế dropout 0; MAX 99,03 ở ep 29). **Chờ 2d** `--front deep6p` (CNN 7×7 cùng cỡ, 500; 17.272). Ngưỡng chốt trước: 2c "ngang deep12p" ≥ 98,84, ghi có đạt 99,00; 2c so 2a +0,30 (cửa sổ 7×7 có ích); 2c so 2d ±0,30 (encoder quantum hơn CNN cùng cỡ nếu ≥ +0,30). Dự đoán Claude: 2c 98,9–99,1; 2d ngang 2c. Điều kiện viết paper "quantum hơn": 2c ≥ 2d + 0,30 (lý tưởng 3 seed) và Mammoth cùng chiều.

### `full_quantum_16x16_level32_qconv_linhead/` (code xong 2026-11-15, chưa chạy test, chưa train)
- Encoder qconv4x3 + mạch level32 giữ nguyên; `--head linear3` (mặc định): mỗi cây Linear(372,10), logit = trung bình, không projection/MLP gộp/dropout: 16.506 (quantum 31,2%). `--head linear-shared`: 9.046 (56,9%). `--head mlp`: 17.264 như cũ.
- `--model cnn3` (đối chứng cổ điển cùng readout): 3 CNN nhỏ độc lập (Conv 1→8, Conv 8→23, 368 đặc trưng mỗi nhánh) thay encoder + cây, head y hệt; mlp 17.291 (C1, so 2c), linear3 16.533 (C2, so L1). Quantum hơn nếu qcnn ≥ cnn3 + 0,30. Dự đoán: cnn3 ~99,2–99,4. Chạy nhanh.
- Lệnh: L1 `--model qcnn --front qconv4x3 --head linear3` (out-dir `front_qconv4x3_linear3`, log phải in `Head=11190, Total=16506`); L2 `--head linear-shared`; C1, C2 trong command.txt. Thứ tự đề xuất: test_model → C1 → L1 → C2/L2.
- Ngưỡng chốt trước (MAX val so 2c 99,03): ≥ 98,73 head MLP không cần; 98,03–98,73 đóng góp vừa; < 98,03 head gánh nhiều. Dự đoán Claude: L1 98,5–98,9, L2 98,3–98,8.

### `qcnn_qconv4x3/` (2026-11-16, chưa chạy test)
- Bản 2c tổ chức lại theo package `qcnn/` (data, encoder_quanvolution, core, layers, model.py); giữ tên module + thứ tự khởi tạo như FusionQCNN → cùng seed giống hệt, nạp được checkpoint 2c. `requirements.txt` cố định torch==2.7.1+cu126 (extra-index-url). `test_model.py` so trực tiếp với code gốc nếu folder gốc nằm cạnh. Lệnh trong `command.txt` (mặc định là cấu hình 2c).

### `classical_cnn_16x16/` (code xong 2026-11-15, chưa chạy)
- Thang SmallCNN (bố cục cnn5k: Conv(1→a) BN ReLU Pool, Conv(a→b) BN ReLU Pool, Linear(16b,b) ReLU, Linear(b,10); tham số 12a+9ab+16b²+14b+10) 934 → 85.258 (cnn17k 17.170 cùng cỡ QCNN), cùng split/aug/Adam/ReduceLROnPlateau/dừng sớm với runner level32; quét lr 3e-3/1e-3/3e-4; `run_sweep.ps1` (30 run, bỏ qua run xong), `tong_hop.py` (bảng + lr tốt nhất, `ket_qua_cnn.csv`). QCNN 2c hơn nếu ≥ cnn17k + 0,30. Dự đoán cnn5k 99,0–99,1; cnn17k 99,2–99,4; cnn85k 99,4–99,5.
- **Người dùng đã bắt đầu chạy trên máy train** (`D:\tensornetwork_mnist\tensornetwork_mnist\classical_cnn_16x16`); lần đầu lỗi vì chưa có `logs` (đã sửa command.txt). Chưa có số.
- Lưu ý: cnn5k cũ (99,11) train bằng `run_bond2.py` lịch 3 pha, khác runner level32; số mới của thang này so sạch hơn.

### `full_quantum_16x16_multilevel/` (code xong 2026-11-14, chưa chạy test, chưa train)
- Level32 không encoder, `--side-level` nhận nhiều tầng (16/32/64/128); trong cây các qubit độc lập nên X, Y, Z mỗi qubit là toàn bộ thông tin của tầng. Projection giữ 12.
- 1a tầng 32+64: 26.030 (quantum 18,5%); 1b 16+32+64: 27.758; 1c 16+32: 20.846; đối chứng cỡ head 2: chỉ 32, projection 18: 26.156.
- Ngưỡng chốt trước (MAX val, so none 98,84): hơn ≥ 99,14; ngang ±0,30; kém ≤ 98,54. Tầng 64 có ích (không chỉ do head lớn) nếu 1a ≥ đối chứng 2 + 0,30. Dự đoán Claude: 1a 98,8–99,1, nhiều khả năng ngang (offline Mammoth: cùng cỡ head đo thêm tầng không tăng).
- Đã giải thích cho người dùng: đọc đa tầng là ý tưởng có cơ sở (multi-scale readout; QCNN Cong et al. 2019 đo ở tầng pooling), độ mới vừa phải, hợp làm ablation; cần đối chứng cỡ head, ablation từng tầng, báo chi phí đo trên máy thật.

### Router MoE: `fullquantum_16x16_frontQuan4x3_Moe/` (ý người dùng 2026-11-17; code xong, mới ast.parse, chưa chạy test/train; zip cùng tên)
- Luồng: encoder qconv4x3 → θ, φ → cả 3 cây (row, column, morton) đều nhận → mỗi cây `Linear(372,10)` → logit l_e; router đọc θ, φ gộp lưới 4×4 → g = softmax → logit = Σ g_e·l_e. Router KHÔNG chặn đường dữ liệu. `--router none` = trung bình đều = linhead L1 (thứ tự tạo module như FusionQCNN linhead: cùng seed khởi tạo giống hệt, nạp được checkpoint L1).
- Router quantum (mặc định): 16 ô → 8 qubit, 2 vòng re-upload RZ(πφ)RY(πθ), mỗi vòng 1 lớp "se" (unitary 1 và 2 qubit học được, 320/lớp), đo ⟨Z⟩ qubit 0, 2, 5, nhân hệ số học được khởi tạo 0 (mọi router bắt đầu g đều). Mô phỏng statevector.
- Tham số: none 16.506 (31,2%); **quantum 17.149** (router 643, mạch 640; quantum 33,75%); **mlp18 17.157** (đối chứng cùng cỡ, 30,0%); linear 16.605; rot 16.557.
- Tùy chọn: `--orders` (vd `row row row`), `--expert-head linear|shared|mlp`, `--router none|quantum|linear|mlp`, `--router-source encoder|pixels`, `--router-detach`, `--router-grid/qubits/layers/ansatz se|rot/init-scale/hidden`, `--mix logits|probs`, `--gate-noise`, `--balance-weight` (0,01), `--entropy-weight`, `--router-lr`. Chế độ suy luận soft / top1 / uniform / tree<i>; history.csv ghi mọi chế độ, oracle, g trung bình, tỷ lệ top1, entropy. `analyze_routing.py` (validation, chạy được trên checkpoint L1 linhead), `evaluate_test.py --mode`, `test_model.py` (gồm so router với mô phỏng ma trận đầy đủ, so với linhead nếu nằm cạnh).
- Lệnh (command.txt): test → D0 analyze L1 → M1 quantum → M2 mlp18 → M4 row row row; M0 none chỉ khi chưa có L1; tùy chọn M3 linear, M5 mix probs, M6 entropy 0,05.
- Ngưỡng chốt trước (MAX val soft): router có ích nếu M1 ≥ L1 + 0,30; quantum hơn cổ điển nếu M1 ≥ M2 + 0,30; top1 dùng được nếu tụt ≤ 0,30; thứ tự quét khác giúp nếu M1 ≥ M4 + 0,30; D0: oracle − uniform < 0,30 thì router khó giúp. Dự đoán Claude: M1 ngang L1, ngang M2; top1 tụt 0,3–1,0; M4 không hơn M1.
- **M1 (router quantum, 17.149; người dùng xác nhận) giữa run ep 22** [CSV `fullquantum_16x16_frontQuan4x3_Moe/ket_qua/history_moe_router_ep22.csv`]: MAX val soft **98,78** (ep 21) = CK; cùng ep: uniform 98,07, top1 82,37, oracle 97,29, từng cây 60,7/40,5/70,3; g TB 0,37/0,25/0,38; entropy 0,968 (ln3 1,099). So 2c 99,03: −0,25 (ngang). Người dùng báo "hơn rõ L1" nhưng **chưa có số L1** → chưa so ngưỡng M1 ≥ L1 + 0,30. **Top1 tụt ~16 điểm: dự đoán Claude (0,3–1,0) SAI xa**; các cây không phải bộ phân loại độc lập (oracle < soft), router là trọng số mềm theo ảnh. Muốn top1 dùng được phải ép chuyên hóa (M5, M6, loss từng cây). Epoch 1.250 s → ~2.700 s từ ep 12 (máy bận). Đang chạy nốt; chờ số cuối, M2 mlp18.
- Đã nói với người dùng: chỉ mới ở mức kết hợp; reviewer dễ coi là ghép khối có sẵn, giá trị nằm ở đối chứng. 3 kiểu quét có lý do chuyên hóa (tầng 32: qubit cây hàng thấy dải 1×8, cột 8×1, Morton 2×4) nhưng encoder 7×7 có thể làm khác biệt nhỏ đi. Accuracy cao hơn bài QMoE không so trực tiếp được (khác thiết lập; model mình ~2/3 cổ điển). [ĐÍNH CHÍNH đã nói] top1 KHÔNG giảm từ 768 xuống 256 mạch: 768 là mạch encoder (vẫn chạy đủ), top1 chỉ bớt 2/3 phần cây.
- Tài liệu: **QMoE** (Nguyen H.-Q. và cộng sự, IEEE QCE 2025, DOI 10.1109/QCE65121.2025.10323; người dùng gửi PDF, đã đọc toàn văn): router lượng tử ĐIỀU KHIỂN các expert PQC, gộp bằng giao thoa (H–CZ–H), một mạch coherent; ảnh 8×8, MNIST 2/4 lớp (MNIST-4 62,92%), không đối chứng cùng cỡ. Bản của mình đo router rồi trộn cổ điển (kiểu mà QMoE gọi là "classical control"), bản coherent không làm được với cây (lấy vết giữa mạch, 3×256 qubit). Cần đọc: Tognini, Banchi, De Palma arXiv 2505.14789 (mixture of quantum experts trên MNIST); HQF-Net 2604.06715; Heddad & Bouanane 2512.22296 (router lượng tử, dữ liệu đồ chơi; không dựa vào claim của bài này).

### Head siêu nhỏ và readout encoder: ĐÃ THÊM làm tùy chọn trong folder Moe (2026-11-17 22:48, mới ast.parse)
- `--expert-head linear|shared|mlp|bloch8|obs10|fusion` (fusion = head MLP 2c, chỉ router none, cùng seed giống hệt 2c); `--obs-terms` (mặc định Z0..Z7 X0 X4); `--front-readout linear|direct` (direct: `models/front_direct.py`, bỏ Linear(27,2), qubit giữa vào thẳng cây dạng ma trận mật độ, 324 góc, KHÔNG bắt đầu như đồng nhất; `front.py` không đổi).
- Router none: bloch8 6.066 (84,9%), obs10 5.376 (95,8%), obs10 + direct 5.208 (98,8%), linear + direct 16.338. Router quantum: 6.709 / 6.019 / 5.851 (98,9%). Lệnh H1–H4, F trong command.txt; ngưỡng: đáng viết nếu H1 hoặc H2 ≥ 98,5; direct có hại nếu H3 ≤ H2 − 0,30. Dự đoán: H1 97–98,5, H2 ~94–97, H3 < H2. Chưa có đối chứng cổ điển cùng head nhỏ (cnn3 + bloch8/obs10, logistic pixel).
- **H2 obs10 router none (5.376, 95,8%)**: ~95 giữa run [NGƯỜI DÙNG BÁO 2026-11-18], trong dự đoán 94–97, dưới ngưỡng 98,5; chưa rõ đã xong.
- **H1 bloch8 + router quantum (6.709, 86,3%)**: đang chạy [NGƯỜI DÙNG BÁO 2026-11-18], chưa có số; log phải in Total=6709. So M1 là so sạch "bỏ cả tương quan lẫn tầng 32" (cùng router); so H2 thì lẫn 2 yếu tố (head + router).
- **`--expert-head nocorr`** (code 2026-11-18, ast.parse; người dùng tự dán lên máy train, sẽ chạy sau): Linear(120,10) = 24 Bloch 8 qubit cuối + 96 tầng 32, bỏ 252 tương quan cặp; lớp `NoCorrLinear` trong `models/moe_model.py` (cat `features[:, :24]` và `features[:, 276:]`), thêm vào `EXPERT_HEADS`, `SIDE_HEADS`. Router quantum 9.589 (~60,4%), router none 8.946. Lệnh **N1** trong command.txt (như M1, out-dir `moe_quantum_nocorr`, log phải in `Head=3630, Total=9589`). Người dùng nghiêng về: nếu H1 tụt là do mất tầng 32 (Mammoth level32 91,50 so qcnn-fusion 89,19). Ngưỡng: tương quan không cần nếu N1 ≥ M1 − 0,30; tầng 32 gánh phần chính nếu N1 ≥ H1 + 0,30. Dự đoán Claude: N1 ngang M1, H1 thấp hơn N1. Chưa từng có thí nghiệm nào bỏ riêng tương quan.
- Vì sao M1 vẫn 17k: head 3 × Linear(372,10) = 11.190 (65%); 372 = 96 tầng 32 + 24 Bloch + 252 tương quan (28 cặp × 9); head MLP 2c chỉ hơn 758.
- Phân tích ban đầu (trước khi code):
- Head 3 × Linear(372,10) = 11.190 = 65% bản MoE. Claude: hướng thuyết phục nhất nếu giữ accuracy. Bậc: **bloch8** mỗi cây Linear(24,10) trên X,Y,Z 8 qubit cuối (750; tổng 6.709; quantum 86,3%); **obs10** 10 quan sát cố định × (hệ số + bias) (60; tổng 6.019; 96,2%). Rẻ hơn trên máy thật (bỏ 252 tương quan cặp).
- Đối chứng cần: hồi quy logistic trên pixel 16×16 (2.570; MNIST thường ~92%), cnn3 cùng head nhỏ. Ngưỡng đề xuất: đáng viết nếu ≥ 98,5 với quantum > 85% và hơn cnn3 cùng head ≥ +0,30. Dự đoán Claude: bloch8 97–98,5; obs10 thấp hơn.

## Tài liệu kiến trúc đã viết (2026-11-15)
- `full_quantum_16x16_level32_qconv/KIEN_TRUC_front_qconv4x3_proj10.txt`: luồng dữ liệu, số tham số từng phần (encoder 492 = góc 324 + Linear 168; 3 cây 4.824; head 11.948; quantum 29,8%).
- `full_quantum_16x16_level32_qconv/kien_truc_qconv4x3.tex` (tài liệu LaTeX chi tiết từ resize ảnh, công thức đầy đủ) và `hinh_kien_truc_qconv4x3.tex` (hình TikZ 3 trang). Chưa biên dịch (máy này không có LaTeX); biên dịch pdfLaTeX trên Overleaf, cần vntex.
- Đã giải thích cho người dùng (chat): qconv = quanvolution học được; mỗi tầng: ô 3×3 → 9 qubit RZ(πφ)RY(πθ) → 4 lớp (RZRYRZ mỗi qubit + CZ 12 cạnh) → ⟨X⟩⟨Y⟩⟨Z⟩ 27 số (tính chính xác, trên máy thật cần 3 cấu hình đo) → Linear(27,2) cộng dư vào θ,φ pixel giữa; 3 tầng song song toàn ảnh (vùng nhìn 3→5→7); không có Linear(256,256). Cây: tầng 32 qubit độc lập (đo Bloch 96), 8 qubit cuối vướng víu (24 + 252 tương quan cặp, 28 cặp). Morton = quét chữ Z. Ví dụ tính bằng code: chỉ CZ, góc 0 → ⟨X₄⟩ = sinπθ₄·∏cos(πθ_j) hàng xóm 1,3,5,7.

## Tra tài liệu độ mới (2026-11-15, chỉ đọc abstract; người dùng sẽ tải toàn văn)
- ĐÃ CÓ: quanvolution trainable + nhiều lớp + residual (ResQuNNs, Kashif & Shafique, arXiv 2402.09146); variational quanvolution (Mattern 2106.07327); quanvolution gốc (Henderson 1904.04767); mã hóa góc học được (Ovalle-Magallanes 2023, ASOC 110307); đọc tầng giữa QCNN làm đặc trưng (Kim, Araujo, Park 2602.21953; Anwar 2508.18161); continual learning lượng tử nhưng khác bài toán (Jiang, Lu, Deng 2108.02786; Situ 2203.14032 GEM; 2601.21318; 2608.10464).
- CHƯA THẤY: quanvolution sinh góc mã hóa cho QCNN cây (gần nhất QCQ-CNN, Sci Rep 2025, s41598-025-13417-1, chưa đọc được); ensemble 3 cây theo thứ tự quét hàng/cột/Morton; QCNN 256 qubit 10 lớp ~99%; QCNN trên rot-MNIST domain-IL với experience replay so CNN/MLP cùng cỡ.
- Danh sách tải ưu tiên: ResQuNNs, QCQ-CNN, Ovalle-Magallanes, 2602.21953, 2108.02786, Mattern, Bowles 2403.07059. Khi người dùng đưa toàn văn (ngoài noread/), đọc và xác nhận lại.
- Nhận định đã nói với người dùng: từng khối đều không mới; cây (lấy vết ngay) tương đương TTN cổ điển bond 4, chỉ 8 qubit cuối vướng víu (nên tự nêu trong bài); 99% cao trong nhóm QCNN nhưng không được so số giữa các bài khác thiết lập; CNN 5k đã 99,11.

## Hướng paper (Claude đề xuất, cập nhật 2026-11-15, chưa chốt, chưa hỏi anh Tùng)
- (1) CHÍNH: bài thực nghiệm có đối chứng chặt "phần nào của QCNN cây thực sự làm việc": mã hóa là nút thắt (none 98,84 → encoder 99,0–99,1); encoder quantum so CNN cùng cỡ (deep6p); cây so head cổ điển cùng cỡ (cnn3, enc-cnn); mạch đóng băng (qconv4x3f); head tuyến tính (linhead); enc-linear (95,77 giữa run: encoder một mình yếu, cây gánh nhiều); 3 seed.
- (2) Ghép continual learning rot-MNIST (ít bài QCNN làm): deep12p thu hẹp khoảng cách cnn-19k từ 3,93 còn 0,85; Forgetting 2,59 so 3,85. Kết quả mạnh nhất có thể: qconv4x3 Mammoth ≥ cnn-17k + 0,30.
- (3) Dự phòng nếu đối chứng ngang/thua: benchmark / kết quả âm kiểu Bowles 2403.07059.
- (4) Mục phụ: tính mô phỏng cổ điển và chi phí phần cứng (768 mạch × 3 cấu hình đo mỗi ảnh; tương quan cặp cần hơn chục cấu hình; tầng 32 cần mạch riêng); thí nghiệm số lần đo hữu hạn.
- Điểm có thể mới để đặt: quanvolution làm bộ sinh trạng thái mã hóa (biên độ + pha) cho QCNN cây, cập nhật dư, khởi tạo đồng nhất; ensemble thứ tự quét; continual learning. Không bán "kiến trúc mới" hay "lợi thế lượng tử". Nơi gửi hợp lý: Quantum Machine Intelligence, IEEE QCE, workshop QML.
- Thêm bộ dữ liệu thứ 2 (Fashion-MNIST 16×16) nếu muốn câu "hơn CNN" thuyết phục.

## Việc tiếp theo khi có số
1. Ghi ngay vào SO và AI_COLLAB (nguồn, đại lượng, MAX/CK).
2. So ngưỡng đã chốt; nói rõ dự đoán đúng hay sai.
3. **Chờ hiện tại (2026-11-21, quan trọng nhất trước):** số cuối **G1** (so ngưỡng 98,80) → có thể G2; **CNN cùng cỡ** `--widths 13 18`, `11 21` / `10 21`, `12 18`, `11 19` (cùng runner); số cuối N1 (ghi TONG_HOP); sau khi chốt model: Fashion-MNIST (hỏi trước khi code), 3 seed cho bản chốt, evaluate_test 1 lần. Danh sách cũ dưới đây phần lớn đã lỗi thời.
   Danh sách cũ: **số cuối M1, H1 + router quantum, H2, MAX val L1; sau đó N1 nocorr** (người dùng chạy sau); **Mammoth cnn-17k và deep6p** (cùng máy với qconv4x3 93,92: quyết định ngưỡng) + `mammoth_tasks.csv`/log qconv4x3 (BWT, Forgetting); **thang CNN tĩnh** (cnn17k ba lr: so 2c); **C1 cnn3** (cùng readout); **số cuối 2c** (epoch dừng, MAX/CK cuối → ghi TONG_HOP); **L1 linhead** (mốc so của MoE) rồi D0 analyze_routing; **test_model.py folder Moe**, rồi H1/H2 (head nhỏ, router none) và M1/M2/M4 (router); đối chứng cổ điển cùng head nhỏ (chưa code); số cuối enc-linear qconv4x3 + config.json (pool 4 hay 2); L1 linhead; 2d deep6p tĩnh (dropout 0); số cuối run readout to; 3b Mammoth enc-cnn-deep12p; TN1 số chính xác (1a–1d); `mammoth_tasks.csv` deep12p; history.csv none (98,84); tên run "Best Val Loss 0.184262".
4. Đề xuất đang chờ người dùng quyết: qconv4x3f (đóng băng), 3 seed cho 2c và đối chứng, ablation ép φ=0, encoder bỏ Linear (lấy thẳng trạng thái qubit giữa, 100% quantum), script phân tích mạch đã train (đưa ô mẫu vào checkpoint 2c). Không nên thêm tầng (qconv4x4) lúc này: ngưỡng ≥ 99,33 mới đáng.
5. Để sau: multilevel 1a (khi có máy trống); hỏi anh Tùng về encoder và hướng paper (đã đề nghị viết tóm tắt 1 trang); task 6 trùng task 11 (góc 178,62 và 178,88).

## Ý tưởng khác đã đánh giá (chưa làm)
- SQUID-GNN (github.com/LeGiangK62/SQUID-GNN): không đưa vào bài. Code GNN quantum anh Tùng hẹn gửi: chưa nhận.

## Lưu ý khác
- `D:\data_system\move_able_atena\` và `D:\moveable_atena\code_v5` là dự án RL khác (movable antenna), không ghi số vào sổ QCNN. Chỉ bàn tiếp nếu người dùng nhắc.
