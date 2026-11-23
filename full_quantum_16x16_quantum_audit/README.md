# Kiểm chứng đóng góp của phần quantum trong s9_meas

Folder này trả lời một câu hỏi: **mạch quantum của s9_meas có đóng góp gì so với lựa chọn cổ điển cùng cỡ không?** Code độc lập, không cần các folder khác.

**Trạng thái:** mới kiểm tra cú pháp, chưa chạy lần nào, chưa có accuracy.

## Các bản

Mọi bản dùng cùng 32 ô 12 pixel, cùng `Linear(128, 16) → LayerNorm → GELU → Linear(16, 10)`, cùng split, augmentation và lịch train. Chỉ lớp đặc trưng trên mỗi ô thay đổi.

| Preset | Lớp đặc trưng trên mỗi ô | Lõi | Head | Tổng | Đo được gì |
|---|---|---:|---:|---:|---|
| `c0_pixel` | Pixel thẳng vào `Linear(12, 4)` | 0 | 2.318 | 2.318 | Mốc cổ điển |
| `c1_encoding` | `sin`, `cos`, tích `cos` lân cận, rồi `Linear(41, 4)` | 0 | 2.434 | 2.434 | Đóng góp của mã hóa |
| `c2_product` | Mạch s9 với mỗi gate là tích hai gate 1 qubit, rồi `Linear(53, 4)` | 32 | 2.482 | 2.514 | So với `s9_meas_random`: đóng góp của vướng víu |
| `c4_local_mlp` | `Linear(12, 16) → GELU → Linear(16, 4)` | 0 | 2.542 | 2.542 | So với s9_meas: đối thủ cổ điển gần cùng tổng tham số |
| `s9_meas` | Mạch đầy đủ, rồi `Linear(53, 4)` | 64 | 2.482 | 2.546 | Bản đang được kiểm chứng |

`c2_product` dùng 2 biên độ cho mỗi qubit thay vì 4.096 biên độ, nên chạy rất nhanh. `test_model.py` so nó với mô phỏng 12 qubit đầy đủ. Không thể dùng khởi tạo balanced vì gate đó tạo vướng víu, nên phải dùng `--init random` và so với `s9_meas_random`.

## Chạy

Chạy theo thứ tự trong `command.txt`, bắt đầu bằng `test_model.py`. Mỗi run dừng ở epoch 25 và in dòng `Phase 2 result`. Gửi lại dòng đó của cả 4 bản. Chạy lại cùng lệnh sẽ resume; bỏ `--stop-after-phase2` để chạy tiếp phase 3.

## Quy tắc quyết định

Chốt trước khi có số. So MAX val hết phase 2, ngưỡng 0,3 điểm. Mốc: s9_meas 98,09%, `s9_meas_random` lấy từ CSV của run đó.

| Kết quả | Kết luận | Làm tiếp |
|---|---|---|
| `c4_local_mlp` từ 97,79% trở lên | Chưa hơn lớp cổ điển cùng cỡ | Nhánh A |
| `c4_local_mlp` thấp hơn rõ, `c2_product` ngang `s9_meas_random` | Mạch có ích nhưng vướng víu thì chưa | Nhánh A |
| Cả hai đều thấp hơn rõ | Có tín hiệu vướng víu đóng góp | Nhánh B: chạy lại với 3 seed |

- **Nhánh A:** dừng tối ưu mô hình quantum. Báo cáo trung thực phần đóng góp, hoặc dùng CNN nếu chỉ cần accuracy.
- **Nhánh B:** 3 seed cho s9_meas, `s9_meas_random`, `c2_product`, `c4_local_mlp`. Nếu vẫn đứng vững thì mới làm head conv.

`c0_pixel` và `c1_encoding` chỉ để giải thích accuracy đến từ đâu, không quyết định nhánh.

## Khi đọc kết quả

- Một seed chỉ dùng để lọc, chưa đủ để kết luận.
- `c0_pixel`, `c1_encoding`, `c2_product` ít tham số hơn s9_meas, nên phải báo kèm số tham số.
- `c4_local_mlp` chỉ là một lớp cổ điển cùng cỡ, không phải lớp cổ điển tốt nhất. Thua nó thì phần quantum không có lợi thế, nhưng thắng nó chưa có nghĩa là thắng mọi lựa chọn cổ điển.
- Ở phase 3, gate học với `--core-lr`, còn lớp cổ điển học với `--readout-lr`. Mốc phase 2 không bị ảnh hưởng vì hai phase đầu chỉ dùng một learning rate.
- Official test không dùng ở bước này. `evaluate_test.py` chỉ dùng một lần cho cấu hình đã chốt.
