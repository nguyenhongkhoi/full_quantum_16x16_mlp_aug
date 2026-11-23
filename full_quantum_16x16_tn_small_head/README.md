# Thử head nhỏ cho cây TN

Folder này trả lời một câu hỏi: **giữ nguyên cây TN của R0, head có thu nhỏ được nhiều mà không mất accuracy không?**

**Trạng thái:** mới kiểm tra cú pháp, chưa chạy lần nào, chưa có accuracy. Chưa có code train đầy đủ; chỉ viết khi bước này có kết quả tốt.

## Vì sao đổi head

Head R0 có 5.402 tham số mỗi nhánh, chủ yếu ở `Linear(324, 16)`. Đặc trưng lại có cấu trúc: 8 qubit, mỗi qubit 3 giá trị; 28 cặp, mỗi cặp 9 tương quan; 16 node side tap, mỗi node 3 giá trị. Head mới dùng chung một lớp nhỏ cho mọi phần tử cùng loại, rồi mới qua MLP.

## Các head

| Head | k | Hidden | Head mỗi nhánh | Tổng 2 nhánh |
|---|---:|---:|---:|---:|
| `current` (R0) | - | 16 | 5.402 | 12.420 |
| `k4_h16` | 4 | 16 | 3.618 | 8.852 |
| `k2_h16` | 2 | 16 | 1.918 | 5.452 |
| `k1_h16` | 1 | 16 | 1.068 | 3.752 |
| `k1_h8` | 1 | 8 | 548 | 2.712 |

Tổng gồm 1.616 tham số quantum của R0.

## Cách thử

`sweep_head.py` nạp checkpoint R0, đóng băng phần quantum, trích đặc trưng của val và 4 bản augmentation của train, rồi train từng head trên đặc trưng đó với 3 seed. Lần chạy lại dùng lại cache `features_cache.pt`.

Script dừng báo lỗi nếu đặc trưng cùng head gốc không cho lại accuracy của checkpoint, 98,32% với R0.

## Quy tắc quyết định

Chốt trước khi có số. So CK trung bình với dòng `current` của cùng lần chạy, không so trực tiếp với 98,32%.

| Kết quả | Làm tiếp |
|---|---|
| Có head dưới 8.000 tổng tham số chênh dưới 0,3 điểm | Train đầy đủ head nhỏ nhất đạt điều kiện, 2 nhánh, đến hết phase 2, so với R0 97,97% |
| Chỉ `k4_h16` đạt điều kiện | Thu nhỏ head không đủ lợi; cân nhắc dừng hướng này |
| Mọi head nhỏ thua trên 0,5 điểm | Head thật sự cần thiết; dừng họ cây |

Các trường hợp còn lại, chênh từ 0,3 đến 0,5 điểm: đọc cùng đường cong rồi mới quyết định.

## Giới hạn

- Đây chỉ là bước lọc. Phần quantum được học cho head R0, nên head nhỏ có thể cho kết quả thấp hơn so với khi train cùng từ đầu.
- Chỉ có 4 bản augmentation cố định, ít hơn augmentation mới ở mỗi epoch khi train thật.
- Lịch train ở đây là cosine 60 epoch, khác 3 phase của R0, nên `current` có thể không đạt đúng 98,32%.
- Official test không dùng ở bước này.
