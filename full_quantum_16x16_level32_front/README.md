# full_quantum_16x16_level32_front

Level32 trên MNIST tĩnh 16×16, thêm một mạng cổ điển nhỏ **trước** mạch. Chép từ `full_quantum_16x16_level32_aug` (folder gốc không bị sửa). Huấn luyện, dữ liệu, augmentation và chia tập giống hệt bản gốc.

## Câu hỏi
Một phần đầu cổ điển rất nhỏ, đổi giá trị mỗi pixel trước khi mã hóa RY, có giúp mạch học tốt hơn không? Phần quantum vẫn chiếm phần lớn tham số không thuộc head.

## Phần đầu (`models/front.py`, tùy chọn `--front`)
| `--front` | Phép tính | Tham số thêm | Tổng | Tỷ lệ quantum |
|---|---|---:|---:|---:|
| `none` | level32 | 0 | 19.118 | 25,2% |
| `conv1` | Conv2d(1,1,3,pad 1) | 10 | 19.128 | 25,2% |
| `cnn4` | x + Conv(4→1)(GELU(Conv(1→4)(x))) | 77 | 19.195 | 25,1% |
| `cnn8` | như cnn4, 8 kênh | 153 | 19.271 | 25,0% |
| `deep12` + `--projection 10` | x + Conv(12→1)(GELU(Conv(12→12)(GELU(Conv(1→12)(x))))) | 1.537 | 18.309 | 26,3% |
| `deep12p` + `--projection 10` | như deep12, thêm kênh phi: mỗi qubit RZ(π·phi) RY(π·theta)\|0⟩ | 1.646 | 18.418 | 26,2% |

`--projection 10` giảm head từ 14.294 xuống 11.948 (mỗi cây Linear(372,10) thay Linear(372,12)). Encoder 1–2k tham số do người dùng chọn (2026-11-12). Khi có encoder lớn như vậy, cần đối chứng cổ điển cùng encoder và cùng tổng tham số trước khi viết là phần quantum có ích.

- Áp lên ảnh đã resize 16×16, ra 256 giá trị, dùng chung cho 3 cây. Mạch vẫn nhận đúng 256 qubit theo 3 thứ tự quét.
- Lúc khởi đầu là đồng nhất (conv1: tâm 1, còn lại 0; cnn: lớp cuối bằng 0), nên output ban đầu giống level32.
- Phần đầu được tạo trong `torch.random.fork_rng`, nên các tham số khác khởi tạo giống level32 và trạng thái ngẫu nhiên lúc train không đổi.
- Khi có phần đầu, giá trị không bị kẹp về [0, 1] trước RY (để luôn có gradient). Ảnh gốc vẫn nằm trong [0, 1], nên lúc khởi đầu không có gì khác.
- Không dùng chung với `--graph-layer` (lớp đồ thị kẹp giá trị).

## Đối chứng cổ điển (`models/classical.py`, tùy chọn `--model`, 2026-11-13)
Cùng encoder (`--front deep12p`, cùng module và cách khởi tạo; giá trị ngẫu nhiên ban đầu khác vì rút ở vị trí khác), bỏ mạch. Đầu ra θ, φ thành ảnh 2 kênh 16×16.

| `--model` | Cấu trúc | Tổng |
|---|---|---:|
| `qcnn` (mặc định) | encoder → mạch level32 → head | 18.418 (deep12p, projection 10) |
| `enc-linear` | encoder → AvgPool 4×4 → Linear(32,10) | 1.976 |
| `enc-cnn` | encoder → Conv(2→20) BN ReLU Pool → Conv(20→32) BN ReLU Pool → Linear(512,20) ReLU Dropout → Linear(20,10) | 18.392 |

Lệnh và ngưỡng: `command.txt` 2f, 2g.

## File
- `models/front.py` (mới), `models/fusion_model.py` (tham số `front`, `count_parameters` thêm `front`, `quantum_share`), `models/encoding.py` (tùy chọn `clamp`).
- `models/classical.py` (mới): `EncoderLinear`, `EncoderCNN`; `models/__init__.py`: `build_model`.
- `run_level32_front.py`: như `run_level32_aug.py`, thêm `--front`, `--model`, `--cnn-widths`, `--cnn-hidden`.
- `evaluate_test.py`: đọc `front` từ checkpoint.
- `test_model.py`: kiểm số tham số, khởi đầu giống level32, gradient.
- `command.txt`: lệnh chạy và ngưỡng chốt trước.

## Lưu ý
- Augmentation (xoay ±5°, dịch ±0,05) giữ như bản level32_aug để so cùng điều kiện.
- Một seed: chênh dưới 0,3 điểm chưa đủ kết luận.
- Official test chỉ chạy một lần, sau khi đã chọn bằng validation.
