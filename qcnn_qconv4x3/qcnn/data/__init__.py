"""Data: tiền xử lý ảnh MNIST (nội suy song tuyến 28x28 -> 16x16, kẹp về [0, 1], làm phẳng theo hàng)."""
from .preprocessing import preprocess_full_quantum

__all__ = ["preprocess_full_quantum"]
