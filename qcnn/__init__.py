"""QCNN có encoder quanvolution học được (qconv4x3) cho MNIST 16x16.

Cấu trúc package:
    data/                  tiền xử lý (resize về 16x16, kẹp [0, 1], làm phẳng)
    encoder_quanvolution/  quanvolution học được qconv<L>x<S> + mã hóa góc 256 qubit lá
    core/                  mạch cây tensor network: unitary, conv-pool, re-upload, strongly-entangling,
                           thứ tự quét, phép đo
    layers/                readout cổ điển (head)
    model.py               QCNNQconv: ghép các phần lại
"""
from .model import QCNNQconv


def build_model(cfg: dict) -> QCNNQconv:
    """Dựng model từ tham số chạy (vars(args)) hoặc từ args lưu trong config của checkpoint (kể cả config của các
    run full_quantum_16x16_level32_qconv với --model qcnn: cùng khóa)."""
    if cfg.get("model", "qcnn") != "qcnn":
        raise ValueError("This folder only has the QCNN model (--model qcnn)")
    return QCNNQconv(front=cfg.get("front", "qconv4x3"), side_level=cfg.get("side_level", 32),
                     projection=cfg.get("projection", 10), hidden=cfg.get("hidden", 16),
                     dropout=cfg.get("dropout", 0.0), head=cfg.get("head", "mlp"),
                     orders=tuple(cfg.get("orders", ["row", "column", "morton"])))


__all__ = ["QCNNQconv", "build_model"]
