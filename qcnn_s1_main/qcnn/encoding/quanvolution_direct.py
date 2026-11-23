"""Encoder quanvolution KHÔNG có readout cổ điển (--front-readout direct): 100% tham số encoder là góc mạch.

Bản thường (qcnn/encoding/quanvolution.py, QuantumFront, --front-readout linear): mỗi tầng đo X, Y, Z của 9 qubit cửa sổ 3x3 (27 số)
rồi Linear(27, 2) (cổ điển) cộng dư vào (theta, phi) của pixel giữa.

Bản direct: cùng mạch 9 qubit mỗi pixel (cùng code QuantumConvStage.features), nhưng lấy thẳng qubit GIỮA (qubit 4):
  - giữa các tầng: vector Bloch (x, y, z) của qubit giữa -> góc mới theta = arccos(z)/pi, phi = atan2(y, x)/pi
    (hướng Bloch; độ dài bị bỏ vì tầng sau mã hóa lại trạng thái thuần);
  - sau tầng cuối: trả về chính vector Bloch (x, y, z) của qubit giữa; model dựng ma trận mật độ
    rho = (I + xX + yY + zZ)/2 (trạng thái hỗn hợp, đúng như "đưa qubit giữa vào cây") làm đầu vào của cây.
Không có tham số cổ điển: qconv4x3 direct = 324 góc.
KHÁC bản thường: KHÔNG bắt đầu như ánh xạ đồng nhất (mạch góc ngẫu nhiên ngay từ đầu làm đổi trạng thái pixel).
Tạo trong trạng thái ngẫu nhiên tách riêng như QuantumFront, nên phần còn lại khởi tạo như cũ.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..layers.gates import cz_signs, grid_edges
from .quanvolution import PATCH_QUBITS, QuantumConvStage, parse_qconv


CENTER = 4  # qubit giữa của cửa sổ 3x3 (theo hàng)


class DirectStage(nn.Module):
    """Một tầng: góc mạch (L, 9, 3) như QuantumConvStage, không readout. Trả về Bloch (x, y, z) của qubit giữa."""

    def __init__(self, layers: int, frozen: bool):
        super().__init__()
        angles = torch.empty(layers, PATCH_QUBITS, 3).uniform_(-math.pi, math.pi)
        if frozen:
            self.register_buffer("angles", angles)
        else:
            self.angles = nn.Parameter(angles)
        self.register_buffer("signs", cz_signs(PATCH_QUBITS, grid_edges()), persistent=False)

    def forward(self, theta, phi, height, width):
        # Dùng nguyên hàm của QuantumConvStage (cùng mạch, đã được test so với ma trận đầy đủ ở level32_qconv).
        values = QuantumConvStage.features(self, theta, phi, height, width)  # (batch, H*W, 27): X0 Y0 Z0 X1 ...
        return values[..., 3 * CENTER:3 * CENTER + 3]


def bloch_to_angles(bloch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., 3) -> theta, phi sao cho RZ(pi phi) RY(pi theta)|0> có hướng Bloch này."""
    x, y, z = bloch.unbind(-1)
    length = (x * x + y * y + z * z + 1e-12).sqrt()  # không dùng norm: gradient của norm tại 0 là nan
    theta = torch.arccos((z / length).clamp(-1 + 1e-6, 1 - 1e-6)) / math.pi
    # atan2(0, 0) có gradient 0/0 = nan: khi x, y gần 0 (qubit gần trục Z, phi không xác định) thay x bằng 1e-3.
    x = torch.where(x * x + y * y < 1e-12, torch.full_like(x, 1e-3), x)
    phi = torch.atan2(y, x) / math.pi
    return theta, phi


class DirectQuantumFront(nn.Module):
    """S tầng DirectStage. forward(values) -> Bloch (batch, H*W, 3) của qubit giữa sau tầng cuối."""

    def __init__(self, mode: str, height: int, width: int):
        super().__init__()
        parsed = parse_qconv(mode)
        if parsed is None:
            raise ValueError(f"--front-readout direct needs a quantum front qconv<L>[x<S>][f], got {mode!r}")
        layers, stages, frozen, hidden = parsed
        if hidden:
            raise ValueError("--front-readout direct has no readout; drop h<H> from --front")
        self.mode, self.height, self.width = mode, height, width
        with torch.random.fork_rng(devices=[]):
            self.stages = nn.ModuleList([DirectStage(layers, frozen) for _ in range(stages)])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        theta, phi = values, torch.zeros_like(values)
        bloch = None
        for index, stage in enumerate(self.stages):
            if self.training and torch.is_grad_enabled():
                bloch = checkpoint(stage, theta, phi, self.height, self.width, use_reentrant=False,
                                   preserve_rng_state=False)
            else:
                bloch = stage(theta, phi, self.height, self.width)
            if index + 1 < len(self.stages):
                theta, phi = bloch_to_angles(bloch)
        return bloch


def direct_circuit_parameters(front: nn.Module) -> int:
    return sum(stage.angles.numel() for stage in front.stages if isinstance(stage.angles, nn.Parameter))
