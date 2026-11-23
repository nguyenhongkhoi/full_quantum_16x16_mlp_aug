"""Encoder quanvolution học được qconv<L>x<S>[h<H>][f] (chép từ full_quantum_16x16_level32_qconv/models/front.py,
chỉ phần quantum; phép tính và số ngẫu nhiên không đổi).

Với mỗi pixel p của ảnh 16x16 (bước 1, đệm 0), một mạch 9 qubit trên ô 3x3 quanh nó, dùng chung cho mọi pixel như
một kernel tích chập:
  1. mã hóa: qubit k = RZ(pi phi_k) RY(pi theta_k)|0>  (ngoài ảnh: |0>)
  2. L lớp: RZ(a) RY(b) RZ(c) trên mỗi qubit (góc riêng), rồi CZ trên 12 cạnh của lưới 3x3
  3. <X>, <Y>, <Z> chính xác của 9 qubit -> 27 số (statevector, 2^9 biên độ)
  4. readout Linear(27, 2) (khởi tạo 0) -> (d_theta, d_phi) cộng vào pixel giữa (cộng dư)
S tầng nối tiếp, mỗi tầng tham số riêng; vùng nhìn tăng 3x3 -> 5x5 -> 7x7 với S = 3.
qconv4x3: L = 4, S = 3; 3 x (108 góc + 56 readout) = 492 tham số (324 góc mạch).
h<H>: readout Linear(27, H) GELU Linear(H, 2). f: góc mạch đóng băng ở giá trị ngẫu nhiên ban đầu.
"""
from __future__ import annotations

import math
import re
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

PATCH_QUBITS = 9  # ô 3x3, đánh số theo hàng; qubit 0 là bit cao nhất của chỉ số trạng thái


def parse_qconv(mode: str) -> Optional[tuple[int, int, bool, int]]:
    """qconv<L>[x<S>][h<H>][f] -> (số lớp, số tầng, đóng băng, hidden); hidden 0 = readout Linear(27, 2) thường.
    Trả về None nếu mode không phải encoder quanvolution."""
    match = re.fullmatch(r"qconv(\d+)(?:x(\d+))?(?:h(\d+))?(f?)", mode)
    if match is None:
        return None
    layers, stages, hidden = int(match.group(1)), int(match.group(2) or 1), int(match.group(3) or 0)
    if layers < 1 or stages < 1:
        raise ValueError(f"{mode!r}: layers and stages must be at least 1")
    if match.group(3) is not None and hidden < 1:
        raise ValueError(f"{mode!r}: the readout hidden size must be at least 1")
    return layers, stages, match.group(4) == "f", hidden


def grid_edges(size: int = 3) -> list[tuple[int, int]]:
    """Các cặp láng giềng gần nhất (ngang, dọc) của lưới size x size (12 cặp với 3x3)."""
    edges = []
    for row in range(size):
        for column in range(size):
            here = row * size + column
            if column + 1 < size:
                edges.append((here, here + 1))
            if row + 1 < size:
                edges.append((here, here + size))
    return edges


def cz_signs(qubits: int, edges) -> torch.Tensor:
    """Đường chéo của tích các cổng CZ trên các cạnh này: (-1)^(tổng b_i b_j trên các cạnh), dạng vector thực."""
    index = torch.arange(2 ** qubits)
    bits = [(index >> (qubits - 1 - q)) & 1 for q in range(qubits)]
    parity = torch.zeros_like(index)
    for a, b in edges:
        parity = parity ^ (bits[a] & bits[b])
    return (1 - 2 * parity).to(torch.float32)


def cis(radius: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """radius * exp(i angle); radius có thể âm (torch.polar chỉ dành cho độ lớn không âm)."""
    return torch.complex(radius * torch.cos(angle), radius * torch.sin(angle))


def rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    """(..., 3) góc (a, b, c) -> (..., 2, 2) ma trận phức RZ(a) RY(b) RZ(c)."""
    a, b, c = angles.unbind(-1)
    cos, sin = torch.cos(b / 2), torch.sin(b / 2)
    plus, minus = (a + c) / 2, (a - c) / 2
    m00 = cis(cos, -plus)
    m01 = cis(-sin, -minus)
    m10 = cis(sin, minus)
    m11 = cis(cos, plus)
    return torch.stack((torch.stack((m00, m01), -1), torch.stack((m10, m11), -1)), -2)


def apply_one_qubit(state: torch.Tensor, gate: torch.Tensor, qubit: int, qubits: int) -> torch.Tensor:
    """Áp cổng gate (2, 2) lên qubit này của state (n, 2^qubits)."""
    view = state.reshape(state.shape[0], 2 ** qubit, 2, 2 ** (qubits - 1 - qubit))
    return torch.einsum("ab,nibj->niaj", gate, view).reshape(state.shape)


def bloch_of_qubits(state: torch.Tensor, qubits: int) -> torch.Tensor:
    """state (n, 2^qubits) -> (n, 3 * qubits): X, Y, Z của mọi qubit, theo thứ tự X0 Y0 Z0 X1 Y1 Z1 ..."""
    out = []
    for qubit in range(qubits):
        view = state.reshape(state.shape[0], 2 ** qubit, 2, 2 ** (qubits - 1 - qubit))
        zero, one = view[:, :, 0], view[:, :, 1]
        coherence = (zero * one.conj()).sum(dim=(1, 2))  # rho_01 (phần chồng chập giữa |0> và |1>)
        z = (zero.abs() ** 2 - one.abs() ** 2).sum(dim=(1, 2))
        out += [2 * coherence.real, -2 * coherence.imag, z]
    return torch.stack(out, dim=1)


class QuantumConvStage(nn.Module):
    """Một tầng quanvolution: với mỗi pixel, một mạch 9 qubit trên ô 3x3 quanh nó, dùng chung cho mọi pixel."""

    def __init__(self, layers: int, frozen: bool, hidden: int = 0):
        super().__init__()
        angles = torch.empty(layers, PATCH_QUBITS, 3).uniform_(-math.pi, math.pi)
        if frozen:
            self.register_buffer("angles", angles)
        else:
            self.angles = nn.Parameter(angles)
        if hidden:
            self.readout = nn.Sequential(nn.Linear(3 * PATCH_QUBITS, hidden), nn.GELU(), nn.Linear(hidden, 2))
            last = self.readout[-1]
        else:
            self.readout = last = nn.Linear(3 * PATCH_QUBITS, 2)
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
        self.register_buffer("signs", cz_signs(PATCH_QUBITS, grid_edges()), persistent=False)

    def features(self, theta: torch.Tensor, phi: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """theta, phi (batch, H*W) -> (batch, H*W, 27) giá trị đo được của mạch trên ô quanh mỗi pixel."""
        batch = theta.shape[0]
        windows = [F.unfold(x.reshape(batch, 1, height, width), 3, padding=1)  # (batch, 9, H*W)
                   .transpose(1, 2).reshape(-1, PATCH_QUBITS) for x in (theta, phi)]
        theta_w, phi_w = windows
        amp0 = cis(torch.cos(torch.pi * theta_w / 2), torch.zeros_like(phi_w))
        amp1 = cis(torch.sin(torch.pi * theta_w / 2), torch.pi * phi_w)
        qubits = torch.stack((amp0, amp1), dim=-1)  # (n, 9, 2)
        gates = rotation_matrices(self.angles)  # (L, 9, 2, 2)
        # Lớp quay đầu tiên tác dụng lên trạng thái tích, nên áp lên từng qubit trước khi lấy tích tensor.
        qubits = torch.einsum("qab,nqb->nqa", gates[0], qubits)
        state = qubits[:, 0]
        for qubit in range(1, PATCH_QUBITS):
            state = (state[:, :, None] * qubits[:, qubit, None, :]).reshape(state.shape[0], -1)
        signs = self.signs.to(state.dtype)
        for layer in range(gates.shape[0]):
            if layer > 0:
                for qubit in range(PATCH_QUBITS):
                    state = apply_one_qubit(state, gates[layer, qubit], qubit, PATCH_QUBITS)
            state = state * signs
        return bloch_of_qubits(state, PATCH_QUBITS).reshape(batch, height * width, -1)

    def forward(self, theta, phi, height, width):
        out = self.readout(self.features(theta, phi, height, width))
        return theta + out[..., 0], phi + out[..., 1]


class QuantumFront(nn.Module):
    """S tầng QuantumConvStage nối tiếp; tầng đầu nhận giá trị pixel làm theta và phi = 0.

    Được tạo trong trạng thái ngẫu nhiên CPU tách riêng (fork_rng), nên phần còn lại của model có cùng giá trị khởi
    tạo (và khi train rút cùng các số ngẫu nhiên) như khi không có encoder."""

    def __init__(self, mode: str, height: int, width: int):
        super().__init__()
        parsed = parse_qconv(mode)
        if parsed is None:
            raise ValueError(f"Unknown encoder {mode!r}; use qconv<L>[x<S>][h<H>][f], e.g. qconv4x3")
        layers, stages, frozen, hidden = parsed
        self.mode, self.height, self.width, self.phase = mode, height, width, True
        self.layers, self.frozen, self.hidden = layers, frozen, hidden
        with torch.random.fork_rng(devices=[]):
            self.stages = nn.ModuleList([QuantumConvStage(layers, frozen, hidden) for _ in range(stages)])

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta, phi = values, torch.zeros_like(values)
        for stage in self.stages:
            if self.training and torch.is_grad_enabled():
                # Tính lại trạng thái 9 qubit trong lượt backward thay vì giữ trong bộ nhớ; kết quả không đổi.
                theta, phi = checkpoint(stage, theta, phi, self.height, self.width, use_reentrant=False,
                                        preserve_rng_state=False)
            else:
                theta, phi = stage(theta, phi, self.height, self.width)
        return theta, phi


def circuit_parameters(front: Optional[nn.Module]) -> int:
    """Số góc mạch học được của QuantumFront (các Linear readout là cổ điển); 0 nếu không có encoder."""
    if not isinstance(front, QuantumFront):
        return 0
    return sum(stage.angles.numel() for stage in front.stages if isinstance(stage.angles, nn.Parameter))
