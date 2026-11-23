"""Router của mixture of experts: cho mỗi ảnh một trọng số g_e (softmax) cho mỗi cây.

Đầu vào chung của mọi router: hai bản đồ 16x16 (theta, phi) của từng pixel (sau encoder, hoặc ảnh gốc với phi = 0),
lấy trung bình theo lưới gh x gw ô (adaptive_avg_pool2d) -> 2 * gh * gw số. Ô đánh số theo hàng của lưới.

    none     không router: g = 1/E cho mọi ảnh (trung bình đều, đúng bằng head linear3 của linhead).
    quantum  mạch Q qubit, mô phỏng statevector chính xác (2^Q biên độ):
               |0...0> ; lặp R = (gh*gw)/Q vòng: mỗi qubit q nhận ô c = r*Q + q bằng cổng mã hóa
               RZ(pi*phi_c) RY(pi*theta_c) (giống cách encoder mã hóa pixel), rồi L lớp học được.
             Lớp "se" (strongly-entangling, giống khối 8 qubit cuối của cây): mỗi qubit một unitary 1 qubit học được
               (8 số), rồi Q unitary 2 qubit học được trên vòng (qubit i, i + r), r = (chỉ số lớp % (Q-1)) + 1
               (32 số mỗi cái): 40Q mỗi lớp (320 với Q = 8).
             Lớp "rot": mỗi qubit RZ RY RZ (3 góc), rồi CZ trên vòng (cố định): 3Q mỗi lớp.
             Đo <Z> của E qubit cách đều (qubit (e*Q)//E), logit_e = s_e * <Z_e>, s_e học được (E số cổ điển,
             khởi tạo 0 -> lúc đầu g đều, model bắt đầu y hệt router none).
             Mặc định lưới 4x4, Q = 8, L = 1, "se": R = 2 vòng, 640 góc mạch + 3 = 643 tham số.
    linear   Linear(2*gh*gw, E), khởi tạo 0: 99 tham số với lưới 4x4, E = 3.
    mlp      Linear(2*gh*gw, H) GELU Linear(H, E), lớp cuối khởi tạo 0: H = 18 -> 651 (đối chứng cùng cỡ router
             quantum 643).

Mọi router được tạo trong trạng thái ngẫu nhiên tách riêng (fork_rng), sau mọi phần khác của model: phần còn lại
khởi tạo giống hệt khi không có router.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .front import apply_one_qubit, cz_signs, rotation_matrices
from .unitary import LearnedUnitary1Q, LearnedUnitary2Q

ROUTERS = ("none", "quantum", "linear", "mlp")
ANSATZE = ("se", "rot")


def pooled_cells(theta: torch.Tensor, phi: torch.Tensor, height: int, width: int,
                 grid: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """theta, phi (batch, H*W) theo hàng -> (batch, gh*gw) mỗi cái: trung bình từng ô của lưới."""
    batch = theta.shape[0]
    maps = torch.stack((theta, phi), dim=1).reshape(batch, 2, height, width)
    pooled = F.adaptive_avg_pool2d(maps, tuple(grid)).flatten(2)  # (batch, 2, cells)
    return pooled[:, 0], pooled[:, 1]


def encoding_gates(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """(...,) -> (..., 2, 2): RZ(pi*phi) RY(pi*theta), bỏ pha toàn cục (RZ = diag(1, e^{i pi phi}))."""
    c = torch.cos(torch.pi * theta / 2)
    s = torch.sin(torch.pi * theta / 2)
    phase = torch.polar(torch.ones_like(phi), torch.pi * phi)
    zero = torch.zeros_like(c)
    top = torch.stack((torch.complex(c, zero), torch.complex(-s, zero)), dim=-1)
    bottom = torch.stack((phase * s, phase * c), dim=-1)
    return torch.stack((top, bottom), dim=-2)


def apply_one_qubit_batched(state: torch.Tensor, gates: torch.Tensor, qubit: int, qubits: int) -> torch.Tensor:
    """state (n, 2^qubits), gates (n, 2, 2): mỗi mẫu một cổng riêng trên qubit này."""
    view = state.reshape(state.shape[0], 2 ** qubit, 2, 2 ** (qubits - 1 - qubit))
    return torch.einsum("nab,nibj->niaj", gates, view).reshape(state.shape)


def apply_two_qubit(state: torch.Tensor, gate: torch.Tensor, first: int, second: int, qubits: int) -> torch.Tensor:
    """state (n, 2^qubits), gate (4, 4) trên (first, second); chỉ số cổng = 2*bit(first) + bit(second).
    Qubit 0 là bit cao nhất của chỉ số trạng thái (như models/front.py)."""
    if first == second:
        raise ValueError("two different qubits needed")
    tensor = state.reshape(state.shape[0], *([2] * qubits))
    axes = [axis for axis in range(1, qubits + 1) if axis not in (1 + first, 1 + second)]
    order = [0, *axes, 1 + first, 1 + second]
    moved = tensor.permute(*order)
    shape = moved.shape
    updated = (moved.reshape(-1, 4) @ gate.transpose(0, 1)).reshape(shape)
    inverse = [0] * len(order)
    for position, axis in enumerate(order):
        inverse[axis] = position
    return updated.permute(*inverse).reshape(state.shape)


def z_expectation(state: torch.Tensor, qubit: int, qubits: int) -> torch.Tensor:
    """state (n, 2^qubits) -> (n,) <Z> của qubit này."""
    probabilities = (state.abs() ** 2).reshape(state.shape[0], 2 ** qubit, 2, 2 ** (qubits - 1 - qubit))
    return probabilities[:, :, 0].sum(dim=(1, 2)) - probabilities[:, :, 1].sum(dim=(1, 2))


class QuantumRouter(nn.Module):
    """Mạch Q qubit có re-upload các ô của lưới; xem docstring của file."""

    def __init__(self, experts: int, grid: Sequence[int] = (4, 4), qubits: int = 8, layers: int = 1,
                 ansatz: str = "se", init_scale: float = 0.01):
        super().__init__()
        cells = grid[0] * grid[1]
        if ansatz not in ANSATZE:
            raise ValueError(f"Unknown router ansatz {ansatz!r}; use one of {ANSATZE}")
        if qubits < 3 or cells % qubits:
            raise ValueError(f"router: {cells} cells cannot be spread evenly over {qubits} qubits (need Q >= 3)")
        if not 1 <= experts <= qubits:
            raise ValueError(f"router: {experts} experts need at most {qubits} qubits")
        if layers < 1:
            raise ValueError("router layers must be at least 1")
        self.experts, self.grid, self.qubits, self.layers, self.ansatz = experts, tuple(grid), qubits, layers, ansatz
        self.rounds = cells // qubits
        self.readout_qubits = [(e * qubits) // experts for e in range(experts)]
        total_layers = self.rounds * layers
        if ansatz == "se":
            self.rotations = nn.ModuleList([
                nn.ModuleList([LearnedUnitary1Q(init_scale=init_scale) for _ in range(qubits)])
                for _ in range(total_layers)])
            self.entanglers = nn.ModuleList([
                nn.ModuleList([LearnedUnitary2Q(init_scale=init_scale) for _ in range(qubits)])
                for _ in range(total_layers)])
        else:
            self.angles = nn.Parameter(init_scale * torch.randn(total_layers, qubits, 3))
            ring = [(q, (q + 1) % qubits) for q in range(qubits)]
            self.register_buffer("signs", cz_signs(qubits, ring), persistent=False)
        # Hệ số đọc ra (cổ điển), khởi tạo 0: lúc đầu mọi logit cổng = 0, g đều.
        self.scale = nn.Parameter(torch.zeros(experts))

    def circuit_parameters(self) -> int:
        return sum(parameter.numel() for name, parameter in self.named_parameters() if name != "scale")

    def _layer(self, state: torch.Tensor, index: int) -> torch.Tensor:
        q = self.qubits
        if self.ansatz == "se":
            for qubit, gate in enumerate(self.rotations[index]):
                state = apply_one_qubit(state, gate(), qubit, q)
            reach = (index % (q - 1)) + 1
            for first, gate in enumerate(self.entanglers[index]):
                state = apply_two_qubit(state, gate(), first, (first + reach) % q, q)
            return state
        gates = rotation_matrices(self.angles[index])  # (Q, 2, 2)
        for qubit in range(q):
            state = apply_one_qubit(state, gates[qubit], qubit, q)
        return state * self.signs.to(state.dtype)

    def expectations(self, theta_cells: torch.Tensor, phi_cells: torch.Tensor) -> torch.Tensor:
        """(batch, cells) mỗi cái -> (batch, E) giá trị <Z> của các qubit đọc ra (chính xác)."""
        batch, q = theta_cells.shape[0], self.qubits
        state = torch.zeros(batch, 2 ** q, dtype=torch.complex64, device=theta_cells.device)
        state[:, 0] = 1
        gates = encoding_gates(theta_cells.float(), phi_cells.float())  # (batch, cells, 2, 2)
        index = 0
        for round_ in range(self.rounds):
            for qubit in range(q):
                state = apply_one_qubit_batched(state, gates[:, round_ * q + qubit], qubit, q)
            for _ in range(self.layers):
                state = self._layer(state, index)
                index += 1
        return torch.stack([z_expectation(state, qubit, q) for qubit in self.readout_qubits], dim=1)

    def forward(self, theta_cells: torch.Tensor, phi_cells: torch.Tensor) -> torch.Tensor:
        return self.expectations(theta_cells, phi_cells) * self.scale


class ClassicalRouter(nn.Module):
    """hidden 0: Linear(2*cells, E); hidden > 0: Linear(2*cells, hidden) GELU Linear(hidden, E). Lớp cuối = 0."""

    def __init__(self, experts: int, grid: Sequence[int] = (4, 4), hidden: int = 0):
        super().__init__()
        inputs = 2 * grid[0] * grid[1]
        if hidden:
            self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, experts))
            last = self.net[-1]
        else:
            self.net = last = nn.Linear(inputs, experts)
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()

    def circuit_parameters(self) -> int:
        return 0

    def forward(self, theta_cells: torch.Tensor, phi_cells: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((theta_cells, phi_cells), dim=1))


def make_router(kind: str, experts: int, grid: Sequence[int], qubits: int, layers: int, ansatz: str,
                init_scale: float, hidden: int) -> nn.Module | None:
    if kind not in ROUTERS:
        raise ValueError(f"Unknown router {kind!r}; use one of {ROUTERS}")
    if kind == "none":
        return None
    # Chỉ rút từ trạng thái ngẫu nhiên CPU tách riêng; không gọi torch.manual_seed (xem models/front.py).
    with torch.random.fork_rng(devices=[]):
        if kind == "quantum":
            return QuantumRouter(experts, grid, qubits, layers, ansatz, init_scale)
        if kind == "linear":
            return ClassicalRouter(experts, grid, 0)
        if hidden < 1:
            raise ValueError("router mlp needs --router-hidden >= 1")
        return ClassicalRouter(experts, grid, hidden)


def router_parameter_count(kind: str, experts: int, grid: Sequence[int], qubits: int, layers: int, ansatz: str,
                           hidden: int) -> tuple[int, int]:
    """(tham số mạch, tham số cổ điển) tính tay, để test so với model."""
    cells = grid[0] * grid[1]
    if kind == "none":
        return 0, 0
    if kind == "quantum":
        per_layer = 40 * qubits if ansatz == "se" else 3 * qubits
        return (cells // qubits) * layers * per_layer, experts
    if kind == "linear":
        return 0, 2 * cells * experts + experts
    return 0, 2 * cells * hidden + hidden + hidden * experts + experts
