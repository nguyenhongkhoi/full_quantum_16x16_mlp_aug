"""Cổng lượng tử cơ bản trên statevector (dùng chung cho encoder quanvolution và router):
lưới cạnh CZ, dấu CZ, ma trận quay, áp cổng 1/2 qubit, vector Bloch, kỳ vọng Z.
"""
from __future__ import annotations

import torch


def grid_edges(size: int = 3) -> list[tuple[int, int]]:
    """Nearest-neighbour pairs of a size x size grid (12 for 3x3)."""
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
    """Diagonal of the product of CZ on these edges: (-1)^(sum over edges of b_i b_j), as a real vector."""
    index = torch.arange(2 ** qubits)
    bits = [(index >> (qubits - 1 - q)) & 1 for q in range(qubits)]
    parity = torch.zeros_like(index)
    for a, b in edges:
        parity = parity ^ (bits[a] & bits[b])
    return (1 - 2 * parity).to(torch.float32)


def cis(radius: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """radius * exp(i angle); radius may be negative (torch.polar is meant for non-negative abs)."""
    return torch.complex(radius * torch.cos(angle), radius * torch.sin(angle))


def rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    """(..., 3) angles (a, b, c) -> (..., 2, 2) complex RZ(a) RY(b) RZ(c)."""
    a, b, c = angles.unbind(-1)
    cos, sin = torch.cos(b / 2), torch.sin(b / 2)
    plus, minus = (a + c) / 2, (a - c) / 2
    m00 = cis(cos, -plus)
    m01 = cis(-sin, -minus)
    m10 = cis(sin, minus)
    m11 = cis(cos, plus)
    return torch.stack((torch.stack((m00, m01), -1), torch.stack((m10, m11), -1)), -2)


def apply_one_qubit(state: torch.Tensor, gate: torch.Tensor, qubit: int, qubits: int) -> torch.Tensor:
    """state (n, 2^qubits), gate (2, 2) on this qubit."""
    view = state.reshape(state.shape[0], 2 ** qubit, 2, 2 ** (qubits - 1 - qubit))
    return torch.einsum("ab,nibj->niaj", gate, view).reshape(state.shape)


def bloch_of_qubits(state: torch.Tensor, qubits: int) -> torch.Tensor:
    """state (n, 2^qubits) -> (n, 3 * qubits): X, Y, Z of every qubit, in the order X0 Y0 Z0 X1 Y1 Z1 ..."""
    out = []
    for qubit in range(qubits):
        view = state.reshape(state.shape[0], 2 ** qubit, 2, 2 ** (qubits - 1 - qubit))
        zero, one = view[:, :, 0], view[:, :, 1]
        coherence = (zero * one.conj()).sum(dim=(1, 2))  # rho_01
        z = (zero.abs() ** 2 - one.abs() ** 2).sum(dim=(1, 2))
        out += [2 * coherence.real, -2 * coherence.imag, z]
    return torch.stack(out, dim=1)


def apply_one_qubit_batched(state: torch.Tensor, gates: torch.Tensor, qubit: int, qubits: int) -> torch.Tensor:
    """state (n, 2^qubits), gates (n, 2, 2): mỗi mẫu một cổng riêng trên qubit này."""
    view = state.reshape(state.shape[0], 2 ** qubit, 2, 2 ** (qubits - 1 - qubit))
    return torch.einsum("nab,nibj->niaj", gates, view).reshape(state.shape)


def apply_two_qubit(state: torch.Tensor, gate: torch.Tensor, first: int, second: int, qubits: int) -> torch.Tensor:
    """state (n, 2^qubits), gate (4, 4) trên (first, second); chỉ số cổng = 2*bit(first) + bit(second).
    Qubit 0 là bit cao nhất của chỉ số trạng thái (như qcnn/encoding/quanvolution.py)."""
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
