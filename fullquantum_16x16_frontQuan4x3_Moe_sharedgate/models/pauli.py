from __future__ import annotations

from itertools import combinations

import torch
from torch import nn

_PAIR_AXES = ("x", "y", "z")


class PauliFeatures(nn.Module):
    """Pauli expectations of a joint density matrix, without building any 2**n x 2**n operator.

    A Pauli string has one non-zero entry per column, so Tr(rho O) only needs 2**n entries of rho.
    Output order: all X, all Y, all Z, then the 9 correlators of each pair. This module has no parameters.
    """

    def __init__(self, final_qubits: int = 8):
        super().__init__()
        terms = [{qubit: axis} for axis in _PAIR_AXES for qubit in range(final_qubits)]
        for first, second in combinations(range(final_qubits), 2):
            terms.extend({first: axis_i, second: axis_j} for axis_i in _PAIR_AXES for axis_j in _PAIR_AXES)
        columns = torch.arange(2**final_qubits)
        rows, phases = [], []
        for term in terms:
            row = columns.clone()
            phase = torch.ones(columns.shape, dtype=torch.complex64)
            for qubit, axis in term.items():
                shift = final_qubits - 1 - qubit  # Qubit 0 is the most significant bit.
                sign = 1 - 2 * ((columns >> shift) & 1)
                if axis in ("x", "y"):
                    row = row ^ (1 << shift)
                if axis == "y":
                    phase = phase * (1j * sign)  # Y|0> = i|1>, Y|1> = -i|0>.
                if axis == "z":
                    phase = phase * sign
            rows.append(row)
            phases.append(phase)
        self.register_buffer("columns", columns, persistent=False)
        self.register_buffer("rows", torch.stack(rows), persistent=False)
        self.register_buffer("phases", torch.stack(phases), persistent=False)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        return (rho[:, self.columns, self.rows] * self.phases).sum(dim=-1).real
