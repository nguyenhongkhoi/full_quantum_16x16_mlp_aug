"""Re-upload variant of the tensor-network core (used by the best full-quantum setting).

After the conv/pool blocks, the mean RY angle of the input pixels that map to each of the
final qubits is re-applied as a learned-scale RY rotation (``reupload_scale``, 8 params per
branch, initialised to 0 so a warm-started checkpoint is reproduced exactly) before the
merge stage. Copied verbatim from Stage 6/9 (scripts/run_stage9_diagnose.py).
"""
from __future__ import annotations

import torch
from torch import nn

from .tn_core import QuantumTNCore, _pure_to_density


class ReuploadQuantumTNCore(QuantumTNCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reupload_scale = nn.Parameter(torch.zeros(self.final_qubits))
        self.param_scale = 1.0  # reupload_scale is used as param_scale * stored (fusion_model.scale_learning_rate)

    def forward(self, qubits):
        return self.forward_levels(qubits)[0]

    def forward_levels(self, qubits, keep=(), rho=None):
        """The core output, and the one-qubit states after each tree level whose qubit count is in `keep`.

        rho, if given, replaces the encoded pixel states as the tree input (one 2x2 state per site, in the same site
        order as qubits); the re-upload angles still come from qubits. Without rho nothing changes."""
        if rho is None:
            rho = _pure_to_density(qubits)
        kept = {}
        for block in self.blocks:
            rho = block(rho)
            if rho.shape[1] in keep:
                kept[rho.shape[1]] = rho
        # Magnitudes, because the amplitude of |1> can carry a position phase. Both are non-negative without it.
        angles = 2 * torch.atan2(qubits[..., 1].abs(), qubits[..., 0].abs())
        # Pooling pairs adjacent sites and keeps the left site at every level, so
        # final qubit q represents the contiguous input interval for that subtree.
        # Average each such interval to align the reupload angle with its core qubit.
        angles = angles.reshape(angles.shape[0], self.final_qubits, -1).mean(dim=2)
        scale = self.reupload_scale if self.param_scale == 1.0 else self.reupload_scale * self.param_scale
        angles = angles * scale[None, :]
        c = torch.cos(angles / 2)
        s = torch.sin(angles / 2)
        gate = torch.stack((torch.stack((c, -s), -1), torch.stack((s, c), -1)), -2).to(rho.dtype)
        rho = gate @ rho @ gate.conj().transpose(-1, -2)
        return self.merge_stage(rho), kept


def add_reupload(model, *, input_sites, final_qubits, init_scale, merge_rounds, entangling_type, n_layers):
    """Swap every branch's tn_core of a FullQuantumQCNN for a ReuploadQuantumTNCore."""
    def make_core():
        return ReuploadQuantumTNCore(
            input_sites=input_sites, final_qubits=final_qubits, init_scale=init_scale,
            merge_rounds=merge_rounds, entangling_type=entangling_type, n_layers=n_layers,
        )

    model.tn_core = make_core()
    for branch in model.extra_branches:
        branch["tn_core"] = make_core()
    return model
