"""Lõi tensor network có re-upload (dùng trong bản full-quantum tốt nhất).

Sau các khối conv-pool, góc RY trung bình của các pixel đầu vào ứng với mỗi qubit cuối được áp lại thành một
phép quay RY có hệ số học được (``reupload_scale``, 8 tham số mỗi cây, khởi tạo 0 nên checkpoint cũ được tái
tạo đúng) trước tầng gộp. Chép nguyên từ Stage 6/9 (scripts/run_stage9_diagnose.py).
"""
from __future__ import annotations

import torch
from torch import nn

from .tn_core import QuantumTNCore, _pure_to_density


class ReuploadQuantumTNCore(QuantumTNCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reupload_scale = nn.Parameter(torch.zeros(self.final_qubits))
        self.param_scale = 1.0  # reupload_scale được dùng dưới dạng param_scale * giá trị lưu (fusion_model.scale_learning_rate)

    def forward(self, qubits):
        return self.forward_levels(qubits)[0]

    def forward_levels(self, qubits, keep=(), rho=None):
        """Đầu ra của lõi, và các trạng thái một qubit sau mỗi tầng của cây có số qubit nằm trong `keep`
        (ví dụ keep = (32,): 32 ma trận 2x2 sau tầng conv-pool thứ 3).

        rho, nếu có, thay các trạng thái pixel đã mã hóa làm đầu vào của cây (mỗi vị trí một trạng thái 2x2, cùng
        thứ tự với qubits); góc re-upload vẫn lấy từ qubits. Không có rho thì không đổi gì."""
        if rho is None:
            rho = _pure_to_density(qubits)
        kept = {}
        for block in self.blocks:
            rho = block(rho)
            if rho.shape[1] in keep:
                kept[rho.shape[1]] = rho
        # Lấy độ lớn, vì biên độ của |1> có thể mang pha (phi). Không có pha thì cả hai đều không âm.
        angles = 2 * torch.atan2(qubits[..., 1].abs(), qubits[..., 0].abs())
        # Mỗi tầng pooling ghép hai vị trí kề nhau và giữ vị trí bên trái, nên qubit cuối q đại diện cho
        # đoạn lá liên tiếp của cây con đó (32 lá). Lấy trung bình mỗi đoạn để góc re-upload khớp với qubit.
        angles = angles.reshape(angles.shape[0], self.final_qubits, -1).mean(dim=2)
        scale = self.reupload_scale if self.param_scale == 1.0 else self.reupload_scale * self.param_scale
        angles = angles * scale[None, :]
        c = torch.cos(angles / 2)
        s = torch.sin(angles / 2)
        gate = torch.stack((torch.stack((c, -s), -1), torch.stack((s, c), -1)), -2).to(rho.dtype)
        rho = gate @ rho @ gate.conj().transpose(-1, -2)
        return self.merge_stage(rho), kept


def add_reupload(model, *, input_sites, final_qubits, init_scale, merge_rounds, entangling_type, n_layers):
    """Thay tn_core của mọi nhánh trong một FullQuantumQCNN bằng ReuploadQuantumTNCore (không dùng trong folder này)."""
    def make_core():
        return ReuploadQuantumTNCore(
            input_sites=input_sites, final_qubits=final_qubits, init_scale=init_scale,
            merge_rounds=merge_rounds, entangling_type=entangling_type, n_layers=n_layers,
        )

    model.tn_core = make_core()
    for branch in model.extra_branches:
        branch["tn_core"] = make_core()
    return model
