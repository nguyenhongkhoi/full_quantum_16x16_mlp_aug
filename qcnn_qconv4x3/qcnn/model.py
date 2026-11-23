"""Model đầy đủ: encoder quanvolution -> 256 qubit lá -> 3 cây QCNN (quét hàng, cột, Morton) có đọc tầng 32
-> head cổ điển. Cùng phép tính, tên module và thứ tự rút số ngẫu nhiên như FusionQCNN trong
full_quantum_16x16_level32_qconv (side_level 32, các tùy chọn của run 2c), nên nạp được checkpoint của folder đó.

Luồng dữ liệu (batch B):
    ảnh (B,1,28,28) --tiền xử lý--> values (B,256) trong [0,1]
    --encoder qconv4x3--> theta, phi (B,256)            [encoder_quanvolution/quanvolution.py]
    --mã hóa góc--> 256 qubit (B,256,2) số phức          [encoder_quanvolution/angle_encoding.py]
    với mỗi cây t (thứ tự lá: hàng / cột / Morton):       [core/]
        5 tầng conv-pool 256->128->64->32->16->8; vector Bloch ở tầng 32 (96 số); re-upload; strongly-entangling
        trên 8 qubit; đặc trưng Pauli (276 số)  -> 372 số
        projection của head (Linear+LN+GELU+Dropout)     [layers/readout.py]
    MLP gộp -> 10 logit
Số tham số với qconv4x3, projection 10, hidden 16, head mlp: quantum 4.824 + encoder 492 (góc 324) + head
11.948 = 17.264.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .core import BRANCH_FEATURES, FINAL_QUBITS, PauliFeatures, ReuploadQuantumTNCore, SPATIAL_ORDERS, bloch, \
    spatial_permutation
from .data import preprocess_full_quantum
from .encoder_quanvolution import QuantumFront, circuit_parameters, encode_qubits
from .layers import build_head, combine

TARGET_SIZE = (16, 16)


class QCNNQconv(nn.Module):
    def __init__(self, front: str = "qconv4x3", num_classes: int = 10, side_level: Optional[int] = 32,
                 projection: int = 10, hidden: int = 16, dropout: float = 0.0, head: str = "mlp",
                 orders: Sequence[str] = SPATIAL_ORDERS):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if side_level is not None and side_level not in (16, 32, 64, 128):
            raise ValueError(f"No tree level has {side_level} qubits")
        self.side_level = side_level
        self.head = head
        branch_inputs = BRANCH_FEATURES + (3 * side_level if side_level else 0)  # 276 + 96 = 372
        # 1. thứ tự lá (buffer, không rút số ngẫu nhiên)
        self.register_buffer("site_permutations", torch.stack([
            spatial_permutation(*TARGET_SIZE, order) for order in orders
        ]), persistent=False)
        # 2. ba mạch cây
        self.cores = nn.ModuleList([
            ReuploadQuantumTNCore(input_sites=TARGET_SIZE[0] * TARGET_SIZE[1], final_qubits=FINAL_QUBITS,
                                  init_scale=0.01, merge_rounds=2, entangling_type="strongly_entangling",
                                  n_layers=4)
            for _ in orders
        ])
        # 3. phép đo 8 qubit cuối (không có tham số)
        self.pauli = PauliFeatures(FINAL_QUBITS)
        # 4. head cổ điển
        self.projections, self.fusion = build_head(head, len(orders), branch_inputs, projection, hidden, dropout,
                                                   num_classes)
        # 5. encoder, tạo sau cùng trong trạng thái ngẫu nhiên tách riêng: mọi phần ở trên khởi tạo như khi không có nó
        self.front_mode = front
        self.front = None if front == "none" else QuantumFront(front, *TARGET_SIZE)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = preprocess_full_quantum(images, TARGET_SIZE, grayscale=True, mode="resize")
        phi = None
        if self.front is not None:
            values, phi = self.front(values)
        qubits = encode_qubits(values.to(images.device), phi, clamp=self.front is None)
        branches = []
        for index, (core, projection) in enumerate(zip(self.cores, self.projections)):
            branch_qubits = qubits.index_select(1, self.site_permutations[index])
            if self.side_level is not None:
                if self.training:
                    features = checkpoint(self._side_features, core, branch_qubits, None, use_reentrant=False,
                                          preserve_rng_state=False)
                else:
                    features = self._side_features(core, branch_qubits, None)
            else:
                rho = (checkpoint(core, branch_qubits, use_reentrant=False, preserve_rng_state=False)
                       if self.training else core(branch_qubits))
                features = self.pauli(rho)
            branches.append(projection(features))
        return combine(branches, self.fusion)

    def _side_features(self, core: ReuploadQuantumTNCore, qubits: torch.Tensor,
                       rho: Optional[torch.Tensor] = None) -> torch.Tensor:
        joint, kept = core.forward_levels(qubits, (self.side_level,), rho=rho)
        return torch.cat((self.pauli(joint), bloch(kept[self.side_level])), dim=1)

    def count_parameters(self) -> Dict[str, float]:
        quantum = sum(p.numel() for p in self.cores.parameters())
        head = sum(p.numel() for module in (self.projections, self.fusion) if module is not None
                   for p in module.parameters())
        front = 0 if self.front is None else sum(p.numel() for p in self.front.parameters())
        front_quantum = circuit_parameters(self.front)
        return {"quantum": quantum, "front": front, "front_quantum": front_quantum, "head": head,
                "total": quantum + front + head, "quantum_share": (quantum + front_quantum) / (quantum + front + head)}
