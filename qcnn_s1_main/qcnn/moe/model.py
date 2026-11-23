"""Encoder quanvolution + router (mixture of experts, tùy chọn) + các cây level32 + head mỗi cây (tùy chọn).

Luồng dữ liệu (mặc định):
    ảnh 28x28 -> 16x16 -> encoder qconv4x3 -> (theta, phi) mỗi pixel -> mỗi pixel một qubit
      -> E cây (một cây mỗi thứ tự quét, mặc định row, column, morton) -> đặc trưng đo của cây -> head -> logit l_e
      -> router đọc (theta, phi) gộp theo lưới -> g = softmax(logit cổng) (E số, tổng 1)
      -> mix "logits": logit = sum_e g_e * l_e          (router none: g = 1/E)
         mix "probs":  log p = log sum_e g_e softmax(l_e) (MoE cổ điển kiểu Jacobs 1991: các cây cạnh tranh)

Đặc trưng đo của một cây: 276 = X, Y, Z của 8 qubit cuối (24) + 9 tương quan của 28 cặp (252), cộng 3 * side_level
(X, Y, Z của các qubit ở tầng side_level; mặc định 32 -> 96): 372.

Head mỗi cây (--expert-head), số với 3 cây, side_level 32:
    linear   Linear(372, 10) mỗi cây                                          11.190
    shared   một Linear(372, 10) chung                                         3.730
    mlp      Linear(372, H) GELU Dropout Linear(H, 10) mỗi cây                 H = 16: 18.414
    bloch8   Linear(24, 10) trên X, Y, Z của 8 qubit cuối (bỏ tương quan và tầng side)    750
    nocorr   Linear(120, 10): 24 Bloch 8 qubit cuối + 96 tầng side, bỏ 252 tương quan cặp   3.630
    nocorr-z        Linear(56, 10): 24 Bloch 8 qubit cuối + chỉ <Z> 32 qubit tầng side      1.710
    nocorr-shared   một Linear(120, 10) chung cho 3 cây                                      1.210
    nocorr-lowrank  Linear(120, r, không bias) -> Linear(r, 10) mỗi cây (--head-rank r)    r=4: 1.590; r=2: 810
    samebasis       Linear(204, 10): 24 Bloch + 84 tương quan cùng trục XX/YY/ZZ + 96 side    6.150
    zz              Linear(148, 10): 24 Bloch + 28 tương quan ZZ + 96 side                    4.470
    obs10    10 quan sát Pauli cố định (--obs-terms), logit_c = a_c * <O_c> + b_c        60
    fusion   head MLP của run 2c: mỗi cây Linear(372, P) LayerNorm GELU Dropout, nối 3P -> Linear(3P, hidden)
             LayerNorm GELU Dropout Linear(hidden, 10) (P = --projection 10, hidden 16: 11.948). Không phải MoE:
             cần --router none; chỉ có đầu ra soft (không có top1, từng cây).

Readout encoder (--front-readout): "linear" = qcnn/encoding/quanvolution.py như cũ (Linear(27, 2) mỗi tầng, cộng dư; qconv4x3: 492,
góc 324); "direct" = qcnn/encoding/quanvolution_direct.py: lấy thẳng qubit giữa, không tham số cổ điển (qconv4x3: 324); cây nhận
ma trận mật độ của qubit giữa.

Router không chặn đường dữ liệu: cả E cây đều nhận đầu ra encoder; router chỉ cho trọng số trộn.
Chế độ suy luận (mode): "soft" (mặc định), "top1" (chỉ cây có g lớn nhất), "uniform", "tree<i>".

Thứ tự tạo module giống FusionQCNN của full_quantum_16x16_level32_qconv_linhead (cùng tên module): cùng seed thì
--expert-head linear / shared với router none khởi tạo y hệt L1 / L2, --expert-head fusion y hệt run 2c, và nạp được
checkpoint của các run đó (với --front-readout linear). Router tạo sau cùng, trong trạng thái ngẫu nhiên tách riêng,
và bắt đầu với g đều.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..core.reupload import ReuploadQuantumTNCore
from ..core.scan import SPATIAL_ORDERS, spatial_permutation
from ..encoding.angle import angle_ry_encode
from ..encoding.preprocessing import preprocess_full_quantum
from ..encoding.quanvolution import front_circuit_parameters, make_front
from ..encoding.quanvolution_direct import DirectQuantumFront, bloch_to_angles, direct_circuit_parameters
from ..layers.counting import free_parameter_count
from ..layers.density import bloch, density_from_bloch
from ..layers.pauli import PauliFeatures
from .heads import (BRANCH_FEATURES, DEFAULT_OBS, FINAL_QUBITS, NoCorrLinear, NoCorrSmall, ObservableHead, SameAxisLinear,
                    SliceLinear)
from .router import ROUTERS, make_router, pooled_cells


TARGET_SIZE = (16, 16)


NOCORR_SMALL = ("nocorr-z", "nocorr-shared", "nocorr-lowrank")  # thêm 2026-11-19


SAME_AXIS = {"samebasis": "XYZ", "zz": "Z"}  # thêm 2026-11-19


EXPERT_HEADS = ("linear", "shared", "mlp", "bloch8", "obs10", "fusion", "nocorr") + NOCORR_SMALL + tuple(SAME_AXIS)


SIDE_HEADS = ("linear", "shared", "mlp", "fusion", "nocorr") + NOCORR_SMALL + tuple(SAME_AXIS)  # head đo tầng side


MIXES = ("logits", "probs")


ROUTER_SOURCES = ("encoder", "pixels")


FRONT_READOUTS = ("linear", "direct")


def mix_logits(expert_logits: torch.Tensor, gates: torch.Tensor, mix: str) -> torch.Tensor:
    """expert_logits (batch, E, C), gates (batch, E), tổng 1 -> đầu ra đưa vào cross_entropy.

    "probs" trả về log xác suất (tổng exp = 1): log_softmax giữ nguyên nên cross_entropy vẫn đúng."""
    if mix == "logits":
        return (gates[:, :, None] * expert_logits).sum(dim=1)
    log_gates = gates.clamp_min(1e-12).log()
    return torch.logsumexp(log_gates[:, :, None] + expert_logits.log_softmax(dim=2), dim=1)


class MoEQCNN(nn.Module):
    MODES = ("soft", "top1", "uniform")

    def __init__(self, num_classes: int = 10, orders: Sequence[str] = SPATIAL_ORDERS, side_level: int = 32,
                 front: str = "qconv4x3", front_readout: str = "linear", expert_head: str = "linear",
                 expert_hidden: int = 16, projection: int = 10, hidden: int = 16, dropout: float = 0.0,
                 obs_terms: Sequence[str] = DEFAULT_OBS, router: str = "quantum", router_source: str = "encoder",
                 router_detach: bool = False, router_grid: Sequence[int] = (4, 4), router_qubits: int = 8,
                 router_layers: int = 1, router_ansatz: str = "se", router_init_scale: float = 0.01,
                 router_hidden: int = 18, mix: str = "logits", gate_noise: float = 0.0, head_rank: int = 4,
                 share_final: bool = False, share_gates: bool = False):
        super().__init__()
        for value, allowed, name in ((expert_head, EXPERT_HEADS, "expert head"), (mix, MIXES, "mix"),
                                     (router_source, ROUTER_SOURCES, "router source"), (router, ROUTERS, "router"),
                                     (front_readout, FRONT_READOUTS, "front readout")):
            if value not in allowed:
                raise ValueError(f"Unknown {name} {value!r}; use one of {allowed}")
        if side_level not in (0, 16, 32, 64, 128):
            raise ValueError(f"No tree level has {side_level} qubits (0 = no side tap)")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if len(orders) < 2 and router != "none":
            raise ValueError("a router needs at least 2 trees (--orders)")
        if expert_head == "fusion" and router != "none":
            raise ValueError("--expert-head fusion is the 2c MLP head, not a mixture: use --router none")
        if front_readout == "direct" and front == "none":
            raise ValueError("--front-readout direct needs a quantum front (e.g. --front qconv4x3)")
        self.orders, self.mix, self.expert_head = tuple(orders), mix, expert_head
        # bloch8 / obs10 chỉ đọc 8 qubit cuối: không đo tầng side (đỡ tính).
        self.side_level = (side_level or None) if expert_head in SIDE_HEADS else None
        self.router_kind, self.router_source, self.router_detach = router, router_source, router_detach
        self.router_grid, self.gate_noise, self.front_readout = tuple(router_grid), gate_noise, front_readout
        experts = len(orders)
        branch_inputs = BRANCH_FEATURES + (3 * self.side_level if self.side_level else 0)

        # Thứ tự tạo như FusionQCNN (linhead) để cùng seed thì cùng giá trị khởi tạo.
        self.register_buffer("site_permutations", torch.stack([
            spatial_permutation(*TARGET_SIZE, order) for order in orders
        ]), persistent=False)
        self.cores = nn.ModuleList([
            ReuploadQuantumTNCore(input_sites=TARGET_SIZE[0] * TARGET_SIZE[1], final_qubits=FINAL_QUBITS,
                                  init_scale=0.01, merge_rounds=2, entangling_type="strongly_entangling",
                                  n_layers=4)
            for _ in orders
        ])
        # --share-final (thêm 2026-11-20): mọi cây dùng chung khối 8 qubit cuối (merge_stage: 4 lớp strongly
        # entangling) của cây đầu. Các cây vẫn được tạo đủ như cũ (cùng seed thì phần còn lại khởi tạo y hệt), rồi
        # thay tham chiếu; khối của cây 2, 3 bị bỏ. Phần conv-pool và hệ số re-upload vẫn riêng từng cây.
        self.share_final = share_final
        # --share-gates (folder _sharedgate, thêm 2026-11-20): trong khối 8 qubit cuối, mỗi lớp dùng CHUNG một unitary
        # 1 qubit cho cả 8 qubit và một unitary 2 qubit cho cả 8 cặp vòng (i, i+r), giống conv-pool dùng chung cho
        # mọi cặp. Mỗi lớp 4 + 16 = 20 tham số thay vì 8 x 4 + 8 x 16 = 160. Tạo đủ cổng như cũ (cùng seed thì phần
        # còn lại khởi tạo y hệt) rồi trỏ mọi vị trí về cổng của qubit / cặp 0.
        self.share_gates = share_gates
        if share_gates:
            for core in self.cores:
                stage = core.merge_stage
                for layer in range(len(stage.rotations)):
                    stage.rotations[layer] = nn.ModuleList([stage.rotations[layer][0]] * len(stage.rotations[layer]))
                    stage.entanglers[layer] = nn.ModuleList(
                        [stage.entanglers[layer][0]] * len(stage.entanglers[layer]))
        if share_final:
            for core in self.cores[1:]:
                core.merge_stage = self.cores[0].merge_stage
        self.pauli = PauliFeatures(FINAL_QUBITS)
        self.fusion = None
        if expert_head == "linear":
            self.projections = nn.ModuleList([nn.Linear(branch_inputs, num_classes) for _ in orders])
        elif expert_head == "shared":
            shared = nn.Linear(branch_inputs, num_classes)
            self.projections = nn.ModuleList([shared] * experts)
        elif expert_head == "mlp":
            self.projections = nn.ModuleList([
                nn.Sequential(nn.Linear(branch_inputs, expert_hidden), nn.GELU(), nn.Dropout(dropout),
                              nn.Linear(expert_hidden, num_classes))
                for _ in orders
            ])
        elif expert_head == "bloch8":
            self.projections = nn.ModuleList([SliceLinear(3 * FINAL_QUBITS, num_classes) for _ in orders])
        elif expert_head == "obs10":
            self.projections = nn.ModuleList([ObservableHead(obs_terms, num_classes) for _ in orders])
        elif expert_head == "nocorr":
            side_features = branch_inputs - BRANCH_FEATURES
            self.projections = nn.ModuleList([NoCorrLinear(side_features, num_classes) for _ in orders])
        elif expert_head in SAME_AXIS:
            side_features = branch_inputs - BRANCH_FEATURES
            self.projections = nn.ModuleList([
                SameAxisLinear(side_features, num_classes, SAME_AXIS[expert_head]) for _ in orders])
        elif expert_head in NOCORR_SMALL:
            side_features = branch_inputs - BRANCH_FEATURES
            if expert_head == "nocorr-shared":
                shared = NoCorrSmall(side_features, num_classes)
                self.projections = nn.ModuleList([shared] * experts)
            elif expert_head == "nocorr-z":
                self.projections = nn.ModuleList([NoCorrSmall(side_features, num_classes, "Z") for _ in orders])
            else:
                if head_rank <= 0:
                    raise ValueError("--expert-head nocorr-lowrank needs --head-rank >= 1")
                self.projections = nn.ModuleList([
                    NoCorrSmall(side_features, num_classes, "XYZ", head_rank) for _ in orders])
        else:  # fusion: như FusionQCNN head "mlp" (run 2c)
            self.projections = nn.ModuleList([
                nn.Sequential(nn.Linear(branch_inputs, projection), nn.LayerNorm(projection), nn.GELU(),
                              nn.Dropout(dropout))
                for _ in orders
            ])
            self.fusion = nn.Sequential(
                nn.Linear(experts * projection, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        self.front_mode = front
        if front == "none":
            self.front = None
        elif front_readout == "direct":
            self.front = DirectQuantumFront(front, *TARGET_SIZE)
        else:
            self.front = make_front(front, *TARGET_SIZE)
        self.router = make_router(router, experts, router_grid, router_qubits, router_layers, router_ansatz,
                                  router_init_scale, router_hidden)
        self.last: Dict[str, torch.Tensor] = {}

    # ---- forward ----
    def gates(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """(batch, E) trọng số trộn, tổng 1."""
        batch, experts = theta.shape[0], len(self.orders)
        if self.router is None:
            return theta.new_full((batch, experts), 1.0 / experts)
        if self.router_detach:
            theta, phi = theta.detach(), phi.detach()
        theta_cells, phi_cells = pooled_cells(theta, phi, *TARGET_SIZE, self.router_grid)
        logits = self.router(theta_cells, phi_cells)
        if self.training and self.gate_noise > 0:
            logits = logits + self.gate_noise * torch.randn_like(logits)
        return logits.softmax(dim=1)

    def encode(self, images: torch.Tensor):
        """-> values (pixel), theta, phi (cho router), qubits (cho re-upload), rho (đầu vào cây, None = thuần)."""
        values = preprocess_full_quantum(images, TARGET_SIZE, grayscale=True, mode="resize")
        if self.front is not None and self.front_readout == "direct":
            vector = self.front(values)  # (batch, 256, 3) Bloch của qubit giữa
            theta, phi = bloch_to_angles(vector)
            qubits = angle_ry_encode(theta, clamp=False).to(images.device)
            return values, theta, phi, qubits, density_from_bloch(vector)
        theta, phi = values, None
        if self.front is not None:
            theta, phi = self.front(values)
        qubits = angle_ry_encode(theta, clamp=self.front is None).to(images.device)
        if phi is not None:
            # RZ(pi * phi) bỏ pha toàn cục: nhân exp(i pi phi) vào |1>. Góc re-upload chỉ dùng độ lớn.
            qubits = torch.stack((qubits[..., 0], qubits[..., 1] * torch.polar(torch.ones_like(phi), torch.pi * phi)),
                                 dim=-1)
        return values, theta, (torch.zeros_like(theta) if phi is None else phi), qubits, None

    def forward(self, images: torch.Tensor, mode: str = "soft") -> torch.Tensor:
        values, theta, phi, qubits, rho = self.encode(images)
        if self.router_source == "pixels":
            gates = self.gates(values, torch.zeros_like(values))
        else:
            gates = self.gates(theta, phi)

        branches = []
        for index, (core, projection) in enumerate(zip(self.cores, self.projections)):
            permutation = self.site_permutations[index]
            branch_qubits = qubits.index_select(1, permutation)
            branch_rho = None if rho is None else rho.index_select(1, permutation)
            if self.training:
                features = checkpoint(self._features, core, branch_qubits, branch_rho, use_reentrant=False,
                                      preserve_rng_state=False)
            else:
                features = self._features(core, branch_qubits, branch_rho)
            branches.append(projection(features))
        if self.fusion is not None:
            self.last = {"gates": gates, "expert_logits": None}
            return self.fusion(torch.cat(branches, dim=1))
        expert_logits = torch.stack(branches, dim=1)  # (batch, E, C)
        self.last = {"gates": gates, "expert_logits": expert_logits}
        return self.combine(expert_logits, gates, mode)

    def combine(self, expert_logits: torch.Tensor, gates: torch.Tensor, mode: str = "soft") -> torch.Tensor:
        """Đầu ra theo chế độ: soft, top1, uniform, tree<i>."""
        experts = expert_logits.shape[1]
        if mode == "soft":
            return mix_logits(expert_logits, gates, self.mix)
        if mode == "uniform":
            return mix_logits(expert_logits, torch.full_like(gates, 1.0 / experts), self.mix)
        if mode == "top1":
            chosen = gates.argmax(dim=1)
        elif mode.startswith("tree") and mode[4:].isdigit() and int(mode[4:]) < experts:
            chosen = torch.full_like(gates[:, 0], int(mode[4:]), dtype=torch.long)
        else:
            raise ValueError(f"Unknown mode {mode!r}; use soft, top1, uniform or tree0..tree{experts - 1}")
        one_hot = F.one_hot(chosen, experts).to(gates.dtype)
        return mix_logits(expert_logits, one_hot, self.mix)

    def _features(self, core: ReuploadQuantumTNCore, qubits: torch.Tensor, rho=None) -> torch.Tensor:
        keep = () if self.side_level is None else (self.side_level,)
        joint, kept = core.forward_levels(qubits, keep, rho=rho)
        if self.side_level is None:
            return self.pauli(joint)
        return torch.cat((self.pauli(joint), bloch(kept[self.side_level])), dim=1)

    # ---- loss phụ của router ----
    def aux_loss(self, balance_weight: float = 0.0, entropy_weight: float = 0.0) -> torch.Tensor:
        """Sau một forward khi train.

        balance = E * sum_e (trung bình batch của g_e)^2 - 1: bằng 0 khi các cây được dùng đều, E - 1 khi mọi ảnh
        dồn vào một cây (dạng mềm của loss cân bằng tải Switch Transformer).
        entropy = trung bình entropy của g mỗi ảnh / log E (0..1): weight > 0 ép mỗi ảnh chọn rõ một cây (giúp top1),
        weight < 0 làm ngược lại."""
        gates = self.last.get("gates")
        if gates is None or self.router is None or (balance_weight == 0 and entropy_weight == 0):
            return torch.zeros((), device=gates.device if gates is not None else None)
        experts = gates.shape[1]
        balance = experts * gates.mean(dim=0).pow(2).sum() - 1
        entropy = -(gates * gates.clamp_min(1e-12).log()).sum(dim=1).mean() / math.log(experts)
        return balance_weight * balance + entropy_weight * entropy

    # ---- đếm tham số ----
    def count_parameters(self) -> Dict[str, float]:
        quantum = sum(parameter.numel() for parameter in self.cores.parameters())
        head_modules = [self.projections] + ([self.fusion] if self.fusion is not None else [])
        head = sum(parameter.numel() for module in head_modules for parameter in module.parameters())
        front = 0 if self.front is None else sum(parameter.numel() for parameter in self.front.parameters())
        if isinstance(self.front, DirectQuantumFront):
            front_quantum = direct_circuit_parameters(self.front)
        else:
            front_quantum = front_circuit_parameters(self.front)
        router = 0 if self.router is None else sum(parameter.numel() for parameter in self.router.parameters())
        router_quantum = 0 if self.router is None else self.router.circuit_parameters()
        total = quantum + front + head + router
        counts = {"quantum": quantum, "front": front, "front_quantum": front_quantum, "head": head,
                  "router": router, "router_quantum": router_quantum, "total": total,
                  "quantum_share": (quantum + front_quantum + router_quantum) / total}
        # Đếm đúng (2026-11-20): bậc tự do thật của các unitary học được; các khóa cũ ở trên giữ nguyên (đếm thừa).
        router_classical = router - router_quantum  # hệ số đọc ra của router quantum, hoặc cả router cổ điển
        for suffix, drop in (("free", False), ("free_nophase", True)):
            trees = free_parameter_count(self.cores, drop)
            router_circuit = 0 if router_quantum == 0 else free_parameter_count(self.router, drop) - router_classical
            total_free = trees + front + head + router_classical + router_circuit
            counts.update({f"quantum_{suffix}": trees, f"router_quantum_{suffix}": router_circuit,
                           f"total_{suffix}": total_free,
                           f"quantum_share_{suffix}": (trees + front_quantum + router_circuit) / total_free})
        return counts
