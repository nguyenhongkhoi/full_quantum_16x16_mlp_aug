from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Dict

import torch
from torch import nn


def _kron_all(matrices: list[torch.Tensor]) -> torch.Tensor:
    result = matrices[0]
    for matrix in matrices[1:]:
        result = torch.kron(result, matrix)
    return result


def _pauli_operator(name: str, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if name == "x":
        values = [[0.0, 1.0], [1.0, 0.0]]
    elif name == "y":
        return torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype, device=device)
    elif name == "z":
        values = [[1.0, 0.0], [0.0, -1.0]]
    else:
        raise ValueError(f"Unsupported Pauli operator: {name}")
    return torch.tensor(values, dtype=dtype, device=device)


def _joint_pauli_expectation(rho: torch.Tensor, names_by_qubit: Dict[int, str], final_qubits: int) -> torch.Tensor:
    identity = torch.eye(2, dtype=rho.dtype, device=rho.device)
    factors = [
        _pauli_operator(names_by_qubit[qubit], dtype=rho.dtype, device=rho.device)
        if qubit in names_by_qubit
        else identity
        for qubit in range(final_qubits)
    ]
    operator = _kron_all(factors)
    return torch.einsum("bij,ji->b", rho, operator).real


class BasisProbabilityReadout(nn.Module):
    def __init__(self, num_classes: int = 10, final_qubits: int = 4, eps: float = 1e-8):
        super().__init__()
        if 2**final_qubits < num_classes:
            raise ValueError("final_qubits cannot represent num_classes basis states")
        self.num_classes = num_classes
        self.final_qubits = final_qubits
        self.eps = eps

    def probabilities(self, rho: torch.Tensor) -> torch.Tensor:
        diag = rho.diagonal(dim1=-2, dim2=-1).real.clamp_min(self.eps)
        outputs = []
        for class_id in range(self.num_classes):
            bits = [(class_id >> bit) & 1 for bit in reversed(range(self.final_qubits))]
            prob = torch.ones(rho.shape[0], device=rho.device, dtype=diag.dtype)
            for q, bit in enumerate(bits):
                prob = prob * diag[:, q, bit]
            outputs.append(prob)
        probs = torch.stack(outputs, dim=-1)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def forward(self, rho: torch.Tensor, return_log: bool = True) -> torch.Tensor:
        probs = self.probabilities(rho)
        return torch.log(probs.clamp_min(self.eps)) if return_log else probs

    def count_parameters(self) -> Dict[str, int]:
        return {"classical_readout": 0}


class JointBasisProbabilityReadout(nn.Module):
    def __init__(self, num_classes: int = 10, final_qubits: int = 4, eps: float = 1e-8):
        super().__init__()
        if 2**final_qubits < num_classes:
            raise ValueError("final_qubits cannot represent num_classes basis states")
        self.num_classes = num_classes
        self.final_qubits = final_qubits
        self.eps = eps

    def probabilities(self, rho: torch.Tensor) -> torch.Tensor:
        dim = 2**self.final_qubits
        if rho.ndim != 3 or rho.shape[-2:] != (dim, dim):
            raise ValueError(f"Expected joint density matrices with shape (batch, {dim}, {dim})")
        probs = rho.diagonal(dim1=-2, dim2=-1).real[:, : self.num_classes].clamp_min(self.eps)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def forward(self, rho: torch.Tensor, return_log: bool = True) -> torch.Tensor:
        probs = self.probabilities(rho)
        return torch.log(probs.clamp_min(self.eps)) if return_log else probs

    def count_parameters(self) -> Dict[str, int]:
        return {"classical_readout": 0}


_PAIR_AXES = ("x", "y", "z")


class PauliObservableReadout(nn.Module):
    """Readout tren cac dac trung Pauli doc tu joint density matrix + 1 lop Linear nho.

    `pair_correlators`:
      - "zz" (mac dinh, tuong thich nguoc voi moi checkpoint da luu truoc day): chi dung tuong
        quan ZZ cho moi cap qubit, giong nguyen ban thiet ke goc.
      - "full9": dung ca 9 to hop tuong quan cap (XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ) thay vi chi ZZ -
        van la quan sat luong tu doc ra tu rho (0 tham so o buoc nay), chi lam giau dau vao cho
        lop Linear phia sau. Xem PLAN.md muc 17 (huong tunning cho full_quantum).
    """

    def __init__(self, num_classes: int = 10, final_qubits: int = 4,
                 pair_correlators: str = "zz", extra_feature_dim: int = 0):
        super().__init__()
        if pair_correlators not in ("zz", "full9"):
            raise ValueError(f"Unsupported pair_correlators: {pair_correlators}")
        self.num_classes = num_classes
        self.final_qubits = final_qubits
        self.pair_correlators = pair_correlators
        self.extra_feature_dim = extra_feature_dim
        n_pairs = len(list(combinations(range(final_qubits), 2)))
        pair_feature_count = n_pairs if pair_correlators == "zz" else n_pairs * 9
        feature_dim = 3 * final_qubits + pair_feature_count + extra_feature_dim
        self.linear = nn.Linear(feature_dim, num_classes)

    def _factorized_features(self, rho: torch.Tensor) -> torch.Tensor:
        if self.pair_correlators != "zz":
            raise NotImplementedError(
                "_factorized_features (rho.ndim==4, per-qubit product state truoc khi ghep "
                "joint) chi ho tro pair_correlators='zz'. Duong dan thuc te dang dung "
                "(_joint_features, rho.ndim==3) ho tro ca 'full9'."
            )
        x = 2.0 * rho[..., 0, 1].real
        y = -2.0 * rho[..., 0, 1].imag
        z = rho[..., 0, 0].real - rho[..., 1, 1].real
        feats = [x, y, z]
        zz = []
        for i, j in combinations(range(self.final_qubits), 2):
            zz.append(z[:, i] * z[:, j])
        if zz:
            feats.append(torch.stack(zz, dim=-1))
        return torch.cat(feats, dim=-1)

    def _joint_features(self, rho: torch.Tensor) -> torch.Tensor:
        dim = 2**self.final_qubits
        if rho.ndim != 3 or rho.shape[-2:] != (dim, dim):
            raise ValueError(f"Expected joint density matrices with shape (batch, {dim}, {dim})")
        x = [_joint_pauli_expectation(rho, {q: "x"}, self.final_qubits) for q in range(self.final_qubits)]
        y = [_joint_pauli_expectation(rho, {q: "y"}, self.final_qubits) for q in range(self.final_qubits)]
        z = [_joint_pauli_expectation(rho, {q: "z"}, self.final_qubits) for q in range(self.final_qubits)]
        pairs = []
        if self.pair_correlators == "zz":
            for i, j in combinations(range(self.final_qubits), 2):
                pairs.append(_joint_pauli_expectation(rho, {i: "z", j: "z"}, self.final_qubits))
        else:  # "full9"
            for i, j in combinations(range(self.final_qubits), 2):
                for axis_i in _PAIR_AXES:
                    for axis_j in _PAIR_AXES:
                        pairs.append(_joint_pauli_expectation(rho, {i: axis_i, j: axis_j}, self.final_qubits))
        return torch.stack([*x, *y, *z, *pairs], dim=-1)

    def features(self, rho: torch.Tensor) -> torch.Tensor:
        if rho.ndim == 4:
            return self._factorized_features(rho)
        return self._joint_features(rho)

    def forward(self, rho: torch.Tensor, extra_features: torch.Tensor | None = None) -> torch.Tensor:
        features = self.features(rho)
        if self.extra_feature_dim:
            if extra_features is None:
                raise ValueError("This readout requires multiscale side features")
            if extra_features.shape[:-1] != features.shape[:-1]:
                raise ValueError("Multiscale feature batch shape does not match the readout")
            if extra_features.shape[-1] != self.extra_feature_dim:
                raise ValueError(
                    f"Expected {self.extra_feature_dim} side features, got {extra_features.shape[-1]}"
                )
            features = torch.cat((features, extra_features), dim=-1)
        elif extra_features is not None:
            raise ValueError("This readout was created without multiscale side features")
        return self.linear(features)

    def count_parameters(self) -> Dict[str, int]:
        return {"classical_readout": sum(p.numel() for p in self.parameters() if p.requires_grad)}


class PauliMLPReadout(PauliObservableReadout):
    """Pauli feature readout with a small nonlinear classical MLP head.

    super().__init__ still builds (and draws RNG for) the original Linear(feature_dim, num_classes),
    and the MLP is initialised inside a forked CPU RNG seeded from the current RNG state. The global
    RNG stream after this readout is therefore exactly the one PauliObservableReadout leaves behind,
    so with the same seed every branch's TN core starts from the same weights as the original model.
    """

    def __init__(self, num_classes: int = 10, final_qubits: int = 4,
                 pair_correlators: str = "zz", hidden_dim: int = 256,
                 dropout: float = 0.1, extra_feature_dim: int = 0):
        super().__init__(num_classes=num_classes, final_qubits=final_qubits,
                         pair_correlators=pair_correlators,
                         extra_feature_dim=extra_feature_dim)
        feature_dim = self.linear.in_features
        mlp_seed = int(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16], 16)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(mlp_seed)
            self.linear = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
