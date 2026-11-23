"""Kiểm tra fullquantum_16x16_frontQuan4x3_Moe. Chạy MỘT lần trên máy train trước khi train, gửi toàn bộ output:
    python .\\test_model.py
Phải kết thúc bằng "ALL TESTS PASSED". Cuối cùng in tốc độ (giây / batch 16, forward + backward) từng kiểu router.

1. Số tham số: router none 16.506 (= linhead L1), quantum 17.149 (mạch 640), linear 16.605, mlp 18 17.157,
   quantum rot 16.557; head shared 9.046, mlp 23.730, fusion 17.264 (= 2c), bloch8 6.709, obs10 6.019,
   obs10 + readout direct 5.851 (quantum 98,9%). Vị trí quan sát của obs10; encoder readout direct.
2. Lúc khởi tạo mọi router cho đầu ra giống hệt router none (g đều), và phần còn lại cùng giá trị khởi tạo.
3. Nếu folder ../full_quantum_16x16_level32_qconv_linhead có mặt: router none giống hệt FusionQCNN --head linear3 /
   linear-shared / mlp (run 2c, với --expert-head fusion)
   (cùng seed: cùng state_dict, cùng đầu ra) và nạp được state_dict của nó.
4. Router lượng tử so với mô phỏng ma trận đầy đủ (complex128), Q = 3 và 4, hai kiểu ansatz.
5. Các chế độ trộn (soft, uniform, top1, tree<i>), mix probs (tổng xác suất 1), loss phụ.
6. Forward / backward chế độ train: mọi tham số có gradient; sau khi hệ số đọc ra khác 0, mạch router có gradient.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from qcnn import MoEQCNN, build_model
from qcnn.encoding.quanvolution import QuantumConvStage
from qcnn.encoding.quanvolution_direct import DirectQuantumFront, bloch_to_angles
from qcnn.layers.density import bloch, density_from_bloch
from qcnn.layers.gates import rotation_matrices
from qcnn.layers.pauli import PauliFeatures
from qcnn.moe.heads import observable_index, same_axis_pair_indices
from qcnn.moe.model import mix_logits
from qcnn.moe.router import QuantumRouter, router_parameter_count

LINHEAD = ROOT.parent / "full_quantum_16x16_level32_qconv_linhead"
IMAGES = torch.rand(4, 1, 28, 28, generator=torch.Generator().manual_seed(1))


def build(seed: int = 0, **kwargs) -> MoEQCNN:
    torch.manual_seed(seed)
    return MoEQCNN(**kwargs)


def test_parameters():
    print("[-] Parameter counts")
    # (kwargs, total, head, front, front angles, router, router circuit)
    cases = [
        (dict(router="none"), 16506, 11190, 492, 324, 0, 0),
        (dict(router="quantum"), 17149, 11190, 492, 324, 643, 640),
        (dict(router="linear"), 16605, 11190, 492, 324, 99, 0),
        (dict(router="mlp", router_hidden=18), 17157, 11190, 492, 324, 651, 0),
        (dict(router="quantum", router_ansatz="rot"), 16557, 11190, 492, 324, 51, 48),
        (dict(router="none", expert_head="shared"), 9046, 3730, 492, 324, 0, 0),
        (dict(router="none", expert_head="mlp"), 23730, 18414, 492, 324, 0, 0),
        (dict(router="none", expert_head="fusion"), 17264, 11948, 492, 324, 0, 0),
        (dict(router="quantum", expert_head="bloch8"), 6709, 750, 492, 324, 643, 640),
        (dict(router="quantum", expert_head="obs10"), 6019, 60, 492, 324, 643, 640),
        (dict(router="quantum", expert_head="nocorr"), 9589, 3630, 492, 324, 643, 640),
        (dict(router="none", expert_head="nocorr"), 8946, 3630, 492, 324, 0, 0),
        (dict(router="quantum", expert_head="nocorr-z"), 7669, 1710, 492, 324, 643, 640),
        (dict(router="quantum", expert_head="nocorr-shared"), 7169, 1210, 492, 324, 643, 640),
        (dict(router="quantum", expert_head="nocorr-lowrank", head_rank=4), 7549, 1590, 492, 324, 643, 640),
        (dict(router="quantum", expert_head="nocorr-lowrank", head_rank=2), 6769, 810, 492, 324, 643, 640),
        (dict(router="quantum", expert_head="samebasis"), 12109, 6150, 492, 324, 643, 640),
        (dict(router="quantum", expert_head="zz"), 10429, 4470, 492, 324, 643, 640),
        (dict(router="quantum", expert_head="obs10", front_readout="direct"), 5851, 60, 324, 324, 643, 640),
        (dict(router="none", front_readout="direct"), 16338, 11190, 324, 324, 0, 0),
    ]
    for kwargs, total, head, front, angles, router, circuit in cases:
        counts = build(**kwargs).count_parameters()
        print(f"    {kwargs}: total {counts['total']}, quantum share {counts['quantum_share'] * 100:.2f}%")
        assert counts["quantum"] == 4824 and counts["front"] == front and counts["front_quantum"] == angles, counts
        assert counts["head"] == head, counts
        assert counts["router"] == router and counts["router_quantum"] == circuit, counts
        assert counts["total"] == total, counts
        kind = kwargs["router"]
        expected = router_parameter_count(kind, 3, (4, 4), 8, 1, kwargs.get("router_ansatz", "se"),
                                          kwargs.get("router_hidden", 18))
        assert expected == (circuit, router - circuit), (expected, circuit, router)
    share = build(router="quantum").count_parameters()["quantum_share"]
    assert abs(share - 5788 / 17149) < 1e-9
    share = build(router="quantum", expert_head="obs10", front_readout="direct").count_parameters()["quantum_share"]
    assert abs(share - 5788 / 5851) < 1e-9
    print("    [PASS] quantum share: default 33.75%, obs10 + direct 98.92%")
    # Đếm đúng (2026-11-20): mỗi cây 5*2*16 + 4*(8*4 + 8*16) + 8 = 808; router 2*(8*4 + 8*16) = 320.
    for kwargs, total_free, total_nophase in ((dict(router="quantum"), 14429, 14175),
                                              (dict(router="quantum", expert_head="samebasis"), 9389, 9135),
                                              (dict(router="quantum", expert_head="nocorr"), 6869, 6615),
                                              (dict(router="none"), 14106, 13884),
                                              (dict(router="mlp", router_hidden=18), 14757, 14535)):
        counts = build(**kwargs).count_parameters()
        assert counts["quantum_free"] == 3 * 808 and counts["quantum_free_nophase"] == 3 * 734, counts
        assert counts["total_free"] == total_free and counts["total_free_nophase"] == total_nophase, (kwargs, counts)
        print(f"    {kwargs}: FREE total {counts['total_free']}, quantum share "
              f"{counts['quantum_share_free'] * 100:.2f}% (no phase {counts['total_free_nophase']})")
    counts = build(router="quantum").count_parameters()
    assert counts["router_quantum_free"] == 320 and abs(counts["quantum_share_free"] - 3068 / 14429) < 1e-9
    print("    [PASS] free parameters: tree 808 (734), router circuit 320 (288), M1 14,429 = 21.26% quantum")
    # --share-final: 3 cây dùng chung khối 8 qubit cuối (FREE 640, cũ 1.280, bỏ pha 576 mỗi khối).
    shared = build(router="quantum", expert_head="samebasis", share_final=True)
    counts = shared.count_parameters()
    assert shared.cores[1].merge_stage is shared.cores[0].merge_stage is shared.cores[2].merge_stage
    assert counts["quantum"] == 4824 - 2 * 1280 and counts["total"] == 12109 - 2 * 1280, counts
    assert counts["total_free"] == 8109 and counts["total_free_nophase"] == 7983, counts
    plain = build(router="quantum", expert_head="samebasis")
    for i in (0, 1, 2):  # phần conv-pool giống hệt bản không chia sẻ (cùng seed)
        for (name, p), (_, q) in zip(shared.cores[i].named_parameters(), plain.cores[i].named_parameters()):
            if not name.startswith("merge_stage"):
                assert torch.equal(p, q), name
    print(f"    [PASS] share-final: samebasis FREE {counts['total_free']} (no phase {counts['total_free_nophase']}), "
          f"quantum share {counts['quantum_share_free'] * 100:.1f}%")
    # --share-gates: khối 8 qubit mỗi lớp 1 unitary 1 qubit + 1 unitary 2 qubit dùng chung (FREE 4 x 20 = 80).
    for kwargs, old, free, nophase in ((dict(share_gates=True), 8749, 7709, 7623),
                                       (dict(share_gates=True, share_final=True), 8429, 7549, 7479)):
        model = build(router="quantum", expert_head="samebasis", **kwargs)
        counts = model.count_parameters()
        stage = model.cores[0].merge_stage
        assert all(gate is stage.rotations[l][0] for l in range(4) for gate in stage.rotations[l])
        assert all(gate is stage.entanglers[l][0] for l in range(4) for gate in stage.entanglers[l])
        assert counts["total"] == old and counts["total_free"] == free and counts["total_free_nophase"] == nophase, \
            (kwargs, counts)
        print(f"    [PASS] {kwargs}: old {old}, FREE {free} (no phase {nophase}), "
              f"quantum share {counts['quantum_share_free'] * 100:.1f}%")
    # Cùng seed: mọi tham số không thuộc khối 8 qubit giống hệt bản không chia sẻ.
    shared, plain = build(router="quantum", expert_head="samebasis", share_gates=True), \
        build(router="quantum", expert_head="samebasis")
    plain_params = dict(plain.named_parameters())
    for name, p in shared.named_parameters():
        if "merge_stage" not in name:
            assert torch.equal(p, plain_params[name]), name
    print("    [PASS] share-gates keeps every other parameter identical (same seed)")


def test_observables():
    print("[-] obs10 observable positions in the PauliFeatures vector")
    assert observable_index("X0") == 0 and observable_index("Y1") == 9 and observable_index("Z3") == 19
    assert observable_index("X0Z5") == 24 + 9 * 4 + 2 == observable_index("Z5X0")
    assert observable_index("Z6Z7") == 24 + 9 * 27 + 8
    for bad in ("Z8", "Q1", "Z1Z1", "Z"):
        try:
            observable_index(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    # samebasis / zz: tương quan cùng trục của cặp p ở vị trí 24 + 9p + {0, 4, 8} (XX, YY, ZZ).
    same = same_axis_pair_indices("XYZ")
    assert same == [24 + 9 * p + k for p in range(28) for k in (0, 4, 8)], same[:6]
    assert same_axis_pair_indices("Z") == [24 + 9 * p + 8 for p in range(28)]
    head = build(router="none", expert_head="samebasis").projections[0]
    assert head.index.numel() == 204 and head.index[:24].tolist() == list(range(24))
    assert head.index[24:108].tolist() == same and head.index[108:].tolist() == list(range(276, 372))
    print("    [PASS] samebasis 204 = 24 + 84 + 96, zz 148 = 24 + 28 + 96 feature positions")
    # Check against PauliFeatures on a random product state of 8 qubits: <Z3> and <X0 Z5> = <X0><Z5>.
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(8, 2, 2, dtype=torch.complex64, generator=generator)
    rhos = torch.stack([s @ s.conj().T / torch.trace(s @ s.conj().T) for s in states])
    joint = rhos[0]
    for r in rhos[1:]:
        joint = torch.kron(joint, r)
    features = PauliFeatures(8)(joint[None])[0]
    single = bloch(rhos[None])[0]  # X0..X7 Y0..Y7 Z0..Z7
    assert torch.allclose(features[:24], single, atol=1e-5)
    assert abs(features[observable_index("Z3")] - single[19]) < 1e-5
    assert abs(features[observable_index("X0Z5")] - single[0] * single[21]) < 1e-5
    assert abs(features[observable_index("Y2X6")] - single[10] * single[6]) < 1e-5
    print("    [PASS]")


def test_direct_front():
    print("[-] --front-readout direct: centre qubit of the qconv circuit, no classical parameters")
    torch.manual_seed(0)
    direct = DirectQuantumFront("qconv4x3", 16, 16).eval()
    values = torch.rand(2, 256, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        got = direct(values)
        theta, phi = values, torch.zeros_like(values)
        for index, stage in enumerate(direct.stages):
            features = QuantumConvStage.features(stage, theta, phi, 16, 16)
            vector = features[..., 12:15]
            if index + 1 < len(direct.stages):
                theta, phi = bloch_to_angles(vector)
    assert torch.allclose(got, vector), (got - vector).abs().max()
    assert got.norm(dim=-1).max() <= 1 + 1e-5
    # bloch_to_angles gives back the direction: RZ(pi phi) RY(pi theta)|0> has Bloch (sin t cos p, sin t sin p, cos t).
    unit = got / got.norm(dim=-1, keepdim=True)
    t, p = bloch_to_angles(got)
    rebuilt = torch.stack((torch.sin(torch.pi * t) * torch.cos(torch.pi * p), torch.sin(torch.pi * t) *
                           torch.sin(torch.pi * p), torch.cos(torch.pi * t)), dim=-1)
    assert torch.allclose(rebuilt, unit, atol=1e-4), (rebuilt - unit).abs().max()
    # The density matrix fed to the trees has this Bloch vector.
    rho = density_from_bloch(got)
    assert torch.allclose(bloch(rho.reshape(-1, 1, 2, 2)).reshape(-1, 3), got.reshape(-1, 3), atol=1e-6)
    print("    [PASS]")


def test_same_start():
    print("[-] Every router starts with uniform gates and the same output as router none")
    base = build(router="none").eval()
    with torch.no_grad():
        expected = base(IMAGES)
    base_state = base.state_dict()
    for kwargs in (dict(router="quantum"), dict(router="quantum", router_ansatz="rot"), dict(router="linear"),
                   dict(router="mlp"), dict(router="quantum", router_source="pixels")):
        model = build(**kwargs).eval()
        state = model.state_dict()
        for key, value in base_state.items():
            assert torch.equal(value, state[key]), f"{kwargs}: {key} differs"
        with torch.no_grad():
            got = model(IMAGES)
        gates = model.last["gates"]
        assert torch.allclose(gates, torch.full_like(gates, 1 / 3)), gates
        difference = (got - expected).abs().max().item()
        print(f"    {kwargs}: max |output - router none| = {difference:.2e}")
        assert difference < 1e-6, difference
    print("    [PASS]")


def load_linhead():
    spec = importlib.util.spec_from_file_location("linhead_models", LINHEAD / "models" / "__init__.py",
                                                  submodule_search_locations=[str(LINHEAD / "models")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["linhead_models"] = module
    spec.loader.exec_module(module)
    return module


def test_linhead_equivalence():
    print("[-] router none = linhead FusionQCNN --head linear3 / linear-shared / mlp (2c) (same seed)")
    if not (LINHEAD / "models" / "fusion_model.py").is_file():
        print(f"    [SKIP] {LINHEAD} not found (copy it next to this folder to run this check)")
        return
    linhead = load_linhead()
    for head, expert_head in (("linear3", "linear"), ("linear-shared", "shared"), ("mlp", "fusion")):
        torch.manual_seed(0)
        old = linhead.FusionQCNN(num_classes=10, side_level=32, projection=10, dropout=0.0, front="qconv4x3",
                                 head=head).eval()
        new = build(router="none", expert_head=expert_head).eval()
        old_state, new_state = old.state_dict(), new.state_dict()
        assert old_state.keys() == new_state.keys(), set(old_state) ^ set(new_state)
        for key in old_state:
            assert torch.equal(old_state[key], new_state[key]), key
        with torch.no_grad():
            difference = (old(IMAGES) - new(IMAGES)).abs().max().item()
        print(f"    {head}: same state_dict; max |output difference| = {difference:.2e}")
        assert difference < 1e-5, difference
        # A linhead config read by build_model (as analyze_routing.py does) loads the linhead weights.
        model = build_model({"model": "qcnn", "front": "qconv4x3", "head": head, "side_level": 32, "projection": 10,
                             "orders": ["row", "column", "morton"], "dropout": 0.0})
        model.load_state_dict(old_state)
    print("    [PASS]")


# ---- mô phỏng ma trận đầy đủ cho router lượng tử ----
def dense_one(gate: torch.Tensor, qubit: int, qubits: int) -> torch.Tensor:
    out = torch.ones(1, 1, dtype=torch.complex128)
    for q in range(qubits):
        out = torch.kron(out, gate.to(torch.complex128) if q == qubit else torch.eye(2, dtype=torch.complex128))
    return out


def dense_two(gate: torch.Tensor, first: int, second: int, qubits: int) -> torch.Tensor:
    dim = 2 ** qubits
    matrix = torch.zeros(dim, dim, dtype=torch.complex128)
    sf, ss = qubits - 1 - first, qubits - 1 - second
    for column in range(dim):
        a = 2 * ((column >> sf) & 1) + ((column >> ss) & 1)
        rest = column & ~(1 << sf) & ~(1 << ss)
        for out in range(4):
            row = rest | ((out >> 1) << sf) | ((out & 1) << ss)
            matrix[row, column] += complex(gate[out, a])
    return matrix


def dense_router(router: QuantumRouter, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    q = router.qubits
    dim = 2 ** q
    results = []
    with torch.no_grad():
        for sample in range(theta.shape[0]):
            state = torch.zeros(dim, dtype=torch.complex128)
            state[0] = 1
            index = 0
            for round_ in range(router.rounds):
                for qubit in range(q):
                    t, p = float(theta[sample, round_ * q + qubit]), float(phi[sample, round_ * q + qubit])
                    ry = torch.tensor([[math.cos(math.pi * t / 2), -math.sin(math.pi * t / 2)],
                                       [math.sin(math.pi * t / 2), math.cos(math.pi * t / 2)]], dtype=torch.complex128)
                    rz = torch.diag(torch.tensor([1, complex(math.cos(math.pi * p), math.sin(math.pi * p))]))
                    state = dense_one(rz @ ry, qubit, q) @ state
                for _ in range(router.layers):
                    if router.ansatz == "se":
                        for qubit in range(q):
                            state = dense_one(router.rotations[index][qubit](), qubit, q) @ state
                        reach = (index % (q - 1)) + 1
                        for first in range(q):
                            gate = router.entanglers[index][first]()
                            state = dense_two(gate, first, (first + reach) % q, q) @ state
                    else:
                        gates = rotation_matrices(router.angles[index])
                        for qubit in range(q):
                            state = dense_one(gates[qubit], qubit, q) @ state
                        for first in range(q):
                            cz = torch.diag(torch.tensor([-1.0 if ((k >> (q - 1 - first)) & 1 and
                                                                   (k >> (q - 1 - (first + 1) % q)) & 1) else 1.0
                                                          for k in range(dim)], dtype=torch.complex128))
                            state = cz @ state
                    index += 1
            probabilities = state.abs() ** 2
            z = [sum(float(probabilities[k]) * (1 - 2 * ((k >> (q - 1 - qubit)) & 1)) for k in range(dim))
                 for qubit in router.readout_qubits]
            results.append(z)
    return torch.tensor(results)


def test_router_dense():
    print("[-] Quantum router <Z> against a dense matrix simulation")
    generator = torch.Generator().manual_seed(3)
    for qubits, grid, ansatz in ((3, (2, 3), "se"), (3, (2, 3), "rot"), (4, (2, 4), "se"), (4, (2, 4), "rot")):
        torch.manual_seed(5)
        router = QuantumRouter(experts=3, grid=grid, qubits=qubits, layers=2, ansatz=ansatz, init_scale=0.7)
        cells = grid[0] * grid[1]
        theta = torch.rand(3, cells, generator=generator) * 1.5 - 0.2
        phi = torch.rand(3, cells, generator=generator) - 0.5
        with torch.no_grad():
            got = router.expectations(theta, phi).double()
        expected = dense_router(router, theta, phi)
        difference = (got - expected).abs().max().item()
        print(f"    Q={qubits} grid={grid} ansatz={ansatz} layers=2: max |difference| = {difference:.2e}")
        assert difference < 1e-5, difference
        assert expected.abs().max() > 1e-3, "trivial test"
    print("    [PASS]")


def test_modes_and_losses():
    print("[-] Mixing modes and auxiliary losses")
    logits = torch.randn(5, 3, 10, generator=torch.Generator().manual_seed(2))
    gates = torch.softmax(torch.randn(5, 3, generator=torch.Generator().manual_seed(4)), dim=1)
    for mix in ("logits", "probs"):
        model = build(router="quantum", mix=mix)
        uniform = model.combine(logits, gates, "uniform")
        assert torch.allclose(uniform, mix_logits(logits, torch.full_like(gates, 1 / 3), mix))
        top1 = model.combine(logits, gates, "top1")
        chosen = gates.argmax(dim=1)
        assert torch.equal(top1.argmax(dim=1), logits[torch.arange(5), chosen].argmax(dim=1))
        for i in range(3):
            assert torch.equal(model.combine(logits, gates, f"tree{i}").argmax(dim=1), logits[:, i].argmax(dim=1))
        if mix == "probs":
            total = model.combine(logits, gates, "soft").exp().sum(dim=1)
            assert torch.allclose(total, torch.ones(5), atol=1e-5), total
        else:
            assert torch.allclose(model.combine(logits, gates, "soft"), (gates[:, :, None] * logits).sum(1))
    model = build(router="quantum")
    model.last = {"gates": torch.full((6, 3), 1 / 3)}
    assert abs(float(model.aux_loss(1.0, 0.0))) < 1e-6
    assert abs(float(model.aux_loss(0.0, 1.0)) - 1.0) < 1e-5
    model.last = {"gates": F.one_hot(torch.zeros(6, dtype=torch.long), 3).float()}
    assert abs(float(model.aux_loss(1.0, 0.0)) - 2.0) < 1e-5
    assert abs(float(model.aux_loss(0.0, 1.0))) < 1e-5
    print("    [PASS] balance 0 when uniform, E-1 = 2 when all on one tree; entropy 1 -> 0")


def test_backward():
    print("[-] Forward and backward in train mode")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")
    for kwargs in (dict(router="quantum"), dict(router="mlp"), dict(router="quantum", mix="probs", gate_noise=0.5),
                   dict(router="quantum", expert_head="bloch8"),
                   dict(router="quantum", expert_head="obs10", front_readout="direct"),
                   dict(router="none", expert_head="fusion")):
        model = build(**kwargs).to(device).train()
        labels = torch.tensor([3, 7], device=device)
        images = torch.rand(2, 1, 28, 28, device=device)
        loss = F.cross_entropy(model(images), labels) + model.aux_loss(0.01, 0.01)
        loss.backward()
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, missing
        bad = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not bad, f"non-finite gradients: {bad}"
        if kwargs["router"] == "quantum":
            assert model.router.scale.grad.abs().sum() > 0, "router scale gets no gradient"
            with torch.no_grad():
                model.router.scale.fill_(1.0)
            model.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            circuit = sum(p.grad.abs().sum().item() for n, p in model.router.named_parameters() if n != "scale")
            print(f"    {kwargs}: loss {loss.item():.4f}, router circuit |grad| {circuit:.2e} (scale 1)")
            assert circuit > 0
        else:
            print(f"    {kwargs}: loss {loss.item():.4f}")
    print("    [PASS]")


def timing():
    print("[-] Speed (seconds per batch of 16, forward + backward, train mode)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for kwargs in (dict(router="none"), dict(router="quantum"), dict(router="mlp"),
                   dict(router="quantum", expert_head="obs10", front_readout="direct")):
        model = build(**kwargs).to(device).train()
        images = torch.rand(16, 1, 28, 28, device=device)
        labels = torch.randint(0, 10, (16,), device=device)
        times = []
        for _ in range(3):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            F.cross_entropy(model(images), labels).backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)
        print(f"    {kwargs}: {min(times[1:]):.3f} s")


if __name__ == "__main__":
    test_parameters()
    test_observables()
    test_direct_front()
    test_same_start()
    test_linhead_equivalence()
    test_router_dense()
    test_modes_and_losses()
    test_backward()
    print("ALL TESTS PASSED")
    timing()
