"""Kiểm tra qcnn_qconv4x3. Chạy một lần trước khi train: python .\\test_model.py
1. số tham số; 2. mạch quanvolution so với ma trận đầy đủ 512x512; 3. encoder khởi đầu là ánh xạ đồng nhất và không
rút số ngẫu nhiên; 4. nếu ../full_quantum_16x16_level32_qconv nằm cạnh folder này: model GIỐNG HỆT FusionQCNN gốc
(cùng trọng số từ cùng seed, cùng đầu ra, cùng gradient, và state_dict gốc nạp được với strict=True);
5. forward/backward trên GPU; 6. tốc độ (chỉ để tham khảo).
"""
import cmath
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from qcnn import QCNNQconv, build_model
from qcnn.encoder_quanvolution import QuantumConvStage, QuantumFront, grid_edges, rotation_matrices

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make(front="qconv4x3", seed=0, **options):
    torch.manual_seed(seed)
    model = QCNNQconv(front=front, **options)
    return model, torch.rand(5)


def test_counts():
    print("[-] Parameter counts")
    cases = (("qconv4x3", "mlp", (4824, 492, 324, 11948, 17264)),
             ("qconv4x3", "linear3", (4824, 492, 324, 11190, 16506)),
             ("qconv4x3", "linear-shared", (4824, 492, 324, 3730, 9046)),
             ("qconv2x2", "mlp", (4824, 220, 108, 11948, 16992)),
             ("none", "mlp", (4824, 0, 0, 11948, 16772)))
    for front, head, expected in cases:
        c = make(front, head=head)[0].count_parameters()
        got = (c["quantum"], c["front"], c["front_quantum"], c["head"], c["total"])
        print(f"    {front:9s} head {head:13s} {c}")
        assert got == expected, (front, head, got, expected)
    share = make()[0].count_parameters()["quantum_share"]
    assert abs(share - 5148 / 17264) < 1e-9, share
    print("    [PASS] qconv4x3 + mlp head: 4,824 + 492 (angles 324) + 11,948 = 17,264, quantum 29.8%")


def _dense_window_features(stage, theta, phi):
    """Đối chiếu cho một pixel: theta, phi (9,) của ô 3x3 -> (27,) X, Y, Z của mọi qubit, tính bằng ma trận đầy đủ."""
    identity = torch.eye(2, dtype=torch.complex128)
    paulis = [torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
              torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
              torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)]

    def kron_all(matrices):
        out = torch.eye(1, dtype=torch.complex128)
        for matrix in matrices:
            out = torch.kron(out, matrix)
        return out

    state = torch.ones(1, dtype=torch.complex128)
    for q in range(9):
        t, p = float(theta[q]), float(phi[q])
        one = torch.tensor([math.cos(math.pi * t / 2), math.sin(math.pi * t / 2) * cmath.exp(1j * math.pi * p)],
                           dtype=torch.complex128)
        state = torch.kron(state, one)
    gates = rotation_matrices(stage.angles.detach().double())
    eye = torch.eye(512, dtype=torch.complex128)
    cz = eye.clone()
    for a, b in grid_edges():
        one_a = (eye - kron_all([paulis[2] if q == a else identity for q in range(9)])) / 2
        one_b = (eye - kron_all([paulis[2] if q == b else identity for q in range(9)])) / 2
        cz = cz @ (eye - 2 * one_a @ one_b)
    for layer in range(gates.shape[0]):
        state = cz @ (kron_all([gates[layer, q] for q in range(9)]) @ state)
    values = []
    for q in range(9):
        for pauli in paulis:
            operator = kron_all([pauli if r == q else identity for r in range(9)])
            values.append((state.conj() @ operator @ state).real)
    return torch.stack(values)


def test_quanvolution_circuit():
    print("[-] Quanvolution circuit vs dense 512x512 matrices (inside the image and at the border)")
    torch.manual_seed(1)
    stage = QuantumConvStage(layers=4, frozen=False)
    height = width = 4
    theta = torch.rand(2, height * width) * 1.5 - 0.2
    phi = torch.rand(2, height * width) * 2 - 1
    fast = stage.features(theta, phi, height, width)
    padded_t = F.pad(theta.reshape(2, height, width), (1, 1, 1, 1))
    padded_p = F.pad(phi.reshape(2, height, width), (1, 1, 1, 1))
    worst = 0.0
    for b, pixel in ((0, 0), (0, 5), (1, 10), (1, 15)):
        row, column = divmod(pixel, width)
        reference = _dense_window_features(stage, padded_t[b, row:row + 3, column:column + 3].flatten(),
                                           padded_p[b, row:row + 3, column:column + 3].flatten())
        worst = max(worst, (fast[b, pixel].double() - reference).abs().max().item())
    print(f"    max difference {worst:.2e}")
    assert worst < 1e-4, worst
    print("    [PASS]")


def test_identity_start():
    print("[-] The encoder starts as the identity and draws no random numbers")
    images = torch.rand(3, 1, 28, 28, generator=torch.Generator().manual_seed(2))
    base, base_next = make("none")
    model, model_next = make("qconv4x3")
    assert torch.equal(base_next, model_next), "building the encoder changed the random state"
    state = model.state_dict()
    for key, value in base.state_dict().items():
        assert torch.equal(value, state[key]), key
    base.eval()
    model.eval()
    with torch.no_grad():
        difference = (model(images) - base(images)).abs().max().item()
    print(f"    max |output - output without encoder| = {difference:.2e}")
    assert difference < 1e-5, difference
    print("    [PASS]")


def test_same_as_original():
    original_root = ROOT.parent / "full_quantum_16x16_level32_qconv"
    if not (original_root / "models" / "fusion_model.py").is_file():
        print(f"[-] SKIP comparison with the original code: {original_root} not found (put this folder next to it)")
        return
    print(f"[-] Identical to the original FusionQCNN in {original_root}")
    sys.path.insert(0, str(original_root))
    from models import FusionQCNN  # package gốc
    for head in ("mlp",):
        torch.manual_seed(0)
        original = FusionQCNN(num_classes=10, side_level=32, projection=10, hidden=16, dropout=0.0,
                              front="qconv4x3")
        original_next = torch.rand(5)
        new, new_next = make("qconv4x3", head=head, projection=10, hidden=16, dropout=0.0)
        assert torch.equal(original_next, new_next), "different random draws while building"
        a, b = original.state_dict(), new.state_dict()
        assert list(a) == list(b), (set(a) ^ set(b))
        for key in a:
            assert torch.equal(a[key], b[key]), key
        new.load_state_dict(original.state_dict(), strict=True)
        # Cùng đầu ra, và cùng gradient sau khi đặt readout của encoder khác 0.
        with torch.no_grad():
            for m in (original, new):
                for stage in m.front.stages:
                    stage.readout.weight.copy_(torch.linspace(-0.2, 0.2, 54).reshape(2, 27))
        images = torch.rand(4, 1, 28, 28, generator=torch.Generator().manual_seed(5))
        labels = torch.tensor([1, 3, 5, 7])
        original.eval()
        new.eval()
        with torch.no_grad():
            difference = (original(images) - new(images)).abs().max().item()
        assert difference <= 1e-6, difference
        original.train()
        new.train()
        F.cross_entropy(original(images), labels).backward()
        F.cross_entropy(new(images), labels).backward()
        grads = dict(original.named_parameters())
        worst = max((p.grad - grads[name].grad).abs().max().item() for name, p in new.named_parameters())
        assert worst <= 1e-6, worst
    print("    [PASS] same weights from seed 0, identical outputs and gradients, original state_dict loads (strict)")


def test_forward_backward():
    print(f"[-] Forward and backward on {DEVICE}, train mode (checkpointing)")
    for head in ("mlp", "linear3"):
        model = make("qconv4x3", head=head)[0].to(DEVICE).train()
        with torch.no_grad():
            for stage in model.front.stages:
                stage.readout.weight.normal_(0, 0.1)
        images = torch.rand(2, 1, 28, 28, device=DEVICE)
        F.cross_entropy(model(images), torch.tensor([3, 7], device=DEVICE)).backward()
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, missing
        norms = [s.angles.grad.norm().item() for s in model.front.stages]
        assert all(n > 0 for n in norms), norms
        print(f"    head {head}: |grad| of the encoder angles per stage {', '.join(f'{n:.2e}' for n in norms)}")
    assert isinstance(build_model({"front": "qconv4x3", "projection": 10}), QCNNQconv)
    print("    [PASS]")


def test_speed():
    print(f"[-] Speed, batch 64, Adam, on {DEVICE} (information only)")
    model = make("qconv4x3")[0].to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters())
    images = torch.rand(64, 1, 28, 28, device=DEVICE)
    labels = torch.randint(0, 10, (64,), device=DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    for repeat in range(3):
        if repeat == 1:
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(images), labels).backward()
        optimizer.step()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    seconds = (time.perf_counter() - started) / 2
    memory = torch.cuda.max_memory_allocated(DEVICE) / 2 ** 30 if DEVICE.type == "cuda" else 0.0
    print(f"    qconv4x3: {seconds:.3f} s/batch, {memory:.2f} GB; about {seconds * 782 / 60:.0f} min per epoch of training")


if __name__ == "__main__":
    print("=" * 60)
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
          + (f", {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    print("=" * 60)
    test_counts()
    test_quanvolution_circuit()
    test_identity_start()
    test_same_as_original()
    test_forward_backward()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    test_speed()
