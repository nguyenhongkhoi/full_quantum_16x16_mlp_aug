"""Checks for full_quantum_16x16_level32_qconv_linhead (copy of level32_qconv + --head linear3 / linear-shared):
parameter counts, the front end starts as the identity, gradients; the quantum encoder against dense 512x512
matrices; the linear heads (counts, logits = mean of the per-tree Linear outputs, gradients).
The checks inherited from level32_qconv build the MLP head (the default of FusionQCNN and build_model).
Run once on the training machine before training: python .\\test_model.py
"""
import cmath
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import EncoderCNN, EncoderLinear, FusionQCNN, build_model
from models.front import QuantumConvStage, QuantumFront, grid_edges, rotation_matrices

EXPECTED = {"none": 0, "conv1": 10, "cnn4": 77, "cnn8": 153, "deep12": 1537, "deep12p": 1646}
HEAD = {12: 14294, 10: 11948}


def build(front: str, seed: int = 0, projection: int = 12) -> tuple[FusionQCNN, torch.Tensor]:
    """The model, and the next random numbers after building it (to check the front end draws none)."""
    torch.manual_seed(seed)
    model = FusionQCNN(num_classes=10, side_level=32, projection=projection, dropout=0.2, front=front)
    return model, torch.rand(5)


def test_parameters():
    print("[-] Parameter counts")
    for projection, head in HEAD.items():
        for front, extra in EXPECTED.items():
            counts = build(front, projection=projection)[0].count_parameters()
            print(f"    projection {projection} {front:8s} {counts}")
            assert counts["quantum"] == 4824, counts
            assert counts["head"] == head, counts
            assert counts["front"] == extra, counts
            assert counts["total"] == 4824 + head + extra, counts
    print("    [PASS] e.g. projection 12 none 19,118; projection 10 deep12 18,309, deep12p 18,418")


def test_identity_start():
    print("[-] With a front end, the model starts exactly as level32")
    images = torch.rand(4, 1, 28, 28, generator=torch.Generator().manual_seed(1))
    for projection, front in ((12, "conv1"), (12, "cnn4"), (10, "deep12"), (10, "deep12p")):
        base, base_next = build("none", projection=projection)
        base.eval()
        with torch.no_grad():
            expected = base(images)
        base_state = base.state_dict()
        model, model_next = build(front, projection=projection)
        assert torch.equal(base_next, model_next), f"{front}: building the front end changed the random state"
        state = model.state_dict()
        for key, value in base_state.items():
            assert torch.equal(value, state[key]), f"{front}: {key} differs from level32"
        model.eval()
        with torch.no_grad():
            got = model(images)
        difference = (got - expected).abs().max().item()
        print(f"    {front}: max |output - level32 output| = {difference:.2e}")
        assert difference < 1e-5, difference
    print("    [PASS]")


def test_forward_and_backward():
    print("[-] Forward and backward, train mode with checkpointing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")
    for front in ("conv1", "cnn4", "deep12p"):
        model = build(front, projection=10 if front.startswith("deep") else 12)[0].to(device)
        model.train()
        images = torch.rand(2, 1, 28, 28, device=device)
        loss = torch.nn.functional.cross_entropy(model(images), torch.tensor([3, 7], device=device))
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad), front
        # At the start only the last front-end layer gets a gradient (cnn<C>: the first conv feeds a zero layer).
        last = model.front.net if front == "conv1" else model.front.net[-1]
        norm = last.weight.grad.norm().item()
        print(f"    {front}: |grad| of the last front-end conv = {norm:.3e}")
        assert norm > 0, front
    print("    [PASS]")


def test_classical_controls():
    print("[-] Classical controls (same deep12p front end, no circuit)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = build("deep12p", projection=10)[0]
    for kind, expected in (("enc-linear", 1976), ("enc-cnn", 18392)):
        torch.manual_seed(0)
        model = build_model({"model": kind, "front": "deep12p", "cnn_widths": [20, 32], "cnn_hidden": 20,
                             "dropout": 0.2})
        counts = model.count_parameters()
        print(f"    {kind:10s} {counts}")
        assert counts["front"] == 1646 and counts["total"] == expected, counts
        # The encoder has the same layers and shapes as in front of the circuit. Its random starting values differ:
        # it is drawn at another point of the random stream. The last conv starts at 0 in both.
        reference_state = reference.front.state_dict()
        for name, value in model.front.state_dict().items():
            assert value.shape == reference_state[name].shape, f"{kind}: front {name} has another shape"
        assert not model.front.net[-1].weight.any(), f"{kind}: the last front-end conv does not start at 0"
        model = model.to(device).train()
        images = torch.rand(4, 1, 28, 28, device=device)
        logits = model(images)
        assert logits.shape == (4, 10), logits.shape
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 3, 5, 7], device=device)).backward()
        assert all(p.grad is not None for p in model.parameters()), kind
        norm = model.front.net[-1].weight.grad.norm().item()
        print(f"    {kind}: |grad| of the last front-end conv = {norm:.3e}")
        assert norm > 0, kind
    assert isinstance(build_model({"model": "enc-linear", "front": "deep12p"}), EncoderLinear)
    assert isinstance(build_model({"model": "enc-cnn", "front": "deep12p"}), EncoderCNN)
    print("    [PASS] enc-linear 1,976; enc-cnn 18,392 (deep12p + circuit + projection 10: 18,418)")


def _dense_window_features(stage, theta, phi):
    """Reference for one pixel: theta, phi (9,) of its window -> (27,) X, Y, Z of every qubit, by dense matrices."""
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


def test_quantum_encoder():
    print("[-] Quantum encoder qconv (9-qubit circuit on each 3x3 window)")
    torch.manual_seed(1)
    stage = QuantumConvStage(layers=2, frozen=False)
    height = width = 4
    theta = torch.rand(2, height * width) * 1.5 - 0.2
    phi = torch.rand(2, height * width) * 2 - 1
    fast = stage.features(theta, phi, height, width)
    assert fast.shape == (2, 16, 27), fast.shape
    padded_t = torch.nn.functional.pad(theta.reshape(2, height, width), (1, 1, 1, 1))
    padded_p = torch.nn.functional.pad(phi.reshape(2, height, width), (1, 1, 1, 1))
    worst = 0.0
    for b, pixel in ((0, 0), (0, 5), (1, 10), (1, 15)):
        row, column = divmod(pixel, width)
        reference = _dense_window_features(stage, padded_t[b, row:row + 3, column:column + 3].flatten(),
                                           padded_p[b, row:row + 3, column:column + 3].flatten())
        worst = max(worst, (fast[b, pixel].double() - reference).abs().max().item())
    print(f"    fast circuit vs dense 512x512 matrices (inside and at the border): max difference {worst:.2e}")
    assert worst < 1e-4, worst
    counts = {mode: sum(p.numel() for p in QuantumFront(mode, 16, 16).parameters())
              for mode in ("qconv2", "qconv2x2", "qconv4x3", "qconv2x2f")}
    print(f"    trainable parameters: {counts}")
    assert counts == {"qconv2": 110, "qconv2x2": 220, "qconv4x3": 492, "qconv2x2f": 112}, counts
    # h<H>: readout Linear(27, H) GELU Linear(H, 2); without h the readout keys are as before.
    big = QuantumFront("qconv2x2h16", 16, 16)
    big_count = sum(p.numel() for p in big.parameters())
    assert big_count == 1072, big_count
    assert "stages.0.readout.0.weight" in big.state_dict() and "stages.0.readout.2.weight" in big.state_dict()
    assert "stages.0.readout.weight" in QuantumFront("qconv2x2", 16, 16).state_dict()
    print(f"    qconv2x2h16 (readout Linear(27,16) GELU Linear(16,2)): {big_count}")
    frozen = QuantumFront("qconv2x2f", 16, 16)
    assert not any(isinstance(s.angles, torch.nn.Parameter) for s in frozen.stages), "frozen angles are parameters"
    assert "stages.0.angles" in frozen.state_dict(), "frozen angles are not saved"
    # Full model: starts as level32 projection 10, draws no random numbers.
    images = torch.rand(3, 1, 28, 28, generator=torch.Generator().manual_seed(2))
    base, base_next = build("none", projection=10)
    model, model_next = build("qconv2x2", projection=10)
    assert torch.equal(base_next, model_next), "building the quantum front end changed the random state"
    state = model.state_dict()
    for key, value in base.state_dict().items():
        assert torch.equal(value, state[key]), f"{key} differs from level32"
    parts = model.count_parameters()
    print(f"    qcnn + qconv2x2, projection 10: {parts}")
    assert (parts["quantum"], parts["front"], parts["front_quantum"], parts["head"], parts["total"]) == \
        (4824, 220, 108, 11948, 16992), parts
    base.eval()
    model.eval()
    with torch.no_grad():
        difference = (model(images) - base(images)).abs().max().item()
    print(f"    max |output - level32 output| = {difference:.2e}")
    assert difference < 1e-5, difference
    # The angles get a gradient once the readouts are not 0 (at the start they are 0, like the last conv of deep12p).
    with torch.no_grad():
        for s in model.front.stages:
            s.readout.weight.normal_(0, 0.1)
    model.train()
    torch.nn.functional.cross_entropy(model(images), torch.tensor([1, 2, 3])).backward()
    norms = [s.angles.grad.norm().item() for s in model.front.stages]
    print(f"    |grad| of the circuit angles per stage: {', '.join(f'{n:.2e}' for n in norms)}")
    assert all(n > 0 for n in norms), norms
    # Checkpointing (train) and the plain path (eval, with grad) give the same values and gradients.
    front = model.front
    values = torch.rand(2, 256)
    front.zero_grad()
    front.train()
    out_ckpt = front(values)
    (out_ckpt[0].sum() + out_ckpt[1].square().sum()).backward()
    grads_ckpt = [p.grad.clone() for p in front.parameters()]
    front.zero_grad()
    front.eval()
    out_plain = front(values)
    (out_plain[0].sum() + out_plain[1].square().sum()).backward()
    for a, b in zip(out_ckpt, out_plain):
        assert torch.allclose(a, b, atol=1e-6), "checkpointed quantum front output differs"
    for p, g in zip(front.parameters(), grads_ckpt):
        assert torch.allclose(p.grad, g, atol=1e-5), "checkpointed quantum front gradient differs"
    # Encoder tests: the same linear readout after each encoder.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for front_mode, expected in (("qconv2x2", 550), ("qconv4x3", 822), ("qconv2x2f", 442), ("cnn8p", 556),
                                 ("deep12p", 1976)):
        torch.manual_seed(0)
        encoder = build_model({"model": "enc-linear", "front": front_mode})
        counts = encoder.count_parameters()
        assert counts["total"] == expected, (front_mode, counts)
        encoder = encoder.to(device).train()
        logits = encoder(torch.rand(4, 1, 28, 28, device=device))
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 3, 5, 7], device=device)).backward()
        print(f"    enc-linear --front {front_mode:9s} {counts}")
    for pool, cases in ((2, (("qconv2x2", 1510), ("cnn8p", 1516), ("qconv2x2f", 1402), ("deep12p", 2936))),
                        (1, (("qconv2x2", 5350),))):
        for front_mode, expected in cases:
            counts = build_model({"model": "enc-linear", "front": front_mode, "linear_pool": pool}).count_parameters()
            assert counts["total"] == expected, (pool, front_mode, counts)
    print("    enc-linear --linear-pool 2: qconv2x2 1,510; cnn8p 1,516; qconv2x2f 1,402; deep12p 2,936")
    for front_mode, expected in (("cnn8p", 16998), ("qconv4x3", 17264), ("deep6p", 17272), ("qconv2x2h16", 17844),
                                 ("cnn38p", 17838)):
        counts = build(front_mode, projection=10)[0].count_parameters()
        assert counts["total"] == expected, (front_mode, counts)
        print(f"    qcnn --front {front_mode:11s} {counts}")
    # Every new front end also starts exactly as level32 and draws no random numbers.
    base, base_next = build("none", projection=10)
    base.eval()
    with torch.no_grad():
        expected_out = base(images)
    for front_mode in ("qconv4x3", "qconv2x2h16", "deep6p"):
        model, model_next = build(front_mode, projection=10)
        assert torch.equal(base_next, model_next), f"{front_mode}: building the front end changed the random state"
        model.eval()
        with torch.no_grad():
            difference = (model(images) - expected_out).abs().max().item()
        assert difference < 1e-5, (front_mode, difference)
    print("    [PASS] qconv2x2 220 (angles 108); enc-linear: qconv2x2 550, cnn8p 556, deep12p 1,976; "
          "qcnn + qconv2x2 16,992, + cnn8p 16,998, + qconv4x3 17,264, + deep6p 17,272, + qconv2x2h16 17,844")


def test_linear_heads():
    print("[-] Linear heads (linhead): one Linear(372, 10) per tree (linear3) or shared, logits = their mean")
    for head, head_count, total in (("linear3", 3 * 3730, 16506), ("linear-shared", 3730, 9046)):
        torch.manual_seed(0)
        model = build_model({"model": "qcnn", "front": "qconv4x3", "projection": 10, "dropout": 0.0, "head": head})
        counts = model.count_parameters()
        print(f"    qconv4x3 --head {head:13s} {counts}")
        assert (counts["quantum"], counts["front"], counts["front_quantum"], counts["head"], counts["total"]) == \
            (4824, 492, 324, head_count, total), counts
        assert model.fusion is None
        assert all(isinstance(p, torch.nn.Linear) and p.in_features == 372 and p.out_features == 10
                   for p in model.projections), model.projections
        if head == "linear-shared":
            assert model.projections[0] is model.projections[1] is model.projections[2]
        # The logits are exactly the mean of the three Linear outputs.
        captured = []
        hooks = [p.register_forward_hook(lambda module, inputs, output: captured.append(output))
                 for p in dict.fromkeys(model.projections)]
        images = torch.rand(3, 1, 28, 28, generator=torch.Generator().manual_seed(3))
        model.eval()
        with torch.no_grad():
            logits = model(images)
        for hook in hooks:
            hook.remove()
        assert len(captured) == 3, len(captured)
        difference = (logits - torch.stack(captured).mean(dim=0)).abs().max().item()
        assert difference < 1e-6, difference
        # Every parameter gets a gradient once the encoder readouts are not 0.
        with torch.no_grad():
            for stage in model.front.stages:
                stage.readout.weight.normal_(0, 0.1)
        model.train()
        torch.nn.functional.cross_entropy(model(images), torch.tensor([1, 2, 3])).backward()
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, missing
    # The MLP head is unchanged: same count, and a config without "head" builds it.
    counts = build_model({"model": "qcnn", "front": "qconv4x3", "projection": 10}).count_parameters()
    assert counts["total"] == 17264 and counts["head"] == 11948, counts
    print("    [PASS] linear3 16,506 (head 11,190, quantum 31.2%); linear-shared 9,046 (head 3,730, 56.9%); "
          "mlp 17,264 unchanged")


def test_cnn3_control():
    print("[-] cnn3: 3 small CNNs in place of encoder + trees, the same head as qcnn")
    for head, total, head_count in (("mlp", 17291, 3 * (3690 + 20) + 698), ("linear3", 16533, 3 * 3690),
                                    ("linear-shared", 9153, 3690)):
        torch.manual_seed(0)
        model = build_model({"model": "cnn3", "head": head, "projection": 10, "hidden": 16, "dropout": 0.0})
        counts = model.count_parameters()
        print(f"    cnn3 --head {head:13s} {counts}")
        assert counts["front"] == 3 * 1821 and counts["head"] == head_count and counts["total"] == total, counts
        # The head has the same layers as the qcnn head for the same --head (input 368 instead of 372).
        qcnn = build_model({"model": "qcnn", "front": "qconv4x3", "projection": 10, "dropout": 0.0, "head": head})
        shapes = [tuple(p.shape) for module in (model.projections, model.fusion) if module is not None
                  for p in module.parameters()]
        qshapes = [tuple(p.shape) for module in (qcnn.projections, qcnn.fusion) if module is not None
                   for p in module.parameters()]
        assert [s if s[-1:] != (368,) else s[:-1] + (372,) for s in shapes] == qshapes, (shapes, qshapes)
        images = torch.rand(4, 1, 28, 28)
        model.train()
        logits = model(images)
        assert logits.shape == (4, 10), logits.shape
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 3, 5, 7])).backward()
        assert all(p.grad is not None for p in model.parameters()), head
    print("    [PASS] cnn3 mlp 17,291 (qcnn 17,264); linear3 16,533 (16,506); linear-shared 9,153 (9,046)")


def test_speed():
    """Information only: seconds per training batch of 64 on this device."""
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[-] Speed, batch 64, Adam, on {device} (information only)")
    images = torch.rand(64, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (64,), device=device)
    cases = [("qcnn", "qconv4x3", "linear3"), ("qcnn", "qconv4x3", "linear-shared"), ("qcnn", "qconv4x3", "mlp"),
             ("cnn3", "none", "mlp"), ("cnn3", "none", "linear3")]
    for kind, front_mode, head in cases:
        model = build_model({"model": kind, "front": front_mode, "projection": 10, "dropout": 0.0,
                             "head": head}).to(device).train()
        optimizer = torch.optim.Adam(model.parameters())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for repeat in range(3):
            if repeat == 1:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(images), labels).backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = (time.perf_counter() - started) / 2
        memory = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
        print(f"    {kind:10s} --front {front_mode:9s} --head {head:13s} {seconds:7.3f} s/batch, {memory:5.2f} GB")
        model = optimizer = None


if __name__ == "__main__":
    print("=" * 60)
    print("Running checks for full_quantum_16x16_level32_qconv_linhead...")
    print("=" * 60)
    test_parameters()
    test_identity_start()
    test_forward_and_backward()
    test_classical_controls()
    test_quantum_encoder()
    test_linear_heads()
    test_cnn3_control()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    test_speed()
