"""Checks for full_quantum_16x16_level32_front: parameter counts, the front end starts as the identity, gradients.
Run once on the training machine before training: python .\\test_model.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import EncoderCNN, EncoderLinear, FusionQCNN, build_model

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


if __name__ == "__main__":
    print("=" * 60)
    print("Running checks for full_quantum_16x16_level32_front...")
    print("=" * 60)
    test_parameters()
    test_identity_start()
    test_forward_and_backward()
    test_classical_controls()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
