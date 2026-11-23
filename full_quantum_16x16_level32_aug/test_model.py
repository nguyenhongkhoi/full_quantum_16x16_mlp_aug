"""Unit tests for full_quantum_16x16_level32_aug models.
Run this script to verify parameter counts, forward pass, and gradient checkpointing.
"""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import FusionQCNN


def test_level32_parameters():
    print("[-] Checking Level32 standard model (19,118 parameters)...")
    model = FusionQCNN(num_classes=10, side_level=32, projection=12)
    counts = model.count_parameters()
    print(f"    Parameter counts: {counts}")
    assert counts["quantum"] == 4824, f"Expected 4824 quantum params, got {counts['quantum']}"
    assert counts["head"] == 14294, f"Expected 14294 head params, got {counts['head']}"
    assert counts["total"] == 19118, f"Expected 19118 total params, got {counts['total']}"
    print("    [PASS] Level32 parameter count is exactly 19,118!")


def test_level32_graph_parameters():
    print("[-] Checking Level32 with Graph Layer (19,148 parameters)...")
    model = FusionQCNN(num_classes=10, side_level=32, projection=12, graph_layer=True)
    counts = model.count_parameters()
    print(f"    Parameter counts: {counts}")
    assert counts["quantum"] == 4854, f"Expected 4854 quantum params, got {counts['quantum']}"
    assert counts["head"] == 14294, f"Expected 14294 head params, got {counts['head']}"
    assert counts["total"] == 19148, f"Expected 19148 total params, got {counts['total']}"
    print("    [PASS] Level32 with Graph Layer parameter count is exactly 19,148 (+30 params)!")


def test_forward_and_backward():
    print("[-] Checking forward and backward pass on dummy batch (batch_size=2)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")
    model = FusionQCNN(num_classes=10, side_level=32, projection=12).to(device)
    dummy_images = torch.rand(2, 1, 28, 28, device=device)
    dummy_labels = torch.tensor([3, 7], device=device)

    # Eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(dummy_images)
        assert out_eval.shape == (2, 10), f"Expected shape (2, 10), got {out_eval.shape}"
    print("    [PASS] Eval forward pass successful!")

    # Train mode with checkpointing
    model.train()
    out_train = model(dummy_images)
    loss = torch.nn.functional.cross_entropy(out_train, dummy_labels)
    loss.backward()

    # Check gradients
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "Some parameters did not receive gradients!"
    print("    [PASS] Train mode & backward pass (gradient flow) successful!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running verification for full_quantum_16x16_level32_aug...")
    print("=" * 60)
    test_level32_parameters()
    test_level32_graph_parameters()
    test_forward_and_backward()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
