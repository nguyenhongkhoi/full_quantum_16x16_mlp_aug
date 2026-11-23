"""Checks for full_quantum_16x16_multilevel (copy of level32_aug; side_level may list several tree levels).
Run once on the training machine before training: python .\\test_model.py
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import FusionQCNN
from models.encoding import angle_ry_encode

# (levels, projection) -> head; quantum is always 4,824.
# head = 3P(F + 3) + 48P + 218 with F = 276 + 3 * sum(levels) features per tree (hidden 16, 10 classes).
EXPECTED_HEAD = {
    ((32,), 12): 14294,       # level32, 19,118
    ((32, 64), 12): 21206,    # 26,030
    ((16, 32), 12): 16022,    # 20,846
    ((16, 32, 64), 12): 22934,  # 27,758
    ((32,), 18): 21332,       # control: level32 with a head as large as 32+64 @12, 26,156
}


def build(levels, projection, seed=0, **kwargs):
    torch.manual_seed(seed)
    return FusionQCNN(num_classes=10, side_level=levels, projection=projection, dropout=0.2, **kwargs)


def test_parameters():
    print("[-] Parameter counts")
    for (levels, projection), head in EXPECTED_HEAD.items():
        counts = build(list(levels), projection).count_parameters()
        share = counts["quantum"] / counts["total"]
        print(f"    levels {str(list(levels)):14s} projection {projection}: {counts}, quantum {share:.1%}")
        assert counts["quantum"] == 4824, counts
        assert counts["head"] == head, counts
        assert counts["total"] == 4824 + head, counts
    graph = build([32], 12, graph_layer=True).count_parameters()
    assert graph["total"] == 19148, graph
    print("    [PASS] level32 19,118; 32+64 26,030; 16+32 20,846; 16+32+64 27,758; control 32 @18 26,156")


def test_same_as_level32():
    print("[-] side_level=32 and side_level=[32] give the same model as level32_aug")
    images = torch.rand(3, 1, 28, 28, generator=torch.Generator().manual_seed(1))
    old, new = build(32, 12), build([32], 12)
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    old.eval()
    new.eval()
    with torch.no_grad():
        assert torch.equal(old(images), new(images))
    print("    [PASS]")


def test_level_features():
    print("[-] Each listed level adds X, Y, Z of its qubits, in the order given")
    model = build([32, 64], 12).eval()
    qubits = angle_ry_encode(torch.rand(2, 256, generator=torch.Generator().manual_seed(3)))
    core = model.cores[0]
    with torch.no_grad():
        features = model._side_features(core, qubits)
        joint, kept = core.forward_levels(qubits, (16, 32, 64, 128))
    assert sorted(kept) == [16, 32, 64, 128], sorted(kept)
    assert features.shape == (2, 276 + 3 * 96), features.shape
    for level in (16, 32, 64, 128):
        trace = kept[level].diagonal(dim1=-2, dim2=-1).sum(-1).real
        assert torch.allclose(trace, torch.ones_like(trace), atol=1e-5), level
    with torch.no_grad():
        single = build([32], 12).eval()._side_features(core, qubits)
    assert torch.allclose(features[:, :276 + 96], single, atol=1e-6), "the level-32 part differs"
    print("    [PASS] 32+64: 276 + 96 + 192 = 564 values per tree; levels 16, 32, 64, 128 all exist")


def test_forward_and_backward():
    print("[-] Forward and backward, train mode with checkpointing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")
    for levels in ([32, 64], [16, 32, 64]):
        model = build(levels, 12).to(device).train()
        images = torch.rand(2, 1, 28, 28, device=device)
        loss = torch.nn.functional.cross_entropy(model(images), torch.tensor([3, 7], device=device))
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad), levels
        model.eval()
        with torch.no_grad():
            assert model(images).shape == (2, 10)
        print(f"    {levels}: ok")
    print("    [PASS]")


def test_speed():
    """Information only: seconds per training batch of 64 on this device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[-] Speed, batch 64, Adam, on {device} (information only)")
    images = torch.rand(64, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (64,), device=device)
    for levels in ([32], [32, 64], [16, 32, 64]):
        model = build(levels, 12).to(device).train()
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
        print(f"    levels {str(levels):14s} {seconds:7.3f} s/batch, {memory:5.2f} GB")
        model = optimizer = None


if __name__ == "__main__":
    print("=" * 60)
    print("Running checks for full_quantum_16x16_multilevel...")
    print("=" * 60)
    test_parameters()
    test_same_as_level32()
    test_level_features()
    test_forward_and_backward()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    test_speed()
