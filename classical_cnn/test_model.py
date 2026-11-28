"""Checks for classical_cnn_16x16. Run once on the training machine: python .\\test_model.py
Parameter counts of every preset, the layout matches the Mammoth SmallCNN (cnn-19k = 19,060), forward/backward,
the preprocessing equals the QCNN one, and speed per batch of 64.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import EXPECTED, PRESETS, STANDARD_EXPECTED, STANDARD_PRESETS, build_model, preprocess, standard_count


def test_counts():
    print("[-] Parameter counts")
    for name, (a, b) in PRESETS.items():
        counts = build_model({"preset": name}).count_parameters()
        formula = 12 * a + 9 * a * b + 16 * b * b + 14 * b + 10
        print(f"    {name:7s} widths ({a:2d}, {b:2d}): {counts}")
        assert counts["total"] == EXPECTED[name] == formula, (name, counts, formula)
    counts = build_model({"widths": [16, 28]}).count_parameters()
    assert counts["total"] == 17170, counts
    counts = build_model({"preset": "cnn5k", "hidden": 32}).count_parameters()
    assert counts["total"] == 1296 + 256 * 32 + 32 + 32 * 10 + 10, counts
    print("    [PASS] cnn1k 934 ... cnn5k 5,578, cnn17k 17,170, cnn19k 19,060, cnn85k 85,258")
    for name, (a, b, h) in STANDARD_PRESETS.items():
        counts = build_model({"arch": "standard", "preset": name}).count_parameters()
        print(f"    {name:7s} standard ({a}, {b}, h {h}): {counts}")
        assert counts["total"] == STANDARD_EXPECTED[name] == standard_count(a, b, h), (name, counts)
    print("    [PASS] std85k 84,754")


def test_forward_backward():
    print("[-] Forward and backward")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in ("cnn1k", "cnn17k", "cnn85k", "std85k"):
        arch = "standard" if name in STANDARD_PRESETS else "small"
        model = build_model({"arch": arch, "preset": name}).to(device).train()
        images = torch.rand(4, 1, 28, 28, device=device)
        logits = model(images)
        assert logits.shape == (4, 10), logits.shape
        F.cross_entropy(logits, torch.tensor([1, 3, 5, 7], device=device)).backward()
        assert all(p.grad is not None for p in model.parameters()), name
    # Same preprocessing as the QCNN (bilinear 16x16, clamp), when that folder is next to this one.
    sibling = ROOT.parent / "full_quantum_16x16_level32_qconv" / "models" / "preprocessing.py"
    if sibling.is_file():
        # Loaded by path: the QCNN package is also called "models", which here is this folder's models.py.
        import importlib.util
        spec = importlib.util.spec_from_file_location("qcnn_preprocessing", sibling)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        preprocess_full_quantum = module.preprocess_full_quantum
        images = torch.rand(3, 1, 28, 28)
        ours = preprocess(images).flatten(1)
        theirs = preprocess_full_quantum(images, (16, 16), grayscale=True, mode="resize")
        assert torch.allclose(ours, theirs), (ours - theirs).abs().max()
        print("    preprocessing equals the QCNN preprocess_full_quantum")
    print("    [PASS]")


def test_speed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[-] Speed, batch 64, Adam, on {device} (information only)")
    images = torch.rand(64, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (64,), device=device)
    for name in ("cnn5k", "cnn17k", "cnn85k"):
        model = build_model({"preset": name}).to(device).train()
        optimizer = torch.optim.Adam(model.parameters())
        for repeat in range(12):
            if repeat == 2:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(images), labels).backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(f"    {name:7s} {(time.perf_counter() - started) / 10 * 1000:6.2f} ms/batch")


if __name__ == "__main__":
    print("=" * 60)
    test_counts()
    test_forward_backward()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    test_speed()
