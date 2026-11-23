"""Quantum backbones and their classical controls for Mammoth. Copy the backbone folder into Mammoth."""

import torch
import torch.nn.functional as F
from torch import nn

from backbone import MammothBackbone, register_backbone
from backbone.quantum_core.front import front_has_phase, make_front
from backbone.quantum_core.fusion_model import SCAN_ANGLE_ORDERS, SPATIAL_ORDERS, FusionQCNN
from backbone.quantum_core.preprocessing import preprocess_full_quantum
from backbone.quantum_core.patch_audit import AuditConfig, PatchAuditModel

EXPECTED_PARAMETERS = {"qcnn-fusion": 19202, "qcnn-fusion-scan": 19202, "qcnn-fusion-moe": 13900,
                       "qcnn-fusion-phase1": 19202, "qcnn-fusion-phase2": 19202, "qcnn-fusion-phase4": 19202,
                       "qcnn-fusion-typed4": 12010, "qcnn-fusion-level32": 19118,
                       "qcnn-fusion-level32-lr4": 19118, "qcnn-fusion-level32-4tree": 19162,
                       "qcnn-fusion-level32-graph": 19148,
                       "qcnn-s9": 2546, "c4-local-mlp": 2542, "mlp-19k": 18858,
                       "cnn-3k": 2988, "cnn-5k": 5578, "cnn-19k": 19060,
                       "mlp-12k": 11970, "cnn-12k": 12070,
                       "qcnn-fusion-level32-deep12p": 18418, "enc-cnn-deep12p": 18392,
                       "enc-linear-deep12p": 1976,
                       # mammoth_qconv_front (2026-11-13)
                       "enc-linear-qconv2x2": 550, "enc-linear-qconv4x3": 822, "enc-linear-qconv2x2f": 442,
                       "enc-linear-cnn8p": 556,
                       "qcnn-fusion-level32-qconv2x2": 16992, "qcnn-fusion-level32-qconv4x3": 17264,
                       "qcnn-fusion-level32-cnn8p": 16998,
                       # mammoth_full_package (2026-11-16): CNN ladder up to 85k, CNN encoder of the size of qconv4x3
                       "cnn-17k": 17170, "cnn-30k": 30726, "cnn-50k": 49984, "cnn-85k": 85258,
                       "qcnn-fusion-level32-deep6p": 17272}


class SmallCNN(nn.Module):
    """The CNN layout of the earlier cnn3k and cnn5k baselines, on the same 16x16 input as the quantum models."""

    def __init__(self, widths, num_classes: int):
        super().__init__()
        first, second = widths
        self.core = nn.Sequential(
            nn.Conv2d(1, first, 3, padding=1),
            nn.BatchNorm2d(first),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(first, second, 3, padding=1),
            nn.BatchNorm2d(second),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(second * 4 * 4, second),
            nn.ReLU(inplace=True),
            nn.Linear(second, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(images, size=(16, 16), mode="bilinear", align_corners=False)
        return self.head(self.core(values))


class SmallMLP(nn.Module):
    """Classical control for qcnn-fusion: the same 16x16 input and about the same parameter count."""

    def __init__(self, num_classes: int, widths=(64, 32)):
        super().__init__()
        first, second = widths
        self.net = nn.Sequential(nn.Linear(256, first), nn.ReLU(), nn.Linear(first, second), nn.ReLU(),
                                 nn.Linear(second, num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixels = F.interpolate(images, size=(16, 16), mode="bilinear", align_corners=False)
        return self.net(pixels.flatten(start_dim=1))


class WrappedBackbone(MammothBackbone):
    """ER and the other models used here only ask for logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, returnt: str = "out") -> torch.Tensor:
        if returnt != "out":
            raise NotImplementedError("These backbones only return logits")
        return self.model(x)


def _check_classes(num_classes: int) -> None:
    if num_classes != 10:
        raise ValueError(f"These backbones expect 10 classes, got {num_classes}. Use --dataset rot-mnist.")


@register_backbone("qcnn-fusion")
def qcnn_fusion(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes))


@register_backbone("qcnn-fusion-scan")
def qcnn_fusion_scan(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, orders=SCAN_ANGLE_ORDERS))


@register_backbone("qcnn-fusion-moe")
def qcnn_fusion_moe(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, experts=4))


@register_backbone("qcnn-fusion-phase1")
def qcnn_fusion_phase1(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, phase_cycles=1.0))


@register_backbone("qcnn-fusion-phase2")
def qcnn_fusion_phase2(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, phase_cycles=2.0))


@register_backbone("qcnn-fusion-phase4")
def qcnn_fusion_phase4(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, phase_cycles=4.0))


@register_backbone("qcnn-fusion-typed4")
def qcnn_fusion_typed4(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, typed_projection=4))


@register_backbone("qcnn-fusion-level32")
def qcnn_fusion_level32(num_classes: int) -> WrappedBackbone:
    """Also reads X, Y, Z of the 32 qubits after tree level 3; each tree is projected to 12 values, not 16."""
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, side_level=32, projection=12))


@register_backbone("qcnn-fusion-level32-lr4")
def qcnn_fusion_level32_lr4(num_classes: int) -> WrappedBackbone:
    """qcnn-fusion-level32 whose quantum parameters learn 4 times faster under Adam. Same parameters, same start."""
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, side_level=32, projection=12, circuit_lr_scale=4.0))


@register_backbone("qcnn-fusion-level32-4tree")
def qcnn_fusion_level32_4tree(num_classes: int) -> WrappedBackbone:
    """qcnn-fusion-level32 with a fourth tree (Morton order shifted by 4 pixels); each tree is projected to 8 values,
    so the total stays near 19k: quantum 6.432, head 12.730."""
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, orders=SPATIAL_ORDERS + ("morton_shift4",), side_level=32,
                                      projection=8))


@register_backbone("qcnn-fusion-level32-graph")
def qcnn_fusion_level32_graph(num_classes: int) -> WrappedBackbone:
    """qcnn-fusion-level32 with an 8-neighbour quantum graph layer before each tree (30 parameters); starts as
    qcnn-fusion-level32."""
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, side_level=32, projection=12, graph_layer=True))


@register_backbone("qcnn-s9")
def qcnn_s9(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(PatchAuditModel(AuditConfig(core="quantum"), init="balanced"))


@register_backbone("c4-local-mlp")
def c4_local_mlp(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(PatchAuditModel(AuditConfig(core="local_mlp"), init="random"))


@register_backbone("mlp-19k")
def mlp_19k(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(SmallMLP(num_classes))


@register_backbone("cnn-3k")
def cnn_3k(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(SmallCNN((8, 11), num_classes))


@register_backbone("cnn-5k")
def cnn_5k(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(SmallCNN((8, 16), num_classes))


@register_backbone("cnn-19k")
def cnn_19k(num_classes: int) -> WrappedBackbone:
    _check_classes(num_classes)
    return WrappedBackbone(SmallCNN((15, 30), num_classes))


@register_backbone("mlp-12k")
def mlp_12k(num_classes: int) -> WrappedBackbone:
    """Same layout as mlp-19k, sized like qcnn-fusion-typed4 (12.010)."""
    _check_classes(num_classes)
    return WrappedBackbone(SmallMLP(num_classes, widths=(42, 22)))


@register_backbone("cnn-12k")
def cnn_12k(num_classes: int) -> WrappedBackbone:
    """Same layout as cnn-19k, sized like qcnn-fusion-typed4 (12.010)."""
    _check_classes(num_classes)
    return WrappedBackbone(SmallCNN((11, 24), num_classes))


# ---- mammoth_level32_front (2026-11-13): level32 with the classical encoder deep12p, and its classical controls ----

class EncoderControl(nn.Module):
    """The same ClassicalFront as in front of the circuit, then a classical readout instead of the circuit.

    The front output (theta, and phi for deep<C>p) becomes a 1- or 2-channel 16x16 image.
    widths None: AvgPool 4x4 -> Linear(channels * 16, 10)  (deep12p: 1,646 + 330 = 1,976).
    widths (a, b): Conv(ch->a) BN ReLU MaxPool -> Conv(a->b) BN ReLU MaxPool -> Linear(16b, hidden) ReLU
                   -> Linear(hidden, 10)  (deep12p, (20, 32), 20: 1,646 + 16,746 = 18,392). No dropout, as in the
                   other Mammoth backbones.
    """

    def __init__(self, front: str, num_classes: int, widths=None, hidden: int = 20):
        super().__init__()
        self.channels = 2 if front_has_phase(front) else 1
        self.front = make_front(front, 16, 16)
        if widths is None:
            self.readout = nn.Sequential(nn.AvgPool2d(4), nn.Flatten(), nn.Linear(self.channels * 16, num_classes))
        else:
            first, second = widths
            self.readout = nn.Sequential(
                nn.Conv2d(self.channels, first, 3, padding=1), nn.BatchNorm2d(first), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(first, second, 3, padding=1), nn.BatchNorm2d(second), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(second * 4 * 4, hidden), nn.ReLU(), nn.Linear(hidden, num_classes),
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = preprocess_full_quantum(images, (16, 16), grayscale=True, mode="resize")
        theta, phi = self.front(values)
        maps = [theta] if phi is None else [theta, phi]
        return self.readout(torch.stack(maps, dim=1).reshape(-1, self.channels, 16, 16))


@register_backbone("qcnn-fusion-level32-deep12p")
def qcnn_fusion_level32_deep12p(num_classes: int) -> WrappedBackbone:
    """qcnn-fusion-level32 with the classical encoder deep12p (1,646) before the circuit: 3 convs 3x3 with 12
    channels set RY(pi theta) and RZ(pi phi) of every qubit. Each tree is projected to 10 values (head 11,948), as
    in the static MNIST run that reached 99.14 val. Total 18,418: quantum 4,824, front 1,646, head 11,948."""
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, side_level=32, projection=10, front="deep12p"))


@register_backbone("enc-cnn-deep12p")
def enc_cnn_deep12p(num_classes: int) -> WrappedBackbone:
    """Classical control of qcnn-fusion-level32-deep12p: the same encoder, the circuit and head replaced by a CNN of
    about the same size (18,392)."""
    _check_classes(num_classes)
    return WrappedBackbone(EncoderControl("deep12p", num_classes, widths=(20, 32), hidden=20))


@register_backbone("enc-linear-deep12p")
def enc_linear_deep12p(num_classes: int) -> WrappedBackbone:
    """The encoder deep12p alone, with the smallest readout (1,976)."""
    _check_classes(num_classes)
    return WrappedBackbone(EncoderControl("deep12p", num_classes))


# ---- mammoth_qconv_front (2026-11-13): a small quantum circuit as the encoder ----
# Encoder test (fast, no tree circuit): each encoder + the same readout AvgPool 4x4 -> Linear(32, 10) (330).
#   enc-linear-deep12p (above, 1,976) is the classical encoder of 1.6k that the quantum encoders are compared with.

@register_backbone("enc-linear-qconv2x2")
def enc_linear_qconv2x2(num_classes: int) -> WrappedBackbone:
    """Quantum encoder qconv2x2 (2 stages of a 9-qubit, 2-layer circuit on each 3x3 window; window 5x5; 220)
    + Linear readout: 550."""
    _check_classes(num_classes)
    return WrappedBackbone(EncoderControl("qconv2x2", num_classes))


@register_backbone("enc-linear-qconv4x3")
def enc_linear_qconv4x3(num_classes: int) -> WrappedBackbone:
    """Quantum encoder qconv4x3 (3 stages, 4 layers; window 7x7 like deep12p; 492) + Linear readout: 822."""
    _check_classes(num_classes)
    return WrappedBackbone(EncoderControl("qconv4x3", num_classes))


@register_backbone("enc-linear-qconv2x2f")
def enc_linear_qconv2x2f(num_classes: int) -> WrappedBackbone:
    """qconv2x2 with the circuit angles frozen at their random start (only the two Linear(27, 2) learn, 112)
    + Linear readout: 442. Shows whether training the circuit matters."""
    _check_classes(num_classes)
    return WrappedBackbone(EncoderControl("qconv2x2f", num_classes))


@register_backbone("enc-linear-cnn8p")
def enc_linear_cnn8p(num_classes: int) -> WrappedBackbone:
    """Classical encoder of the same size as qconv2x2: cnn8p (226, window 5x5) + Linear readout: 556."""
    _check_classes(num_classes)
    return WrappedBackbone(EncoderControl("cnn8p", num_classes))


@register_backbone("qcnn-fusion-level32-qconv2x2")
def qcnn_fusion_level32_qconv2x2(num_classes: int) -> WrappedBackbone:
    """Full quantum pipeline: the quantum encoder qconv2x2 (220) before the unchanged level32 circuit (4,824),
    projection 10 as deep12p (head 11,948). Total 16,992; quantum 5,040 of them (circuit 4,824 + encoder angles
    108; the encoder's two Linear(27, 2), 112, are counted in front)."""
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, side_level=32, projection=10, front="qconv2x2"))


@register_backbone("qcnn-fusion-level32-qconv4x3")
def qcnn_fusion_level32_qconv4x3(num_classes: int) -> WrappedBackbone:
    """As qcnn-fusion-level32-qconv2x2 with the encoder qconv4x3 (492, window 7x7). Total 17,264."""
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, side_level=32, projection=10, front="qconv4x3"))


@register_backbone("qcnn-fusion-level32-cnn8p")
def qcnn_fusion_level32_cnn8p(num_classes: int) -> WrappedBackbone:
    """Classical control of qcnn-fusion-level32-qconv2x2: the same circuit and head, a classical encoder of the same
    size (cnn8p, 226). Total 16,998."""
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, side_level=32, projection=10, front="cnn8p"))


# ---- mammoth_full_package (2026-11-16) ----
# The same SmallCNN layout as cnn-3k/5k/12k/19k, more sizes (also in classical_cnn_16x16 for static MNIST):
# parameters 12a + 9ab + 16b^2 + 14b + 10 for widths (a, b).

@register_backbone("cnn-17k")
def cnn_17k(num_classes: int) -> WrappedBackbone:
    """SmallCNN (16, 28): 17,170, the size of qcnn-fusion-level32-qconv4x3 (17,264)."""
    _check_classes(num_classes)
    return WrappedBackbone(SmallCNN((16, 28), num_classes))


@register_backbone("cnn-30k")
def cnn_30k(num_classes: int) -> WrappedBackbone:
    """SmallCNN (20, 38): 30,726."""
    _check_classes(num_classes)
    return WrappedBackbone(SmallCNN((20, 38), num_classes))


@register_backbone("cnn-50k")
def cnn_50k(num_classes: int) -> WrappedBackbone:
    """SmallCNN (24, 49): 49,984."""
    _check_classes(num_classes)
    return WrappedBackbone(SmallCNN((24, 49), num_classes))


@register_backbone("cnn-85k")
def cnn_85k(num_classes: int) -> WrappedBackbone:
    """SmallCNN (32, 64): 85,258."""
    _check_classes(num_classes)
    return WrappedBackbone(SmallCNN((32, 64), num_classes))


@register_backbone("qcnn-fusion-level32-deep6p")
def qcnn_fusion_level32_deep6p(num_classes: int) -> WrappedBackbone:
    """Classical control of qcnn-fusion-level32-qconv4x3: the same circuit and head, a classical encoder of the same
    window (7x7) and about the same size (deep6p: 3 convs 3x3 with 6 channels, 500). Total 17,272."""
    _check_classes(num_classes)
    return WrappedBackbone(FusionQCNN(num_classes, side_level=32, projection=10, front="deep6p"))
