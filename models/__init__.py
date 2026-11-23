from .classical import EncoderCNN, EncoderLinear
from .fusion_model import FusionQCNN, SPATIAL_ORDERS, SCAN_ANGLE_ORDERS

__all__ = ["FusionQCNN", "EncoderLinear", "EncoderCNN", "build_model", "SPATIAL_ORDERS", "SCAN_ANGLE_ORDERS"]

MODELS = ("qcnn", "enc-linear", "enc-cnn")


def build_model(cfg: dict):
    """Build a model from the run arguments (vars(args) or the args saved in a checkpoint config)."""
    kind = cfg.get("model", "qcnn")
    front = cfg.get("front", "none")
    if kind == "enc-linear":
        return EncoderLinear(front=front)
    if kind == "enc-cnn":
        return EncoderCNN(front=front, widths=tuple(cfg.get("cnn_widths", (20, 32))),
                          hidden=cfg.get("cnn_hidden", 20), dropout=cfg.get("dropout", 0.2))
    if kind != "qcnn":
        raise ValueError(f"Unknown model {kind!r}; use one of {MODELS}")
    return FusionQCNN(
        num_classes=10,
        hidden=cfg.get("hidden", 16),
        orders=tuple(cfg.get("orders", ["row", "column", "morton"])),
        side_level=cfg.get("side_level", 32),
        projection=cfg.get("projection", 12),
        graph_layer=cfg.get("graph_layer", False),
        circuit_lr_scale=cfg.get("circuit_lr_scale", 1.0),
        dropout=cfg.get("dropout", 0.2),
        front=front,
    )
