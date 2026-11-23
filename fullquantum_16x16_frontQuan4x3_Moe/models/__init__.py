from .moe_model import (DEFAULT_OBS, EXPERT_HEADS, FRONT_READOUTS, MIXES, ROUTER_SOURCES, SPATIAL_ORDERS,
                        MoEQCNN)
from .router import ANSATZE, ROUTERS

__all__ = ["MoEQCNN", "build_model", "EXPERT_HEADS", "FRONT_READOUTS", "MIXES", "ROUTER_SOURCES", "SPATIAL_ORDERS",
           "ROUTERS", "ANSATZE", "DEFAULT_OBS"]

LEGACY_HEADS = {"linear3": "linear", "linear-shared": "shared", "mlp": "fusion"}


def build_model(cfg: dict) -> MoEQCNN:
    """Dựng model từ tham số run (vars(args) hoặc args lưu trong config của checkpoint).

    Nhận cả config của full_quantum_16x16_level32_qconv(_linhead) (không có "router"): --head linear3 /
    linear-shared / mlp thành router none với head linear / shared / fusion, để analyze_routing.py đọc được checkpoint
    L1, L2, 2c."""
    if "router" not in cfg:
        head = cfg.get("head", "mlp")
        if cfg.get("model", "qcnn") != "qcnn" or head not in LEGACY_HEADS:
            raise ValueError(f"Checkpoint config with model={cfg.get('model')!r}, head={head!r} is not supported")
        cfg = {**cfg, "router": "none", "expert_head": LEGACY_HEADS[head], "mix": "logits",
               "projection": cfg.get("projection", 12), "front_readout": "linear"}
    return MoEQCNN(
        num_classes=10,
        orders=tuple(cfg.get("orders", SPATIAL_ORDERS)),
        side_level=cfg.get("side_level", 32) or 0,
        front=cfg.get("front", "qconv4x3"),
        front_readout=cfg.get("front_readout", "linear"),
        expert_head=cfg.get("expert_head", "linear"),
        expert_hidden=cfg.get("expert_hidden", 16),
        projection=cfg.get("projection", 10),
        hidden=cfg.get("hidden", 16),
        dropout=cfg.get("dropout", 0.0),
        obs_terms=tuple(cfg.get("obs_terms", DEFAULT_OBS)),
        router=cfg.get("router", "quantum"),
        router_source=cfg.get("router_source", "encoder"),
        router_detach=cfg.get("router_detach", False),
        router_grid=tuple(cfg.get("router_grid", (4, 4))),
        router_qubits=cfg.get("router_qubits", 8),
        router_layers=cfg.get("router_layers", 1),
        router_ansatz=cfg.get("router_ansatz", "se"),
        router_init_scale=cfg.get("router_init_scale", 0.01),
        router_hidden=cfg.get("router_hidden", 18),
        mix=cfg.get("mix", "logits"),
        gate_noise=cfg.get("gate_noise", 0.0),
        head_rank=cfg.get("head_rank", 4),
        share_final=cfg.get("share_final", False),
    )
