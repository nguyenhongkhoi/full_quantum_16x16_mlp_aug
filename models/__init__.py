from __future__ import annotations

from .patch_audit import AuditConfig, PatchAuditModel, is_head_parameter

# Every preset uses the same 32 patches of 12 pixels and the same readout; only the feature layer changes.
PRESETS = {
    "s9_meas": AuditConfig(core="quantum"),
    "c0_pixel": AuditConfig(core="pixel"),
    "c1_encoding": AuditConfig(core="encoding"),
    "c2_product": AuditConfig(core="product"),
    "c4_local_mlp": AuditConfig(core="local_mlp"),
}
EXPECTED_PARAMETERS = {
    "s9_meas": 2546,
    "c0_pixel": 2318,
    "c1_encoding": 2434,
    "c2_product": 2514,
    "c4_local_mlp": 2542,
}


def build_model(preset: str, init: str = "balanced", init_scale: float = 0.1, freeze_core: bool = False):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    return PatchAuditModel(PRESETS[preset], init=init, init_scale=init_scale, freeze_core=freeze_core)


__all__ = ["AuditConfig", "EXPECTED_PARAMETERS", "PRESETS", "PatchAuditModel", "build_model", "is_head_parameter"]
