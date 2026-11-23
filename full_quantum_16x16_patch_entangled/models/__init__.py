from __future__ import annotations

from .patch_qcnn import MeasureConfig, PatchConfig, PatchMeasureQCNN, PatchQCNN, is_head_parameter

# Each step changes one thing relative to the step it names in the README.
PRESETS = {
    "s1_p8": PatchConfig(patch=(2, 4), grid=(8, 4)),
    "s2_p12": PatchConfig(patch=(3, 4), grid=(8, 4)),
    "s3_ru2": PatchConfig(patch=(3, 4), grid=(8, 4), layers=2),
    "s4_ru3": PatchConfig(patch=(3, 4), grid=(8, 4), layers=3),
    "s5_overlap": PatchConfig(patch=(3, 4), grid=(8, 8), layers=2),
    "s6_k4": PatchConfig(patch=(3, 4), grid=(8, 8), layers=2, channels=4, grad_checkpoint=True),
    "s7_p16": PatchConfig(patch=(4, 4), grid=(4, 4), grad_checkpoint=True),
    "s8_xl": PatchConfig(patch=(3, 4), grid=(8, 8), layers=2, channels=8, projection=8, hidden=32,
                         grad_checkpoint=True),
    "s9_meas": MeasureConfig(patch=(3, 4), grid=(8, 4)),
    "s10_meas_ru2": MeasureConfig(patch=(3, 4), grid=(8, 4), layers=2),
}
EXPECTED_PARAMETERS = {
    "s1_p8": {"mlp": 2522, "linear": 1594},
    "s2_p12": {"mlp": 2538, "linear": 1610},
    "s3_ru2": {"mlp": 2602, "linear": 1674},
    "s4_ru3": {"mlp": 2666, "linear": 1738},
    "s5_overlap": {"mlp": 2650, "linear": 1722},
    "s6_k4": {"mlp": 4150, "linear": 3222},
    "s7_p16": {"mlp": 2490, "linear": 1562},
    "s8_xl": {"mlp": 10042, "linear": 6106},
    "s9_meas": {"mlp": 2546, "linear": 1570},
    "s10_meas_ru2": {"mlp": 2610, "linear": 1634},
}


def build_model(preset: str, readout: str = "mlp", init: str = "balanced",
                init_scale: float = 0.1, freeze_core: bool = False):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    config = PRESETS[preset]
    model_class = PatchMeasureQCNN if isinstance(config, MeasureConfig) else PatchQCNN
    return model_class(config, readout=readout, init=init, init_scale=init_scale, freeze_core=freeze_core)


__all__ = ["EXPECTED_PARAMETERS", "MeasureConfig", "PRESETS", "PatchConfig", "PatchMeasureQCNN", "PatchQCNN",
           "build_model", "is_head_parameter"]
