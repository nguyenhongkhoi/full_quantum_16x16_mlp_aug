"""Đánh giá trên một loader: gom logit mọi chế độ (soft / top1 / uniform / từng cây), cổng router, tóm tắt."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def collect(model, loader: DataLoader, device: torch.device, progress=None) -> Dict[str, torch.Tensor]:
    """Một lượt eval: nhãn, đầu ra soft, g (batch, E), logit từng cây (batch, E, C; None với head fusion),
    loss soft từng ảnh. Trên CPU."""
    model.eval()
    labels, outputs, gates, expert_logits, losses = [], [], [], [], []
    for index, (images, target) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        target = target.to(device, non_blocking=device.type == "cuda")
        output = model(images, mode="soft")
        losses.append(F.cross_entropy(output, target, reduction="none").cpu())
        labels.append(target.cpu())
        outputs.append(output.float().cpu())
        gates.append(model.last["gates"].float().cpu())
        if model.last["expert_logits"] is not None:
            expert_logits.append(model.last["expert_logits"].float().cpu())
        if progress is not None:
            progress(index, len(loader))
    return {"labels": torch.cat(labels), "outputs": torch.cat(outputs), "gates": torch.cat(gates),
            "expert_logits": torch.cat(expert_logits) if expert_logits else None, "losses": torch.cat(losses)}


def predictions(model, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Dự đoán (nhãn) của mọi chế độ: soft, top1, uniform, tree<i>. Head fusion: chỉ soft."""
    out = {"soft": data["outputs"].argmax(dim=1)}
    gates, logits = data["gates"], data["expert_logits"]
    if logits is None:
        return out
    for mode in ("top1", "uniform"):
        out[mode] = model.combine(logits, gates, mode).argmax(dim=1)
    for index in range(logits.shape[1]):
        out[f"tree{index}"] = logits[:, index].argmax(dim=1)
    return out


def summarize(model, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Các số của một lượt eval (accuracy theo tỷ lệ 0..1). Head fusion: top1, uniform, từng cây, oracle = nan."""
    labels, gates = data["labels"], data["gates"]
    experts = gates.shape[1]
    preds = predictions(model, data)
    result = {"loss": data["losses"].mean().item()}
    for mode in ["soft", "top1", "uniform", *[f"tree{i}" for i in range(experts)]]:
        result[f"acc_{mode}"] = (preds[mode] == labels).float().mean().item() if mode in preds else math.nan
    if data["expert_logits"] is None:
        result["acc_oracle"] = math.nan
    else:
        any_correct = torch.stack([preds[f"tree{i}"] == labels for i in range(experts)], dim=1).any(dim=1)
        result["acc_oracle"] = any_correct.float().mean().item()
    chosen = gates.argmax(dim=1)
    for index in range(experts):
        result[f"gate_mean{index}"] = gates[:, index].mean().item()
        result[f"top1_share{index}"] = (chosen == index).float().mean().item()
    entropy = -(gates * gates.clamp_min(1e-12).log()).sum(dim=1) / math.log(experts) if experts > 1 else gates[:, 0] * 0
    result["gate_entropy"] = entropy.mean().item()
    return result
