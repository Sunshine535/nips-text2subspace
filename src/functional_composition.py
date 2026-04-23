"""
Functional LoRA Composition (FLC): Calibration-Optimal Adapter Merging.

Instead of heuristic weight-space merging (TA/TIES/DARE) or feature-space
composition (SFC), FLC finds the rank-r merged adapter that minimally distorts
each adapter's actual activation effects on its calibration data.

    min_{rank-r ΔW} Σ_i ||ΔW · x - ΔW_i · x||²  for x ~ D_i

Solved exactly via weighted least squares + truncated SVD.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from pathlib import Path

logger = logging.getLogger(__name__)


def load_lora_delta_w(adapter_path: str) -> Dict[str, torch.Tensor]:
    """Load adapter and compute delta_W = B @ A for each LoRA module."""
    p = Path(adapter_path)
    if (p / "adapter_model.safetensors").exists():
        weights = load_file(str(p / "adapter_model.safetensors"))
    else:
        weights = torch.load(str(p / "adapter_model.bin"), map_location="cpu")

    modules = set()
    for key in weights:
        if ".lora_A.weight" in key:
            modules.add(key.replace(".lora_A.weight", ""))

    delta_ws = {}
    for mod in sorted(modules):
        A = weights[mod + ".lora_A.weight"].float()
        B = weights[mod + ".lora_B.weight"].float()
        clean_mod = mod.replace("base_model.model.", "", 1)
        delta_ws[clean_mod] = B @ A
    return delta_ws


def collect_module_inputs(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    target_modules: List[str],
    batch_size: int = 4,
    max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    """Collect input activations at each LoRA target module."""
    input_cache = {mod: [] for mod in target_modules}
    handles = []

    for name, module in model.named_modules():
        for target in target_modules:
            if name == target or name.endswith(target):
                def make_hook(mod_name):
                    def hook_fn(module, input, output):
                        if isinstance(input, tuple):
                            x = input[0]
                        else:
                            x = input
                        input_cache[mod_name].append(x.detach().cpu().reshape(-1, x.shape[-1]))
                    return hook_fn
                handles.append(module.register_forward_hook(make_hook(target)))
                break

    model.eval()
    device = next(model.parameters()).device

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                          max_length=max_length, padding="max_length").to(device)
        with torch.no_grad():
            model(**{k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")})

    for h in handles:
        h.remove()

    result = {}
    for mod, tensors in input_cache.items():
        if tensors:
            result[mod] = torch.cat(tensors, dim=0)
    return result


def functional_merge(
    delta_ws: List[Dict[str, torch.Tensor]],
    calibration_inputs: List[Dict[str, torch.Tensor]],
    rank: int = 16,
    reg: float = 1e-6,
    weights: Optional[List[float]] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Functional LoRA Composition: find optimal rank-r merge.

    For each module, solves:
        min_{ΔW} Σ_i w_i ||ΔW · X_i - ΔW_i · X_i||²_F
    then truncates to rank r.

    Args:
        delta_ws: list of per-adapter {module: delta_W} dicts
        calibration_inputs: list of per-adapter {module: X} activation dicts
        rank: target LoRA rank
        reg: regularization for least-squares stability
        weights: per-adapter importance weights (default: uniform)

    Returns:
        Dict mapping module_name → (B, A) LoRA factors
    """
    N = len(delta_ws)
    if weights is None:
        weights = [1.0 / N] * N

    all_modules = sorted(set.intersection(*[set(d.keys()) for d in delta_ws]))
    merged = {}
    diagnostics = {}

    for mod in all_modules:
        X_parts = []
        Y_parts = []

        for i in range(N):
            if mod not in calibration_inputs[i]:
                continue
            dw_i = delta_ws[i][mod]
            x_i = calibration_inputs[i][mod].float()

            n_samples = min(x_i.shape[0], 2048)
            x_i = x_i[:n_samples]

            y_i = (dw_i @ x_i.T).T
            X_parts.append(x_i * (weights[i] ** 0.5))
            Y_parts.append(y_i * (weights[i] ** 0.5))

        if not X_parts:
            continue

        X = torch.cat(X_parts, dim=0)
        Y = torch.cat(Y_parts, dim=0)

        XtX = X.T @ X
        d_in = XtX.shape[0]
        XtX_reg = XtX + reg * torch.eye(d_in)
        YtX = Y.T @ X

        delta_W_opt = YtX @ torch.linalg.solve(XtX_reg, torch.eye(d_in))

        U, S, Vh = torch.linalg.svd(delta_W_opt, full_matrices=False)
        r = min(rank, U.shape[1])
        sqrt_S = torch.sqrt(S[:r].clamp(min=0))

        B = U[:, :r] @ torch.diag(sqrt_S)
        A = torch.diag(sqrt_S) @ Vh[:r, :]
        merged[mod] = (B, A)

        total_energy = float(S.sum())
        kept_energy = float(S[:r].sum())
        residual = float((S[r:]).sum()) if r < len(S) else 0.0
        diagnostics[mod] = {
            "total_sv_energy": total_energy,
            "kept_energy": kept_energy,
            "residual_energy": residual,
            "energy_ratio": kept_energy / max(total_energy, 1e-10),
        }

    mean_ratio = sum(d["energy_ratio"] for d in diagnostics.values()) / max(len(diagnostics), 1)
    logger.info("FLC merged %d modules, rank=%d, mean energy retained=%.4f",
                len(merged), rank, mean_ratio)

    return merged, diagnostics


def save_functional_adapter(
    merged: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    template_adapter_path: str,
    output_dir: str,
    dtype=torch.bfloat16,
):
    """Save FLC-merged adapter in PEFT format."""
    import shutil, json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    state = {}
    for mod, (B, A) in merged.items():
        state[mod + ".lora_A.weight"] = A.to(dtype)
        state[mod + ".lora_B.weight"] = B.to(dtype)

    save_file(state, str(out / "adapter_model.safetensors"))

    template = Path(template_adapter_path)
    cfg_src = template / "adapter_config.json"
    if cfg_src.exists():
        shutil.copy2(str(cfg_src), str(out / "adapter_config.json"))
