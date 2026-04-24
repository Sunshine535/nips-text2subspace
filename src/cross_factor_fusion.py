"""
Bilinear Cross-Factor Fusion (BCFF): Compose LoRA Adapters via Factor Algebra.

When merging two LoRA adapters (B₁A₁, B₂A₂), the full composition space includes
cross-terms B₁A₂ and B₂A₁ that standard methods (TA/TIES/DARE) ignore.

BCFF learns optimal coefficients for all 4 terms via ridge regression on calibration
data, then reconstructs a merged adapter via SVD.

    ΔW_merged = c₁·B₁A₁ + c₂·B₂A₂ + c₃·B₁A₂ + c₄·B₂A₁

The cross-terms can capture positive transfer that diagonal merging cannot express.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from pathlib import Path
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def load_lora_factors(adapter_path: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Load raw LoRA A and B factors from adapter."""
    p = Path(adapter_path)
    if (p / "adapter_model.safetensors").exists():
        weights = load_file(str(p / "adapter_model.safetensors"))
    else:
        weights = torch.load(str(p / "adapter_model.bin"), map_location="cpu")

    factors = {}
    for key in weights:
        if ".lora_A.weight" in key:
            mod = key.replace(".lora_A.weight", "").replace("base_model.model.", "", 1)
            A = weights[key].float()
            B = weights[mod.replace("model.", "base_model.model.model.", 1) + ".lora_B.weight"]
            if B is None:
                for bkey in weights:
                    if ".lora_B.weight" in bkey and mod.split(".")[-1] in bkey:
                        clean_bmod = bkey.replace(".lora_B.weight", "").replace("base_model.model.", "", 1)
                        if clean_bmod == mod:
                            B = weights[bkey]
                            break
            if B is not None:
                factors[mod] = (B.float(), A.float())
    return factors


def load_lora_factors_v2(adapter_path: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Load raw LoRA A and B factors — robust version. Factors are UNSCALED.

    Use get_lora_scaling(adapter_path) + scale B by alpha/r before using (B @ A)
    as a substitute for the LoRA output.
    """
    p = Path(adapter_path)
    if (p / "adapter_model.safetensors").exists():
        weights = load_file(str(p / "adapter_model.safetensors"))
    else:
        weights = torch.load(str(p / "adapter_model.bin"), map_location="cpu")

    modules = {}
    for key in weights:
        if ".lora_A.weight" in key:
            mod_raw = key.replace(".lora_A.weight", "")
            mod_clean = mod_raw.replace("base_model.model.", "", 1)
            b_key = mod_raw + ".lora_B.weight"
            if b_key in weights:
                modules[mod_clean] = (weights[b_key].float(), weights[key].float())
    return modules


def get_lora_scaling(adapter_path: str) -> float:
    """Return lora_alpha / r (the PEFT scaling factor) from adapter_config.json.

    CARR hook / eval_carr / train_carr_router must apply this factor to
    (B @ A) to match PEFT single-adapter inference.
    """
    import json as _json
    cfg_path = Path(adapter_path) / "adapter_config.json"
    if not cfg_path.is_file():
        logger.warning("No adapter_config.json at %s; assuming scale=1.0", adapter_path)
        return 1.0
    with cfg_path.open() as f:
        cfg = _json.load(f)
    alpha = float(cfg.get("lora_alpha", 1))
    r = float(cfg.get("r", 1) or 1)
    return alpha / max(r, 1.0)


def load_scaled_delta_ws(adapter_path: str) -> Dict[str, "torch.Tensor"]:
    """Convenience: returns {module_name: (B @ A) * (alpha / r)} matching PEFT output."""
    factors = load_lora_factors_v2(adapter_path)
    scale = get_lora_scaling(adapter_path)
    return {mod: (B @ A) * scale for mod, (B, A) in factors.items()}


def collect_module_inputs_for_bcff(
    model, tokenizer, texts, target_modules, batch_size=4, max_length=128,
):
    """Collect input activations at LoRA target modules."""
    input_cache = {mod: [] for mod in target_modules}
    handles = []

    for name, module in model.named_modules():
        for target in target_modules:
            if name == target or name.endswith(target):
                def make_hook(mod_name):
                    def hook_fn(module, inp, out):
                        x = inp[0] if isinstance(inp, tuple) else inp
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
            stacked = torch.cat(tensors, dim=0)
            result[mod] = stacked[:min(2048, stacked.shape[0])]
    return result


def bcff_merge(
    factors_1: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    factors_2: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    calibration_inputs: Dict[str, torch.Tensor],
    rank: int = 16,
    reg: float = 1e-4,
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict]:
    """
    Bilinear Cross-Factor Fusion.

    For each module, learns optimal coefficients c1..c4 for:
        ΔW = c1·B1A1 + c2·B2A2 + c3·B1A2 + c4·B2A1

    via ridge regression on calibration activations, then truncates to target rank.
    """
    common_modules = sorted(set(factors_1.keys()) & set(factors_2.keys()))
    merged = {}
    diagnostics = {}

    for mod in common_modules:
        B1, A1 = factors_1[mod]
        B2, A2 = factors_2[mod]

        # 4 candidate delta_W matrices
        dw_11 = B1 @ A1  # diagonal: adapter 1
        dw_22 = B2 @ A2  # diagonal: adapter 2
        dw_12 = B1 @ A2  # cross: B1 with A2's input projection
        dw_21 = B2 @ A1  # cross: B2 with A1's input projection

        if mod in calibration_inputs:
            X = calibration_inputs[mod].float()
            n = X.shape[0]

            # Compute effects of each candidate on calibration inputs
            # Y_target = combined effect we want to match (sum of individual effects)
            y_1 = (dw_11 @ X.T).T  # (n, d_out)
            y_2 = (dw_22 @ X.T).T
            y_12 = (dw_12 @ X.T).T
            y_21 = (dw_21 @ X.T).T

            Y_target = y_1 + y_2  # target: reproduce both adapters' effects

            # Stack candidates as features for ridge regression
            # For each output dimension, solve: Y_target[:,j] = c1*y_1[:,j] + c2*y_2[:,j] + c3*y_12[:,j] + c4*y_21[:,j]
            F = torch.stack([y_1, y_2, y_12, y_21], dim=2)  # (n, d_out, 4)
            d_out = F.shape[1]

            # Solve per output dimension (vectorized)
            # Reshape: F -> (n*d_out, 4), Y_target -> (n*d_out,)
            F_flat = F.reshape(-1, 4)
            Y_flat = Y_target.reshape(-1)

            # Ridge regression: c = (F^T F + λI)^{-1} F^T Y
            FtF = F_flat.T @ F_flat  # (4, 4)
            FtY = F_flat.T @ Y_flat  # (4,)
            coeffs = torch.linalg.solve(FtF + reg * torch.eye(4), FtY)
        else:
            # No calibration data: equal weight, no cross terms
            coeffs = torch.tensor([0.5, 0.5, 0.0, 0.0])

        c1, c2, c3, c4 = coeffs.tolist()

        # Compose merged delta_W
        dw_merged = c1 * dw_11 + c2 * dw_22 + c3 * dw_12 + c4 * dw_21

        # Truncate to target rank via SVD
        U, S, Vh = torch.linalg.svd(dw_merged, full_matrices=False)
        r = min(rank, U.shape[1])
        sqrt_S = torch.sqrt(S[:r].clamp(min=0))
        B_merged = U[:, :r] @ torch.diag(sqrt_S)
        A_merged = torch.diag(sqrt_S) @ Vh[:r, :]

        merged[mod] = (B_merged, A_merged)

        total_energy = float(S.sum())
        kept_energy = float(S[:r].sum())
        diagnostics[mod] = {
            "coefficients": [c1, c2, c3, c4],
            "cross_magnitude": abs(c3) + abs(c4),
            "diagonal_magnitude": abs(c1) + abs(c2),
            "energy_ratio": kept_energy / max(total_energy, 1e-10),
        }

    # Summary
    if diagnostics:
        mean_cross = sum(d["cross_magnitude"] for d in diagnostics.values()) / len(diagnostics)
        mean_diag = sum(d["diagonal_magnitude"] for d in diagnostics.values()) / len(diagnostics)
        mean_energy = sum(d["energy_ratio"] for d in diagnostics.values()) / len(diagnostics)
        mean_c = [
            sum(d["coefficients"][i] for d in diagnostics.values()) / len(diagnostics)
            for i in range(4)
        ]
        logger.info(
            "BCFF merged %d modules: mean_coeffs=[%.3f, %.3f, %.3f, %.3f], "
            "cross/diag=%.3f/%.3f, energy_retained=%.4f",
            len(merged), *mean_c, mean_cross, mean_diag, mean_energy
        )

    return merged, diagnostics


def save_bcff_adapter(
    merged: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    template_adapter_path: str,
    output_dir: str,
    dtype=torch.bfloat16,
):
    """Save BCFF-merged adapter in PEFT format."""
    from safetensors.torch import save_file
    import shutil

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    state = {}
    for mod, (B, A) in merged.items():
        peft_mod = "base_model.model." + mod
        state[peft_mod + ".lora_A.weight"] = A.to(dtype)
        state[peft_mod + ".lora_B.weight"] = B.to(dtype)

    save_file(state, str(out / "adapter_model.safetensors"))

    template = Path(template_adapter_path)
    cfg_src = template / "adapter_config.json"
    if cfg_src.exists():
        shutil.copy2(str(cfg_src), str(out / "adapter_config.json"))
