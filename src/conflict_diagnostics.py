"""
Activation-Conditioned Conflict Diagnostics for LoRA Adapter Composition.

Computes per-layer, per-adapter-pair conflict metrics using activation-conditioned
Gram matrices:

    G_ij^l = E_h [<ΔW_i^l h, ΔW_j^l h>]

This captures functional conflict, not just parameter-space geometry. Two adapters
with identical left singular spaces but different right spaces will show different
conflict under different input distributions — which parameter-only CRS misses.
"""

import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


def compute_activation_gram(
    delta_ws: List[Dict[str, torch.Tensor]],
    module_inputs: Dict[str, torch.Tensor],
    max_samples: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    Compute activation-conditioned Gram matrix per module.

    G_ij^l = (1/N) Σ_n (ΔW_i^l h_n)^T (ΔW_j^l h_n)

    Args:
        delta_ws: list of per-adapter delta_W dicts {module_name: (d_out, d_in)}
        module_inputs: {module_name: (N, d_in)} calibration activations

    Returns:
        {module_name: (n_adapters, n_adapters) Gram matrix}
    """
    n_adapters = len(delta_ws)
    grams = {}

    common_modules = sorted(set.intersection(*[set(d.keys()) for d in delta_ws]))

    for mod in common_modules:
        if mod not in module_inputs:
            continue

        H = module_inputs[mod].float()[:max_samples]
        N = H.shape[0]

        effects = []
        for i in range(n_adapters):
            dw_i = delta_ws[i][mod].float()
            e_i = H @ dw_i.T
            effects.append(e_i)

        G = torch.zeros(n_adapters, n_adapters)
        for i in range(n_adapters):
            for j in range(i, n_adapters):
                g_ij = (effects[i] * effects[j]).sum() / max(N, 1)
                G[i, j] = g_ij
                G[j, i] = g_ij

        grams[mod] = G

    return grams


def compute_pair_conflict(
    gram: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute conflict metrics from a 2×2 Gram matrix.

    Args:
        gram: (2, 2) Gram matrix for adapter pair

    Returns:
        Dict with cosine similarity, conflict score, energy ratio
    """
    g_ii = gram[0, 0].item()
    g_jj = gram[1, 1].item()
    g_ij = gram[0, 1].item()

    norm_i = max(g_ii ** 0.5, 1e-10)
    norm_j = max(g_jj ** 0.5, 1e-10)

    cosine = g_ij / (norm_i * norm_j)

    interference_energy = abs(g_ij)
    total_energy = g_ii + g_jj

    return {
        "cosine_similarity": float(cosine),
        "interference_energy": float(interference_energy),
        "total_energy": float(total_energy),
        "interference_ratio": float(interference_energy / max(total_energy, 1e-10)),
        "adapter_0_energy": float(g_ii),
        "adapter_1_energy": float(g_jj),
    }


def compute_conflict_projector(
    delta_ws: List[Dict[str, torch.Tensor]],
    module_inputs: Dict[str, torch.Tensor],
    conflict_threshold: float = 0.3,
    max_rank: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Compute projectors onto conflict directions per module.

    For each module, identifies directions where adapter effects overlap significantly,
    and returns a projector P_conflict such that P_conflict @ h extracts the
    conflicting component.

    Args:
        delta_ws: per-adapter delta_W dicts
        module_inputs: calibration activations
        conflict_threshold: cosine threshold for "conflicting" direction
        max_rank: maximum rank of conflict projector

    Returns:
        {module_name: (d_out, d_out) projector matrix}
    """
    n_adapters = len(delta_ws)
    common_modules = sorted(set.intersection(*[set(d.keys()) for d in delta_ws]))
    projectors = {}

    for mod in common_modules:
        if mod not in module_inputs:
            continue

        H = module_inputs[mod].float()[:2048]

        effect_dirs = []
        for i in range(n_adapters):
            dw_i = delta_ws[i][mod].float()
            U, S, _ = torch.linalg.svd(dw_i, full_matrices=False)
            top_dirs = U[:, :max_rank]
            effect_dirs.append(top_dirs)

        if len(effect_dirs) < 2:
            continue

        all_dirs = torch.cat(effect_dirs, dim=1)
        cosines = all_dirs.T @ all_dirs
        n_per = effect_dirs[0].shape[1]

        conflict_mask = torch.zeros(all_dirs.shape[1], dtype=torch.bool)
        for i in range(n_adapters):
            for j in range(i + 1, n_adapters):
                block = cosines[i*n_per:(i+1)*n_per, j*n_per:(j+1)*n_per]
                high_overlap = block.abs() > conflict_threshold
                if high_overlap.any():
                    rows, cols = high_overlap.nonzero(as_tuple=True)
                    for r in rows:
                        conflict_mask[i*n_per + r] = True
                    for c in cols:
                        conflict_mask[j*n_per + c] = True

        if conflict_mask.sum() == 0:
            continue

        conflict_dirs = all_dirs[:, conflict_mask]
        U_c, _, _ = torch.linalg.svd(conflict_dirs, full_matrices=False)
        r = min(max_rank, U_c.shape[1])
        P = U_c[:, :r] @ U_c[:, :r].T
        projectors[mod] = P

    return projectors


def compute_adapter_reliability(
    base_logprobs: torch.Tensor,
    adapter_logprobs: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute adapter reliability from logprob comparison.

    Args:
        base_logprobs: (N,) per-token or per-sample logprobs from base model
        adapter_logprobs: (N,) per-token or per-sample logprobs from adapter model

    Returns:
        Dict with reliability metrics
    """
    improvement = adapter_logprobs - base_logprobs
    n_better = (improvement > 0).sum().item()
    n_total = improvement.numel()

    return {
        "mean_improvement": float(improvement.mean()),
        "fraction_better": float(n_better / max(n_total, 1)),
        "std_improvement": float(improvement.std()),
        "reliability_score": float(n_better / max(n_total, 1)),
    }
