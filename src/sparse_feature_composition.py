"""
Sparse Feature Composition (SFC): Compose Features, Not Weights.

Core algorithm: decompose LoRA adapter effects into SAE feature space,
compose by feature-level max-pool, reconstruct composed adapter.

Two variants:
  - SFC-Exact: hook-based inference, applies composed features per layer
  - SFC-Merge: pre-compute composed weight update, deploy as standard LoRA
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from .sae_decomposition import (
    AdapterFeatureMap,
    FeatureProfile,
    SparseAutoencoder,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Disentanglement Score (FDS)
# ---------------------------------------------------------------------------

def compute_fds_pairwise(
    profile_a: FeatureProfile,
    profile_b: FeatureProfile,
) -> float:
    """Compute Feature Disentanglement Score between two adapter profiles at one layer.

    FDS = 1 - |S_a ∩ S_b| / |S_a ∪ S_b|  (Jaccard distance)

    Returns:
        FDS ∈ [0, 1]. 1 = fully disentangled (no overlap). 0 = identical supports.
    """
    set_a = set(profile_a.support.tolist())
    set_b = set(profile_b.support.tolist())

    if not set_a and not set_b:
        return 1.0  # both empty = no interference

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return 1.0 - intersection / union


def compute_fds(
    map_a: AdapterFeatureMap,
    map_b: AdapterFeatureMap,
    layer_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute Feature Disentanglement Score across all layers.

    Args:
        map_a, map_b: feature maps for two adapters
        layer_weights: optional per-layer weights (default: uniform)

    Returns:
        Dict with per-layer FDS, global FDS, overlap count, union count
    """
    common_layers = sorted(set(map_a.profiles.keys()) & set(map_b.profiles.keys()))

    per_layer = {}
    total_overlap = 0
    total_union = 0

    for layer in common_layers:
        fds = compute_fds_pairwise(map_a.profiles[layer], map_b.profiles[layer])
        per_layer[layer] = fds

        set_a = set(map_a.profiles[layer].support.tolist())
        set_b = set(map_b.profiles[layer].support.tolist())
        total_overlap += len(set_a & set_b)
        total_union += len(set_a | set_b)

    # Global FDS
    if layer_weights:
        w_sum = sum(layer_weights.get(l, 1.0) for l in common_layers)
        global_fds = sum(
            per_layer[l] * layer_weights.get(l, 1.0) for l in common_layers
        ) / max(w_sum, 1e-10)
    else:
        global_fds = 1.0 - total_overlap / max(total_union, 1)

    return {
        "global_fds": global_fds,
        "per_layer_fds": per_layer,
        "total_overlap": total_overlap,
        "total_union": total_union,
        "n_layers": len(common_layers),
    }


def compute_fds_matrix(
    feature_maps: List[AdapterFeatureMap],
) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise FDS matrix for N adapters.

    Returns:
        fds_matrix: (N, N) array of pairwise FDS values
        names: adapter names in order
    """
    N = len(feature_maps)
    matrix = np.ones((N, N))
    names = [m.adapter_name for m in feature_maps]

    for i in range(N):
        for j in range(i + 1, N):
            result = compute_fds(feature_maps[i], feature_maps[j])
            matrix[i, j] = result["global_fds"]
            matrix[j, i] = result["global_fds"]

    return matrix, names


# ---------------------------------------------------------------------------
# SFC Composition: Feature-level max-pool
# ---------------------------------------------------------------------------

@dataclass
class ComposedFeatureProfile:
    """Result of composing multiple adapter feature profiles at one layer."""
    layer_name: str
    support: torch.Tensor          # (|S_union|,) indices of composed features
    coefficients: torch.Tensor     # (|S_union|,) composed coefficients
    full_coefficients: torch.Tensor  # (n_features,) full composed vector
    source_adapters: Dict[int, List[str]]  # feature_idx → which adapters contributed
    overlap_indices: torch.Tensor   # features where multiple adapters overlap
    n_features: int


def compose_feature_profiles(
    profiles: List[FeatureProfile],
    weights: Optional[List[float]] = None,
) -> ComposedFeatureProfile:
    """Compose multiple adapter feature profiles at one layer via max-pool.

    For each feature:
    - If only one adapter modifies it: take that adapter's coefficient
    - If multiple adapters modify it: take the weighted max coefficient

    Args:
        profiles: feature profiles from different adapters (same layer)
        weights: optional per-adapter weights (default: uniform)

    Returns:
        ComposedFeatureProfile with the merged feature representation
    """
    if not profiles:
        raise ValueError("No profiles to compose")

    n_features = profiles[0].total_features
    N = len(profiles)
    if weights is None:
        weights = [1.0] * N

    # Build full coefficient matrix (N, n_features) — now signed
    coeff_matrix = torch.zeros(N, n_features)
    for i, prof in enumerate(profiles):
        coeff_matrix[i] = prof.full_coefficients * weights[i]

    # Compose: take coefficient with largest absolute magnitude (preserving sign)
    abs_matrix = coeff_matrix.abs()
    _, source_indices = abs_matrix.max(dim=0)
    composed = coeff_matrix.gather(0, source_indices.unsqueeze(0)).squeeze(0)

    # Determine support (non-zero composed features)
    active_mask = composed.abs() > 0
    support = active_mask.nonzero(as_tuple=False).squeeze(-1)
    coefficients = composed[support]

    # Track which adapters contributed to each feature
    source_adapters = {}
    overlap_list = []
    for idx in support.tolist():
        contributors = [
            profiles[i].adapter_name
            for i in range(N)
            if coeff_matrix[i, idx].abs() > 0
        ]
        source_adapters[idx] = contributors
        if len(contributors) > 1:
            overlap_list.append(idx)

    overlap_indices = torch.tensor(overlap_list, dtype=torch.long)

    return ComposedFeatureProfile(
        layer_name=profiles[0].layer_name,
        support=support,
        coefficients=coefficients,
        full_coefficients=composed,
        source_adapters=source_adapters,
        overlap_indices=overlap_indices,
        n_features=n_features,
    )


# ---------------------------------------------------------------------------
# SFC-Exact: Hook-based inference
# ---------------------------------------------------------------------------

class SFCExactHook:
    """Applies composed feature modifications during inference via hooks.

    For each layer with a SAE:
    1. Intercept the residual stream activation
    2. Encode through SAE to get features
    3. Apply composed feature modifications
    4. Decode back and replace the activation
    """

    def __init__(
        self,
        saes: Dict[str, SparseAutoencoder],
        composed_profiles: Dict[str, ComposedFeatureProfile],
    ):
        self.saes = saes
        self.composed_profiles = composed_profiles
        self.handles: List = []

    def _make_hook(self, layer_name: str):
        sae = self.saes[layer_name]
        composed = self.composed_profiles[layer_name]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
                rest = output[1:]
            else:
                act = output
                rest = None

            device = act.device
            dtype = act.dtype
            original_shape = act.shape

            # Flatten to (tokens, d_model)
            flat = act.reshape(-1, act.shape[-1]).float()

            # Encode base features
            sae_device = sae.W_enc.device
            flat_on_sae = flat.to(sae_device)
            f_base = sae.encode(flat_on_sae)

            # Apply composed modifications
            f_composed = f_base.clone()
            support = composed.support.to(sae_device)
            coeffs = composed.coefficients.to(sae_device)
            f_composed[:, support] = f_composed[:, support] + coeffs.unsqueeze(0)

            # Decode
            delta = sae.decode(f_composed) - sae.decode(f_base)
            delta = delta.to(device).to(dtype)

            # Apply
            modified = act + delta.reshape(original_shape)

            if rest is not None:
                return (modified,) + rest
            return modified

        return hook_fn

    def attach(self, model: nn.Module):
        """Attach SFC hooks to the model."""
        for name, module in model.named_modules():
            if name in self.composed_profiles:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)
                logger.info(f"Attached SFC hook to {name}")

    def detach(self):
        """Remove all hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# SFC-Merge: Reconstruct composed adapter as LoRA weights
# ---------------------------------------------------------------------------

def reconstruct_lora_from_features(
    base_activations: torch.Tensor,
    sae: SparseAutoencoder,
    composed_profile: ComposedFeatureProfile,
    rank: int = 16,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct a LoRA (B, A) factorization from composed feature profile.

    Method: compute the target activation modification δh for each probe input,
    then fit a rank-r ΔW via truncated SVD.

    Args:
        base_activations: (n_tokens, d_model) base activations at this layer
        sae: the SAE for this layer
        composed_profile: the composed feature profile
        rank: target LoRA rank
        device: computation device

    Returns:
        B: (d_model, rank) LoRA B matrix
        A: (rank, d_model) LoRA A matrix (operates on input activations)
    """
    sae = sae.to(device)
    n_tokens = base_activations.shape[0]
    d_model = base_activations.shape[1]

    # Compute target δh for each token
    chunk_size = 512
    all_delta_h = []

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        h_base = base_activations[start:end].to(device).float()

        # Current features
        f_base = sae.encode(h_base)

        # Add composed modifications
        f_mod = f_base.clone()
        support = composed_profile.support.to(device)
        coeffs = composed_profile.coefficients.to(device)
        f_mod[:, support] = f_mod[:, support] + coeffs.unsqueeze(0)

        # Target activation difference
        delta_h = sae.decode(f_mod) - sae.decode(f_base)
        all_delta_h.append(delta_h.cpu())

    delta_H = torch.cat(all_delta_h, dim=0)  # (n_tokens, d_model)

    # Fit ΔW via SVD of the expected outer product
    # ΔW ≈ δH · X^+ where X = base_activations
    # For rank-r: truncated SVD of δH
    # Since we want ΔW such that ΔW · x ≈ δh, and we observe (x, δh) pairs:
    # Direct approach: SVD of the correlation matrix
    X = base_activations.float()  # (n_tokens, d_model) — input activations
    Y = delta_H.float()           # (n_tokens, d_model) — target deltas

    # Least squares: ΔW = Y^T X (X^T X)^{-1}, but for rank-r, just SVD of Y
    # Since Y = ΔW · X, and X varies, we use SVD(Y) as the primary approximation
    # For the LoRA case where ΔW operates on the SAME residual stream:
    # δh = ΔW_out · x where x is the input to this layer
    # We approximate ΔW_out ≈ B · A via SVD of the mean effect

    # Compute mean feature modification vector (simpler, more stable)
    mean_delta_h = Y.mean(dim=0)  # (d_model,)

    # For a richer approximation: SVD of the delta matrix
    U, S, Vh = torch.linalg.svd(Y.T, full_matrices=False)  # Y.T is (d_model, n_tokens)

    r = min(rank, U.shape[1])
    sqrt_S = torch.sqrt(S[:r].clamp(min=0))

    B = U[:, :r] @ torch.diag(sqrt_S)   # (d_model, r)
    A = torch.diag(sqrt_S) @ Vh[:r, :]  # (r, n_tokens) — NOT what we want

    # Actually, we need A to operate on input activations, not token indices.
    # Use the proper least-squares solution: ΔW = (Y^T X) (X^T X)^{-1}
    # Then truncate ΔW to rank r.
    XtX = X.T @ X  # (d_model, d_model)
    YtX = Y.T @ X  # (d_model, d_model) — this IS ΔW (least squares)
    reg = 1e-6 * torch.eye(d_model)
    delta_W = YtX @ torch.linalg.solve(XtX + reg, torch.eye(d_model))

    # Truncated SVD of ΔW
    U_w, S_w, Vh_w = torch.linalg.svd(delta_W, full_matrices=False)
    r = min(rank, U_w.shape[1])
    sqrt_S_w = torch.sqrt(S_w[:r].clamp(min=0))

    B = U_w[:, :r] @ torch.diag(sqrt_S_w)   # (d_model, r)
    A = torch.diag(sqrt_S_w) @ Vh_w[:r, :]  # (r, d_model)

    return B, A


# ---------------------------------------------------------------------------
# Full SFC pipeline
# ---------------------------------------------------------------------------

@dataclass
class SFCResult:
    """Result of Sparse Feature Composition."""
    method: str  # "sfc_exact" or "sfc_merge"
    composed_profiles: Dict[str, ComposedFeatureProfile]
    fds_matrix: Optional[np.ndarray]
    adapter_names: List[str]
    sparsity_stats: Dict[str, float]
    # For SFC-Merge only:
    merged_lora_B: Optional[Dict[str, torch.Tensor]] = None
    merged_lora_A: Optional[Dict[str, torch.Tensor]] = None


def sfc_compose(
    feature_maps: List[AdapterFeatureMap],
    weights: Optional[List[float]] = None,
) -> Dict[str, ComposedFeatureProfile]:
    """Core SFC algorithm: compose multiple adapters in feature space.

    This is the ONE operation that replaces TIES/DARE/TA.

    Args:
        feature_maps: feature maps from multiple adapters
        weights: optional per-adapter weights

    Returns:
        Dict mapping layer_name → ComposedFeatureProfile
    """
    # Find common layers
    all_layers = [set(fm.profiles.keys()) for fm in feature_maps]
    common_layers = sorted(set.intersection(*all_layers))

    composed = {}
    for layer in common_layers:
        profiles = [fm.profiles[layer] for fm in feature_maps]
        composed[layer] = compose_feature_profiles(profiles, weights)

    # Log composition stats
    total_overlap = sum(c.overlap_indices.numel() for c in composed.values())
    total_support = sum(c.support.numel() for c in composed.values())
    logger.info(
        f"SFC composed {len(feature_maps)} adapters across {len(common_layers)} layers: "
        f"{total_support} active features, {total_overlap} overlapping "
        f"({total_overlap/max(total_support,1)*100:.1f}%)"
    )

    return composed


# ---------------------------------------------------------------------------
# Interference measurement
# ---------------------------------------------------------------------------

def compute_interference(
    profile_a: FeatureProfile,
    profile_b: FeatureProfile,
    sae: SparseAutoencoder,
) -> Dict[str, float]:
    """Compute the exact interference between two adapters at one layer.

    Interference = Σ_{k ∈ S_a ∩ S_b} ψ(c_a,k, c_b,k) · ||d_k||²

    where ψ(a, b) = (a + b - max(a, b))² = min(a, b)² for non-negative coefficients.

    Returns:
        Dict with total_interference, n_overlapping, interference_per_feature, etc.
    """
    set_a = set(profile_a.support.tolist())
    set_b = set(profile_b.support.tolist())
    overlap = set_a & set_b

    if not overlap:
        return {
            "total_interference": 0.0,
            "n_overlapping": 0,
            "max_interference_feature": -1,
            "overlap_fraction": 0.0,
        }

    overlap_idx = torch.tensor(sorted(overlap), dtype=torch.long)
    c_a = profile_a.full_coefficients[overlap_idx]
    c_b = profile_b.full_coefficients[overlap_idx]

    # For non-negative coefficients, SFC max-pool interference:
    # ψ(a, b) = a + b - max(a, b) = min(a, b)
    psi = torch.min(c_a, c_b)

    # Weight by decoder norm ||d_k||²
    d_norms_sq = (sae.W_dec[overlap_idx] ** 2).sum(dim=1)  # (|overlap|,)
    interference_per_feature = psi ** 2 * d_norms_sq

    total = float(interference_per_feature.sum())
    max_feat_idx = int(overlap_idx[interference_per_feature.argmax()])

    union_size = len(set_a | set_b)

    return {
        "total_interference": total,
        "n_overlapping": len(overlap),
        "max_interference_feature": max_feat_idx,
        "overlap_fraction": len(overlap) / max(union_size, 1),
        "per_feature_interference": {
            int(idx): float(val)
            for idx, val in zip(overlap_idx.tolist(), interference_per_feature.tolist())
        },
    }


# ---------------------------------------------------------------------------
# Comparison with weight-space methods
# ---------------------------------------------------------------------------

def sfc_vs_baselines_summary(
    feature_maps: List[AdapterFeatureMap],
) -> Dict[str, float]:
    """Quick summary stats for SFC composition of given adapters.

    Returns sparsity and FDS stats useful for comparison tables.
    """
    N = len(feature_maps)
    names = [fm.adapter_name for fm in feature_maps]

    # Sparsity stats
    sparsities = [fm.global_sparsity for fm in feature_maps]

    # Pairwise FDS
    fds_values = []
    for i in range(N):
        for j in range(i + 1, N):
            result = compute_fds(feature_maps[i], feature_maps[j])
            fds_values.append(result["global_fds"])

    return {
        "n_adapters": N,
        "adapter_names": names,
        "mean_sparsity": float(np.mean(sparsities)),
        "max_sparsity": float(np.max(sparsities)),
        "min_sparsity": float(np.min(sparsities)),
        "mean_pairwise_fds": float(np.mean(fds_values)) if fds_values else 1.0,
        "min_pairwise_fds": float(np.min(fds_values)) if fds_values else 1.0,
        "predicted_composition_quality": "high" if np.mean(fds_values) > 0.8 else
                                          "medium" if np.mean(fds_values) > 0.5 else "low",
    }
