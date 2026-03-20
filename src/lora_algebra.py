"""
LoRA Algebra: algebraic operations on LoRA weight spaces with Grassmann manifold mapping.

Supports compose, subtract, interpolate, project, and comparison with TIES/DARE/Task Arithmetic.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel, get_peft_model_state_dict

logger = logging.getLogger(__name__)


@dataclass
class LoRAWeights:
    """Container for a single LoRA adapter's A and B matrices keyed by layer."""
    name: str
    lora_A: Dict[str, torch.Tensor] = field(default_factory=dict)
    lora_B: Dict[str, torch.Tensor] = field(default_factory=dict)
    rank: int = 0
    alpha: float = 1.0

    @classmethod
    def from_state_dict(cls, name: str, state_dict: Dict[str, torch.Tensor], alpha: float = 1.0) -> "LoRAWeights":
        lora_A, lora_B = {}, {}
        for key, val in state_dict.items():
            if "lora_A" in key:
                layer_key = key.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
                lora_A[layer_key] = val.clone().float()
            elif "lora_B" in key:
                layer_key = key.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
                lora_B[layer_key] = val.clone().float()
        rank = next(iter(lora_A.values())).shape[0] if lora_A else 0
        return cls(name=name, lora_A=lora_A, lora_B=lora_B, rank=rank, alpha=alpha)

    @classmethod
    def from_peft_dir(cls, name: str, peft_dir: str, device: str = "cpu") -> "LoRAWeights":
        import json, os, safetensors.torch
        adapter_path = os.path.join(peft_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            state_dict = safetensors.torch.load_file(adapter_path, device=device)
        else:
            bin_path = os.path.join(peft_dir, "adapter_model.bin")
            state_dict = torch.load(bin_path, map_location=device)
        config_path = os.path.join(peft_dir, "adapter_config.json")
        alpha = 1.0
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            alpha = cfg.get("lora_alpha", 1.0) / cfg.get("r", 1)
        return cls.from_state_dict(name, state_dict, alpha=alpha)

    def to_delta_weight(self) -> Dict[str, torch.Tensor]:
        """Compute delta_W = B @ A * scaling for each layer."""
        deltas = {}
        for key in self.lora_A:
            if key in self.lora_B:
                deltas[key] = (self.lora_B[key] @ self.lora_A[key]) * self.alpha
        return deltas

    def to_state_dict(self, prefix: str = "base_model.model.") -> Dict[str, torch.Tensor]:
        sd = {}
        for key in self.lora_A:
            sd[f"{prefix}{key}.lora_A.weight"] = self.lora_A[key]
        for key in self.lora_B:
            sd[f"{prefix}{key}.lora_B.weight"] = self.lora_B[key]
        return sd


class GrassmannProjector:
    """Map LoRA weight deltas to points on the Grassmann manifold G(r, d)."""

    def __init__(self, svd_rank: int = 32):
        self.svd_rank = svd_rank

    def to_grassmann(self, delta_W: torch.Tensor) -> torch.Tensor:
        """Project delta_W onto Grassmann manifold via truncated SVD → orthonormal basis."""
        U, S, Vh = torch.linalg.svd(delta_W.float(), full_matrices=False)
        k = min(self.svd_rank, U.shape[1])
        return U[:, :k]

    def grassmann_distance(self, U1: torch.Tensor, U2: torch.Tensor) -> float:
        """Geodesic distance on Grassmann manifold via principal angles."""
        M = U1.T @ U2
        _, sigmas, _ = torch.linalg.svd(M)
        sigmas = torch.clamp(sigmas, -1.0, 1.0)
        angles = torch.acos(sigmas)
        return torch.norm(angles).item()

    def grassmann_mean(self, bases: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
        """Karcher/Fréchet mean on the Grassmann manifold (iterative)."""
        if weights is None:
            weights = [1.0 / len(bases)] * len(bases)
        mu = bases[0].clone()
        for _ in range(20):
            tangent_sum = torch.zeros_like(mu)
            for U, w in zip(bases, weights):
                tangent_sum += w * self._log_map(mu, U)
            mu = self._exp_map(mu, tangent_sum)
        return mu

    def _log_map(self, U_base: torch.Tensor, U_target: torch.Tensor) -> torch.Tensor:
        M = U_base.T @ U_target
        W = U_target - U_base @ M
        Q, R = torch.linalg.qr(W)
        Usvd, Ssvd, Vhsvd = torch.linalg.svd(torch.cat([M, R], dim=0), full_matrices=False)
        k = U_base.shape[1]
        angles = torch.atan2(Ssvd[:k], torch.ones_like(Ssvd[:k]))
        return Q @ (Vhsvd[:k, :].T * angles.unsqueeze(0))

    def _exp_map(self, U_base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        Q, R = torch.linalg.qr(tangent)
        Usvd, Ssvd, Vhsvd = torch.linalg.svd(R, full_matrices=False)
        cos_S = torch.diag(torch.cos(Ssvd))
        sin_S = torch.diag(torch.sin(Ssvd))
        result = U_base @ Vhsvd.T @ cos_S + Q @ Usvd @ sin_S
        new_U, _, _ = torch.linalg.svd(result, full_matrices=False)
        return new_U[:, :U_base.shape[1]]


class LoRAAlgebra:
    """Core algebraic operations on LoRA weight spaces."""

    def __init__(self, grassmann_rank: int = 32):
        self.projector = GrassmannProjector(svd_rank=grassmann_rank)

    @staticmethod
    def compose(lora_a: LoRAWeights, lora_b: LoRAWeights, name: str = "composed") -> LoRAWeights:
        """A + B: compose two LoRAs by adding their delta weights, then re-factorize."""
        delta_a = lora_a.to_delta_weight()
        delta_b = lora_b.to_delta_weight()
        rank = max(lora_a.rank, lora_b.rank)
        new_A, new_B = {}, {}
        for key in set(delta_a.keys()) | set(delta_b.keys()):
            d_a = delta_a.get(key, torch.zeros_like(next(iter(delta_a.values()))))
            d_b = delta_b.get(key, torch.zeros_like(d_a))
            combined = d_a + d_b
            U, S, Vh = torch.linalg.svd(combined.float(), full_matrices=False)
            new_A[key] = (torch.diag(torch.sqrt(S[:rank])) @ Vh[:rank, :])
            new_B[key] = (U[:, :rank] @ torch.diag(torch.sqrt(S[:rank])))
        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)

    @staticmethod
    def subtract(lora_a: LoRAWeights, lora_b: LoRAWeights, name: str = "subtracted") -> LoRAWeights:
        """A - B: subtract delta weights, then re-factorize."""
        delta_a = lora_a.to_delta_weight()
        delta_b = lora_b.to_delta_weight()
        rank = max(lora_a.rank, lora_b.rank)
        new_A, new_B = {}, {}
        for key in set(delta_a.keys()) | set(delta_b.keys()):
            d_a = delta_a.get(key, torch.zeros_like(next(iter(delta_a.values()))))
            d_b = delta_b.get(key, torch.zeros_like(d_a))
            diff = d_a - d_b
            U, S, Vh = torch.linalg.svd(diff.float(), full_matrices=False)
            new_A[key] = (torch.diag(torch.sqrt(S[:rank])) @ Vh[:rank, :])
            new_B[key] = (U[:, :rank] @ torch.diag(torch.sqrt(S[:rank])))
        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)

    @staticmethod
    def interpolate(
        lora_a: LoRAWeights, lora_b: LoRAWeights, alpha: float = 0.5, name: str = "interpolated"
    ) -> LoRAWeights:
        """alpha * A + (1 - alpha) * B: linear interpolation in delta-weight space."""
        delta_a = lora_a.to_delta_weight()
        delta_b = lora_b.to_delta_weight()
        rank = max(lora_a.rank, lora_b.rank)
        new_A, new_B = {}, {}
        for key in set(delta_a.keys()) | set(delta_b.keys()):
            d_a = delta_a.get(key, torch.zeros_like(next(iter(delta_a.values()))))
            d_b = delta_b.get(key, torch.zeros_like(d_a))
            interp = alpha * d_a + (1.0 - alpha) * d_b
            U, S, Vh = torch.linalg.svd(interp.float(), full_matrices=False)
            new_A[key] = (torch.diag(torch.sqrt(S[:rank])) @ Vh[:rank, :])
            new_B[key] = (U[:, :rank] @ torch.diag(torch.sqrt(S[:rank])))
        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)

    def project_onto_subspace(
        self, lora: LoRAWeights, basis_loras: List[LoRAWeights], name: str = "projected"
    ) -> LoRAWeights:
        """Project lora onto the subspace spanned by basis_loras on the Grassmann manifold."""
        delta = lora.to_delta_weight()
        basis_deltas = [b.to_delta_weight() for b in basis_loras]
        rank = lora.rank
        new_A, new_B = {}, {}
        for key in delta:
            target_basis = self.projector.to_grassmann(delta[key])
            component_bases = [self.projector.to_grassmann(bd[key]) for bd in basis_deltas if key in bd]
            if not component_bases:
                new_A[key] = lora.lora_A[key].clone()
                new_B[key] = lora.lora_B[key].clone()
                continue
            all_basis = torch.cat(component_bases, dim=1)
            Q, _ = torch.linalg.qr(all_basis)
            projected = Q @ (Q.T @ delta[key])
            U, S, Vh = torch.linalg.svd(projected.float(), full_matrices=False)
            new_A[key] = (torch.diag(torch.sqrt(S[:rank])) @ Vh[:rank, :])
            new_B[key] = (U[:, :rank] @ torch.diag(torch.sqrt(S[:rank])))
        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)

    def grassmann_interpolate(
        self, lora_a: LoRAWeights, lora_b: LoRAWeights, t: float = 0.5, name: str = "grassmann_interp"
    ) -> LoRAWeights:
        """Geodesic interpolation on Grassmann manifold (more principled than linear)."""
        delta_a = lora_a.to_delta_weight()
        delta_b = lora_b.to_delta_weight()
        rank = max(lora_a.rank, lora_b.rank)
        new_A, new_B = {}, {}
        for key in set(delta_a.keys()) & set(delta_b.keys()):
            U_a = self.projector.to_grassmann(delta_a[key])
            U_b = self.projector.to_grassmann(delta_b[key])
            log_vec = self.projector._log_map(U_a, U_b)
            U_interp = self.projector._exp_map(U_a, t * log_vec)
            scale_a = torch.norm(delta_a[key], "fro")
            scale_b = torch.norm(delta_b[key], "fro")
            scale = (1 - t) * scale_a + t * scale_b
            S_diag = torch.ones(min(rank, U_interp.shape[1])) * (scale / np.sqrt(rank))
            Vh = torch.eye(delta_a[key].shape[1])[:rank, :]
            new_B[key] = U_interp[:, :rank] @ torch.diag(torch.sqrt(S_diag))
            new_A[key] = torch.diag(torch.sqrt(S_diag)) @ Vh
        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


class MergingBaselines:
    """Baselines: TIES, DARE, Task Arithmetic for comparison."""

    @staticmethod
    def task_arithmetic(loras: List[LoRAWeights], scaling: float = 1.0) -> Dict[str, torch.Tensor]:
        """Task Arithmetic: simple sum of task vectors."""
        all_deltas = [lora.to_delta_weight() for lora in loras]
        all_keys = set()
        for d in all_deltas:
            all_keys |= set(d.keys())
        merged = {}
        for key in all_keys:
            stack = [d[key] for d in all_deltas if key in d]
            merged[key] = scaling * torch.stack(stack).sum(dim=0)
        return merged

    @staticmethod
    def ties_merging(
        loras: List[LoRAWeights], density: float = 0.5, scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """TIES: Trim, Elect Sign, Disjoint Merge."""
        all_deltas = [lora.to_delta_weight() for lora in loras]
        all_keys = set()
        for d in all_deltas:
            all_keys |= set(d.keys())
        merged = {}
        for key in all_keys:
            stack = torch.stack([d[key] for d in all_deltas if key in d])
            # Trim: keep top-k% by magnitude per task
            k = int(density * stack[0].numel())
            trimmed = stack.clone()
            for i in range(trimmed.shape[0]):
                flat = trimmed[i].abs().flatten()
                if k < flat.numel():
                    threshold = flat.topk(k).values[-1]
                    trimmed[i][trimmed[i].abs() < threshold] = 0.0
            # Elect sign: majority vote
            sign_votes = torch.sign(trimmed).sum(dim=0)
            elected_sign = torch.sign(sign_votes)
            elected_sign[elected_sign == 0] = 1.0
            # Disjoint merge: average only agreeing values
            mask = torch.sign(trimmed) == elected_sign.unsqueeze(0)
            trimmed[~mask] = 0.0
            counts = mask.float().sum(dim=0).clamp(min=1.0)
            merged[key] = scaling * trimmed.sum(dim=0) / counts
        return merged

    @staticmethod
    def dare_merging(
        loras: List[LoRAWeights], drop_rate: float = 0.5, scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """DARE: Drop And REscale."""
        all_deltas = [lora.to_delta_weight() for lora in loras]
        all_keys = set()
        for d in all_deltas:
            all_keys |= set(d.keys())
        merged = {}
        for key in all_keys:
            stack = torch.stack([d[key] for d in all_deltas if key in d])
            # Random drop + rescale per task vector
            mask = torch.bernoulli(torch.full_like(stack, 1.0 - drop_rate))
            rescaled = stack * mask / (1.0 - drop_rate)
            merged[key] = scaling * rescaled.mean(dim=0)
        return merged


def compute_similarity_matrix(loras: List[LoRAWeights], projector: GrassmannProjector) -> np.ndarray:
    """Compute pairwise Grassmann distance matrix between LoRAs."""
    n = len(loras)
    dist_matrix = np.zeros((n, n))
    deltas = [lora.to_delta_weight() for lora in loras]
    common_keys = set(deltas[0].keys())
    for d in deltas[1:]:
        common_keys &= set(d.keys())
    first_key = sorted(common_keys)[0]
    for i in range(n):
        U_i = projector.to_grassmann(deltas[i][first_key])
        for j in range(i + 1, n):
            U_j = projector.to_grassmann(deltas[j][first_key])
            dist = projector.grassmann_distance(U_i, U_j)
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    return dist_matrix
