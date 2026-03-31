"""
Grassmannian Composition: geometry-aware merging of LoRA adapters on fixed-rank manifolds.

Core algorithm: GrassMerge — compose LoRA adapters by averaging subspaces on the Grassmann
manifold G(r, d_out) × G(r, d_in) with projected-core spectral interpolation.

Also provides baselines (TIES, DARE, Task Arithmetic) and legacy LoRA algebra operations.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class LoRAWeights:
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
                if layer_key.startswith("base_model.model."):
                    layer_key = layer_key[len("base_model.model."):]
                lora_A[layer_key] = val.clone().float()
            elif "lora_B" in key:
                layer_key = key.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
                if layer_key.startswith("base_model.model."):
                    layer_key = layer_key[len("base_model.model."):]
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
            state_dict = torch.load(bin_path, map_location=device, weights_only=True)
        config_path = os.path.join(peft_dir, "adapter_config.json")
        alpha = 1.0
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            alpha = cfg.get("lora_alpha", 1.0) / cfg.get("r", 1)
        return cls.from_state_dict(name, state_dict, alpha=alpha)

    def to_delta_weight(self) -> Dict[str, torch.Tensor]:
        deltas = {}
        for key in self.lora_A:
            if key in self.lora_B:
                deltas[key] = (self.lora_B[key] @ self.lora_A[key]) * self.alpha
        return deltas

    def to_state_dict(self, prefix: str = "base_model.model.") -> Dict[str, torch.Tensor]:
        sd = {}
        for key in self.lora_A:
            clean_key = key[len(prefix):] if key.startswith(prefix) else key
            sd[f"{prefix}{clean_key}.lora_A.weight"] = self.lora_A[key]
        for key in self.lora_B:
            clean_key = key[len(prefix):] if key.startswith(prefix) else key
            sd[f"{prefix}{clean_key}.lora_B.weight"] = self.lora_B[key]
        return sd

    def save_peft_dir(
        self,
        peft_dir: str,
        base_model_name: str = "",
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.0,
        task_type: str = "CAUSAL_LM",
    ) -> None:
        """Write a PEFT-compatible adapter directory (adapter_model.safetensors + adapter_config.json)."""
        import json
        import os
        import safetensors.torch
        os.makedirs(peft_dir, exist_ok=True)
        sd = self.to_state_dict()
        safetensors.torch.save_file(sd, os.path.join(peft_dir, "adapter_model.safetensors"))
        cfg = {
            "peft_type": "LORA",
            "base_model_name_or_path": base_model_name,
            "r": self.rank,
            "lora_alpha": self.rank,
            "target_modules": target_modules or [],
            "lora_dropout": lora_dropout,
            "bias": "none",
            "task_type": task_type,
        }
        with open(os.path.join(peft_dir, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)


class GrassmannOps:
    """Grassmann manifold G(r, d) operations: log map, exp map, distance, Karcher mean."""

    @staticmethod
    def log_map(U_base: torch.Tensor, U_target: torch.Tensor) -> torch.Tensor:
        M = U_base.T @ U_target
        W = U_target - U_base @ M
        Q, R = torch.linalg.qr(W)
        k = U_base.shape[1]
        stacked = torch.cat([M, R], dim=0)
        Usvd, Ssvd, Vhsvd = torch.linalg.svd(stacked, full_matrices=False)
        angles = torch.atan2(Ssvd[:k], torch.ones_like(Ssvd[:k]))
        return Q @ (Vhsvd[:k, :].T * angles.unsqueeze(0))

    @staticmethod
    def exp_map(U_base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        Q, R = torch.linalg.qr(tangent)
        Usvd, Ssvd, Vhsvd = torch.linalg.svd(R, full_matrices=False)
        cos_S = torch.diag(torch.cos(Ssvd))
        sin_S = torch.diag(torch.sin(Ssvd))
        result = U_base @ Vhsvd.T @ cos_S + Q @ Usvd @ sin_S
        new_U, _, _ = torch.linalg.svd(result, full_matrices=False)
        return new_U[:, :U_base.shape[1]]

    @staticmethod
    def geodesic_distance(U1: torch.Tensor, U2: torch.Tensor) -> float:
        M = U1.T @ U2
        _, sigmas, _ = torch.linalg.svd(M)
        sigmas = torch.clamp(sigmas, -1.0, 1.0)
        angles = torch.acos(sigmas)
        return torch.norm(angles).item()

    @staticmethod
    def principal_angles(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
        M = U1.T @ U2
        _, sigmas, _ = torch.linalg.svd(M)
        sigmas = torch.clamp(sigmas, -1.0, 1.0)
        return torch.acos(sigmas)

    @classmethod
    def karcher_mean(
        cls,
        bases: List[torch.Tensor],
        weights: Optional[List[float]] = None,
        max_iter: int = 20,
        tol: float = 1e-7,
    ) -> torch.Tensor:
        if weights is None:
            weights = [1.0 / len(bases)] * len(bases)
        mu = bases[0].clone()
        for _ in range(max_iter):
            tangent_sum = torch.zeros_like(mu)
            for U, w in zip(bases, weights):
                tangent_sum += w * cls.log_map(mu, U)
            tangent_norm = torch.norm(tangent_sum).item()
            if tangent_norm < tol:
                break
            mu = cls.exp_map(mu, tangent_sum)
        return mu


def bilateral_grassmann_distance(
    delta_i: torch.Tensor, delta_j: torch.Tensor, rank: int
) -> float:
    U_i, _, Vh_i = torch.linalg.svd(delta_i.float(), full_matrices=False)
    U_j, _, Vh_j = torch.linalg.svd(delta_j.float(), full_matrices=False)
    r = min(rank, U_i.shape[1], U_j.shape[1])
    d_left = GrassmannOps.geodesic_distance(U_i[:, :r], U_j[:, :r])
    d_right = GrassmannOps.geodesic_distance(Vh_i[:r, :].T, Vh_j[:r, :].T)
    return float(np.sqrt(d_left**2 + d_right**2))


def spectral_weighted_bgd(
    delta_i: torch.Tensor, delta_j: torch.Tensor, rank: int
) -> float:
    """BGD variant that weights subspace distances by spectral importance."""
    U_i, S_i, Vh_i = torch.linalg.svd(delta_i.float(), full_matrices=False)
    U_j, S_j, Vh_j = torch.linalg.svd(delta_j.float(), full_matrices=False)
    r = min(rank, U_i.shape[1], U_j.shape[1])
    angles_left = GrassmannOps.principal_angles(U_i[:, :r], U_j[:, :r])
    angles_right = GrassmannOps.principal_angles(Vh_i[:r, :].T, Vh_j[:r, :].T)
    w_i = S_i[:r] / (S_i[:r].sum() + 1e-8)
    w_j = S_j[:r] / (S_j[:r].sum() + 1e-8)
    w = (w_i + w_j) / 2.0
    d_left = float(torch.sqrt((w * angles_left ** 2).sum()).item())
    d_right = float(torch.sqrt((w * angles_right ** 2).sum()).item())
    return float(np.sqrt(d_left**2 + d_right**2))


def cosine_interference(
    delta_i: torch.Tensor, delta_j: torch.Tensor
) -> float:
    """Simple cosine-based interference metric (baseline for BGD comparison)."""
    flat_i = delta_i.flatten().float()
    flat_j = delta_j.flatten().float()
    cos = torch.nn.functional.cosine_similarity(
        flat_i.unsqueeze(0), flat_j.unsqueeze(0)
    ).item()
    return 1.0 - cos


def frobenius_interference(
    delta_i: torch.Tensor, delta_j: torch.Tensor
) -> float:
    """Frobenius-distance-based interference metric (baseline for BGD comparison)."""
    norm_i = torch.norm(delta_i.float(), "fro").item()
    norm_j = torch.norm(delta_j.float(), "fro").item()
    norm_diff = torch.norm(delta_i.float() - delta_j.float(), "fro").item()
    return norm_diff / max(norm_i + norm_j, 1e-8)


class GrassMerge:
    """
    Grassmannian Composition of LoRA adapters.

    For each layer: SVD → Karcher mean on G(r, d_out) and G(r, d_in) →
    projected-core averaging → reconstruct → refactorize to PEFT format.
    """

    def __init__(self, karcher_max_iter: int = 20, karcher_tol: float = 1e-7):
        self.karcher_max_iter = karcher_max_iter
        self.karcher_tol = karcher_tol

    def merge(
        self,
        loras: List[LoRAWeights],
        weights: Optional[List[float]] = None,
        name: str = "grassmerge",
    ) -> LoRAWeights:
        N = len(loras)
        if N < 2:
            raise ValueError(f"Need >= 2 adapters, got {N}")
        if weights is None:
            weights = [1.0 / N] * N
        assert abs(sum(weights) - 1.0) < 1e-6, f"Weights must sum to 1, got {sum(weights)}"

        rank = max(lora.rank for lora in loras)
        all_deltas = [lora.to_delta_weight() for lora in loras]
        common_keys = set(all_deltas[0].keys())
        for d in all_deltas[1:]:
            common_keys &= set(d.keys())

        new_A, new_B = {}, {}
        for key in sorted(common_keys):
            merged_A, merged_B = self._merge_layer(
                [d[key] for d in all_deltas], weights, rank
            )
            new_A[key] = merged_A
            new_B[key] = merged_B

        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)

    def _merge_layer(
        self,
        deltas: List[torch.Tensor],
        weights: List[float],
        rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = len(deltas)

        Us, Ss, Vs = [], [], []
        for dW in deltas:
            U, S, Vh = torch.linalg.svd(dW.float(), full_matrices=False)
            r = min(rank, U.shape[1])
            Us.append(U[:, :r])
            Ss.append(S[:r])
            Vs.append(Vh[:r, :].T)

        r = min(rank, Us[0].shape[1])
        for i in range(len(Us)):
            if Us[i].shape[1] < r:
                pad = r - Us[i].shape[1]
                Us[i] = torch.cat([Us[i], torch.zeros(Us[i].shape[0], pad, device=Us[i].device)], dim=1)
                Vs[i] = torch.cat([Vs[i], torch.zeros(Vs[i].shape[0], pad, device=Vs[i].device)], dim=1)
                Ss[i] = torch.cat([Ss[i], torch.zeros(pad, device=Ss[i].device)])

        U_star = GrassmannOps.karcher_mean(Us, weights, self.karcher_max_iter, self.karcher_tol)
        V_star = GrassmannOps.karcher_mean(Vs, weights, self.karcher_max_iter, self.karcher_tol)

        S_avg = torch.zeros(r, r, device=deltas[0].device)
        for i in range(N):
            S_i = U_star.T @ deltas[i].float() @ V_star
            S_avg += weights[i] * S_i

        U_core, sigma_star, Vh_core = torch.linalg.svd(S_avg, full_matrices=False)
        U_final = U_star @ U_core
        V_final = V_star @ Vh_core.T

        sqrt_sigma = torch.sqrt(sigma_star.clamp(min=0))
        B_merged = U_final @ torch.diag(sqrt_sigma)
        A_merged = torch.diag(sqrt_sigma) @ V_final.T

        return A_merged, B_merged

    def compute_bgd_matrix(self, loras: List[LoRAWeights]) -> np.ndarray:
        N = len(loras)
        bgd_matrix = np.zeros((N, N))
        all_deltas = [lora.to_delta_weight() for lora in loras]
        common_keys = set(all_deltas[0].keys())
        for d in all_deltas[1:]:
            common_keys &= set(d.keys())

        rank = max(lora.rank for lora in loras)
        all_keys = sorted(common_keys)

        for i in range(N):
            for j in range(i + 1, N):
                bgd_sum = 0.0
                for key in all_keys:
                    bgd_sum += bilateral_grassmann_distance(
                        all_deltas[i][key], all_deltas[j][key], rank
                    )
                bgd_avg = bgd_sum / len(all_keys)
                bgd_matrix[i][j] = bgd_avg
                bgd_matrix[j][i] = bgd_avg

        return bgd_matrix


class SVDProcrustesMerge:
    """Baseline: SVD-Procrustes alignment then averaging."""

    @staticmethod
    def merge(loras: List[LoRAWeights], weights: Optional[List[float]] = None, name: str = "procrustes") -> LoRAWeights:
        N = len(loras)
        if weights is None:
            weights = [1.0 / N] * N
        rank = max(lora.rank for lora in loras)
        all_deltas = [lora.to_delta_weight() for lora in loras]
        common_keys = set(all_deltas[0].keys())
        for d in all_deltas[1:]:
            common_keys &= set(d.keys())

        new_A, new_B = {}, {}
        for key in sorted(common_keys):
            ref = all_deltas[0][key].float()
            U_ref, S_ref, Vh_ref = torch.linalg.svd(ref, full_matrices=False)
            r = min(rank, U_ref.shape[1])

            aligned_sum = weights[0] * ref
            for i in range(1, N):
                dW = all_deltas[i][key].float()
                U_i, S_i, Vh_i = torch.linalg.svd(dW, full_matrices=False)
                M_left = U_ref[:, :r].T @ U_i[:, :r]
                Ul, _, Vhl = torch.linalg.svd(M_left)
                R_left = Vhl.T @ Ul.T

                M_right = Vh_ref[:r, :] @ Vh_i[:r, :].T
                Ur, _, Vhr = torch.linalg.svd(M_right)
                R_right = Vhr.T @ Ur.T

                aligned = (U_i[:, :r] @ R_left) @ torch.diag(S_i[:r]) @ (R_right @ Vh_i[:r, :])
                aligned_sum += weights[i] * aligned

            U, S, Vh = torch.linalg.svd(aligned_sum, full_matrices=False)
            new_B[key] = U[:, :r] @ torch.diag(torch.sqrt(S[:r]))
            new_A[key] = torch.diag(torch.sqrt(S[:r])) @ Vh[:r, :]

        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


class ColumnOnlyGrassmannMerge:
    """Ablation baseline: Grassmann merge using only the left subspace (column-only)."""

    def __init__(self, karcher_max_iter: int = 20, karcher_tol: float = 1e-7):
        self.karcher_max_iter = karcher_max_iter
        self.karcher_tol = karcher_tol

    def merge(self, loras: List[LoRAWeights], weights: Optional[List[float]] = None, name: str = "col_grassmann") -> LoRAWeights:
        N = len(loras)
        if weights is None:
            weights = [1.0 / N] * N
        rank = max(lora.rank for lora in loras)
        all_deltas = [lora.to_delta_weight() for lora in loras]
        common_keys = set(all_deltas[0].keys())
        for d in all_deltas[1:]:
            common_keys &= set(d.keys())

        new_A, new_B = {}, {}
        for key in sorted(common_keys):
            Us = []
            for dW in [d[key] for d in all_deltas]:
                U, S, Vh = torch.linalg.svd(dW.float(), full_matrices=False)
                r = min(rank, U.shape[1])
                Us.append(U[:, :r])

            U_star = GrassmannOps.karcher_mean(Us, weights, self.karcher_max_iter, self.karcher_tol)

            avg_delta = sum(w * d[key].float() for w, d in zip(weights, all_deltas))
            projected = U_star @ (U_star.T @ avg_delta)
            U, S, Vh = torch.linalg.svd(projected, full_matrices=False)
            r = min(rank, U.shape[1])
            new_B[key] = U[:, :r] @ torch.diag(torch.sqrt(S[:r]))
            new_A[key] = torch.diag(torch.sqrt(S[:r])) @ Vh[:r, :]

        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


class KnOTSMerge:
    """KnOTS-style merging: align LoRAs via orthogonal Procrustes on the B factors,
    then average in the aligned space."""

    @staticmethod
    def merge(loras: List[LoRAWeights], weights: Optional[List[float]] = None,
              name: str = "knots") -> LoRAWeights:
        N = len(loras)
        if weights is None:
            weights = [1.0 / N] * N
        rank = max(lora.rank for lora in loras)
        all_deltas = [lora.to_delta_weight() for lora in loras]
        common_keys = set(all_deltas[0].keys())
        for d in all_deltas[1:]:
            common_keys &= set(d.keys())

        new_A, new_B = {}, {}
        for key in sorted(common_keys):
            ref = all_deltas[0][key].float()
            U_ref, S_ref, Vh_ref = torch.linalg.svd(ref, full_matrices=False)
            r = min(rank, U_ref.shape[1])
            aligned_sum = weights[0] * ref
            for i in range(1, N):
                dW = all_deltas[i][key].float()
                U_i, _, _ = torch.linalg.svd(dW, full_matrices=False)
                M = U_ref[:, :r].T @ U_i[:, :r]
                Ul, _, Vhl = torch.linalg.svd(M)
                R = Vhl.T @ Ul.T
                aligned_sum += weights[i] * (U_i[:, :r] @ R @ U_ref[:, :r].T @ dW)
            U, S, Vh = torch.linalg.svd(aligned_sum, full_matrices=False)
            new_B[key] = U[:, :r] @ torch.diag(torch.sqrt(S[:r]))
            new_A[key] = torch.diag(torch.sqrt(S[:r])) @ Vh[:r, :]
        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


class TSPAMerge:
    """TSPA-style merging: task-specific parameter alignment.
    
    Key difference from SVD-Procrustes: TSPA aligns per-component (each singular
    direction independently) using spectral-weighted rotation, whereas Procrustes
    does a single global rotation. TSPA also applies sign correction per component
    before averaging to resolve sign ambiguities.
    """

    @staticmethod
    def merge(loras: List[LoRAWeights], weights: Optional[List[float]] = None,
              name: str = "tspa") -> LoRAWeights:
        N = len(loras)
        if weights is None:
            weights = [1.0 / N] * N
        rank = max(lora.rank for lora in loras)
        all_deltas = [lora.to_delta_weight() for lora in loras]
        common_keys = set(all_deltas[0].keys())
        for d in all_deltas[1:]:
            common_keys &= set(d.keys())

        new_A, new_B = {}, {}
        for key in sorted(common_keys):
            svd_data = []
            for dW in [d[key].float() for d in all_deltas]:
                U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
                r = min(rank, U.shape[1])
                svd_data.append((U[:, :r], S[:r], Vh[:r, :]))

            U_ref, S_ref, Vh_ref = svd_data[0]
            r = U_ref.shape[1]

            aligned_Us = [U_ref]
            aligned_Ss = [S_ref]
            aligned_Vhs = [Vh_ref]

            for i in range(1, N):
                U_i, S_i, Vh_i = svd_data[i]
                signs_U = torch.sign((U_ref.T @ U_i).diag())
                signs_U[signs_U == 0] = 1.0
                U_aligned = U_i * signs_U.unsqueeze(0)
                signs_V = torch.sign((Vh_ref @ Vh_i.T).diag())
                signs_V[signs_V == 0] = 1.0
                Vh_aligned = Vh_i * signs_V.unsqueeze(1)
                perm = torch.zeros(r, r, device=U_i.device)
                cost = -(U_ref.T @ U_aligned).abs()
                used_cols = set()
                for row in range(r):
                    best_col = -1
                    best_val = float("inf")
                    for col in range(r):
                        if col not in used_cols and cost[row, col] < best_val:
                            best_val = cost[row, col].item()
                            best_col = col
                    perm[row, best_col] = 1.0
                    used_cols.add(best_col)
                U_aligned = U_aligned @ perm.T
                Vh_aligned = perm @ Vh_aligned
                S_aligned = perm @ S_i.unsqueeze(1)
                S_aligned = S_aligned.squeeze(1)

                aligned_Us.append(U_aligned)
                aligned_Ss.append(S_aligned)
                aligned_Vhs.append(Vh_aligned)

            merged_U = sum(w * u for w, u in zip(weights, aligned_Us))
            merged_S = sum(w * s for w, s in zip(weights, aligned_Ss))
            merged_Vh = sum(w * vh for w, vh in zip(weights, aligned_Vhs))

            merged_delta = merged_U @ torch.diag(merged_S) @ merged_Vh
            U, S, Vh = torch.linalg.svd(merged_delta, full_matrices=False)
            new_B[key] = U[:, :r] @ torch.diag(torch.sqrt(S[:r]))
            new_A[key] = torch.diag(torch.sqrt(S[:r])) @ Vh[:r, :]
        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


class MergingBaselines:
    @staticmethod
    def task_arithmetic(loras: List[LoRAWeights], scaling: float = 1.0) -> Dict[str, torch.Tensor]:
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
    def task_arithmetic_avg(loras: List[LoRAWeights], scaling: float = 1.0) -> Dict[str, torch.Tensor]:
        all_deltas = [lora.to_delta_weight() for lora in loras]
        all_keys = set()
        for d in all_deltas:
            all_keys |= set(d.keys())
        merged = {}
        for key in all_keys:
            stack = [d[key] for d in all_deltas if key in d]
            merged[key] = scaling * torch.stack(stack).mean(dim=0)
        return merged

    @staticmethod
    def ties_merging(
        loras: List[LoRAWeights], density: float = 0.5, scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        all_deltas = [lora.to_delta_weight() for lora in loras]
        all_keys = set()
        for d in all_deltas:
            all_keys |= set(d.keys())
        merged = {}
        for key in all_keys:
            stack = torch.stack([d[key] for d in all_deltas if key in d])
            k = int(density * stack[0].numel())
            trimmed = stack.clone()
            for i in range(trimmed.shape[0]):
                flat = trimmed[i].abs().flatten()
                if k < flat.numel():
                    threshold = flat.topk(k).values[-1]
                    trimmed[i][trimmed[i].abs() < threshold] = 0.0
            sign_votes = torch.sign(trimmed).sum(dim=0)
            elected_sign = torch.sign(sign_votes)
            elected_sign[elected_sign == 0] = 1.0
            mask = torch.sign(trimmed) == elected_sign.unsqueeze(0)
            trimmed[~mask] = 0.0
            counts = mask.float().sum(dim=0).clamp(min=1.0)
            merged[key] = scaling * trimmed.sum(dim=0) / counts
        return merged

    @staticmethod
    def dare_merging(
        loras: List[LoRAWeights], drop_rate: float = 0.5, scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        all_deltas = [lora.to_delta_weight() for lora in loras]
        all_keys = set()
        for d in all_deltas:
            all_keys |= set(d.keys())
        merged = {}
        for key in all_keys:
            stack = torch.stack([d[key] for d in all_deltas if key in d])
            mask = torch.bernoulli(torch.full_like(stack, 1.0 - drop_rate))
            rescaled = stack * mask / (1.0 - drop_rate)
            merged[key] = scaling * rescaled.mean(dim=0)
        return merged


class LoRAAlgebra:
    """Legacy interface — kept for backward compatibility with existing experiment scripts."""

    def __init__(self, grassmann_rank: int = 32):
        self.grassmann_rank = grassmann_rank
        self._grassmerge = GrassMerge()

    def grassmann_compose(
        self,
        loras: List[LoRAWeights],
        weights: Optional[List[float]] = None,
        name: str = "grassmerge",
    ) -> LoRAWeights:
        return self._grassmerge.merge(loras, weights, name)

    @staticmethod
    def compose(lora_a: LoRAWeights, lora_b: LoRAWeights, name: str = "composed") -> LoRAWeights:
        delta_a = lora_a.to_delta_weight()
        delta_b = lora_b.to_delta_weight()
        rank = max(lora_a.rank, lora_b.rank)
        new_A, new_B = {}, {}
        for key in set(delta_a.keys()) | set(delta_b.keys()):
            d_a = delta_a.get(key, torch.zeros_like(next(iter(delta_a.values()))))
            d_b = delta_b.get(key, torch.zeros_like(d_a))
            combined = d_a + d_b
            U, S, Vh = torch.linalg.svd(combined.float(), full_matrices=False)
            new_A[key] = torch.diag(torch.sqrt(S[:rank])) @ Vh[:rank, :]
            new_B[key] = U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))
        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)

    @staticmethod
    def subtract(lora_a: LoRAWeights, lora_b: LoRAWeights, name: str = "subtracted") -> LoRAWeights:
        delta_a = lora_a.to_delta_weight()
        delta_b = lora_b.to_delta_weight()
        rank = max(lora_a.rank, lora_b.rank)
        new_A, new_B = {}, {}
        for key in set(delta_a.keys()) | set(delta_b.keys()):
            d_a = delta_a.get(key, torch.zeros_like(next(iter(delta_a.values()))))
            d_b = delta_b.get(key, torch.zeros_like(d_a))
            diff = d_a - d_b
            U, S, Vh = torch.linalg.svd(diff.float(), full_matrices=False)
            new_A[key] = torch.diag(torch.sqrt(S[:rank])) @ Vh[:rank, :]
            new_B[key] = U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))
        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)

    @staticmethod
    def interpolate(
        lora_a: LoRAWeights, lora_b: LoRAWeights, alpha: float = 0.5, name: str = "interpolated"
    ) -> LoRAWeights:
        delta_a = lora_a.to_delta_weight()
        delta_b = lora_b.to_delta_weight()
        rank = max(lora_a.rank, lora_b.rank)
        new_A, new_B = {}, {}
        for key in set(delta_a.keys()) | set(delta_b.keys()):
            d_a = delta_a.get(key, torch.zeros_like(next(iter(delta_a.values()))))
            d_b = delta_b.get(key, torch.zeros_like(d_a))
            interp = alpha * d_a + (1.0 - alpha) * d_b
            U, S, Vh = torch.linalg.svd(interp.float(), full_matrices=False)
            new_A[key] = torch.diag(torch.sqrt(S[:rank])) @ Vh[:rank, :]
            new_B[key] = U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))
        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)

    def grassmann_interpolate(
        self, lora_a: LoRAWeights, lora_b: LoRAWeights, t: float = 0.5, name: str = "geodesic"
    ) -> LoRAWeights:
        """Geodesic interpolation on the Grassmann manifold between two LoRA adapters."""
        delta_a = lora_a.to_delta_weight()
        delta_b = lora_b.to_delta_weight()
        rank = max(lora_a.rank, lora_b.rank)
        new_A, new_B = {}, {}
        for key in set(delta_a.keys()) & set(delta_b.keys()):
            dW_a = delta_a[key].float()
            dW_b = delta_b[key].float()
            U_a, S_a, Vh_a = torch.linalg.svd(dW_a, full_matrices=False)
            U_b, S_b, Vh_b = torch.linalg.svd(dW_b, full_matrices=False)
            r = min(rank, U_a.shape[1], U_b.shape[1])
            U_a, V_a = U_a[:, :r], Vh_a[:r, :].T
            U_b, V_b = U_b[:, :r], Vh_b[:r, :].T
            S_a, S_b = S_a[:r], S_b[:r]

            tangent_U = GrassmannOps.log_map(U_a, U_b)
            U_t = GrassmannOps.exp_map(U_a, t * tangent_U)
            tangent_V = GrassmannOps.log_map(V_a, V_b)
            V_t = GrassmannOps.exp_map(V_a, t * tangent_V)
            S_t = (1.0 - t) * S_a + t * S_b

            sqrt_S = torch.sqrt(S_t.clamp(min=0))
            new_B[key] = U_t @ torch.diag(sqrt_S)
            new_A[key] = torch.diag(sqrt_S) @ V_t.T
        return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


def compute_bgd_matrix(loras: List[LoRAWeights]) -> np.ndarray:
    merger = GrassMerge()
    return merger.compute_bgd_matrix(loras)


def compute_similarity_matrix(loras: List[LoRAWeights], projector: "GrassmannProjector") -> np.ndarray:
    n = len(loras)
    dist_matrix = np.zeros((n, n))
    deltas = [lora.to_delta_weight() for lora in loras]
    common_keys = set(deltas[0].keys())
    for d in deltas[1:]:
        common_keys &= set(d.keys())
    all_keys = sorted(common_keys)
    for i in range(n):
        for j in range(i + 1, n):
            dist_sum = 0.0
            for key in all_keys:
                U_i, _, _ = torch.linalg.svd(deltas[i][key].float(), full_matrices=False)
                r = min(projector.svd_rank, U_i.shape[1])
                U_i_k = U_i[:, :r]
                U_j, _, _ = torch.linalg.svd(deltas[j][key].float(), full_matrices=False)
                U_j_k = U_j[:, :r]
                dist_sum += GrassmannOps.geodesic_distance(U_i_k, U_j_k)
            dist_avg = dist_sum / len(all_keys)
            dist_matrix[i][j] = dist_avg
            dist_matrix[j][i] = dist_avg
    return dist_matrix


class GrassmannProjector:
    """Legacy wrapper for backward compatibility."""

    def __init__(self, svd_rank: int = 32):
        self.svd_rank = svd_rank

    def to_grassmann(self, delta_W: torch.Tensor) -> torch.Tensor:
        U, S, Vh = torch.linalg.svd(delta_W.float(), full_matrices=False)
        k = min(self.svd_rank, U.shape[1])
        return U[:, :k]

    def grassmann_distance(self, U1: torch.Tensor, U2: torch.Tensor) -> float:
        return GrassmannOps.geodesic_distance(U1, U2)

    def grassmann_mean(self, bases: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
        return GrassmannOps.karcher_mean(bases, weights)
