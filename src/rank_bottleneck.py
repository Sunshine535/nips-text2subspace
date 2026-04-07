"""
Rank Bottleneck of Adapter Composition.

Theory: composition rank r_c determines the tight lower bound on static merge loss.
Method: Bottleneck-Aware Composition (BAC) — static merge on non-bottleneck directions,
        lightweight router on bottleneck directions.
Diagnostic: Composition Rank Score (CRS) — normalized mergeability index.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .lora_algebra import LoRAWeights, GrassmannOps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core theory: Composition Rank estimation
# ---------------------------------------------------------------------------

@dataclass
class LayerBottleneckInfo:
    """Per-layer rank bottleneck analysis results."""
    key: str
    rank: int                       # original LoRA rank r
    composition_rank: int           # r_c (at tolerance eps)
    bottleneck_dim: int             # k = max(0, r_c - r)
    eigvals: torch.Tensor           # eigenvalues of Gram matrix (descending)
    tail_energy: float              # Σ_{j>r} σ_j² — lower bound on merge loss
    total_energy: float             # Σ_j σ_j²
    crs: float                      # per-layer CRS ∈ [0, 1]
    # Subspace bases (populated by identify_bottleneck_directions)
    shared_basis: Optional[torch.Tensor] = None    # (d_out, k_static)
    bottleneck_basis: Optional[torch.Tensor] = None  # (d_out, k_bottleneck)


@dataclass
class CompositionAnalysis:
    """Full composition rank analysis across all layers."""
    layers: Dict[str, LayerBottleneckInfo]
    global_crs: float
    total_tail_energy: float
    total_energy: float
    adapter_names: List[str]
    weights: List[float]


def compute_composition_rank(
    loras: List[LoRAWeights],
    weights: Optional[List[float]] = None,
    eps: float = 0.0,
    eps_rank: float = 1e-8,
) -> CompositionAnalysis:
    """Compute composition rank r_c and CRS per layer (weight-space only, no activations).

    Uses the efficient Gram matrix approach exploiting LoRA's low-rank structure.
    For N adapters of rank r, computes an Nr×Nr Gram matrix per layer (not d×d).

    The singular values of the stacked whitened operator G^l = [√π_1 ΔW_1, ..., √π_N ΔW_N]
    equal the square roots of eigenvalues of the Gram matrix K = Ũ^T Ũ, where
    Ũ = [√π_1 U_1 diag(S_1), ..., √π_N U_N diag(S_N)].

    By Theorem 1, for any rank-k static merge M:
        E_l(M) >= Σ_{j>k} eigenvalue_j(K)
    """
    N = len(loras)
    if weights is None:
        weights = [1.0 / N] * N

    all_svds = [lora.fast_svd() for lora in loras]
    common_keys = sorted(set.intersection(*[set(s.keys()) for s in all_svds]))

    layers = {}
    total_tail = 0.0
    total_energy = 0.0

    for key in common_keys:
        Us = [all_svds[i][key][0] for i in range(N)]  # each (d_out, r_i)
        Ss = [all_svds[i][key][1] for i in range(N)]  # each (r_i,)
        r = Us[0].shape[1]

        # Build Nr×Nr Gram matrix K where K_{(i,a),(j,b)} = √π_i S_{i,a} (U_i^T U_j)_{a,b} S_{j,b} √π_j
        Nr = N * r
        K = torch.zeros(Nr, Nr, dtype=torch.float64)
        for i in range(N):
            ui = Us[i].double()
            si = Ss[i].double()
            wi = math.sqrt(weights[i])
            for j in range(i, N):
                uj = Us[j].double()
                sj = Ss[j].double()
                wj = math.sqrt(weights[j])
                # block = (wi * si) ⊗ (U_i^T U_j) ⊗ (wj * sj)
                cross = ui.T @ uj  # (r, r)
                block = (wi * si).unsqueeze(1) * cross * (wj * sj).unsqueeze(0)
                K[i * r:(i + 1) * r, j * r:(j + 1) * r] = block
                if i != j:
                    K[j * r:(j + 1) * r, i * r:(i + 1) * r] = block.T

        # Eigendecompose (symmetric positive semi-definite)
        eigvals, eigvecs = torch.linalg.eigh(K)
        eigvals = eigvals.flip(0).clamp(min=0)  # descending
        eigvecs = eigvecs.flip(1)

        # Composition rank at tolerance eps
        if eps == 0:
            rc = int((eigvals > eps_rank * eigvals[0]).sum().item())
        else:
            cumtail = eigvals.sum() - eigvals.cumsum(0)
            mask = cumtail <= eps
            rc = int(mask.nonzero()[0].item()) + 1 if mask.any() else Nr

        rc = max(rc, 1)
        tail_e = float(eigvals[r:].sum().item()) if rc > r else 0.0
        tot_e = float(eigvals.sum().item())
        k_bot = max(0, rc - r)
        crs_val = 1.0 - k_bot / ((N - 1) * r) if N > 1 and r > 0 else 1.0
        crs_val = max(0.0, min(1.0, crs_val))

        layers[key] = LayerBottleneckInfo(
            key=key,
            rank=r,
            composition_rank=rc,
            bottleneck_dim=k_bot,
            eigvals=eigvals.float(),
            tail_energy=tail_e,
            total_energy=tot_e,
            crs=crs_val,
        )
        total_tail += tail_e
        total_energy += tot_e

    # Global CRS (energy-weighted)
    if total_energy > 0:
        global_crs = 1.0 - total_tail / total_energy
    else:
        global_crs = 1.0

    return CompositionAnalysis(
        layers=layers,
        global_crs=global_crs,
        total_tail_energy=total_tail,
        total_energy=total_energy,
        adapter_names=[l.name for l in loras],
        weights=weights,
    )


# ---------------------------------------------------------------------------
# Bottleneck direction identification
# ---------------------------------------------------------------------------

def identify_bottleneck_directions(
    loras: List[LoRAWeights],
    weights: Optional[List[float]] = None,
    static_rank: Optional[int] = None,
    tail_energy_coverage: float = 0.95,
) -> CompositionAnalysis:
    """Identify shared (static-mergeable) vs bottleneck directions per layer.

    Returns CompositionAnalysis with shared_basis and bottleneck_basis populated.

    The stacked operator Ũ = [√π_i U_i diag(S_i)] has Nr columns in R^{d_out}.
    Top-k singular vectors = shared static subspace.
    Next-b singular vectors = bottleneck subspace (covering tail_energy_coverage of tail).
    """
    N = len(loras)
    if weights is None:
        weights = [1.0 / N] * N

    analysis = compute_composition_rank(loras, weights)

    all_svds = [lora.fast_svd() for lora in loras]

    for key, info in analysis.layers.items():
        r = info.rank
        k_static = static_rank if static_rank is not None else r

        # Reconstruct Ũ for this layer
        Us = [all_svds[i][key][0].float() for i in range(N)]
        Ss = [all_svds[i][key][1].float() for i in range(N)]

        # Ũ = [√π_1 U_1 diag(S_1), ..., √π_N U_N diag(S_N)]  (d_out × Nr)
        U_tilde_cols = []
        for i in range(N):
            col = math.sqrt(weights[i]) * Us[i] * Ss[i].unsqueeze(0)  # (d_out, r)
            U_tilde_cols.append(col)
        U_tilde = torch.cat(U_tilde_cols, dim=1)  # (d_out, Nr)

        # SVD of Ũ (thin, since Nr << d_out typically)
        U_full, S_full, _ = torch.linalg.svd(U_tilde, full_matrices=False)
        # U_full: (d_out, Nr), S_full: (Nr,)

        # Shared basis = top k_static directions
        info.shared_basis = U_full[:, :k_static]

        # Bottleneck basis = next directions covering tail_energy_coverage of tail energy
        tail_eigvals = S_full[k_static:] ** 2
        if tail_eigvals.numel() > 0 and tail_eigvals.sum() > 0:
            cum = tail_eigvals.cumsum(0) / tail_eigvals.sum()
            b = int((cum < tail_energy_coverage).sum().item()) + 1
            b = min(b, tail_eigvals.numel())
            info.bottleneck_basis = U_full[:, k_static:k_static + b]
            info.bottleneck_dim = b
        else:
            info.bottleneck_basis = U_full[:, :0]  # empty
            info.bottleneck_dim = 0

    return analysis


# ---------------------------------------------------------------------------
# BAC: Bottleneck-Aware Composition
# ---------------------------------------------------------------------------

class CompositionRouter(nn.Module):
    """Lightweight MLP router for bottleneck directions.

    Maps hidden state → N-dimensional softmax weights for each bottleneck layer.
    """

    def __init__(self, input_dim: int, n_adapters: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_adapters),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: (batch, seq_len, input_dim) or (batch, input_dim)
        Returns:
            weights: (batch, [seq_len,] n_adapters) softmax probabilities
        """
        logits = self.net(hidden_state)
        return torch.softmax(logits, dim=-1)


@dataclass
class BACConfig:
    """Configuration for Bottleneck-Aware Composition."""
    static_rank: int = 16           # rank of static merged component
    max_bottleneck_dim: int = 32    # max bottleneck dimensions per layer
    tail_energy_coverage: float = 0.95
    router_hidden_dim: int = 64
    router_lr: float = 1e-3
    router_epochs: int = 5
    router_batch_size: int = 32


class BottleneckAwareComposition:
    """BAC: Static merge on shared directions + lightweight router on bottleneck directions.

    Resolves the Adapter Composition Trilemma by sacrificing minimal staticness (k-dim signal)
    to achieve full faithfulness while remaining compact (rank r output).
    """

    def __init__(self, config: BACConfig = BACConfig()):
        self.config = config
        self.analysis: Optional[CompositionAnalysis] = None
        self.static_merge: Optional[LoRAWeights] = None
        self.residuals: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        self.router: Optional[CompositionRouter] = None

    def analyze(self, loras: List[LoRAWeights], weights: Optional[List[float]] = None):
        """Step 1: Analyze composition rank and identify bottleneck directions."""
        self.analysis = identify_bottleneck_directions(
            loras, weights,
            static_rank=self.config.static_rank,
            tail_energy_coverage=self.config.tail_energy_coverage,
        )
        logger.info(
            f"Composition analysis: global CRS={self.analysis.global_crs:.3f}, "
            f"total tail energy={self.analysis.total_tail_energy:.4f}"
        )
        for key, info in self.analysis.layers.items():
            if info.bottleneck_dim > 0:
                logger.info(f"  {key}: r_c={info.composition_rank}, k_bot={info.bottleneck_dim}, crs={info.crs:.3f}")

    def build_static_merge(
        self, loras: List[LoRAWeights], weights: Optional[List[float]] = None,
    ) -> LoRAWeights:
        """Step 2: Build the static merged adapter (non-bottleneck directions).

        For each layer, computes the best rank-k_static approximation of the
        weighted adapter sum projected through the shared subspace.
        """
        N = len(loras)
        if weights is None:
            weights = [1.0 / N] * N
        if self.analysis is None:
            self.analyze(loras, weights)

        all_svds = [lora.fast_svd() for lora in loras]
        common_keys = sorted(set.intersection(*[set(s.keys()) for s in all_svds]))

        new_A, new_B = {}, {}
        rank = self.config.static_rank

        for key in common_keys:
            info = self.analysis.layers.get(key)
            if info is None or info.shared_basis is None:
                continue

            U_shared = info.shared_basis  # (d_out, k_static)
            k_static = U_shared.shape[1]

            # Compute weighted average in shared subspace: S_avg = Σ_i w_i U_shared^T ΔW_i
            # Using compact factors: U_shared^T B_i A_i * alpha_i
            S_avg = torch.zeros(k_static, loras[0].lora_A[key].shape[1], dtype=torch.float32)
            for i in range(N):
                B_i = loras[i].lora_B.get(key)
                A_i = loras[i].lora_A.get(key)
                if B_i is None or A_i is None:
                    continue
                alpha_i = loras[i].alpha
                # (k_static, r) @ (r, d_in) * alpha * w
                proj = (U_shared.T @ B_i.float()) @ A_i.float() * alpha_i
                S_avg += weights[i] * proj

            # Best rank-r approximation of the projected sum
            U_s, S_s, Vh_s = torch.linalg.svd(S_avg, full_matrices=False)
            r_use = min(rank, U_s.shape[1])
            sqrt_S = torch.sqrt(S_s[:r_use].clamp(min=0))

            # Reconstruct in original space
            B_merged = (U_shared @ U_s[:, :r_use]) @ torch.diag(sqrt_S)  # (d_out, r)
            A_merged = torch.diag(sqrt_S) @ Vh_s[:r_use, :]              # (r, d_in)

            new_B[key] = B_merged
            new_A[key] = A_merged

        self.static_merge = LoRAWeights(
            name="bac_static", lora_A=new_A, lora_B=new_B,
            rank=rank, alpha=1.0,
        )
        return self.static_merge

    def compute_residuals(self, loras: List[LoRAWeights]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Step 3: Compute per-adapter residuals projected onto bottleneck subspace.

        For each adapter i and bottleneck layer:
            R_i = U_bot^T @ (ΔW_i - S_static) — the residual in bottleneck coordinates.
        Stored as small matrices (b × d_in) per adapter per layer.
        """
        if self.static_merge is None:
            raise RuntimeError("Call build_static_merge first")

        residuals = {}  # {adapter_name: {layer_key: (b, d_in) tensor}}
        static_deltas = self.static_merge.to_delta_weight()

        for lora in loras:
            adapter_resid = {}
            deltas = lora.to_delta_weight()
            for key, info in self.analysis.layers.items():
                if info.bottleneck_dim == 0 or info.bottleneck_basis is None:
                    continue
                if key not in deltas or key not in static_deltas:
                    continue
                U_bot = info.bottleneck_basis  # (d_out, b)
                residual_full = deltas[key].float() - static_deltas[key].float()
                # Project to bottleneck coords: (b, d_out) @ (d_out, d_in) = (b, d_in)
                adapter_resid[key] = U_bot.T @ residual_full
            residuals[lora.name] = adapter_resid

        self.residuals = residuals
        return residuals

    def get_bottleneck_layers(self) -> List[str]:
        """Return layer keys that have bottleneck dimensions > 0."""
        if self.analysis is None:
            return []
        return [k for k, v in self.analysis.layers.items() if v.bottleneck_dim > 0]

    def get_total_bottleneck_params(self) -> int:
        """Total additional parameters for BAC over static merge."""
        if self.analysis is None:
            return 0
        N = len(self.analysis.adapter_names)
        total = 0
        for key, info in self.analysis.layers.items():
            if info.bottleneck_dim > 0 and info.bottleneck_basis is not None:
                b = info.bottleneck_dim
                d_in = list(self.static_merge.lora_A.values())[0].shape[1] if self.static_merge else 0
                # Per-adapter residual: N × b × d_in (stored, not params)
                # Bottleneck basis: d_out × b (stored, not params)
                # Router params counted separately
                total += b  # approximate per-layer router output dim
        return total

    def summary(self) -> str:
        """Print summary of composition analysis."""
        if self.analysis is None:
            return "No analysis computed yet."
        lines = [
            f"=== Rank Bottleneck Analysis ===",
            f"Adapters: {self.analysis.adapter_names}",
            f"Weights: {self.analysis.weights}",
            f"Global CRS: {self.analysis.global_crs:.4f}",
            f"Total tail energy: {self.analysis.total_tail_energy:.6f}",
            f"Total energy: {self.analysis.total_energy:.6f}",
            f"Tail ratio: {self.analysis.total_tail_energy / max(self.analysis.total_energy, 1e-10):.4f}",
            f"",
            f"Per-layer breakdown:",
            f"{'Layer':<50} {'r':>3} {'r_c':>4} {'k_bot':>5} {'CRS':>6} {'Tail E':>10}",
            f"{'-'*85}",
        ]
        for key in sorted(self.analysis.layers.keys()):
            info = self.analysis.layers[key]
            lines.append(
                f"{key:<50} {info.rank:>3} {info.composition_rank:>4} "
                f"{info.bottleneck_dim:>5} {info.crs:>6.3f} {info.tail_energy:>10.6f}"
            )
        bot_layers = self.get_bottleneck_layers()
        lines.append(f"\nLayers with bottleneck: {len(bot_layers)} / {len(self.analysis.layers)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CRS comparison with existing metrics
# ---------------------------------------------------------------------------

def compare_diagnostics(
    loras: List[LoRAWeights],
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute CRS alongside existing metrics (BGD, cosine, Frobenius) for comparison."""
    from .lora_algebra import bilateral_grassmann_distance, cosine_interference, frobenius_interference

    analysis = compute_composition_rank(loras, weights)

    # Pairwise metrics (for N=2)
    result = {"crs": analysis.global_crs, "tail_energy": analysis.total_tail_energy}

    if len(loras) == 2:
        deltas = [lora.to_delta_weight() for lora in loras]
        common_keys = sorted(set(deltas[0].keys()) & set(deltas[1].keys()))
        bgd_sum, cos_sum, frob_sum = 0.0, 0.0, 0.0
        r = max(l.rank for l in loras)
        for key in common_keys:
            bgd_sum += bilateral_grassmann_distance(deltas[0][key], deltas[1][key], r)
            cos_sum += cosine_interference(deltas[0][key], deltas[1][key])
            frob_sum += frobenius_interference(deltas[0][key], deltas[1][key])
        n = max(len(common_keys), 1)
        result["bgd"] = bgd_sum / n
        result["cosine_interference"] = cos_sum / n
        result["frobenius_interference"] = frob_sum / n

    return result


# ---------------------------------------------------------------------------
# Synthetic verification utilities
# ---------------------------------------------------------------------------

def create_synthetic_adapters(
    d_out: int = 256,
    d_in: int = 256,
    rank: int = 16,
    n_adapters: int = 2,
    overlap_dim: int = 8,
    spectral_decay: float = 0.9,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], int]:
    """Create synthetic adapter delta weights with controlled subspace overlap.

    Args:
        d_out, d_in: dimensions
        rank: adapter rank
        n_adapters: number of adapters
        overlap_dim: number of shared column-space dimensions
        spectral_decay: exponential decay of singular values
        seed: random seed

    Returns:
        deltas: list of (d_out, d_in) delta weight matrices
        expected_rc: expected composition rank
    """
    torch.manual_seed(seed)

    # Create a shared subspace of dimension overlap_dim
    shared_U = torch.linalg.qr(torch.randn(d_out, overlap_dim))[0]  # (d_out, overlap_dim)

    deltas = []
    for i in range(n_adapters):
        # Private subspace
        private_dim = rank - overlap_dim
        private_U = torch.linalg.qr(torch.randn(d_out, private_dim))[0]
        # Orthogonalize private against shared
        private_U = private_U - shared_U @ (shared_U.T @ private_U)
        private_U = torch.linalg.qr(private_U)[0]

        U_i = torch.cat([shared_U, private_U], dim=1)  # (d_out, rank)

        # Random right subspace
        V_i = torch.linalg.qr(torch.randn(d_in, rank))[0]  # (d_in, rank)

        # Spectral profile
        S_i = torch.tensor([spectral_decay ** j for j in range(rank)])

        # Delta weight
        delta = U_i @ torch.diag(S_i) @ V_i.T
        deltas.append(delta)

    # Expected composition rank:
    # shared_dim contributes rank (same subspace → merge for free)
    # each adapter's private_dim is unique → adds to r_c
    expected_rc = overlap_dim + n_adapters * (rank - overlap_dim)

    return deltas, expected_rc


def verify_theorem_synthetic(
    d_out: int = 256,
    d_in: int = 256,
    rank: int = 16,
    overlap_dims: Optional[List[int]] = None,
    seed: int = 42,
) -> List[Dict]:
    """Verify Theorem 1 on synthetic adapters with varying overlap.

    Returns list of results with predicted and actual merge losses.
    """
    if overlap_dims is None:
        overlap_dims = [0, 4, 8, 12, 16]

    results = []
    for overlap in overlap_dims:
        overlap = min(overlap, rank)
        deltas, expected_rc = create_synthetic_adapters(
            d_out=d_out, d_in=d_in, rank=rank,
            n_adapters=2, overlap_dim=overlap, seed=seed,
        )

        # Create LoRAWeights from synthetic deltas
        loras = []
        for i, delta in enumerate(deltas):
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            r = min(rank, U.shape[1])
            sqrt_S = torch.sqrt(S[:r])
            B = U[:, :r] @ torch.diag(sqrt_S)
            A = torch.diag(sqrt_S) @ Vh[:r, :]
            lw = LoRAWeights(
                name=f"synthetic_{i}",
                lora_A={"layer.0": A},
                lora_B={"layer.0": B},
                rank=r, alpha=1.0,
            )
            loras.append(lw)

        # Compute composition rank
        analysis = compute_composition_rank(loras)
        info = analysis.layers["layer.0"]

        # Compute actual best rank-r merge loss
        avg_delta = 0.5 * (deltas[0] + deltas[1])
        U_avg, S_avg, Vh_avg = torch.linalg.svd(avg_delta, full_matrices=False)
        # Best rank-r approximation
        merged = U_avg[:, :rank] @ torch.diag(S_avg[:rank]) @ Vh_avg[:rank, :]
        actual_loss = sum(
            0.5 * torch.norm(merged - d, "fro").item() ** 2 for d in deltas
        )

        # Predicted lower bound = tail energy
        predicted_bound = info.tail_energy

        results.append({
            "overlap_dim": overlap,
            "expected_rc": expected_rc,
            "measured_rc": info.composition_rank,
            "predicted_lower_bound": predicted_bound,
            "actual_loss": actual_loss,
            "bound_tight": actual_loss > 0 and predicted_bound / actual_loss if actual_loss > 0 else 1.0,
            "crs": info.crs,
        })

    return results
