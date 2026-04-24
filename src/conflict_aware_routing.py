"""
CARR: Conflict-Aware Reliability-Gated Residual Routing.

Core new method from GPT-5.5 Pro diagnosis. Decomposes multi-adapter composition into:
1. Static compatible component (safe TA/TIES merge)
2. Conflict residual per adapter
3. Reliability-calibrated, input-conditioned gate with base fallback

The router is a small module trained on calibration data while base model and
adapters remain frozen.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CARRConfig:
    n_adapters: int = 2
    d_model: int = 4096
    gate_hidden_dim: int = 128
    use_reliability: bool = True
    use_conflict: bool = True
    use_base_fallback: bool = True
    top_k: int = 1
    base_kl_weight: float = 0.1
    conflict_weight: float = 0.05
    sparsity_weight: float = 0.01
    temperature: float = 1.0


class ReliabilityFeatures(nn.Module):
    """Compute per-adapter reliability features from input hidden states."""

    def __init__(self, d_model: int, n_adapters: int, hidden_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden_dim)
        self.reliability_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_adapters)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, seq, d_model) hidden states
        Returns:
            reliability: (batch, seq, n_adapters) reliability scores in [0, 1]
        """
        features = F.relu(self.proj(h))
        scores = torch.cat([head(features) for head in self.reliability_heads], dim=-1)
        return torch.sigmoid(scores)


class ConflictAwareResidualRouter(nn.Module):
    """
    CARR Router: decides per-input/token how to compose adapter residuals.

    Outputs gates:
    - g_base: probability of keeping base (no adapter)
    - g_static: probability of using static merged component
    - g_i: probability of routing adapter i's conflict residual

    Gate is softmax over [base, static, adapter_1, ..., adapter_n].
    """

    def __init__(self, config: CARRConfig):
        super().__init__()
        self.config = config
        n_choices = 2 + config.n_adapters

        input_dim = config.d_model
        if config.use_reliability:
            input_dim += config.n_adapters
        if config.use_conflict:
            input_dim += config.n_adapters

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, config.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gate_hidden_dim, n_choices),
        )

        if config.use_reliability:
            self.reliability = ReliabilityFeatures(
                config.d_model, config.n_adapters, config.gate_hidden_dim // 2
            )

    def forward(
        self,
        h: torch.Tensor,
        adapter_effects: Optional[List[torch.Tensor]] = None,
        conflict_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gate logits.

        Args:
            h: (batch, seq, d_model) current hidden states
            adapter_effects: optional list of (batch, seq, d_out) adapter residuals
            conflict_scores: optional (batch, seq, n_adapters) conflict features

        Returns:
            gates: (batch, seq, 2 + n_adapters) softmax probabilities
                   [base, static, adapter_0, ..., adapter_n-1]
        """
        features = [h]

        if self.config.use_reliability:
            rel = self.reliability(h)
            features.append(rel)

        if self.config.use_conflict and conflict_scores is not None:
            features.append(conflict_scores)
        elif self.config.use_conflict:
            features.append(torch.zeros(
                h.shape[0], h.shape[1], self.config.n_adapters,
                device=h.device, dtype=h.dtype
            ))

        gate_input = torch.cat(features, dim=-1)
        logits = self.gate_net(gate_input) / self.config.temperature
        gates = F.softmax(logits, dim=-1)
        return gates

    def compute_composed_output(
        self,
        h: torch.Tensor,
        static_delta: torch.Tensor,
        adapter_residuals: List[torch.Tensor],
        conflict_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply CARR routing to produce composed hidden states.

        Args:
            h: (batch, seq, d_model) base hidden states
            static_delta: (batch, seq, d_model) static merge component
            adapter_residuals: list of (batch, seq, d_model) per-adapter conflict residuals
            conflict_scores: optional conflict features

        Returns:
            h_composed: (batch, seq, d_model) composed output
            gate_stats: dict of gate diagnostics
        """
        gates = self.forward(h, conflict_scores=conflict_scores)

        g_base = gates[:, :, 0:1]
        g_static = gates[:, :, 1:2]
        g_adapters = [gates[:, :, 2+i:3+i] for i in range(len(adapter_residuals))]

        h_composed = h.clone()
        h_composed = h_composed + g_static * static_delta

        for i, g_i in enumerate(g_adapters):
            h_composed = h_composed + g_i * adapter_residuals[i]

        gate_stats = {
            "base_gate_mean": float(g_base.detach().mean()),
            "static_gate_mean": float(g_static.detach().mean()),
            "adapter_gate_means": [float(g.detach().mean()) for g in g_adapters],
            "gate_entropy": float(-(gates.detach() * (gates.detach() + 1e-10).log()).sum(-1).mean()),
        }

        return h_composed, gate_stats


class CARRHook:
    """
    Applies CARR routing at LoRA target modules during inference.

    Intercepts the forward pass, computes adapter residuals,
    routes them through the CARR gate, and modifies hidden states.
    """

    def __init__(
        self,
        router: ConflictAwareResidualRouter,
        static_delta_ws: Dict[str, torch.Tensor],
        adapter_delta_ws: List[Dict[str, torch.Tensor]],
    ):
        self.router = router
        self.static_delta_ws = static_delta_ws
        self.adapter_delta_ws = adapter_delta_ws
        self.handles: List = []
        self.gate_stats: Dict[str, List] = {}

    def _make_hook(self, module_name: str):
        static_dw = self.static_delta_ws.get(module_name)
        adapter_dws = [adw.get(module_name) for adw in self.adapter_delta_ws]

        if static_dw is None or any(dw is None for dw in adapter_dws):
            return None

        router = self.router

        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                h = input[0]
            else:
                h = input

            if isinstance(output, tuple):
                out = output[0]
                rest = output[1:]
            else:
                out = output
                rest = None

            device = h.device
            dtype = h.dtype
            h_flat = h.float()

            static_delta = (h_flat @ static_dw.to(device).float().T)
            adapter_residuals = [
                (h_flat @ dw.to(device).float().T) for dw in adapter_dws
            ]

            with torch.no_grad():
                composed, stats = router.compute_composed_output(
                    torch.zeros_like(h_flat),
                    static_delta,
                    adapter_residuals,
                )

            modified = out + composed.to(dtype)

            if module_name not in self.gate_stats:
                self.gate_stats[module_name] = []
            self.gate_stats[module_name].append(stats)

            if rest is not None:
                return (modified,) + rest
            return modified

        return hook_fn

    def attach(self, model: nn.Module):
        for name, module in model.named_modules():
            for target in self.static_delta_ws.keys():
                if name == target or name.endswith(target):
                    hook_fn = self._make_hook(target)
                    if hook_fn is not None:
                        self.handles.append(module.register_forward_hook(hook_fn))
                    break

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def get_aggregated_stats(self) -> Dict:
        agg = {}
        for mod, stats_list in self.gate_stats.items():
            if not stats_list:
                continue
            agg[mod] = {
                "base_gate_mean": sum(s["base_gate_mean"] for s in stats_list) / len(stats_list),
                "static_gate_mean": sum(s["static_gate_mean"] for s in stats_list) / len(stats_list),
                "gate_entropy": sum(s["gate_entropy"] for s in stats_list) / len(stats_list),
                "n_forward": len(stats_list),
            }
        return agg
