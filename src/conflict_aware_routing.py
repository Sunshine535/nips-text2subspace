"""
CARR: Conflict-Aware Reliability-Gated Residual Routing.

Core new method from GPT-5.5 Pro diagnosis. Decomposes multi-adapter composition into:
1. Static compatible component (safe TA/TIES merge)
2. Conflict residual per adapter
3. Reliability-calibrated, input-conditioned gate with base fallback

The router is a small module trained on calibration data while base model and
adapters remain frozen.

GPT-5.5 Pro Review Fixes (Round 2):
- Hook passes real h to router (not zeros) — input-conditioned gate
- conflict_scores propagated from precomputed module-level Gram
- use_base_fallback=False removes base from softmax choices
- top_k masks adapter gates to sparse selection
- Training mode: no torch.no_grad wrapper (gradient flows through router)
- Eval mode: torch.no_grad for inference-only stats
"""

import logging
from dataclasses import dataclass
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
    top_k: int = 0
    base_kl_weight: float = 0.1
    conflict_weight: float = 0.05
    sparsity_weight: float = 0.01
    temperature: float = 1.0
    # Round 3 Task 6: conflict mode. "module" = precomputed module-level prior
    # (broadcast to all tokens); "token" = per-token activation-effect cosine, computed
    # inside CARRHook from adapter_residuals.
    conflict_mode: str = "module"


class ReliabilityFeatures(nn.Module):
    """Compute per-adapter reliability features from input hidden states."""

    def __init__(self, d_model: int, n_adapters: int, hidden_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden_dim)
        self.reliability_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_adapters)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.proj(h))
        scores = torch.cat([head(features) for head in self.reliability_heads], dim=-1)
        return torch.sigmoid(scores)


class ConflictAwareResidualRouter(nn.Module):
    """
    CARR Router: decides per-input/token how to compose adapter residuals.

    Gate choices depend on config:
    - use_base_fallback=True:  [base, static, adapter_0, ..., adapter_n-1]
    - use_base_fallback=False: [static, adapter_0, ..., adapter_n-1]
    """

    def __init__(self, config: CARRConfig):
        super().__init__()
        self.config = config

        self.n_choices = (1 if config.use_base_fallback else 0) + 1 + config.n_adapters
        self.base_idx = 0 if config.use_base_fallback else None
        self.static_idx = 1 if config.use_base_fallback else 0
        self.adapter_start_idx = self.static_idx + 1

        input_dim = config.d_model
        if config.use_reliability:
            input_dim += config.n_adapters
        if config.use_conflict:
            input_dim += config.n_adapters

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, config.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gate_hidden_dim, self.n_choices),
        )

        if config.use_reliability:
            self.reliability = ReliabilityFeatures(
                config.d_model, config.n_adapters, config.gate_hidden_dim // 2
            )

    def forward(
        self,
        h: torch.Tensor,
        conflict_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gate probabilities. h MUST be the real input hidden states.

        Args:
            h: (batch, seq, d_model) current hidden states (NOT zeros)
            conflict_scores: (batch, seq, n_adapters) precomputed conflict features

        Returns:
            gates: (batch, seq, n_choices) softmax probabilities
        """
        features = [h]

        if self.config.use_reliability:
            rel = self.reliability(h)
            features.append(rel)

        if self.config.use_conflict:
            if conflict_scores is not None:
                features.append(conflict_scores)
            else:
                logger.warning("use_conflict=True but conflict_scores is None; filling zeros")
                features.append(torch.zeros(
                    h.shape[0], h.shape[1], self.config.n_adapters,
                    device=h.device, dtype=h.dtype
                ))

        gate_input = torch.cat(features, dim=-1)
        logits = self.gate_net(gate_input) / self.config.temperature

        if self.config.top_k > 0 and self.config.top_k < self.config.n_adapters:
            adapter_logits = logits[:, :, self.adapter_start_idx:]
            topk_vals, topk_idx = adapter_logits.topk(self.config.top_k, dim=-1)
            mask = torch.full_like(adapter_logits, float('-inf'))
            mask.scatter_(-1, topk_idx, 0.0)
            logits = torch.cat([logits[:, :, :self.adapter_start_idx], adapter_logits + mask], dim=-1)

        gates = F.softmax(logits, dim=-1)
        return gates

    def compute_composed_output(
        self,
        h: torch.Tensor,
        static_delta: torch.Tensor,
        adapter_residuals: List[torch.Tensor],
        conflict_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply CARR routing. h is the REAL input hidden state for input-conditioned gating.
        """
        gates = self.forward(h, conflict_scores=conflict_scores)

        g_static = gates[:, :, self.static_idx:self.static_idx+1]
        result = g_static * static_delta

        for i in range(len(adapter_residuals)):
            idx = self.adapter_start_idx + i
            g_i = gates[:, :, idx:idx+1]
            result = result + g_i * adapter_residuals[i]

        gate_stats = self._compute_gate_stats(gates)
        return result, gate_stats

    def _compute_gate_stats(self, gates: torch.Tensor) -> Dict[str, float]:
        g = gates.detach()
        stats = {
            "static_gate_mean": float(g[:, :, self.static_idx].mean()),
            "adapter_gate_means": [
                float(g[:, :, self.adapter_start_idx + i].mean())
                for i in range(self.config.n_adapters)
            ],
            "gate_entropy": float(-(g * (g + 1e-10).log()).sum(-1).mean()),
            "n_choices": self.n_choices,
        }
        if self.base_idx is not None:
            stats["base_gate_mean"] = float(g[:, :, self.base_idx].mean())
        return stats


class CARRHook:
    """
    Applies CARR routing at LoRA target modules during inference.

    CRITICAL: passes REAL input hidden states to router for input-conditioned gating.
    Precomputed per-module conflict scores are broadcast to all tokens.
    """

    def __init__(
        self,
        router: ConflictAwareResidualRouter,
        static_delta_ws: Dict[str, torch.Tensor],
        adapter_delta_ws: List[Dict[str, torch.Tensor]],
        module_conflict_scores: Optional[Dict[str, torch.Tensor]] = None,
        training: bool = False,
    ):
        self.router = router
        self.static_delta_ws = static_delta_ws
        self.adapter_delta_ws = adapter_delta_ws
        self.module_conflict_scores = module_conflict_scores or {}
        self.training = training
        self.handles: List = []
        self.gate_stats: Dict[str, List] = {}
        # Round 3 Task 4: keep differentiable gate tensors per module for loss
        self.last_gates: Dict[str, torch.Tensor] = {}
        # Round 3 Task 5: keep reliability outputs per module for L_cal
        self.last_reliability: Dict[str, torch.Tensor] = {}

    def _make_hook(self, module_name: str):
        static_dw = self.static_delta_ws.get(module_name)
        adapter_dws = [adw.get(module_name) for adw in self.adapter_delta_ws]

        if static_dw is None or any(dw is None for dw in adapter_dws):
            logger.warning("Skipping %s: missing static or adapter delta_W", module_name)
            return None

        d_out = static_dw.shape[0]
        d_in = static_dw.shape[1]
        router = self.router
        is_training = self.training

        conflict_score_per_module = self.module_conflict_scores.get(module_name)

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

            device = out.device
            dtype = out.dtype
            h_float = h.float()

            static_delta = h_float @ static_dw.to(device).float().T
            adapter_residuals = [
                h_float @ dw.to(device).float().T for dw in adapter_dws
            ]

            batch, seq = h_float.shape[0], h_float.shape[1]
            conflict_scores = None
            conflict_mode = getattr(router.config, "conflict_mode", "module")
            if conflict_mode == "token" and len(adapter_residuals) == 2:
                # Per-token activation-effect cosine conflict feature
                d0 = adapter_residuals[0]
                d1 = adapter_residuals[1]
                d0_n = d0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                d1_n = d1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                cos = ((d0 * d1).sum(dim=-1, keepdim=True)) / (d0_n * d1_n)
                interf = ((1.0 - cos) * 0.5).clamp(0.0, 1.0)  # (B, S, 1)
                conflict_scores = torch.cat([interf, 1.0 - interf], dim=-1)
            elif conflict_score_per_module is not None:
                cs = conflict_score_per_module.to(device).float()
                conflict_scores = cs.unsqueeze(0).unsqueeze(0).expand(batch, seq, -1)

            if is_training:
                gates_tensor = router(h_float, conflict_scores=conflict_scores)
                composed = self._compose_from_gates(
                    gates_tensor, static_delta, adapter_residuals)
                stats = router._compute_gate_stats(gates_tensor)
                # Expose differentiable gates + reliability for multi-term loss
                self.last_gates[module_name] = gates_tensor
                if router.config.use_reliability:
                    self.last_reliability[module_name] = router.reliability(h_float)
            else:
                with torch.no_grad():
                    composed, stats = router.compute_composed_output(
                        h_float, static_delta, adapter_residuals, conflict_scores
                    )

            modified = out + composed.to(dtype)

            if module_name not in self.gate_stats:
                self.gate_stats[module_name] = []
            self.gate_stats[module_name].append(stats)

            if rest is not None:
                return (modified,) + rest
            return modified

        return hook_fn

    def _compose_from_gates(
        self, gates: torch.Tensor,
        static_delta: torch.Tensor,
        adapter_residuals: List[torch.Tensor],
    ) -> torch.Tensor:
        g_static = gates[:, :, self.router.static_idx:self.router.static_idx + 1]
        result = g_static * static_delta
        for i in range(len(adapter_residuals)):
            idx = self.router.adapter_start_idx + i
            g_i = gates[:, :, idx:idx + 1]
            result = result + g_i * adapter_residuals[i]
        return result

    def clear_step_buffers(self):
        """Call this between training steps to release held tensors."""
        self.last_gates.clear()
        self.last_reliability.clear()
        self.gate_stats.clear()

    def mean_reliability_across_modules(self) -> Optional[torch.Tensor]:
        """Mean of reliability head outputs (B, seq, n_adapters) across modules.

        Returns (B, n_adapters) token-mean pooled, or None if reliability disabled.
        """
        if not self.last_reliability:
            return None
        stacked = torch.stack(list(self.last_reliability.values()), dim=0)  # (M,B,S,A)
        return stacked.mean(dim=(0, 2))  # (B, A)

    def attach(self, model: nn.Module):
        for name, module in model.named_modules():
            for target in self.static_delta_ws.keys():
                if name == target or name.endswith(target):
                    hook_fn = self._make_hook(target)
                    if hook_fn is not None:
                        self.handles.append(module.register_forward_hook(hook_fn))
                        logger.info("CARR hook attached: %s → %s", name, target)
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
            n = len(stats_list)
            agg[mod] = {
                "base_gate_mean": sum(s.get("base_gate_mean", 0) for s in stats_list) / n,
                "static_gate_mean": sum(s["static_gate_mean"] for s in stats_list) / n,
                "gate_entropy": sum(s["gate_entropy"] for s in stats_list) / n,
                "n_forward": n,
            }
        return agg
