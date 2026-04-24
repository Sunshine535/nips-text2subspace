"""CARR run config loader. Single source of truth for train_carr_router + eval_carr."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


_CARR_FIELDS = {
    "n_adapters", "d_model", "gate_hidden_dim",
    "use_reliability", "use_conflict", "use_base_fallback",
    "top_k", "temperature",
    "base_kl_weight", "conflict_weight", "sparsity_weight",
    "lr", "max_steps", "calib_samples", "batch_size", "max_length",
}

MODE_OVERRIDES = {
    "static_only": None,  # no router
    "carr_full": {"use_reliability": True, "use_conflict": True, "use_base_fallback": True},
    "carr_no_mechanism": {"use_reliability": False, "use_conflict": False, "use_base_fallback": True},
    "no_reliability": {"use_reliability": False, "use_conflict": True, "use_base_fallback": True},
    "no_conflict": {"use_reliability": True, "use_conflict": False, "use_base_fallback": True},
    "no_base_fallback": {"use_reliability": True, "use_conflict": True, "use_base_fallback": False},
}


@dataclass
class CARRRunConfig:
    base_model: str
    adapter_dir: str
    dataset_dir: str
    carr: Dict[str, Any] = field(default_factory=dict)
    eval_cfg: Dict[str, Any] = field(default_factory=dict)
    safe_pairs: List[List[str]] = field(default_factory=list)
    leaky_pairs: List[List[str]] = field(default_factory=list)
    source_yaml: Optional[str] = None

    def carr_kwargs_for_mode(self, mode: str) -> Dict[str, Any]:
        base = {k: v for k, v in self.carr.items() if k in _CARR_FIELDS}
        overlay = MODE_OVERRIDES.get(mode)
        if overlay:
            base.update(overlay)
        return base

    @property
    def top_k(self) -> int:
        return int(self.carr.get("top_k", 0))

    @property
    def sparsity_weight(self) -> float:
        return float(self.carr.get("sparsity_weight", 0.01))

    @property
    def base_kl_weight(self) -> float:
        return float(self.carr.get("base_kl_weight", 0.1))

    @property
    def conflict_weight(self) -> float:
        return float(self.carr.get("conflict_weight", 0.05))

    @property
    def metric_mode(self) -> str:
        return str(self.eval_cfg.get("metric_mode", "generation"))

    @property
    def eval_max_samples(self) -> int:
        return int(self.eval_cfg.get("max_samples", 50))


def load_run_config(yaml_path: str) -> CARRRunConfig:
    if yaml is None:
        raise RuntimeError("PyYAML not installed; install with `pip install pyyaml`")
    p = Path(yaml_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {yaml_path}")
    with p.open() as f:
        raw = yaml.safe_load(f) or {}
    return CARRRunConfig(
        base_model=raw.get("base_model", ""),
        adapter_dir=raw.get("adapter_dir", ""),
        dataset_dir=raw.get("dataset_dir", ""),
        carr=raw.get("carr", {}) or {},
        eval_cfg=raw.get("eval", {}) or {},
        safe_pairs=raw.get("safe_pairs", []) or [],
        leaky_pairs=raw.get("leaky_pairs", []) or [],
        source_yaml=str(p.resolve()),
    )


def assert_config_applied(requested: Dict[str, Any], actual: Dict[str, Any]) -> None:
    """Hard-fail if yaml-specified field does not match realized CARRConfig."""
    mismatched = []
    for k, v in requested.items():
        if k not in actual:
            continue
        a = actual[k]
        if isinstance(v, float) or isinstance(a, float):
            if abs(float(a) - float(v)) > 1e-6:
                mismatched.append((k, v, a))
        elif a != v:
            mismatched.append((k, v, a))
    if mismatched:
        lines = [f"  {k}: yaml={y!r} actual={a!r}" for k, y, a in mismatched]
        raise RuntimeError("CARR config not applied:\n" + "\n".join(lines))


def save_effective_config(cfg: CARRRunConfig, run_dir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "source_yaml": cfg.source_yaml,
        "base_model": cfg.base_model,
        "adapter_dir": cfg.adapter_dir,
        "dataset_dir": cfg.dataset_dir,
        "carr": cfg.carr,
        "eval": cfg.eval_cfg,
        "safe_pairs": cfg.safe_pairs,
        "leaky_pairs": cfg.leaky_pairs,
    }
    if extra:
        payload["extra"] = extra
    out = os.path.join(run_dir, "effective_config.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def save_sample_manifest(
    manifest: List[Dict[str, Any]], run_dir: str, fname: str = "sample_manifest.jsonl"
) -> str:
    os.makedirs(run_dir, exist_ok=True)
    out = os.path.join(run_dir, fname)
    with open(out, "w") as f:
        for row in manifest:
            f.write(json.dumps(row, default=str) + "\n")
    return out


def build_manifest_entry(
    domain: str, row_idx: int, prompt: str, gold: str, sample_seed: int, eval_type: str
) -> Dict[str, Any]:
    return {
        "domain": domain,
        "eval_type": eval_type,
        "sample_seed": int(sample_seed),
        "row_idx": int(row_idx),
        "question_hash": _stable_hash(prompt),
        "label_hash": _stable_hash(str(gold)),
    }
