"""Unit tests for nips-text2subspace core modules."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lora_algebra import (
    GrassMerge,
    GrassmannOps,
    LoRAWeights,
    compute_bgd_matrix,
    compute_similarity_matrix,
    GrassmannProjector,
)


def _make_lora(name: str, num_layers: int = 4, rank: int = 4,
               d_out: int = 16, d_in: int = 16) -> LoRAWeights:
    """Create a synthetic LoRAWeights with random A/B matrices."""
    torch.manual_seed(hash(name) % 2**31)
    lora_A, lora_B = {}, {}
    for i in range(num_layers):
        key = f"model.layers.{i}.self_attn.q_proj"
        lora_A[key] = torch.randn(rank, d_in)
        lora_B[key] = torch.randn(d_out, rank)
    return LoRAWeights(name=name, lora_A=lora_A, lora_B=lora_B, rank=rank, alpha=1.0)


# ---------------------------------------------------------------------------
# 1. Adapter state_dict round-trip
# ---------------------------------------------------------------------------


class TestAdapterRoundTrip:
    """from_state_dict → to_state_dict round-trip must preserve weights."""

    def test_round_trip_no_prefix(self):
        """Keys without PEFT prefix survive a round-trip."""
        lora = _make_lora("rt_no_prefix")
        sd = lora.to_state_dict()
        restored = LoRAWeights.from_state_dict("restored", sd, alpha=lora.alpha)
        for key in lora.lora_A:
            assert key in restored.lora_A, f"Missing A key: {key}"
            assert torch.allclose(lora.lora_A[key], restored.lora_A[key])
        for key in lora.lora_B:
            assert key in restored.lora_B, f"Missing B key: {key}"
            assert torch.allclose(lora.lora_B[key], restored.lora_B[key])

    def test_round_trip_with_prefix(self):
        """Keys that already have base_model.model. prefix must not double-prefix."""
        lora = _make_lora("rt_prefix")
        sd = lora.to_state_dict(prefix="base_model.model.")
        for k in sd:
            assert not k.startswith("base_model.model.base_model.model."), \
                f"Double prefix detected: {k}"

        restored = LoRAWeights.from_state_dict("restored", sd, alpha=lora.alpha)
        for key in lora.lora_A:
            assert key in restored.lora_A, f"Missing A key after prefix round-trip: {key}"
            assert torch.allclose(lora.lora_A[key], restored.lora_A[key])

    def test_from_state_dict_strips_prefix(self):
        """from_state_dict strips leading 'base_model.model.' from keys."""
        raw_sd = {
            "base_model.model.layer.0.lora_A.weight": torch.randn(4, 8),
            "base_model.model.layer.0.lora_B.weight": torch.randn(8, 4),
        }
        lora = LoRAWeights.from_state_dict("test", raw_sd)
        assert "layer.0" in lora.lora_A
        assert "layer.0" in lora.lora_B
        assert not any(k.startswith("base_model.model.") for k in lora.lora_A)

    def test_double_round_trip(self):
        """Two consecutive round-trips must be idempotent."""
        lora = _make_lora("double_rt")
        sd1 = lora.to_state_dict()
        lora2 = LoRAWeights.from_state_dict("r2", sd1)
        sd2 = lora2.to_state_dict()
        assert set(sd1.keys()) == set(sd2.keys())
        for k in sd1:
            assert torch.allclose(sd1[k], sd2[k])


# ---------------------------------------------------------------------------
# 2. BGD computation — verify all layers are used
# ---------------------------------------------------------------------------


class TestBGDAllLayers:
    """compute_bgd_matrix must aggregate over ALL shared layers."""

    def test_bgd_uses_all_layers(self):
        """BGD result changes when an extra layer is added (proves it's not capped)."""
        lora_a = _make_lora("bgd_a", num_layers=3)
        lora_b = _make_lora("bgd_b", num_layers=3)
        bgd_3 = compute_bgd_matrix([lora_a, lora_b])
        assert bgd_3.shape == (2, 2)
        assert bgd_3[0, 1] > 0

        lora_a6 = _make_lora("bgd_a", num_layers=6)
        lora_b6 = _make_lora("bgd_b", num_layers=6)
        bgd_6 = compute_bgd_matrix([lora_a6, lora_b6])
        assert bgd_6[0, 1] != pytest.approx(bgd_3[0, 1], abs=1e-6), \
            "BGD unchanged with more layers — likely not using all layers"

    def test_bgd_symmetric(self):
        lora_a = _make_lora("sym_a", num_layers=4)
        lora_b = _make_lora("sym_b", num_layers=4)
        bgd = compute_bgd_matrix([lora_a, lora_b])
        assert bgd[0, 1] == pytest.approx(bgd[1, 0], abs=1e-8)

    def test_bgd_self_distance_near_zero(self):
        lora_a = _make_lora("self_a", num_layers=4)
        bgd = compute_bgd_matrix([lora_a, lora_a])
        assert bgd[0, 1] == pytest.approx(0.0, abs=1e-2)

    def test_similarity_matrix_uses_all_layers(self):
        """compute_similarity_matrix must aggregate over all layers, not just first."""
        proj = GrassmannProjector(svd_rank=4)
        lora_a = _make_lora("sim_a", num_layers=3)
        lora_b = _make_lora("sim_b", num_layers=3)
        sim_3 = compute_similarity_matrix([lora_a, lora_b], proj)
        assert sim_3.shape == (2, 2)
        assert sim_3[0, 1] > 0

        lora_a6 = _make_lora("sim_a", num_layers=6)
        lora_b6 = _make_lora("sim_b", num_layers=6)
        sim_6 = compute_similarity_matrix([lora_a6, lora_b6], proj)
        assert sim_6[0, 1] != pytest.approx(sim_3[0, 1], abs=1e-6), \
            "Similarity unchanged with more layers — likely not using all layers"


# ---------------------------------------------------------------------------
# 3. Domain coverage
# ---------------------------------------------------------------------------


class TestDomainCoverage:
    """All 12 YAML domains must have a corresponding benchmark entry."""

    @pytest.fixture()
    def yaml_domains(self):
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "domains.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return set(cfg["domains"].keys())

    @pytest.fixture()
    def eval_domains(self):
        import importlib, types
        stub_modules = {}
        for mod_name in ["datasets", "peft", "transformers"]:
            stub_modules[mod_name] = types.ModuleType(mod_name)
            sys.modules.setdefault(mod_name, stub_modules[mod_name])
        if not hasattr(sys.modules["datasets"], "load_dataset"):
            sys.modules["datasets"].load_dataset = lambda *a, **kw: None
        if not hasattr(sys.modules["peft"], "PeftModel"):
            sys.modules["peft"].PeftModel = type("PeftModel", (), {})
        for attr in ["AutoModelForCausalLM", "AutoTokenizer"]:
            if not hasattr(sys.modules["transformers"], attr):
                setattr(sys.modules["transformers"], attr, type(attr, (), {}))
        spec = importlib.util.spec_from_file_location(
            "eval_domain_accuracy",
            str(Path(__file__).resolve().parent.parent / "scripts" / "eval_domain_accuracy.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return set(mod.ALL_EVAL_DOMAINS), set(mod.DOMAIN_BENCHMARKS.keys())

    def test_all_yaml_domains_have_benchmarks(self, yaml_domains, eval_domains):
        all_eval, bench_keys = eval_domains
        missing = yaml_domains - bench_keys
        assert not missing, f"YAML domains without benchmark config: {missing}"

    def test_all_yaml_domains_in_default_eval(self, yaml_domains, eval_domains):
        all_eval, _ = eval_domains
        missing = yaml_domains - all_eval
        assert not missing, f"YAML domains not in default eval list: {missing}"

    def test_twelve_domains(self, yaml_domains):
        assert len(yaml_domains) == 12, f"Expected 12 domains, got {len(yaml_domains)}"


# ---------------------------------------------------------------------------
# 4. GrassMerge basic sanity
# ---------------------------------------------------------------------------


class TestGrassMerge:
    def test_merge_preserves_rank(self):
        lora_a = _make_lora("gm_a", rank=4)
        lora_b = _make_lora("gm_b", rank=4)
        merger = GrassMerge()
        merged = merger.merge([lora_a, lora_b])
        assert merged.rank == 4

    def test_merge_all_keys_present(self):
        lora_a = _make_lora("gm_ka", num_layers=3)
        lora_b = _make_lora("gm_kb", num_layers=3)
        merger = GrassMerge()
        merged = merger.merge([lora_a, lora_b])
        assert len(merged.lora_A) == 3
        assert len(merged.lora_B) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
