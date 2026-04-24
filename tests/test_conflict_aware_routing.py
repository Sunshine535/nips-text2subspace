"""Tests for CARR router: base equivalence, single-adapter match, gate behavior."""
import torch
import sys
sys.path.insert(0, ".")
from src.conflict_aware_routing import CARRConfig, ConflictAwareResidualRouter


def test_base_equivalence():
    """When all adapter residuals are zero, output should equal input."""
    torch.manual_seed(42)
    config = CARRConfig(n_adapters=2, d_model=32, gate_hidden_dim=16)
    router = ConflictAwareResidualRouter(config)
    router.eval()

    h = torch.randn(2, 5, 32)
    static_delta = torch.zeros(2, 5, 32)
    adapter_residuals = [torch.zeros(2, 5, 32), torch.zeros(2, 5, 32)]

    h_composed, stats = router.compute_composed_output(h, static_delta, adapter_residuals)
    diff = (h_composed - h).abs().max().item()
    assert diff < 1e-5, f"With zero residuals, output should equal input. Max diff={diff}"


def test_gate_outputs_valid_distribution():
    """Gates should be non-negative and sum to 1."""
    torch.manual_seed(42)
    config = CARRConfig(n_adapters=3, d_model=32, gate_hidden_dim=16)
    router = ConflictAwareResidualRouter(config)

    h = torch.randn(2, 5, 32)
    gates = router.forward(h)

    assert gates.shape == (2, 5, 5), f"Expected shape (2,5,5), got {gates.shape}"
    assert (gates >= 0).all(), "Gates should be non-negative"
    sums = gates.sum(dim=-1)
    assert (sums - 1.0).abs().max() < 1e-5, f"Gates should sum to 1, max deviation={sums.max()}"


def test_router_with_reliability_disabled():
    """Router should work without reliability features."""
    config = CARRConfig(n_adapters=2, d_model=32, gate_hidden_dim=16, use_reliability=False)
    router = ConflictAwareResidualRouter(config)

    h = torch.randn(1, 3, 32)
    gates = router.forward(h)
    assert gates.shape == (1, 3, 4)


def test_router_with_conflict_disabled():
    """Router should work without conflict features."""
    config = CARRConfig(n_adapters=2, d_model=32, gate_hidden_dim=16, use_conflict=False)
    router = ConflictAwareResidualRouter(config)

    h = torch.randn(1, 3, 32)
    gates = router.forward(h)
    assert gates.shape == (1, 3, 4)


def test_gate_stats_logged():
    """compute_composed_output should return gate statistics."""
    config = CARRConfig(n_adapters=2, d_model=16, gate_hidden_dim=8)
    router = ConflictAwareResidualRouter(config)

    h = torch.randn(1, 3, 16)
    static_delta = torch.randn(1, 3, 16) * 0.1
    adapter_residuals = [torch.randn(1, 3, 16) * 0.1 for _ in range(2)]

    _, stats = router.compute_composed_output(h, static_delta, adapter_residuals)

    assert "base_gate_mean" in stats
    assert "static_gate_mean" in stats
    assert "gate_entropy" in stats
    assert "adapter_gate_means" in stats
    assert len(stats["adapter_gate_means"]) == 2


if __name__ == "__main__":
    test_base_equivalence()
    print("PASS: base equivalence (zero residuals → output = input)")
    test_gate_outputs_valid_distribution()
    print("PASS: gate outputs valid probability distribution")
    test_router_with_reliability_disabled()
    print("PASS: router works without reliability")
    test_router_with_conflict_disabled()
    print("PASS: router works without conflict")
    test_gate_stats_logged()
    print("PASS: gate stats logged correctly")
    print("\nAll CARR router tests PASSED")
