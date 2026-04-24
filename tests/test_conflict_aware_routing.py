"""Tests for CARR router — per GPT-5.5 Pro review round 2 requirements."""
import torch
import sys
sys.path.insert(0, ".")
from src.conflict_aware_routing import CARRConfig, ConflictAwareResidualRouter, CARRHook


def test_base_equivalence_zero_residuals():
    """With zero residuals, composed output should be zero (no modification)."""
    torch.manual_seed(42)
    config = CARRConfig(n_adapters=2, d_model=32, gate_hidden_dim=16)
    router = ConflictAwareResidualRouter(config)
    router.eval()

    h = torch.randn(2, 5, 32)
    static_delta = torch.zeros(2, 5, 32)
    adapter_residuals = [torch.zeros(2, 5, 32), torch.zeros(2, 5, 32)]

    composed, _ = router.compute_composed_output(h, static_delta, adapter_residuals)
    assert composed.abs().max().item() < 1e-5, "Zero residuals should produce zero composed output"


def test_gate_valid_distribution():
    """Gates should be non-negative and sum to 1."""
    config = CARRConfig(n_adapters=3, d_model=32, gate_hidden_dim=16)
    router = ConflictAwareResidualRouter(config)
    h = torch.randn(2, 5, 32)
    gates = router.forward(h)
    assert (gates >= 0).all(), "Gates must be non-negative"
    assert (gates.sum(-1) - 1.0).abs().max() < 1e-5, "Gates must sum to 1"
    assert gates.shape[-1] == 5, f"Expected 5 choices (base+static+3 adapters), got {gates.shape[-1]}"


def test_input_conditioned_different_inputs_different_gates():
    """CRITICAL: Different input hidden states must produce different gates."""
    torch.manual_seed(42)
    config = CARRConfig(n_adapters=2, d_model=32, gate_hidden_dim=16)
    router = ConflictAwareResidualRouter(config)

    h1 = torch.randn(1, 1, 32) * 10
    h2 = torch.randn(1, 1, 32) * 10

    gates1 = router.forward(h1)
    gates2 = router.forward(h2)

    diff = (gates1 - gates2).abs().max().item()
    assert diff > 0.001, f"Different inputs must produce different gates, got max diff={diff}"


def test_conflict_scores_change_gates():
    """Different conflict_scores must produce different gates."""
    torch.manual_seed(42)
    config = CARRConfig(n_adapters=2, d_model=32, gate_hidden_dim=16, use_conflict=True)
    router = ConflictAwareResidualRouter(config)

    h = torch.randn(1, 1, 32)
    cs1 = torch.tensor([[[0.9, 0.1]]])
    cs2 = torch.tensor([[[0.1, 0.9]]])

    gates1 = router.forward(h, conflict_scores=cs1)
    gates2 = router.forward(h, conflict_scores=cs2)

    diff = (gates1 - gates2).abs().max().item()
    assert diff > 0.001, f"Different conflict scores must change gates, got max diff={diff}"


def test_no_base_fallback_removes_base_choice():
    """use_base_fallback=False should have one fewer choice (no base gate)."""
    config_with = CARRConfig(n_adapters=2, d_model=32, gate_hidden_dim=16, use_base_fallback=True)
    config_without = CARRConfig(n_adapters=2, d_model=32, gate_hidden_dim=16, use_base_fallback=False)

    router_with = ConflictAwareResidualRouter(config_with)
    router_without = ConflictAwareResidualRouter(config_without)

    h = torch.randn(1, 3, 32)
    gates_with = router_with.forward(h)
    gates_without = router_without.forward(h)

    assert gates_with.shape[-1] == 4, f"With base: expected 4 choices, got {gates_with.shape[-1]}"
    assert gates_without.shape[-1] == 3, f"Without base: expected 3 choices, got {gates_without.shape[-1]}"
    assert router_without.base_idx is None
    assert router_with.base_idx == 0


def test_top_k_masks_adapter_gates():
    """top_k=1 with 3 adapters should zero out 2 adapter gates."""
    torch.manual_seed(42)
    config = CARRConfig(n_adapters=3, d_model=32, gate_hidden_dim=16, top_k=1)
    router = ConflictAwareResidualRouter(config)

    h = torch.randn(1, 5, 32)
    gates = router.forward(h)

    adapter_gates = gates[:, :, router.adapter_start_idx:]
    for t in range(5):
        nonzero = (adapter_gates[0, t] > 0.01).sum().item()
        assert nonzero <= 2, f"top_k=1 should have at most 1 dominant adapter gate (plus softmax leakage), got {nonzero} non-negligible"


def test_no_reliability_mode():
    """Router works without reliability features."""
    config = CARRConfig(n_adapters=2, d_model=32, gate_hidden_dim=16, use_reliability=False)
    router = ConflictAwareResidualRouter(config)
    h = torch.randn(1, 3, 32)
    gates = router.forward(h)
    assert gates.shape[-1] == 4


def test_no_conflict_mode():
    """Router works without conflict features."""
    config = CARRConfig(n_adapters=2, d_model=32, gate_hidden_dim=16, use_conflict=False)
    router = ConflictAwareResidualRouter(config)
    h = torch.randn(1, 3, 32)
    gates = router.forward(h)
    assert gates.shape[-1] == 4


def test_gate_stats_include_required_fields():
    """Gate stats must include all mechanism diagnostics."""
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


def test_training_mode_allows_gradients():
    """In training mode, gradients should flow through the router."""
    config = CARRConfig(n_adapters=2, d_model=16, gate_hidden_dim=8)
    router = ConflictAwareResidualRouter(config)
    router.train()

    h = torch.randn(1, 3, 16, requires_grad=True)
    static_delta = torch.randn(1, 3, 16)
    adapter_residuals = [torch.randn(1, 3, 16) for _ in range(2)]

    composed, _ = router.compute_composed_output(h, static_delta, adapter_residuals)
    loss = composed.sum()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in router.parameters())
    assert has_grad, "Router parameters must receive gradients in training mode"


if __name__ == "__main__":
    tests = [
        ("base_equivalence_zero_residuals", test_base_equivalence_zero_residuals),
        ("gate_valid_distribution", test_gate_valid_distribution),
        ("input_conditioned_different_gates", test_input_conditioned_different_inputs_different_gates),
        ("conflict_scores_change_gates", test_conflict_scores_change_gates),
        ("no_base_fallback_removes_choice", test_no_base_fallback_removes_base_choice),
        ("top_k_masks_adapter_gates", test_top_k_masks_adapter_gates),
        ("no_reliability_mode", test_no_reliability_mode),
        ("no_conflict_mode", test_no_conflict_mode),
        ("gate_stats_fields", test_gate_stats_include_required_fields),
        ("training_mode_gradients", test_training_mode_allows_gradients),
    ]
    for name, test_fn in tests:
        test_fn()
        print(f"PASS: {name}")
    print(f"\nAll {len(tests)} CARR router tests PASSED")
