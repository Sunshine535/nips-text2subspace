"""Tests for activation-conditioned conflict diagnostics."""
import torch
import pytest
import sys
sys.path.insert(0, ".")
from src.conflict_diagnostics import compute_activation_gram, compute_pair_conflict


def test_identical_adapters_high_conflict():
    """Two identical adapters should have cosine similarity ~1."""
    torch.manual_seed(42)
    d_in, d_out = 32, 32
    dw = torch.randn(d_out, d_in)
    H = torch.randn(100, d_in)

    grams = compute_activation_gram(
        delta_ws=[{"mod": dw}, {"mod": dw.clone()}],
        module_inputs={"mod": H},
    )
    metrics = compute_pair_conflict(grams["mod"])
    assert metrics["cosine_similarity"] > 0.99, f"Identical adapters should have cosine ~1, got {metrics['cosine_similarity']}"


def test_orthogonal_adapters_low_conflict():
    """Orthogonal adapters should have near-zero cosine similarity."""
    torch.manual_seed(42)
    d = 64
    U = torch.linalg.qr(torch.randn(d, d))[0]
    dw1 = U[:, :8] @ U[:, :8].T
    dw2 = U[:, 8:16] @ U[:, 8:16].T

    H = torch.randn(200, d)

    grams = compute_activation_gram(
        delta_ws=[{"mod": dw1}, {"mod": dw2}],
        module_inputs={"mod": H},
    )
    metrics = compute_pair_conflict(grams["mod"])
    assert abs(metrics["cosine_similarity"]) < 0.15, f"Orthogonal adapters should have low cosine, got {metrics['cosine_similarity']}"


def test_same_left_different_right_different_under_activation():
    """
    Critical test from GPT-5.5 diagnosis:
    Two adapters with same U (left space) but different V (right space) should
    produce DIFFERENT conflict under different activation covariance.
    This is what parameter-only CRS misses.
    """
    torch.manual_seed(42)
    d = 32
    r = 4

    U = torch.linalg.qr(torch.randn(d, d))[0][:, :r]
    V1 = torch.linalg.qr(torch.randn(d, d))[0][:, :r]
    V2 = torch.linalg.qr(torch.randn(d, d))[0][:, :r]

    S = torch.ones(r)
    dw1 = U @ torch.diag(S) @ V1.T
    dw2 = U @ torch.diag(S) @ V2.T

    # Activation distribution strongly aligned with V1
    H_v1 = torch.randn(500, r) @ V1.T + 0.001 * torch.randn(500, d)

    # Activation distribution strongly aligned with V2
    H_v2 = torch.randn(500, r) @ V2.T + 0.001 * torch.randn(500, d)

    grams_v1 = compute_activation_gram(
        delta_ws=[{"mod": dw1}, {"mod": dw2}],
        module_inputs={"mod": H_v1},
    )
    grams_v2 = compute_activation_gram(
        delta_ws=[{"mod": dw1}, {"mod": dw2}],
        module_inputs={"mod": H_v2},
    )

    conflict_v1 = compute_pair_conflict(grams_v1["mod"])
    conflict_v2 = compute_pair_conflict(grams_v2["mod"])

    cos_diff = abs(conflict_v1["cosine_similarity"] - conflict_v2["cosine_similarity"])
    energy_diff = abs(conflict_v1["interference_energy"] - conflict_v2["interference_energy"])
    either_differs = cos_diff > 0.01 or energy_diff > 0.01
    assert either_differs, (
        f"Same-U different-V adapters should show different conflict under different activations. "
        f"cosine diff={cos_diff:.6f}, energy diff={energy_diff:.6f}"
    )


if __name__ == "__main__":
    test_identical_adapters_high_conflict()
    print("PASS: identical adapters → high conflict")
    test_orthogonal_adapters_low_conflict()
    print("PASS: orthogonal adapters → low conflict")
    test_same_left_different_right_different_under_activation()
    print("PASS: same-left-different-right → activation-dependent conflict")
    print("\nAll conflict diagnostic tests PASSED")
