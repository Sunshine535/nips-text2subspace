"""Regression test: BCFF target Y=y1+y2 is tautological.

The current bcff_merge objective sets Y_target = y_1 + y_2 where y_1 and y_2
are already among the candidates. Ridge regression therefore always returns
coefficients approximately [1, 1, 0, 0], making cross-terms vacuous.

This test documents the bug and prevents future reuse of the self-target.
"""
import torch
import pytest


def test_bcff_tautology_random_data():
    """With random A/B factors and random calibration, coefficients should be ~[1,1,0,0]."""
    torch.manual_seed(42)
    d_in, d_out, r = 64, 64, 8
    n_samples = 200

    B1 = torch.randn(d_out, r)
    A1 = torch.randn(r, d_in)
    B2 = torch.randn(d_out, r)
    A2 = torch.randn(r, d_in)

    dw_11 = B1 @ A1
    dw_22 = B2 @ A2
    dw_12 = B1 @ A2
    dw_21 = B2 @ A1

    X = torch.randn(n_samples, d_in)

    y_1 = (dw_11 @ X.T).T
    y_2 = (dw_22 @ X.T).T
    y_12 = (dw_12 @ X.T).T
    y_21 = (dw_21 @ X.T).T

    Y_target = y_1 + y_2

    F_flat = torch.stack([y_1, y_2, y_12, y_21], dim=2).reshape(-1, 4)
    Y_flat = Y_target.reshape(-1)

    reg = 1e-4
    FtF = F_flat.T @ F_flat
    FtY = F_flat.T @ Y_flat
    coeffs = torch.linalg.solve(FtF + reg * torch.eye(4), FtY)

    c1, c2, c3, c4 = coeffs.tolist()
    assert abs(c1 - 1.0) < 0.01, f"c1={c1}, expected ~1.0"
    assert abs(c2 - 1.0) < 0.01, f"c2={c2}, expected ~1.0"
    assert abs(c3) < 0.01, f"c3={c3}, expected ~0.0"
    assert abs(c4) < 0.01, f"c4={c4}, expected ~0.0"


def test_bcff_tautology_is_structural():
    """Tautology holds regardless of data distribution (uniform, skewed, etc)."""
    for seed in [1, 7, 42, 100, 999]:
        torch.manual_seed(seed)
        d_in, d_out, r = 32, 32, 4
        n = 100

        B1, A1 = torch.randn(d_out, r), torch.randn(r, d_in)
        B2, A2 = torch.randn(d_out, r), torch.randn(r, d_in)

        X = torch.randn(n, d_in) * (seed % 5 + 1)

        y_1 = (B1 @ A1 @ X.T).T
        y_2 = (B2 @ A2 @ X.T).T
        y_12 = (B1 @ A2 @ X.T).T
        y_21 = (B2 @ A1 @ X.T).T

        F_flat = torch.stack([y_1, y_2, y_12, y_21], dim=2).reshape(-1, 4)
        Y_flat = (y_1 + y_2).reshape(-1)

        coeffs = torch.linalg.solve(F_flat.T @ F_flat + 1e-4 * torch.eye(4), F_flat.T @ Y_flat)

        assert abs(coeffs[0] - 1.0) < 0.02, f"seed={seed}: c1={coeffs[0]}"
        assert abs(coeffs[1] - 1.0) < 0.02, f"seed={seed}: c2={coeffs[1]}"
        assert abs(coeffs[2]) < 0.02, f"seed={seed}: c3={coeffs[2]}"
        assert abs(coeffs[3]) < 0.02, f"seed={seed}: c4={coeffs[3]}"


if __name__ == "__main__":
    test_bcff_tautology_random_data()
    test_bcff_tautology_is_structural()
    print("BCFF tautology regression test PASSED")
    print("This confirms the BCFF objective is vacuous and should NOT be used as a method")
