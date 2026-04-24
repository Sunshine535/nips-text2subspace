# Minimal Experiment Results

## Local Tests (no GPU required)

| Experiment | Command | Config | Dataset | Seed | Metric | Result | Expected | Pass/Fail | Interpretation |
|------------|---------|--------|---------|------|--------|--------|----------|-----------|----------------|
| Split sanity | `check_splits.py` | splits.yaml | all | N/A | overlap count | 0 safe overlap, 4 leaky flagged | 0 overlap | PASS | MMLU leakage confirmed |
| BCFF tautology | `test_bcff_tautology.py` | N/A | synthetic | 5 seeds | coeff error | all ≈[1,1,0,0] | [1,1,0,0] | PASS | BCFF objective vacuous |
| Conflict identical | `test_conflict_diagnostics.py` | N/A | synthetic | 42 | cosine | >0.99 | >0.99 | PASS | Gram correctly detects identity |
| Conflict orthogonal | same | N/A | synthetic | 42 | cosine | <0.15 | <0.15 | PASS | Gram correctly detects orthogonality |
| Conflict activation-dep | same | N/A | synthetic | 42 | energy diff | >0.01 | >0.01 | PASS | Activation covariance matters |
| CARR base equiv | `test_conflict_aware_routing.py` | N/A | synthetic | 42 | max diff | <1e-5 | <1e-5 | PASS | Zero residuals → no change |
| CARR gate dist | same | N/A | synthetic | 42 | sum=1 | ✓ | sum=1 | PASS | Valid probability distribution |
| Import smoke | inline python | N/A | N/A | N/A | import success | ✓ | ✓ | PASS | All new modules importable |

## GPU-Required Tests (PENDING)

| Experiment | Command | Status | Notes |
|------------|---------|--------|-------|
| One-batch overfit | `train_carr_router.py --overfit_one_batch` | NOT RUN | Needs pod GPU |
| Checkpoint integrity | `check_checkpoint_integrity.py` | NOT RUN | Needs pod |
| A: Static only | `eval_carr.py --mode static_only` | NOT RUN | Needs pod |
| B: CARR no mechanism | `eval_carr.py --mode carr_no_reliability_conflict` | NOT RUN | Needs pod |
| C: Full CARR | `eval_carr.py --mode carr_full` | NOT RUN | Needs pod |
| Ablations | various | NOT RUN | Needs pod |
