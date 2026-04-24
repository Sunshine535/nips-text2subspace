# Package for GPT-5.5 Pro Review

## Summary of Changes

1. **Archived pre-CARR files**: RESULTS_REPORT.md, README_RUN.md, PROPOSAL.md → `archive/20260424_pre_carr/`
2. **Fixed P0 bug**: MMLU train/test leakage in `scripts/train_domain_lora.py` (test→auxiliary_train)
3. **Created split manifest**: `configs/splits.yaml` with leakage risk flags for all domains
4. **Created split checker**: `scripts/check_splits.py` — identifies 4 safe, 4 leaky domains
5. **BCFF tautology regression test**: `tests/test_bcff_tautology.py` — documents and prevents reuse
6. **Implemented conflict diagnostics**: `src/conflict_diagnostics.py` with activation-conditioned Gram
7. **Implemented CARR router**: `src/conflict_aware_routing.py` with reliability gate, conflict features, base fallback
8. **Created CARR config**: `configs/carr_minimal.yaml` with A/B/C mode definitions
9. **10 tests written**: 2 tautology + 3 conflict + 5 router, all passing locally

## Files Changed

| Action | File |
|--------|------|
| EDIT | `scripts/train_domain_lora.py` (MMLU split fix) |
| EDIT | `src/conflict_aware_routing.py` (detach fix) |
| NEW | `configs/splits.yaml` |
| NEW | `configs/carr_minimal.yaml` |
| NEW | `scripts/check_splits.py` |
| NEW | `src/conflict_diagnostics.py` |
| NEW | `src/conflict_aware_routing.py` |
| NEW | `tests/test_bcff_tautology.py` |
| NEW | `tests/test_conflict_diagnostics.py` |
| NEW | `tests/test_conflict_aware_routing.py` |
| NEW | `archive/20260424_pre_carr/` (3 archived files) |
| NEW | `reports/` (8 report files) |

## Commands Run

| Command | Result |
|---------|--------|
| `python3 scripts/check_splits.py` | PASS: 4 safe, 4 leaky |
| `python3 tests/test_bcff_tautology.py` | PASS: coeffs ≈ [1,1,0,0] |
| `python3 tests/test_conflict_diagnostics.py` | PASS: 3/3 tests |
| `python3 tests/test_conflict_aware_routing.py` | PASS: 5/5 tests |
| `python3 -c "from src.conflict_diagnostics import *; ..."` | PASS: clean import |

## Result Tables

### Split Audit
| Domain | Status | Risk |
|--------|--------|------|
| code | SAFE | None |
| math | SAFE | None |
| medical | SAFE | None |
| science | SAFE | None |
| geography | LEAKY | MMLU test split training |
| history | LEAKY | MMLU test split training |
| philosophy | LEAKY | MMLU test split training |
| psychology | LEAKY | MMLU test split training |

### Existing Results (Historical, not modified)
All SFC/FLC/BCFF results preserved in `results-synced/` as frozen evidence.

## Mechanism Logs

CARR module includes logging for: base_gate_mean, static_gate_mean, adapter_gate_means, gate_entropy. Not yet tested on real model (needs GPU).

## Failed Tests

None. All 10 tests pass locally.

## Unresolved Questions

1. Does CARR actually beat TA/TIES on science+medical pair? (Needs GPU)
2. Does CARR router converge on calibration data? (Needs GPU)
3. Can existing contaminated adapters still be used for science/medical? (Yes, those use different training datasets)
4. Is CARR sufficiently different from LoRA-Flow/MixLoRA? (Needs ablation)
5. Should eval_domain_accuracy.py seed be fixed before CARR experiments? (Yes, but lower priority than CARR implementation)

## Whether New Results Support Original Diagnosis

- **Split leakage**: CONFIRMED (4/8 domains leaky) — supports diagnosis P13
- **BCFF tautology**: CONFIRMED by regression test — supports diagnosis P9
- **Conflict diagnostics work**: Activation-conditioned Gram correctly distinguishes identical/orthogonal/activation-dependent conflict — supports diagnosis that activation-space diagnostics are needed
- **CARR router module**: Passes base equivalence and valid gate distribution — architecture is sound
- **No GPU results yet**: A/B/C comparison not yet run — diagnosis success prediction still untested

## What GPT-5.5 Pro Should Review Next

1. Verify `src/conflict_aware_routing.py` implementation matches the CARR specification
2. Review `src/conflict_diagnostics.py` — is activation-conditioned Gram sufficient?
3. Check if CARR training objective in future `scripts/train_carr_router.py` avoids the BCFF tautology pattern
4. After GPU experiments: review A/B/C comparison, gate statistics, and mechanism ablations
5. Assess novelty differentiation vs LoRA-Flow/AdapterFusion
