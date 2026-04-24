# Claude Execution Plan

**Date**: 2026-04-24
**Diagnosis file**: `GPT55_DIAGNOSIS.md` (repository root)

## 1. MAIN METHOD PATH

**CARR: Conflict-Aware Reliability-Gated Residual Routing**

Static compatible merge as backbone + input-conditioned reliability gate + conflict-aware residual router + base fallback.

## 2. Missing Mechanism to Implement

**Reliability-Calibrated, Input-Conditioned Conflict Arbitration**

A small router (frozen base + frozen adapters) that decides per-input/layer/token: use base, use static merge, or route a specific adapter's conflict residual.

## 3. Current Evidence Supporting the Diagnosis

| Evidence | Source | Supports |
|----------|--------|----------|
| SFC static steering fails | PROGRESS.md, SFC logs | Missing input conditioning |
| FLC rank-r collapse (18-20% energy) | FLC logs | Missing conflict-aware routing |
| BCFF coefficients [1,1,0,0] | BCFF logs, code audit | Missing downstream-aligned objective |
| Science adapter +8% | Single adapter eval | Some residuals are reliably useful |
| TA/TIES strong baselines | BCFF results | Static compatible directions work |
| Math/medical no improvement | Single adapter eval | Need reliability calibration and abstention |
| MMLU split leakage risk | train_domain_lora.py code | Evaluation reliability undermined |

## 4. Current Evidence Contradicting or Weakening the Diagnosis

| Evidence | Source | Concern |
|----------|--------|---------|
| Confidence "medium-low" for success | GPT55 diagnosis itself | CARR may not produce positive results |
| High novelty risk (LoRA-Flow, MixLoRA, AdapterFusion close) | Related work analysis | CARR differentiation may be insufficient |
| Current adapters may be too weak | Single adapter evals | Even with perfect router, adapters don't help on math/medical |
| Only 32 LoRA modules (8 of 32 layers) | Adapter config | Limited adapter coverage |
| 50-sample evaluation noise | Current eval setup | Differences may be statistical noise |

## 5. Files to Inspect (before editing)

- `scripts/train_domain_lora.py` — MMLU split handling
- `scripts/eval_domain_accuracy.py` — seed, MCQ extraction
- `src/cross_factor_fusion.py` — BCFF tautology
- `src/lora_algebra.py` — DARE seed
- `src/rank_bottleneck.py` — CRS implementation
- `configs/domains.yaml` — current config
- `results-synced/*.json` — frozen results
- `tests/test_text2subspace.py` — existing tests

## 6. Files to Edit

| File | Change | Task |
|------|--------|------|
| `scripts/train_domain_lora.py` | Fix MMLU split | Task 2 |
| `scripts/eval_domain_accuracy.py` | Fix seed, add logprob MCQ | Task 3 |
| NEW `configs/splits.yaml` | Split manifest | Task 2 |
| NEW `scripts/check_splits.py` | Split overlap check | Task 2 |
| NEW `src/data_sanity.py` | Calibration fail-fast | Task 4 |
| NEW `tests/test_bcff_tautology.py` | BCFF regression test | Task 5 |
| NEW `src/conflict_diagnostics.py` | Activation-conditioned conflict | Task 6 |
| NEW `tests/test_conflict_diagnostics.py` | Conflict metric tests | Task 6 |
| NEW `src/conflict_aware_routing.py` | CARR router module | Task 7 |
| NEW `tests/test_conflict_aware_routing.py` | Base equivalence tests | Task 7 |
| NEW `scripts/train_carr_router.py` | CARR training | Task 8 |
| NEW `scripts/eval_carr.py` | CARR eval with A/B/C modes | Task 8 |
| NEW `configs/carr_minimal.yaml` | CARR config | Task 8 |

## 7. Files to Archive

| File | Reason | Archive Location |
|------|--------|-----------------|
| `RESULTS_REPORT.md` | Low-confidence historical | `archive/20260424_pre_carr/` |
| `README_RUN.md` | Stale paths | `archive/20260424_pre_carr/` |
| `PROPOSAL.md` | Claims invalidated | `archive/20260424_pre_carr/` |

## 8. Files NOT to Touch

- `results-synced/` — frozen historical evidence
- `logs/` — historical logs
- `PROGRESS.md` — critical evidence (append only)
- `PROOF_AUDIT.md` — theory audit
- `review-stage/AUTO_REVIEW.md` — review history
- `src/lora_algebra.py` — baselines (only expose DARE seed param, no logic change)

## 9. Tests Before Changes

- `pytest tests/test_text2subspace.py -q` — existing tests should still pass

## 10. Tests After Changes

- `pytest tests/ -q` — all tests including new ones
- `python scripts/check_splits.py` — no train/test overlap
- `pytest tests/test_bcff_tautology.py -q` — BCFF tautology documented
- `pytest tests/test_conflict_diagnostics.py -q` — conflict metrics work
- `pytest tests/test_conflict_aware_routing.py -q` — base equivalence holds

## 11. Rollback Conditions

- If split checks reveal that ALL domain adapters are leaky → stop, report, do not implement CARR
- If base equivalence test fails → stop, fix PEFT integration first
- If existing tests break → revert and fix
- If conflict diagnostics are numerically unstable → simplify before proceeding to router

## Execution Order

1. Create result registry (Task 1) — no code changes, just documentation
2. Fix split and sample manifest (Task 2) — P0 bug fix
3. Fix evaluation reliability (Task 3) — P0 bug fix
4. Add calibration fail-fast (Task 4) — P0 infrastructure
5. Archive BCFF as main method (Task 5) — documentation + test
6. Implement conflict diagnostics (Task 6) — new module
7. Implement CARR router (Task 7) — core new method
8. Add CARR train/eval scripts (Task 8) — experiment infrastructure
9. Add mechanism logging (Task 9) — verification
10. Run minimal verification queue (Task 10) — experiments

**Rule**: Each task must pass its verification before proceeding to the next.
