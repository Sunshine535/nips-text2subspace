# Patch Summary — Rounds 1-3 Combined

## Files Changed
- `src/conflict_aware_routing.py` — CARR router (created R1, fixed R2: real h, conflict scores, base_fallback, top_k, grad flow)
- `src/conflict_diagnostics.py` — Activation-conditioned Gram conflict metrics (R1)
- `src/data_sanity.py` — Calibration fail-fast (R3)
- `scripts/train_carr_router.py` — CARR training with one-batch overfit (R3)
- `scripts/eval_carr.py` — A/B/C comparison: static_only, carr_no_mechanism, carr_full (R3)
- `scripts/train_domain_lora.py` — MMLU split fix test→auxiliary_train (R1)
- `scripts/eval_domain_accuracy.py` — sample_seed parameterized (R3)
- `scripts/check_splits.py` — Split checker with audit_all/require_safe modes (R1, fixed R2)
- `configs/splits.yaml` — Split manifest with historical vs current splits (R1, updated R2)
- `configs/carr_minimal.yaml` — CARR experiment config (R1)
- `PROPOSAL.md` — Deprecation banner added (R2)
- `RESULTS_REPORT.md` — Deprecation banner added (R2)
- `README_RUN.md` — Deprecation banner added (R2)

## Files Added
- `tests/test_bcff_tautology.py` — 2 tests (R1)
- `tests/test_conflict_diagnostics.py` — 3 tests (R1)
- `tests/test_conflict_aware_routing.py` — 10 tests (R1+R2)
- `archive/20260424_pre_carr/` — Archived old files (R1)
- `reports/` — 12 report files (R1-R3)
- `results-synced/carr_abc_seed{1,2,3}.json` — 3-seed A/B/C results (R3)
- `results-synced/carr_train_overfit.log` — One-batch overfit log (R3)

## Files Archived
- RESULTS_REPORT.md, README_RUN.md, PROPOSAL.md → `archive/20260424_pre_carr/`

## Bugs Fixed
1. MMLU train/test leakage (test→auxiliary_train)
2. BCFF tautological target (documented via test)
3. CARRHook zeros input (→ real h)
4. CARRHook missing conflict_scores (→ propagated from Gram)
5. use_base_fallback decoration (→ removes base from softmax)
6. top_k unused (→ masks adapter gates)
7. Router no_grad in training mode (→ gradient flow)
8. eval_domain_accuracy hardcoded seed (→ parameterized)

## GPU Experiments Run
- 1 one-batch overfit (50 steps) — PASS
- 9 A/B/C evaluations (3 modes × 3 seeds) — ALL PASS
- 6 CARR router trainings (200 steps each)

## Key Result
```
C: Full CARR    mean=0.763±0.006
A: Static TA    mean=0.740±0.000   (+2.3%)
B: No mechanism mean=0.740±0.010   (+2.3%)
```
C > A and C > B in all 3 seeds. Mechanism confirmed active.

## Decision: CONTINUE
Minimal evidence supports the new mechanism. Ready for broader baselines and ablations.
