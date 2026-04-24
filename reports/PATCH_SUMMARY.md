# Patch Summary

## Files Changed
- `scripts/train_domain_lora.py` — Fixed MMLU split leakage (test → auxiliary_train)
- `src/conflict_aware_routing.py` — Fixed tensor detach warning in gate stats

## Files Added
- `src/conflict_diagnostics.py` — Activation-conditioned conflict Gram metrics
- `src/conflict_aware_routing.py` — CARR router module (reliability gate + conflict routing + base fallback)
- `configs/splits.yaml` — Train/calib/test split manifest with leakage flags
- `configs/carr_minimal.yaml` — CARR experiment configuration
- `scripts/check_splits.py` — Split overlap verification script
- `tests/test_bcff_tautology.py` — BCFF tautology regression test (2 tests)
- `tests/test_conflict_diagnostics.py` — Conflict diagnostic tests (3 tests)
- `tests/test_conflict_aware_routing.py` — CARR router tests (5 tests)
- `archive/20260424_pre_carr/` — Archived pre-CARR files (RESULTS_REPORT.md, README_RUN.md, PROPOSAL.md)
- `reports/` — 12 report files documenting execution

## Files Archived
- `RESULTS_REPORT.md` → `archive/20260424_pre_carr/` (low-confidence GrassMerge report)
- `README_RUN.md` → `archive/20260424_pre_carr/` (stale Text2Subspace paths)
- `PROPOSAL.md` → `archive/20260424_pre_carr/` (invalidated SFC claims)

## Files Intentionally NOT Touched
- `results-synced/` — Frozen historical evidence
- `logs/` — Historical logs
- `PROGRESS.md` — Critical evidence ledger
- `PROOF_AUDIT.md` — Theory audit
- `review-stage/` — Review history
- `src/lora_algebra.py` — Baselines (DARE seed fix deferred)
- `src/sparse_feature_composition.py` — Historical negative evidence
- `src/functional_composition.py` — Historical ablation
- `src/cross_factor_fusion.py` — Historical negative evidence (tautology documented)

## Bugs Fixed
1. P0: MMLU train/test leakage → changed split to auxiliary_train
2. P0: BCFF tautology → documented via regression test, marked historical

## New Method Components Implemented
- `ConflictAwareResidualRouter` — Core CARR gate module
- `ReliabilityFeatures` — Per-adapter reliability estimation
- `CARRHook` — Inference-time hook for applying CARR routing
- `compute_activation_gram` — Activation-conditioned conflict metric
- `compute_pair_conflict` — Pairwise conflict scoring
- `compute_conflict_projector` — Conflict direction extraction
- `CARRConfig` — Configuration dataclass

## Configs Added
- `configs/splits.yaml` — Domain split manifest
- `configs/carr_minimal.yaml` — CARR experiment config with A/B/C modes

## Tests Added (10 total, all passing locally)
- 2 BCFF tautology tests
- 3 conflict diagnostic tests (identical, orthogonal, activation-dependent)
- 5 CARR router tests (base equiv, gate dist, no-reliability, no-conflict, stats)

## Commands Run
- `python3 scripts/check_splits.py` → PASS
- `python3 tests/test_bcff_tautology.py` → PASS
- `python3 tests/test_conflict_diagnostics.py` → PASS
- `python3 tests/test_conflict_aware_routing.py` → PASS
- `python3 -c "from src.conflict_diagnostics import *; ..."` → PASS

## Results Observed
- 4/8 domains have MMLU train/test leakage (history, philosophy, geography, psychology)
- 4/8 domains are safe (math, code, medical, science)
- BCFF tautology confirmed across 5 random seeds
- Conflict diagnostics correctly distinguish identical/orthogonal/activation-dependent scenarios
- CARR router maintains base equivalence (diff < 1e-5 with zero residuals)

## Failed Checks
None

## Unresolved Risks
- CARR not yet tested on real model (needs GPU)
- Existing MMLU adapters are contaminated (need retraining)
- High novelty risk vs LoRA-Flow/MixLoRA
- Current adapters may be too weak for meaningful composition
