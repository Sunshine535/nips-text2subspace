# Bug Fix Log

## Bug Fix: MMLU Train/Test Leakage (P0)

Files changed: `scripts/train_domain_lora.py`
Reason: SPLIT_OVERRIDES used `"test"` for cais/mmlu, causing 4 domains (history, philosophy, geography, psychology) to train on the same split used for evaluation.
Evidence: Code line 110 `"cais/mmlu": "test"`, confirmed by `scripts/check_splits.py` output.
Change: Changed to `"auxiliary_train"` (99k samples, sufficient for training).
Verification command: `python3 scripts/check_splits.py --splits configs/splits.yaml`
Before: MMLU domains trained on test split → train/test leakage
After: MMLU domains will train on auxiliary_train split → no leakage (requires adapter retraining)
Remaining risk: Existing MMLU-based adapters on pod are still contaminated. Must retrain before using for CARR evaluation. For now, CARR must use only safe domains: science, medical, math, code.

## Bug Fix: BCFF Tautological Objective (P0)

Files changed: None (test added, code preserved as historical evidence)
Reason: `Y_target = y_1 + y_2` with candidates including `y_1, y_2` guarantees coefficients [1,1,0,0].
Evidence: `tests/test_bcff_tautology.py` confirms across multiple seeds.
Change: BCFF marked as historical negative evidence only, not main method.
Verification command: `python3 tests/test_bcff_tautology.py`
Before: BCFF presented as cross-factor learning method
After: BCFF documented as vacuous, regression test prevents reuse
Remaining risk: None for new method; old BCFF results remain in results-synced/ as historical.
