# GPT-5.5 Pro Report Extraction

## Diagnosis File Used

`GPT55_DIAGNOSIS.md` (repository root, 1244 lines)

## Recommended MAIN METHOD PATH

**CARR: Conflict-Aware Reliability-Gated Residual Routing**

Decompose multi-adapter composition into:
1. Static compatible component (TA/TIES/GrassMerge safe merge)
2. Conflict residual component (per-adapter residuals projected to bottleneck/conflict directions)
3. Reliability/conflict gate (small router with base fallback)
4. Composed hidden update: `h' = h + g_static * d_static + Σ g_i * d_i_conflict`

## Missing Mechanism

**Reliability-Calibrated, Input-Conditioned Conflict Arbitration**

All previous methods (SFC, FLC, BCFF) assume global static merge. The missing mechanism is per-input, per-layer, per-token decision-making about when to use base, when to use static merge, and when to route a specific adapter's conflict residual.

## Evidence From Positive Results

- Science adapter sometimes helps (+8% over base) → some adapter residuals are reliable
- TA/TIES strong on some pairs → static compatible directions can be safely merged
- GrassMerge old report beat TA/TIES/DARE mean (but not base mean) → geometry may help selectively

## Evidence From Negative Results

- SFC: static feature steering fails (P5, P6) → missing input conditioning
- FLC: rank-r static compression destroys information (P7) → missing conflict-aware residual routing
- BCFF: tautological target learns [1,1,0,0] (P9) → missing downstream-aligned objective
- Weak math/medical adapters hurt composition (P3, P11) → missing reliability calibration and abstention

## Evidence From Unstable Results

- BCFF sometimes helps science but hurts philosophy (P10) → uncalibrated composition exploits strong residual but harms weak domains
- GrassMerge positive relative to TA but negative relative to base (P4) → geometry insufficient without reliability

## Evidence From Failed Ablations

- Max-pool SFC, FLC LS merge, BCFF cross-factor → all three expose different facets of the same problem: static methods cannot handle heterogeneous adapter reliability and inter-adapter conflict

## Why Existing Best Positive Fragment Is Insufficient

TA/TIES/GrassMerge are static — they output one merged adapter for all inputs. They cannot explain why math/medical/philosophy get hurt, cannot provide per-input abstention, and cannot route conflict residuals selectively.

## Files to Inspect

- `scripts/train_domain_lora.py` — MMLU split leakage (P0)
- `scripts/eval_domain_accuracy.py` — fixed sampling seed, MCQ extraction (P0)
- `src/cross_factor_fusion.py` — BCFF tautology (P0)
- `src/rank_bottleneck.py` — CRS completeness
- `src/lora_algebra.py` — DARE seed issue
- `configs/domains.yaml` — split configuration

## Files to Edit

- Add `configs/splits.yaml` — train/calib/test split manifest
- Add `scripts/check_splits.py` — split overlap verification
- Add `src/conflict_diagnostics.py` — activation-conditioned conflict metrics
- Add `src/conflict_aware_routing.py` — CARR router module
- Add `scripts/train_carr_router.py` — CARR training
- Add `scripts/eval_carr.py` — CARR evaluation with A/B/C modes
- Add `configs/carr_minimal.yaml` — CARR config
- Add `src/data_sanity.py` — calibration fail-fast
- Add `tests/test_bcff_tautology.py` — regression test
- Add `tests/test_conflict_diagnostics.py` — conflict metric tests
- Add `tests/test_conflict_aware_routing.py` — CARR base equivalence tests
- Edit `scripts/eval_domain_accuracy.py` — fix seed, add logprob MCQ
- Edit `scripts/train_domain_lora.py` — fix MMLU split

## Files to Archive

- SFC as main method → historical negative evidence
- FLC as main method → ablation only
- BCFF as main method → historical negative evidence
- `RESULTS_REPORT.md` → low-confidence historical
- `README_RUN.md` → stale paths
- Text2Subspace pilot results → not true text-to-adapter

## Files to Keep

- `PROGRESS.md` — critical evidence ledger
- `PROOF_AUDIT.md` — theory audit
- `review-stage/AUTO_REVIEW.md` — review history
- `results-synced/` — frozen historical evidence
- `src/lora_algebra.py` — baselines (after DARE seed fix)

## Files to Keep Only as Baseline

- TA, TIES, DARE, GrassMerge implementations in `src/lora_algebra.py`

## Files to Keep Only as Ablation

- SFC static steering
- FLC static LS merge
- BCFF tautological merge
- BAC static variant

## Suspected Bugs

| Priority | Bug | File | Impact |
|----------|-----|------|--------|
| P0 | MMLU test split used for training | `scripts/train_domain_lora.py` | Train/test leakage |
| P0 | BCFF Y_target = y1+y2 tautology | `src/cross_factor_fusion.py` | Method vacuous |
| P0 | FLC empty calibration not caught | FLC eval scripts | Invalid results |
| P0 | Eval dataset sampling seed=42 hardcoded | `scripts/eval_domain_accuracy.py` | Multi-seed meaningless |
| P1 | DARE hardcoded seed 42 | `src/lora_algebra.py` | Baseline unfairness |
| P1 | SFC interference assumes non-negative coefficients | `src/sparse_feature_composition.py` | Diagnostic invalid |

## Required Logging

- Base gate rate, adapter gate rate, gate entropy
- Reliability calibration curve (per-adapter ECE)
- Conflict scores before/after routing
- Routed residual norm
- Base KL divergence
- Per-domain metrics with sample IDs
- Seed, checkpoint hash, config, command

## Required Minimal Experiments

15-item queue (P0-P2), starting with:
1. Smoke/import tests
2. Split sanity
3. Metric sanity
4. One-batch router overfit
5. Checkpoint/base equivalence
6. BCFF tautology regression
7. Reproduce FLC negative
8. Static baseline reproduction
9-11. A/B/C comparison
12-14. Ablations (no reliability, no conflict, no base fallback)
15. 3-seed stability

## Required Core Comparison

A. Existing Best Positive Fragment Only (static TA/TIES/GrassMerge)
B. New MAIN METHOD Without New Mechanism (CARR architecture but no reliability/conflict)
C. Full New MAIN METHOD (CARR with reliability + conflict + base fallback)

## Required Baselines

Base model, Individual LoRA, TA, TIES, DARE, RegMean, KnOTS, GrassMerge, AdapterFusion, LoraHub, LoRA-Flow, MixLoRA (where feasible)

## Stop / Continue / Pivot Criteria

- **Continue**: Split/metric checks pass; ≥2 reliable non-leaky domains; Full CARR > A and B over 3 seeds; mechanism logs show nontrivial gate use
- **Stop**: Split leakage unfixable; no adapter beats base; Full CARR ≤ static baseline
- **Pivot**: Oracle routing works but learned routing fails; CARR only works by domain labels; official LoRA-Flow/MixLoRA clearly beats CARR
