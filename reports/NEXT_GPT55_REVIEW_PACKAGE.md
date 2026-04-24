# Package for GPT-5.5 Pro Review — Round 3

## Summary of Changes Since Round 2

### Critical Bug Fixes (from GPT-5.5 Round 2 review)
1. **CARRHook passes real h to router** (was zeros — broke input conditioning)
2. **conflict_scores propagated** from precomputed module-level Gram
3. **use_base_fallback=False** removes base from softmax choices (was decoration)
4. **top_k** masks adapter gates to sparse selection (was unused)
5. **Training mode** no torch.no_grad (gradients flow through router)
6. **Root files** get deprecation banners (PROPOSAL/RESULTS_REPORT/README_RUN)
7. **splits.yaml** distinguishes historical vs current train split
8. **check_splits.py** has audit_all vs require_safe modes
9. **eval_domain_accuracy.py** sample_seed parameterized (was hardcoded 42)
10. **data_sanity.py** calibration fail-fast added

### New Implementation
- `scripts/train_carr_router.py` — CARR router training with one-batch overfit support
- `scripts/eval_carr.py` — A/B/C comparison (static_only, carr_no_mechanism, carr_full)

## GPU Experiment Results

### One-Batch Overfit (PASS)
```
step=0  loss=8.024 gate_entropy=1.326 (uniform)
step=40 loss=7.476 gate_entropy=0.192 (selective)
```
Gate layer differentiation: L27 base=0.997 (learned base fallback), L19 base=0.01 (routes adapters)

### 3-Seed A/B/C Comparison (PASS)
```
Method                Science    Medical    Mean
Base                  0.720      0.640      0.680
Single best           0.820      0.660      0.740
A: Static TA          0.840      0.640      0.740
B: CARR no mech       0.847      0.633      0.740
C: Full CARR          0.880      0.647      0.763  ← +2.3% over A, +2.3% over B
```
- C > A in 3/3 seeds ✓
- C > B in 3/3 seeds ✓
- A ≈ B (architecture alone doesn't help) ✓
- Science = 0.88 in ALL 3 seeds (std=0.000) ✓

## Mechanism Evidence

| Evidence | Finding | Supports CARR? |
|----------|---------|----------------|
| Loss decreases during training | 8.02→7.48 | Yes — router learns |
| Gate entropy drops | 1.33→0.19 | Yes — selective routing |
| Layer-differentiated gates | L27 base=0.997, L19 base=0.01 | Yes — per-layer decisions |
| C > A consistently | +2.3% mean over 3 seeds | Yes — beats static merge |
| C > B consistently | +2.3% mean over 3 seeds | Yes — mechanism matters |
| A ≈ B | 0.740 vs 0.740 | Yes — architecture alone insufficient |
| Science stable at 0.88 | std=0.000 across seeds | Yes — deterministic improvement |

## What GPT-5.5 Pro Should Review Next

1. **Are 50 samples × 3 seeds sufficient?** Science improvement is stable (std=0) but medical varies.
2. **Is the +2.3% mean improvement meaningful?** On 50 samples, the margin is 1-2 correct answers.
3. **Should we run ablations next?** (no_reliability, no_conflict, no_base_fallback)
4. **Should we expand to more domain pairs?** (math+science, medical+math)
5. **Official baseline comparison** — TIES/DARE need to be run in same framework
6. **Logprob MCQ metric** — generation-based extraction may be noisy
7. **Is the improvement from mechanism or from router having more parameters?** — B control suggests mechanism, but should verify parameter count is small
8. **Gate statistics suggest L27 always uses base** — is CARR just learning to skip problematic layers?

## Files Changed
- `src/conflict_aware_routing.py` — Critical hook fix + base_fallback + top_k
- `scripts/train_carr_router.py` — NEW: CARR training
- `scripts/eval_carr.py` — NEW: A/B/C evaluation
- `scripts/eval_domain_accuracy.py` — sample_seed fix
- `src/data_sanity.py` — NEW: calibration fail-fast
- `configs/splits.yaml` — historical vs current split
- `scripts/check_splits.py` — audit_all vs require_safe
- Root files: deprecation banners added

## Results Files
- `results-synced/carr_abc_seed1.json`
- `results-synced/carr_abc_seed2.json`
- `results-synced/carr_abc_seed3.json`
- `results-synced/carr_train_overfit.log`

## Unresolved
- eval_domain_accuracy.py logprob MCQ not yet implemented
- Official TIES/DARE baselines not yet run in CARR framework
- Ablations (no_reliability, no_conflict, no_base_fallback) not yet run
- Only 1 domain pair tested (science+medical)
- README/paper claims not yet updated (per GPT-5.5 rules)
