# Minimal Experiment Results

## Local Tests (no GPU)

| Experiment | Command | Result | Pass/Fail |
|------------|---------|--------|-----------|
| Split sanity (safe domains) | `check_splits.py --domains science,medical --mode require_safe` | 0 leaky | PASS |
| Split sanity (all) | `check_splits.py --mode audit_all` | 4 safe, 4 leaky | PASS (audit) |
| BCFF tautology | `test_bcff_tautology.py` | coeffs≈[1,1,0,0] | PASS |
| Conflict: identical | `test_conflict_diagnostics.py` | cosine>0.99 | PASS |
| Conflict: orthogonal | same | cosine<0.15 | PASS |
| Conflict: activation-dep | same | energy diff>0.01 | PASS |
| CARR: base equivalence | `test_conflict_aware_routing.py` | diff<1e-5 | PASS |
| CARR: valid gates | same | sum=1, non-negative | PASS |
| CARR: input-conditioned | same | different h → different gates | PASS |
| CARR: conflict-conditioned | same | different cs → different gates | PASS |
| CARR: no_base_fallback | same | removes base choice | PASS |
| CARR: top_k masks | same | sparse adapter selection | PASS |
| CARR: gradient flow | same | router params get gradients | PASS |
| Import smoke | inline | all modules import | PASS |

## GPU Experiments (pod, 4×H200)

| Experiment | Config | Dataset | Seeds | Metric | Result | Expected | Pass/Fail |
|------------|--------|---------|-------|--------|--------|----------|-----------|
| One-batch overfit | CARR full, 50 steps | science+medical calib | 1 | loss/entropy | loss 8.02→7.48, entropy 1.33→0.19 | loss↓, entropy↓ | **PASS** |
| Gate layer differentiation | same | same | 1 | per-layer base gate | L27=0.997, L19=0.01 | non-uniform | **PASS** |
| A: Static TA (seed 1) | TA merge+SVD | ARC+MedMCQA 50s | 1 | accuracy | sci=0.84, med=0.64, mean=0.74 | baseline | **PASS** |
| B: CARR no mech (seed 1) | no rel/conf | same | 1 | accuracy | sci=0.84, med=0.64, mean=0.74 | ≤ C | **PASS** |
| C: Full CARR (seed 1) | full | same | 1 | accuracy | sci=0.88, med=0.66, mean=0.77 | > A, > B | **PASS** |
| A: Static TA (seed 2) | TA merge+SVD | same | 2 | accuracy | sci=0.84, med=0.64, mean=0.74 | stable | **PASS** |
| B: CARR no mech (seed 2) | no rel/conf | same | 2 | accuracy | sci=0.86, med=0.60, mean=0.73 | ≤ C | **PASS** |
| C: Full CARR (seed 2) | full | same | 2 | accuracy | sci=0.88, med=0.64, mean=0.76 | > A, > B | **PASS** |
| A: Static TA (seed 3) | TA merge+SVD | same | 3 | accuracy | sci=0.84, med=0.64, mean=0.74 | stable | **PASS** |
| B: CARR no mech (seed 3) | no rel/conf | same | 3 | accuracy | sci=0.84, med=0.66, mean=0.75 | ≤ C | **PASS** |
| C: Full CARR (seed 3) | full | same | 3 | accuracy | sci=0.88, med=0.64, mean=0.76 | > A, > B | **PASS** |

## 3-Seed Aggregate

| Method | Mean±Std | C>Method? |
|--------|----------|-----------|
| Base | 0.680±0.000 | Yes (+0.083) |
| Single best | 0.740±0.000 | Yes (+0.023) |
| A: Static TA | 0.740±0.000 | Yes (+0.023) |
| B: CARR no mech | 0.740±0.010 | Yes (+0.023) |
| **C: Full CARR** | **0.763±0.006** | — |
