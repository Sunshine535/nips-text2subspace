# Current Result Audit

## Result Table

| Result | File | Dataset | Config | Seed | Metric | Value | Compared Against | Supports GPT-5.5 Diagnosis? | Notes |
|--------|------|---------|--------|------|--------|-------|------------------|------------------------------|-------|
| Base math | bcff_results.json | GSM8K 50s | default | unknown | accuracy | 0.02 | — | Yes (weak adapter) | Near-random |
| Base medical | bcff_results.json | MedMCQA 50s | default | unknown | accuracy | 0.64 | — | Yes | Moderate base |
| Base philosophy | bcff_results.json | MMLU 50s | default | unknown | accuracy | 0.42 | — | Yes | MMLU leakage risk |
| Base science | bcff_results.json | ARC 50s | default | unknown | accuracy | 0.72 | — | Yes | Strongest domain |
| Single(math) | bcff_results.json | GSM8K 50s | LoRA r16 | unknown | accuracy | 0.02 | base 0.02 | Yes (P3: weak adapter) | No improvement |
| Single(medical) | bcff_results.json | MedMCQA 50s | LoRA r16 | unknown | accuracy | 0.66 | base 0.64 | Yes | Marginal +0.02 |
| Single(philosophy) | bcff_results.json | MMLU 50s | LoRA r16 | unknown | accuracy | 0.44 | base 0.42 | Yes | LEAKAGE RISK +0.02 |
| Single(science) | bcff_results.json | ARC 50s | LoRA r16 | unknown | accuracy | 0.82 | base 0.72 | Yes (P2: useful) | **+0.10 real signal** |
| SFC math+sci | sfc_remaining.json | 50s | SFC hooks | unknown | accuracy | 0.00/0.76 | TA 0.00/0.84 | Yes (P5/P6: static fails) | Worse than TA |
| SFC sci+phil | sfc_remaining.json | 50s | SFC hooks | unknown | accuracy | 0.66/0.40 | TA 0.82/0.42 | Yes | Below base on science |
| FLC math+med | PROGRESS.md | 50s | rank-16 LS | unknown | accuracy | 0.00/0.16 | base 0.02/0.66 | Yes (P7: collapse) | Catastrophic |
| BCFF math+med | bcff_results.json | 50s | sum+SVD | unknown | accuracy | 0.02/0.66 | TA 0.02/0.66 | Yes (P9: tautology) | Equivalent to TA |
| BCFF sci+phil | bcff_results.json | 50s | sum+SVD | unknown | accuracy | 0.88/0.34 | TA 0.82/0.42 | Yes (P10: imbalance) | Helps science, hurts phil |
| BCFF med+phil | bcff_results.json | 50s | sum+SVD | unknown | accuracy | 0.66/0.46 | TA 0.66/0.72 | Yes | TA much better on phil |
| TIES math+sci | sfc_remaining.json | 50s | delta_W | unknown | accuracy | 0.00/0.86 | TA 0.00/0.84 | Yes (P1: strong static) | TIES is strong baseline |

## Variant Existence Check

| Variant | Exists? | Evidence |
|---------|---------|----------|
| A. Existing Best Positive Fragment Only | Partial | TA/TIES results exist but with leakage risk, no multi-seed |
| B. New MAIN METHOD Without New Mechanism | No | CARR not yet implemented |
| C. Full New MAIN METHOD | No | CARR not yet implemented |

## Result-Based Execution Decision

**Decision: FIX BUG FIRST**

**Reason**: P0 bug (MMLU test split leakage) contaminates philosophy and history adapter training AND evaluation. Before implementing CARR, we must:
1. Fix the split leakage
2. Fix the hardcoded eval sampling seed
3. Verify that at least science (ARC, non-leaky) and medical (MedMCQA, separate validation split) are trustworthy domains for CARR evaluation

Science adapter (+10% over base on ARC) is the strongest non-contaminated signal. Medical is marginal (+2%). Math is useless. Philosophy/history are MMLU-contaminated.

CARR minimal verification should use science+medical pair on non-leaky splits as the primary test case.
