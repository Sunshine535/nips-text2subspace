# Core Comparison

## Status: PENDING GPU EXPERIMENTS

The core A/B/C comparison has not yet been run. CARR router module is implemented and passes local tests, but requires GPU for real-model evaluation.

| Variant | Config | Dataset | Seeds | Metric Mean | Std | Compared To | Result | Interpretation |
|---------|--------|---------|-------|-------------|-----|-------------|--------|----------------|
| A. Static TA/TIES | carr_minimal.yaml mode=static_only | science+medical | 1,2,3 | — | — | — | NOT RUN | Baseline |
| B. CARR no mechanism | carr_minimal.yaml mode=carr_no_reliability_conflict | same | 1,2,3 | — | — | A | NOT RUN | Architecture-only control |
| C. Full CARR | carr_minimal.yaml mode=carr_full | same | 1,2,3 | — | — | A, B | NOT RUN | Full method |

## Decision: Cannot evaluate yet

No GPU experiments have been run. All interpretations below are hypothetical:

- If C > A and C > B: New mechanism likely adds value → proceed to baselines
- If C ≈ A: CARR router is just learning static merge → diagnose
- If C ≈ B: Reliability/conflict features not helping → check mechanism logs
- If C < A: CARR hurts performance → debug or stop
