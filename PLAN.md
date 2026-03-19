# Plan: Text2Subspace (Stage-Gate v2)

## Gate 0: Adapter Zoo (Pending)
- [x] Added runnable low-rank/full-rank pilot on trace-derived supervision.
- [ ] Collect standardized adapters across task families.
- Go criterion: reproducible metadata and checkpoints.

## Gate 1: Canonicalization (Pilot proxy done, full pending)
- [x] Low-rank subspace proxy baseline implemented (`run_text2subspace_pilot.py`).
- [ ] Implement true basis alignment/canonicalization over adapter checkpoints.
- Go criterion: canonicalized target variance lower than raw targets.

## Gate 2: Generator (Pending)
- [ ] Train conditional generator and evaluate unseen tasks.
- Go criterion: at least 90% of LoRA quality with `>= 10x` faster adaptation.

## Gate 3: Composition
- Add merge-aware objective and conflict diagnostics.
- Go criterion: lower interference than naive merge baseline.

## Gate 4: Paper Package
- Transfer tables, latency-quality frontier, composition analysis, artifacts.

## Kill Criteria
- If quality floor cannot be reached, pivot from pure generation to retrieval+selection over adapter bank.
