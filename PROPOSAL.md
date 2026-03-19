# Proposal: Text2Subspace (Revised v2)

## Thesis
Generating full model weights from text is intractable, but generating adapters in a canonicalized low-rank subspace can be practical.

## Falsifiable Questions
1. Can text-conditioned adapter generation approach gradient LoRA quality with much lower adaptation time?
2. Does canonicalization improve unseen-task transfer?
3. Does merge-aware generation reduce interference in multi-skill composition?

## Quantitative Success Criteria
- Primary: achieve at least 90% of LoRA quality with `>= 10x` lower adaptation latency.
- Secondary: unseen-family transfer `>= +2` absolute over non-canonical baseline.

## Method
- Build diverse adapter zoo.
- Canonicalize basis/alignment.
- Train conditional generator `G(task_text, exemplars) -> LoRA weights`.
- Add merge-aware conflict penalty.

## What Was Unreasonable Before and Is Corrected
- Weight-generation idea too unconstrained -> limited to low-rank adapter subspace.
- Identifiability not addressed -> canonicalization requirement added.
- Composition claim unsupported -> merge interference metrics added.

## Current Gap
- Pilot low-rank subspace experiment exists (`run_text2subspace_pilot.py`).
- True text-to-adapter generator over checkpoint weights is still pending.
