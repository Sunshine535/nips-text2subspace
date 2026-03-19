# Experiments: Text2Subspace (Revised v2)

## Task Families
- Math reasoning.
- Code generation.
- Open QA.
- Summarization.
- Tool-use instruction following.

## Baselines
- Per-task LoRA fine-tuning.
- Adapter retrieval.
- Direct adapter merge heuristics.

## Metrics
- Task quality per family.
- Adaptation latency and GPU time.
- Additional trainable params/memory.
- Unseen-task transfer score.
- Merge interference score.

## Statistical Protocol
- Family-wise held-out split.
- 3 replications minimum.
- Bootstrap CIs for transfer metrics.

## NeurIPS Minimum Publishable Standard
- 90% of LoRA quality with `>= 10x` lower adaptation latency.
- Significant transfer gains on unseen task families.
- Reproducible canonicalization and inversion scripts.

## Current Status
- Pilot implementation and first result are now available.

## Implemented Pilot (2026-02-27)
- Script:
  - `methods/05_text2subspace/scripts/run_text2subspace_pilot.py`
- Command:
  ```bash
  python methods/05_text2subspace/scripts/run_text2subspace_pilot.py
  ```
- Input:
  - `methods/01_adathink/results/per_sample_Qwen3_8B_20260227_140410.csv`
- Output:
  - `methods/05_text2subspace/results/text2subspace_pilot_20260227_151439.json`

## Pilot Snapshot
- Full-rank policy head:
  - action-match `0.775`
  - acc `0.400`
  - avg tokens `25.925`
  - utility `0.3848`
- Low-rank (rank=8) policy head:
  - action-match `0.775`
  - acc `0.400`
  - avg tokens `25.925`
  - utility `0.3848`

## Limitation
- This is a low-rank policy-head proxy, not true text-conditioned adapter generation in checkpoint space.
