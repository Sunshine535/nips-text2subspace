---
type: idea
node_id: idea:failed_001
title: "GrassMerge: Grassmannian Geometry for LoRA Merging"
stage: failed
outcome: negative
created_at: 2026-03-20T00:00:00Z
updated_at: 2026-04-07T00:00:00Z
tags: [grassmann, lora-merging, abandoned]
---

# One-line thesis

Merge LoRA adapters by averaging subspaces on the Grassmann manifold.

## Why It Failed

1. **Crowded space**: KnOTS, Core Space, TSPA, StelLA, ESM all compete in geometric merging
2. **Individual LoRAs degraded base model**: math -13.5%, medical -13% on Qwen3-8B
3. **Mixed results**: 50% win rate vs Task Arithmetic, 43% vs TIES — not convincing
4. **Scored 3/10 at Round 7 auto-review**: "too incomplete empirically for any acceptance tier"
5. **No deep theory**: Grassmann geodesic midpoint is a heuristic, not a theorem

## Lesson Learned

- Weight-space geometry is the **wrong abstraction level** for composition
- Need to go deeper: feature space, not weight space
- Method-only papers in crowded space need 10x performance gap to stand out
