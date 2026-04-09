---
type: idea
node_id: idea:failed_002
title: "Rank Bottleneck of Adapter Composition + BAC"
stage: failed
outcome: negative
created_at: 2026-04-07T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
tags: [rank-bottleneck, trilemma, bac, theory, abandoned]
---

# One-line thesis

Rank-constrained LoRA merging has a tight spectral lower bound; the Adapter Composition Trilemma is resolved by BAC.

## Why It Failed (for Best Paper standard)

1. **Theorem is Eckart-Young in disguise**: E_l(M) ≥ Σ_{j>k} σ_j(G^l)² — any reviewer who knows matrix approximation will see this immediately
2. **Trilemma is trivially obvious**: "Compact + Static + Faithful: pick 2" is true of ANY capacity-constrained approximation problem
3. **BAC is lightweight MoE**: SMEAR (ICLR'25) already does input-dependent routing for adapters
4. **Assessment**: Strong paper (7-7.5/10) but NOT paradigm-shifting

## Lesson Learned

- Repackaging known results (Eckart-Young) with new notation ≠ new theory
- Best Paper needs genuinely surprising results, not "filling gaps with standard tools"
- Theory-first papers need the theory to be DEEP, not just correct
- Need to change the FRAME, not just solve within the existing frame

## Reusable Assets

- CRS computation code (src/rank_bottleneck.py) — can be used for comparison with FDS
- BAC static merge code — partially reusable
- Baseline implementations (TIES, DARE, TA) — fully reusable
