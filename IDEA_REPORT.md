# Idea Discovery Report

**Direction**: nips-text2subspace (restart from GrassMerge failure)
**Date**: 2026-04-07
**Pipeline**: research-lit → idea-creator → novelty-check → research-review (3 rounds)

## Executive Summary

After surveying 60+ papers (2024-2026) and 3 rounds of GPT-5.4 xhigh review + novelty checking, we converge on **"The Rank Bottleneck of Adapter Composition"** — a theory+method paper that proves LoRA merging has a tight spectral lower bound, introduces the Adapter Composition Trilemma, and provides a minimal-overhead method that achieves the bound.

Previous GrassMerge approach abandoned due to: (a) crowded geometric merging space (Core Space, StelLA, ESM, KnOTS, TSPA all compete), (b) individual LoRAs degraded strong base models, (c) scored 3/10 at Round 7 review.

## Literature Landscape

### Saturated Directions (AVOID)
- **Parameter-space merging heuristics**: TIES, DARE, KnOTS, DO-Merging, FREE-Merging, FW-Merging, TARA — ~15 papers in 2024-2026
- **Geometry-aware training**: StelLA (NeurIPS'25 Spotlight), RiemannLoRA, LoRA-S — training-time manifold methods
- **Representation-space merging**: ESM (Feb'26), AIM (NeurIPS'25), ARM (Feb'26) — activation-guided merging

### Active but with Theoretical Gaps
- **Mergeability prediction**: Demystifying Mergeability (Jan'26), Will it Merge? (Jan'26) — empirical, no formal theory
- **Merging theory**: "Why More Experts Fail" (May'25, saturation bounds), Unified Generalization Framework (Jan'26), rank collapse proof (ACL'25) — bounds exist but no impossibility results
- **Merging vs routing trade-off**: Lotfi et al. (Mar'26) — empirical comparison, no formal optimality theory
- **Input-dependent merge**: SMEAR (ICLR'25) — operational method, no theory

### Open Gaps (OUR TARGET)
1. **No formal impossibility result** for input-independent adapter merging
2. **No characterization of WHEN merging must fail** tied to adapter/input structure
3. **No formal bridge between merging and routing** with optimality guarantees
4. **No composition-specific diagnostic** with causal/structural interpretation

## Ranked Ideas

### Idea 1: "The Rank Bottleneck of Adapter Composition" — RECOMMENDED

**Status**: 3 rounds of external review, novelty confirmed, score 7.5/10 for NeurIPS acceptance

**One-Line**: Rank-constrained LoRA merging has a tight spectral lower bound; the Adapter Composition Trilemma (Compact + Static + Faithful: pick 2) is resolved by a minimal-overhead composition router.

**Core Contributions**:

**C1 — The Adapter Composition Trilemma**
When composing N rank-r LoRA adapters:
- Compact (rank-r output) + Static (input-independent) + Faithful (capability-preserving) — PICK 2
- Merging = Compact + Static, sacrifices Faithful
- Routing = Faithful, sacrifices Compact + Static
- Our method (BAC) = Compact + Faithful, minimal sacrifice of Static (k-dim signal, k << r)

**C2 — Rank Bottleneck Theorem**
- "Composition rank" r_c = rank of domain-conditional weighted adapter sum
- When r_c > r: ANY rank-r merge incurs loss ≥ Σ_{j=r+1}^{r_c} σ_j² (tight, Eckart-Young extension)
- LoRA-specific: the rank constraint IS the bottleneck
- Explains ALL prior results: merging works when r_c ≤ r (aligned subspaces), fails when r_c >> r

**C2.1 — Composition Rank Characterization (strengthening)**
- r_c depends on principal angles between domain-conditional adapter subspaces
- Geometric conditions: when adapters are aligned → r_c = r (free merge); orthogonal → r_c = Nr (maximum bottleneck)
- Perturbation bound extending to nonlinear multi-layer transformers

**C3 — Bottleneck-Aware Composition (BAC)**
- Static merge for non-bottleneck directions (0 overhead)
- Tiny MLP router (k outputs per layer) for bottleneck directions
- k = r_c - r (typically 3-10 per layer)
- Formally optimal: achieves the lower bound
- Bridges merging (k=0) and routing (k=Nr) as formal spectrum

**C4 — Composition Rank Score (CRS)**
- Pre-merge diagnostic: CRS = normalized composition rank excess
- Predicts merge success from adapter weights + small probe set
- Phase diagram: CRS vs performance = canonical figure organizing the subfield

**Novelty Assessment**:
- Impossibility theorem: STRONG (no prior formal impossibility for merging)
- Merging-routing bridge: STRONG (Lotfi et al. do empirical only, we provide theory)
- Composition rank diagnostic: MODERATE-STRONG (structural, not just proxy)
- BAC method: MODERATE (SMEAR is operational precedent, but we have optimality)

**Reviewer Scores**: 6-6.5/10 best paper, 7.5/10 strong paper (GPT-5.4 xhigh AC-level review)

**Reviewer-Identified Strengths**:
- Trilemma is quotable and organizing
- Phase diagram could become canonical
- Method is sensibly targeted (dynamic capacity only where needed)
- Explains why existing methods sometimes work and sometimes fail

**Reviewer-Identified Weaknesses (to address in execution)**:
1. Theorem must go beyond plain truncated SVD — need geometric characterization of r_c and nonlinear perturbation bounds
2. Theory-to-method gap: BAC's MLP router must be shown to approximate the oracle selector
3. Corollary 3 (k-dim sufficiency) needs precise statement of assumptions
4. "Information-theoretic" is overclaimed — should say "approximation-theoretic" unless formally proven in a channel model
5. Empirical phase transition must be sharp and reproducible

**Mandatory Citations**: Lotfi et al. (Mar'26), ESM (Feb'26), SMEAR (ICLR'25), "Why More Experts Fail" (May'25), Unified Generalization Framework (Jan'26), PaCA/CaLoRA (NeurIPS'25)

**Kill Criteria**:
- r_c does not predict actual merge quality in practice (spectral gap is not the real bottleneck)
- BAC with k-dim router does not outperform ESM or SMEAR
- Phase transition is not sharp (CRS is just another noisy metric)

---

### Idea 2: BACKUP — "Causal Adapter Decomposition via Interchange Interventions"

Systematic causal analysis (not correlational PCA) of adapter effects for merging guidance. Bridges mechanistic interpretability + adapter composition. Can be incorporated as a module within Idea 1 (for identifying bottleneck directions causally rather than spectrally).

**Status**: Novelty moderate-strong (PaCA does parameter-level causal analysis, "Reasoning Traces" does single-direction intervention, but no systematic interchange-intervention decomposition for merging).

**Risk**: Computationally expensive, "causal" language invites reviewer attacks, may not clearly dominate ESM in practice.

**Decision**: Incorporate as ablation in Idea 1 (CAD vs spectral identification of bottleneck directions), not as standalone contribution.

---

### Eliminated Ideas

| Idea | Phase Eliminated | Reason |
|------|-----------------|--------|
| GrassMerge (original) | Gate 1 | 3/10 score, LoRAs degrade strong base models, crowded space |
| Subspace-conditioned generation (EigenLoRAx + T2L) | Phase 2 | Incremental combination, "task arithmetic in different space" |
| Representation-Theoretic Adapter Algebra (RTAA) | Phase 3 | ESM (Feb'26) occupies same niche, TSV + SD-MoE crowd the space |
| Pure impossibility theory (no method) | Phase 4 | Reviewer: "too theoretical, need constructive method" |
| Scaling laws for composition | Phase 2 | May not have clean laws; more empirical than theoretical |

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Old GrassMerge proposal: `refine-logs/FINAL_PROPOSAL_v1_grassmerge.md`

### Theory Summary (4 Theorems)
1. **Rank Bottleneck Lower Bound**: E_l(M) ≥ Σ_{j>k} σ_j(G^l)² (tight, on stacked whitened operator)
2. **Geometric Characterization**: r_c = rank of block Gram matrix K^l; free merge iff all subspaces identical; max bottleneck iff pairwise orthogonal
3. **Multi-Layer Perturbation**: layerwise errors accumulate multiplicatively via sensitivity Γ_l
4. **Trilemma**: Compact + Static + Faithful — pick 2 (formal impossibility when r_c > r)

### Method Summary (BAC)
- Static merge for non-bottleneck directions (zero overhead)
- Tiny MLP router (k = r_c - r outputs/layer) for bottleneck directions
- Formally optimal: achieves lower bound
- Bridges merging (k=0) and routing (k=Nr) as continuous spectrum

### Experiment Plan Summary
- 7 blocks, ~700-1150 GPU-hours, 16 weeks
- P0: synthetic verification, CRS phase diagram, BAC eval, N-way scaling, Llama replication, held-out domains, failure analysis
- Key figure: CRS vs performance phase diagram

## Next Steps

- [ ] Implement CRS estimation + BAC on existing codebase
- [ ] Block 0: train Llama LoRAs + held-out domains + multitask baseline
- [ ] Block 1: synthetic theorem verification
- [ ] Block 2-3: CRS + BAC evaluation
- [ ] /auto-review-loop with nightmare difficulty after initial results
