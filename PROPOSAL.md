# Proposal: LoRA Algebra — Algebraic Operations on LoRA Weight Spaces

## One-Sentence Summary

Define principled algebraic operations (compose, interpolate, project, subtract)
on LoRA adapter weight spaces via Grassmann manifold geometry, providing
theoretical guarantees that heuristic methods (TIES, DARE, Task Arithmetic) lack.

## Problem Statement

The LoRA ecosystem produces thousands of task-specific adapters, but combining
them remains ad-hoc. Current merging methods operate directly in weight space:

- **Task Arithmetic** adds/subtracts weight differences linearly — this assumes
  task vectors are in a shared linear subspace, which is generally false.
- **TIES** resolves sign conflicts via magnitude-weighted election — a heuristic
  with no optimality guarantee.
- **DARE** randomly drops parameters before merging — reduces interference
  empirically but can destroy important directions.

All three methods ignore a fundamental problem: **LoRA matrices span low-rank
subspaces, and these subspaces have rotational/permutation symmetry**. Merging
in the ambient weight space conflates basis-dependent artifacts with meaningful
task information.

## Critical Differentiation

| | Text-to-LoRA / LoRAGen | TIES / DARE / Task Arithmetic | **Ours** |
|---|---|---|---|
| **Goal** | GENERATE new LoRA | MERGE existing LoRAs | ALGEBRA on LoRA space |
| **Input** | Task description text | Set of LoRA weights | Set of LoRA weights |
| **Output** | New LoRA weights | Merged LoRA weights | Algebraically composed LoRA |
| **Theory** | None | None (heuristic) | Grassmann manifold geometry |

We are **orthogonal** to LoRA generation — T2L/LoRAGen produce individual LoRAs;
we define operations for combining, interpolating, and decomposing them.

## Thesis (Falsifiable)

> Mapping LoRA adapters to the Grassmann manifold Gr(r, d) via a learned
> canonicalization map, then performing algebraic operations (composition,
> interpolation, projection) using geodesic geometry, produces combined adapters
> with provably lower interference than weight-space heuristics.

## Falsifiable Hypotheses

1. **H1 (Composition superiority):** Grassmann composition preserves >= 98% of
   individual LoRA accuracy on both source tasks, while TIES/DARE preserve <= 95%.

2. **H2 (Interpolation smoothness):** Geodesic interpolation on Gr(r,d) yields
   monotonically smooth accuracy curves between two tasks, while linear
   interpolation in weight space shows quality collapse at intermediate points.

3. **H3 (Projection precision):** Projecting away a task subspace removes >= 90%
   of that capability while preserving >= 95% of other capabilities. TIES-based
   removal achieves <= 80% removal precision.

4. **H4 (Rank preservation):** All algebraic operations output LoRAs of the
   original rank r, while naive weight-space operations may increase effective
   rank or introduce rank deficiency.

5. **H5 (Canonicalization necessity):** Without canonicalization, Grassmann
   operations degrade to weight-space heuristic quality. The learned canonical
   map accounts for >= 5pt accuracy gain in composition.

## Quantitative Success Criteria

| Criterion | Metric | Target | Comparison |
|-----------|--------|--------|------------|
| Primary | Composition accuracy retention | >= 98% on both tasks | vs TIES <= 95% |
| Primary | Interpolation smoothness | Monotonic (0 quality collapses) | vs linear (>= 1 collapse) |
| Secondary | Projection removal precision | >= 90% target removal | vs TIES <= 80% |
| Secondary | Other-task preservation | >= 95% accuracy retained | vs TIES <= 90% |
| Tertiary | Rank preservation | Exact rank r maintained | vs weight-space (rank varies) |

## Method

### Grassmann Manifold Framework

The Grassmann manifold Gr(r, d) is the set of all r-dimensional subspaces of R^d.
A LoRA adapter with matrices A ∈ R^{d×r} and B ∈ R^{r×d} spans an r-dimensional
subspace of the weight space. We map each LoRA to a point on Gr(r, d):

1. **Canonicalization φ**: Learn a map that resolves basis ambiguity.
   LoRA (A, B) and (AQ, Q⁻¹B) for any invertible Q span the same subspace.
   φ maps both to the same canonical representative.

2. **Composition ⊕**: Given canonical reps φ(L₁), φ(L₂) on Gr(r, d),
   compute the geodesic midpoint. This is the subspace that minimizes
   total Grassmann distance to both inputs.

3. **Interpolation**: Move along the geodesic from φ(L₁) to φ(L₂) with
   parameter t ∈ [0, 1]. Unlike linear interpolation, this stays on the
   manifold.

4. **Projection π_S**: Given a task subspace S ⊂ Gr(r, d), project a LoRA
   onto the orthogonal complement S⊥ to remove task S while preserving others.

5. **Subtraction ⊖**: Use the logarithmic map to compute the tangent vector
   from φ(L₁) to φ(L₂), representing the "task difference."

### Learning the Canonicalization Map

- Train a neural network f_θ that takes LoRA weight matrices (A, B) and outputs
  a canonical orthonormal basis for the column space of AB^T.
- Training objective: for LoRAs that represent the same task (data augmentation
  via random basis rotations), f_θ should produce identical outputs.
- Contrastive loss: same-task LoRAs → close on Gr(r,d); different-task → far.

### Implementation

1. **SVD-based Grassmann embedding**: Compute thin SVD of AB^T = UΣV^T, use
   U[:, :r] as the Grassmann representative.
2. **Geodesic computation**: Use the matrix exponential of the skew-symmetric
   log map for geodesics on Gr(r, d).
3. **Efficient implementation**: For r << d (typical: r=16, d=4096), all
   operations are O(d·r²) — negligible compared to LoRA training.

## Why Not Just Align Bases Then Average?

Git Re-Basin aligns neural network bases via permutation search. This is:
1. NP-hard in general (Git Re-Basin uses approximations)
2. Designed for full networks, not low-rank adapters
3. Permutation-only, ignoring continuous rotational symmetry of LoRA subspaces
4. No guarantees on the quality of the merged result

Our Grassmann approach handles continuous symmetry natively and provides
distance-minimizing guarantees for all operations.

## Risk Analysis

| Risk | Mitigation |
|------|------------|
| Grassmann operations too expensive | O(d·r²) with r=16, d=4096 → ~1M FLOPs, negligible |
| Canonicalization map doesn't converge | SVD-based fallback always works; learned map is an improvement |
| Domain LoRAs have very different structure | Include diverse domains in training; analyze failure modes |
| Benefits disappear at high rank | Test r ∈ {4, 8, 16, 32, 64}; theory predicts larger gains at lower rank |
| Grassmann composition ≈ simple averaging | Verify with ablation: Grassmann vs weight-space average |

## Compute Budget

| Phase | GPUs | Duration | GPU-Hours |
|-------|------|----------|-----------|
| Train 12 domain LoRAs | 4× A100 | 3 days | 288 |
| Learn canonicalization map | 2× A100 | 2 days | 96 |
| Algebraic operation experiments | 4× A100 | 3 days | 288 |
| Baselines (TIES, DARE, TA) | 4× A100 | 2 days | 192 |
| Ablations and analysis | 4× A100 | 2 days | 192 |
| **Total** | | **12 days** | **1056 GPU-hours** |

## Kill Criteria

Abandon if:
1. Grassmann composition accuracy <= weight-space average ± 1pt on >= 3 domain pairs
2. Canonicalization fails to converge (same-task LoRAs not mapped to same point)
3. Benefits vanish at rank >= 16 (only works at impractically low ranks)
