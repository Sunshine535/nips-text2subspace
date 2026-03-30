# Round 2 Refinement

## Problem Anchor
(Copied verbatim)
- **Bottom-line problem**: Current LoRA merging methods operate by naively combining parameter-space vectors, ignoring the geometric structure of the low-rank subspaces.
- **Must-solve bottleneck**: No mathematical framework treats LoRA adapters as geometric objects with operations that provably preserve task-relevant information.
- **Non-goals**: NOT new training methods, architectures, or adapter designs.
- **Constraints**: NeurIPS 2026, 8× A100-80GB, PEFT-compatible.
- **Success condition**: Theorems on operator-distortion + empirical advantage + practical algorithms.

## Anchor Check
- **Original bottleneck**: Heuristic parameter-space merging ignores subspace geometry.
- **Why revised method still addresses it**: All changes tighten the same mechanism — geometry-aware post-training composition.
- **Drift rejected**: None. Reviewer confirmed anchor preserved.

## Simplicity Check
- **Dominant contribution**: GrassMerge algorithm with operator-distortion analysis.
- **Components removed**: "Parallel transport" language replaced with honest "rotation-aligned spectral averaging." BGD made explicitly secondary.
- **Unnecessary complexity rejected**: None — reviewer asked for more honesty in naming, not less substance.
- **Smallest adequate route**: Still zero new parameters. Cleaner spectral step.

## Changes Made

### 1. Fixed spectral core step — honest naming and cleaner mechanism
- Reviewer said: Absolute value + diagonal extraction is not true parallel transport. Either formalize properly or rename.
- Action: Renamed from "parallel-transported spectral averaging" to "rotation-aligned spectral interpolation." New formulation: compute the full r×r rotation matrices R_U, R_V between each adapter's subspaces and the mean subspace, then apply the rotation to the full spectral matrix (not just diagonal) before averaging, and take the SVD of the averaged spectral matrix to extract the merged spectrum. This is geometrically honest: it aligns spectra in the common frame before averaging.
- Reasoning: The mechanism was always doing rotation alignment + averaging; the name was misleading. Clean naming improves credibility.
- Impact: Theory credibility improved.

### 2. Defined merge weighting policy
- Reviewer said: Merge weights w_i are assumed but never specified.
- Action: Default: equal weights w_i = 1/N. Supported alternatives: (a) performance-proportional: w_i ∝ acc_i on a calibration set, (b) distance-aware: w_i ∝ 1/Σ_{j≠i} BGD(i,j) (closer adapters get more weight). The paper focuses on equal weights for all main experiments; alternatives are ablated.
- Reasoning: Equal weights is the simplest default. Alternatives are natural extensions but not core to the contribution.
- Impact: Completeness improved.

### 3. Reframed theorems honestly
- Reviewer said: Theorem 1 bounds deviation from linear, not preservation. Theorem 2 is coarse.
- Action: Theorem 1 now states: "GrassMerge has smaller operator distortion from the rank-r truncation of the ideal average than linear averaging." This is provable because Grassmannian averaging preserves rank-r structure while linear averaging can exceed it. Theorem 2 is reframed: "BGD provides a computable, theoretically motivated upper bound on interference. We prove the bound under Lipschitz conditions and empirically validate its tightness."
- Reasoning: Honest claims that are provable > grand claims that are not.
- Impact: Theory credibility strengthened without overclaiming.

### 4. Addressed SVD ambiguities
- Reviewer said: Sign/permutation ambiguities in SVD not discussed.
- Action: Added a sign-alignment pre-processing step: for each adapter i, choose the SVD sign convention that maximizes ⟨U_i, U_1⟩_F (align all to first adapter's orientation). This is standard in Procrustes-like methods and costs O(r) per layer. Permutation ambiguity is resolved by the Grassmannian representation itself (subspace is invariant to column permutation of U).
- Reasoning: SVD sign ambiguity is a well-known issue with a well-known fix.
- Impact: Robustness improved.

### 5. Toned down novelty language
- Reviewer said: Do not claim the manifold object as the discovery; claim the operator + diagnostic.
- Action: Rewritten novelty statement: "Our contribution is not that Grassmannian geometry exists, but that we operationalize it into a PEFT-compatible post-training merge algorithm with computable interference diagnostics and provable operator-distortion bounds." Removed "first to" language except where verifiable.
- Reasoning: Honest positioning is more credible.
- Impact: Venue readiness improved.

### 6. Added failure mode analysis
- Reviewer said: Need paragraph on mixed ranks, degenerate singular values, etc.
- Action: Added detailed failure modes section covering: (a) mixed ranks → pad smaller adapter with zeros before SVD, (b) near-zero singular values → threshold at ε·σ_max before processing, (c) highly dissimilar spectra → report BGD warning and recommend task-specific rather than merged adapter, (d) high number of adapters → Karcher mean may converge slowly, use single-step tangent approximation.
- Reasoning: Practical robustness discussion strengthens the paper.
- Impact: Feasibility improved.

## Revised Proposal

# Research Proposal: Grassmannian Composition — Geometry-Aware Merging of LoRA Adapters on Fixed-Rank Manifolds

## Problem Anchor
(Identical to all previous rounds — see above)

## Technical Gap
LoRA ΔW = BA defines rank-r operators. Existing merging treats these as flat parameter vectors, ignoring left/right subspace geometry and spectral structure. Linear averaging can: (a) exceed rank r, (b) rotate subspaces away from both originals, (c) destructively interfere in the spectral domain. No prior work formulates LoRA composition as optimization on G(r,d_out)×G(r,d_in) with explicit spectral handling.

## Method Thesis
LoRA adapters as fixed-rank operators (U,Σ,V) can be composed via subspace averaging on Grassmann manifolds with rotation-aligned spectral interpolation, yielding provably smaller operator distortion than parameter-space methods. Zero new parameters; changes only post-training composition.

## Contribution Focus
- **Dominant**: GrassMerge — a PEFT-compatible algorithm for geometry-aware LoRA composition with operator-distortion analysis.
- **Secondary**: Bilateral Grassmann Distance (BGD) as an interference diagnostic.

## Algorithm 1: GrassMerge (Revised)

**Input**: N rank-r LoRA adapters {(A_i, B_i)}_{i=1}^N, weights {w_i} (default: equal)

**Pre-processing**: Sign alignment — for each adapter i, flip SVD signs to maximize alignment with adapter 1.

**Per layer l**:
```
1. ΔW_i = (α/r) · B_i · A_i

2. Truncated SVD: ΔW_i = U_i · diag(σ_i) · V_i^T
   Sign-align: if ⟨U_i, U_1⟩_F < 0, flip U_i ← -U_i, V_i ← -V_i

3. Karcher mean on G(r, d_out):
   U* ← U_1
   for iter = 1..K:
     τ ← Σ_i w_i · Log_{U*}(U_i)
     U* ← Exp_{U*}(τ)
     if ‖τ‖_F < ε: break

4. Karcher mean on G(r, d_in):
   V* ← V_1  (same iteration as step 3)

5. Rotation-aligned spectral interpolation:
   For each i:
     R_U^i = (U*)^T · U_i  ∈ R^{r×r}     (rotation: U_i ≈ U* · R_U^i)
     R_V^i = (V*)^T · V_i  ∈ R^{r×r}
     Σ_aligned^i = R_U^i · diag(σ_i) · (R_V^i)^T  ∈ R^{r×r}
   
   Σ_avg = Σ_i w_i · Σ_aligned^i
   
   Û, σ*, V̂^T = SVD(Σ_avg)  (r×r SVD, cheap)
   
   U_final = U* · Û
   V_final = V* · V̂

6. Reconstruct: ΔW* = U_final · diag(σ*) · V_final^T

7. Refactorize: B* = U_final · diag(√σ*), A* = diag(√σ*) · V_final^T

8. Output PEFT-compatible (A*, B*)
```

**Complexity**: O(N·L·(d·r² + r³·K)) per merge. Typically 5-30 seconds for 9B model.

**Rank handling**: If adapters have different ranks r_i, pad smaller ones to r_max = max(r_i) with zero singular values before step 2.

## Theoretical Results

**Assumption A1**: All adapters have rank r (or padded to same rank).
**Assumption A2**: L(θ + ΔW) is M-Lipschitz in ΔW within ‖ΔW‖_F ≤ R.

**Theorem 1 (Rank-Preserving Distortion Bound)**:
Let ΔW_lin = (ΔW₁ + ΔW₂)/2 be the parameter-space average, and ΔW_G = GrassMerge(ΔW₁, ΔW₂). Let [·]_r denote best rank-r approximation. Then:

  ‖ΔW_G - [ΔW_lin]_r‖_F ≤ (σ_max/2) · f(Θ_U, Θ_V)

where f is a computable function of principal angles, and σ_max = max(‖Σ₁‖, ‖Σ₂‖). Critically, ΔW_G is already rank-r by construction, while ΔW_lin generally has rank > r. GrassMerge produces the geometry-aware rank-r average; linear averaging requires a lossy truncation to return to rank r.

**Proof approach**: The Karcher mean on G(r,d) is the minimizer of summed squared geodesic distances. Comparing to the linear average projected back to rank r, the distortion is bounded by the curvature of the Grassmannian (which involves sin of principal angles). Full proof in Appendix A with explicit constants.

**Theorem 2 (Interference Upper Bound)**:
Under A2:
  |L(θ + ΔW_G) - L(θ + ΔW_i)| ≤ M · ‖ΔW_G - ΔW_i‖_F

By the structure of GrassMerge:
  ‖ΔW_G - ΔW_i‖_F ≤ σ_max · g(BGD, N)

where g is explicitly computable. Thus BGD provides a computable upper bound on per-task interference.

**Presentation**: Theorem 2 is presented as a theoretically motivated bound that we empirically validate. We measure the actual tightness ratio (predicted/actual interference) and report it.

**BGD (secondary contribution)**:
  BGD(i,j) = √(d_G(U_i,U_j)² + d_G(V_i,V_j)²)

Computable in O(d·r²). Monotonic with interference bound. Useful as a pre-merge diagnostic: high BGD → expect interference → consider separate adapters or reduced merge weight.

## Novelty Statement

Our contribution is not that Grassmannian geometry exists, but that we:
1. Operationalize it into a PEFT-compatible, constant-time post-training merge algorithm (GrassMerge)
2. Prove operator-distortion bounds connecting subspace geometry to merge quality
3. Provide BGD as a computable, theoretically grounded interference diagnostic

**Unique position vs prior work**: Only method that combines full (U,Σ,V) representation + post-training composition + operator-level bounds.

## Failure Modes
- **Mixed ranks**: Pad to r_max with zero SVs. Ablate impact.
- **Near-zero singular values**: Threshold at ε·σ_max before processing.
- **Highly dissimilar spectra**: BGD flags high interference risk → recommend separate adapters.
- **Many adapters (N>12)**: Karcher mean may converge slowly → use single-step tangent approximation.
- **Near-degenerate principal angles**: Stable via QR-based log/exp maps rather than direct arccos.

## Validation

### A0: Representation ablation (decisive)
Column-only Grassmann vs bi-Grassmann vs SVD-Procrustes vs parameter-avg. 66 pairs on Qwen3.5-9B.

### C1: GrassMerge vs baselines
66 pairs × 3 models (Qwen3.5-9B, Llama-3.1-8B, Mistral-7B-v0.3). Compare: GrassMerge, TaskArith (λ=0.5,1.0), TIES (d=0.3,0.5,0.7), DARE (p=0.3,0.5,0.7), SVD-Procrustes.

### C2: BGD interference prediction
198 pairs total. Spearman ρ between BGD and actual performance drop. Report R², predicted vs actual scatter plot, and tightness ratio.

### Ablations
- Weight policy: equal vs performance-proportional vs distance-aware
- Rank: r ∈ {4, 8, 16, 32, 64}
- Number of adapters: N ∈ {2, 3, 4, 6, 12}
- Karcher iterations: K ∈ {1, 5, 10, 20}

## Compute: ~1010 GPU-hours, 8× A100-80GB, 6-8 weeks
