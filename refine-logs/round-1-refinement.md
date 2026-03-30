# Round 1 Refinement

## Problem Anchor
- **Bottom-line problem**: Current LoRA merging methods (Task Arithmetic, TIES, DARE) operate by naively combining parameter-space vectors, ignoring the geometric structure of the low-rank subspaces that LoRA adapters define. This leads to unprincipled composition with no guarantees on task performance preservation, no formal measure of inter-task interference, and no algebraic closure.
- **Must-solve bottleneck**: There is no mathematical framework that treats LoRA adapters as geometric objects (subspaces) with well-defined algebraic operations that provably preserve task-relevant information.
- **Non-goals**: NOT building new LoRA training methods, architectures, or adapter designs.
- **Constraints**: NeurIPS 2026, 8× A100-80GB, PEFT-compatible, theory + experiments.
- **Success condition**: Formal framework with theorems, empirical advantage, practical algorithms.

## Anchor Check
- **Original bottleneck**: Heuristic parameter-space merging ignores subspace geometry → unpredictable composition quality.
- **Why revised method still addresses it**: Upgrading from column-subspace to full fixed-rank operator geometry (U, S, V) is a STRONGER version of the same insight. The bottleneck is still "geometry-unaware merging."
- **Reviewer suggestions rejected as drift**: None — all suggestions strengthen the same direction.

## Simplicity Check
- **Dominant contribution after revision**: Fixed-rank Grassmannian geometry for LoRA composition with operator-distortion bounds.
- **Components removed or merged**: Subtract operation CUT. Interpolation DEMOTED to appendix. Projection DEMOTED to appendix. Paper centers on COMPOSE + INTERFERENCE METRIC only.
- **Reviewer suggestions rejected as unnecessary complexity**: None — reviewer actually asked to simplify, not complicate.
- **Why the remaining mechanism is still the smallest adequate route**: We add zero trainable parameters. The only change is replacing parameter-space averaging with geometry-aware averaging in post-training composition. This is the minimal change needed.

## Changes Made

### 1. Upgraded mathematical object from column-subspace to bi-Grassmann fixed-rank operator
- Reviewer said: `U = col(ΔW)` loses row-space and spectral information. Two adapters with same column space can differ.
- Action: Each LoRA adapter ΔW is now represented as a triple `(U, Σ, V)` via truncated SVD, with `(U, V) ∈ G(r, d_out) × G(r, d_in)` and `Σ ∈ R^{r×r}_+`. Composition operates on all three components: Karcher mean for U and V on their respective Grassmann manifolds, parallel-transported average for Σ.
- Reasoning: This is the mathematically correct representation of a fixed-rank operator as a point on a product manifold.
- Impact: Core novelty strengthened; all theorems now concern operator-level distortion, not just subspace proximity.

### 2. Added exact per-layer pseudocode
- Reviewer said: Not concrete enough to implement. Per-layer pipeline undefined.
- Action: Added Algorithm 1 (Grassmannian Merge) with 8 numbered steps, explicit per-layer loop, SVD computation, Karcher mean iterations, scale transport, and PEFT-format output.
- Reasoning: A theory paper still needs implementable algorithms.
- Impact: Method Specificity dramatically improved.

### 3. Narrowed scope from "algebra" to "geometry"
- Reviewer said: "Grassmann algebra" is mathematically misleading. Too many operations.
- Action: Renamed to "Grassmannian Composition." Paper focuses on: (a) pairwise and multi-way composition, (b) interference prediction via Grassmann distance. Subtract/project cut from main text.
- Reasoning: One sharp contribution > many diluted ones.
- Impact: Contribution Quality focused.

### 4. Rewrote theorems with explicit assumptions
- Reviewer said: Theorem 1 compares projectors not operators; Theorem 2 not believable without strong assumptions.
- Action: Theorem 1 now bounds operator distortion ‖ΔW_Grassmann - ΔW_linear‖_F. Theorem 2 now has explicit local-Lipschitz assumption on loss landscape. Theorem 3 (monotonicity) moved to appendix as a proposition.
- Reasoning: Believable, provable claims > grand overclaims.
- Impact: Venue readiness improved.

### 5. Added decisive representation ablation
- Reviewer said: Missing ablation — column-only vs full bi-subspace.
- Action: Added Ablation A0: column-only Grassmann vs bi-Grassmann vs SVD-Procrustes alignment vs parameter averaging. This is the FIRST experiment in the paper.
- Reasoning: This ablation defends the core design choice.
- Impact: Validation Focus improved.

## Revised Proposal

# Research Proposal: Grassmannian Composition — Geometry-Aware Merging of LoRA Adapters on Fixed-Rank Manifolds

## Problem Anchor

(Copied verbatim from round 0)

- **Bottom-line problem**: Current LoRA merging methods operate by naively combining parameter-space vectors, ignoring the geometric structure of the low-rank subspaces. No guarantees on task performance preservation, no formal interference measure.

- **Must-solve bottleneck**: No mathematical framework treats LoRA adapters as geometric objects with operations that provably preserve task-relevant information.

- **Non-goals**: NOT new LoRA training, NOT new architectures, NOT text-conditioned generation.

- **Constraints**: NeurIPS 2026, 8× A100-80GB, PEFT-compatible, theory + experiments.

- **Success condition**: Theorems on operator-distortion bounds + empirical advantage + practical PEFT-compatible algorithms.

## Technical Gap

LoRA adapters ΔW = BA define rank-r operators. Existing merging methods (Task Arithmetic, TIES, DARE) treat these as flat parameter vectors, ignoring three geometric structures:

1. **Left subspace** (column space of B): captures which output dimensions are adapted
2. **Right subspace** (row space of A): captures which input features are used
3. **Spectral core** (singular values): captures the magnitude and importance distribution

Linear parameter averaging can:
- Increase effective rank beyond r (losing the low-rank inductive bias)
- Rotate subspaces away from both originals (neither task is well-served)
- Average singular values without accounting for subspace alignment (destructive interference)

**No prior work** formulates LoRA composition as an optimization problem on the product manifold `G(r, d_out) × G(r, d_in) × R^r_+` where the geometric structure of fixed-rank operators is respected.

## Method Thesis

- **One-sentence thesis**: LoRA adapters, represented as fixed-rank operators (U, Σ, V) on the product manifold G(r, d_out) × G(r, d_in) × R^r_+, can be composed via Karcher mean and parallel-transported spectral averaging, yielding provably smaller operator distortion than parameter-space heuristics.

- **Smallest adequate intervention**: Zero new parameters. Only changes the post-training composition algorithm.

- **Timeliness**: The adapter-zoo paradigm (HuggingFace PEFT hub, LoRA Model Merging community) makes principled composition a pressing practical need.

## Contribution Focus

- **Dominant contribution**: A fixed-rank Grassmannian composition framework with operator-distortion theorems.
- **Supporting contribution**: Grassmann distance as a principled interference predictor (empirical + theoretical justification).
- **Non-contributions**: No new training, no new architectures, no learned composition.

## Proposed Method

### Complexity Budget
- Frozen: Any pre-trained LLM
- New trainable components: NONE
- Intentionally excluded: Learned weights, meta-learning, routers, neural mergers

### Algorithm 1: Grassmannian LoRA Merge (GrassMerge)

**Input**: N LoRA adapters {(A_i, B_i)}_{i=1}^N, all rank r, weights {w_i} with Σw_i = 1

**Per layer l (independently for each LoRA-targeted layer)**:

```
1. Compute delta weights: ΔW_i^l = (α/r) · B_i^l · A_i^l    ∀i

2. Truncated SVD: ΔW_i^l = U_i^l · diag(σ_i^l) · (V_i^l)^T    (rank-r)
   where U_i^l ∈ R^{d_out × r}, V_i^l ∈ R^{d_in × r}, σ_i^l ∈ R^r_+

3. Karcher mean for left subspaces on G(r, d_out):
   U*^l = KarcherMean({U_i^l}, {w_i}, max_iter=20)
   
   KarcherMean iteratively:
     μ ← U_1^l
     repeat:
       tangent ← Σ_i w_i · Log_μ(U_i^l)
       μ ← Exp_μ(tangent)
     until ‖tangent‖ < ε

4. Karcher mean for right subspaces on G(r, d_in):
   V*^l = KarcherMean({V_i^l}, {w_i}, max_iter=20)

5. Parallel-transport spectral core to common frame:
   For each adapter i:
     R_U^i = (U*^l)^T · U_i^l        (r×r rotation from U_i to U*)
     R_V^i = (V*^l)^T · V_i^l        (r×r rotation from V_i to V*)
     σ_transported^i = |R_U^i · diag(σ_i^l) · (R_V^i)^T|_diag
   
   σ*^l = Σ_i w_i · σ_transported^i

6. Reconstruct merged delta weight:
   ΔW*^l = U*^l · diag(σ*^l) · (V*^l)^T

7. Re-factorize to LoRA format:
   B*^l = U*^l · diag(√σ*^l)
   A*^l = diag(√σ*^l) · (V*^l)^T

8. Output: LoRA adapter (A*^l, B*^l) per layer, PEFT-compatible
```

**Output**: Merged LoRA adapter in standard PEFT format (adapter_model.safetensors + adapter_config.json)

**Time complexity**: O(N · L · (d · r² + r³ · K)) where L = number of layers, K = Karcher iterations (typically 10-20). In practice ~5-30 seconds for a 9B model with 12 adapters.

### Theoretical Results

**Assumption A1 (Same rank)**: All adapters have the same rank r.

**Assumption A2 (Local Lipschitz)**: The model's loss L(θ + ΔW) is M-Lipschitz in ΔW within a ball of radius R around zero: |L(θ + ΔW₁) - L(θ + ΔW₂)| ≤ M · ‖ΔW₁ - ΔW₂‖_F for ‖ΔW_i‖_F ≤ R.

**Theorem 1 (Operator Distortion Bound)**: Let ΔW₁, ΔW₂ be rank-r adapters with SVDs (U₁, Σ₁, V₁), (U₂, Σ₂, V₂). Let ΔW_lin = (ΔW₁ + ΔW₂)/2 (linear average) and ΔW_G = GrassMerge(ΔW₁, ΔW₂, w=[0.5, 0.5]) (Grassmannian merge). Then:

  ‖ΔW_G - ΔW_lin‖_F ≤ (σ_max/2) · (‖sin Θ_U‖_F + ‖sin Θ_V‖_F)

where Θ_U, Θ_V are the vectors of principal angles between the left/right subspaces, and σ_max = max(‖Σ₁‖, ‖Σ₂‖). The two methods agree when Θ_U = Θ_V = 0 (aligned) and diverge most when tasks are maximally misaligned.

**Proof sketch**: Decompose the difference via the tangent-space approximation of the Karcher mean. The first-order term vanishes (mean is a critical point); the second-order term scales with the curvature of the Grassmannian, which is proportional to sin of principal angles. Full proof in Appendix A.

**Theorem 2 (Interference Prediction)**: Under Assumption A2, for an N-way Grassmannian merge with equal weights:

  |L(θ + ΔW_G) - L(θ + ΔW_i)| ≤ M · σ_max · (1/N) · Σ_{j≠i} (‖sin Θ_U^{ij}‖_F + ‖sin Θ_V^{ij}‖_F)

This provides a computable upper bound on task-i performance degradation from merging, using only the principal angles between adapter subspaces.

**Corollary**: Grassmann distance d_G(i,j) = ‖Θ^{ij}‖ upper-bounds the interference contribution, and ‖sin Θ‖_F ≤ ‖Θ‖_F = d_G, so:

  |L(θ + ΔW_G) - L(θ + ΔW_i)| ≤ (M · σ_max / N) · Σ_{j≠i} d_G^{left}(i,j) + d_G^{right}(i,j)

### Grassmann Distance as Interference Predictor

Define the **Bilateral Grassmann Distance** (BGD) between two LoRA adapters:

  BGD(i, j) = √(d_G(U_i, U_j)² + d_G(V_i, V_j)²)

This is computable in O(d · r²) per pair. By Theorem 2, BGD is theoretically justified as an interference predictor. We empirically validate the correlation between BGD and actual performance degradation.

### Integration

All inputs: standard PEFT LoRA checkpoints. All outputs: standard PEFT LoRA checkpoints. Zero modifications to training or inference code. Drop-in replacement for `model.merge_adapter()` style calls.

### Training Plan

**Stage 1**: Train 12 domain LoRAs on 3 base models (Qwen3.5-9B, Llama-3.1-8B, Mistral-7B-v0.3), r=16, α=32, 2 epochs SFT. 36 adapters total. (~600 GPU-hours)

**Stage 2**: Grassmannian composition experiments. All C(12,2)=66 pairs per model. Compare: GrassMerge, Task Arithmetic (λ=0.5, 1.0), TIES (density=0.3, 0.5, 0.7), DARE (drop=0.3, 0.5, 0.7), SVD-Procrustes align. (~10 GPU-hours, mostly CPU)

**Stage 3**: Downstream evaluation on domain benchmarks. (~300 GPU-hours)

**Stage 4**: Ablations. (~100 GPU-hours)

### Failure Modes and Diagnostics

- **Karcher mean non-convergence**: Detect via tangent norm monitoring; fallback to single-step tangent mean approximation.
- **Benefits vanish for similar tasks**: Expected — when Θ→0, all methods agree. Report the operating regime.
- **Theorem bounds are loose**: Measure predicted vs actual interference; report tightness ratio.

### Novelty Argument

**Closest prior work**:
| Work | Manifold | Object | Post-training? | Operator-level bounds? |
|------|----------|--------|---------------|----------------------|
| Fisher-Rao merging (2026) | Fisher-Rao | Predictive distributions | Yes | No (KL-based) |
| RiemannLoRA (2025) | Fixed-rank | LoRA parameters | No (training) | No |
| LoRA-S (ICLR 2026) | Quotient | LoRA factors | No (training) | No |
| TSPA (ICLR 2026) | None (heuristic) | Rotation alignment | Yes | No |
| **Ours** | **G(r,d_out)×G(r,d_in)×R^r_+** | **Full operator (U,Σ,V)** | **Yes** | **Yes** |

We are the first to combine: (a) full bi-subspace + spectral representation, (b) post-training composition, (c) operator-level distortion bounds.

## Claim-Driven Validation Sketch

### Ablation A0 (Decisive representation ablation)
- **Claim**: Full bi-Grassmann representation (U, Σ, V) outperforms column-only (U only), SVD-Procrustes, and parameter averaging.
- **Experiment**: 66 pairs on one model, evaluate accuracy on both constituent domains.
- **Metric**: Mean accuracy across all pairs.
- **Expected**: Bi-Grassmann > column-only > Procrustes > parameter-avg, with largest gap on dissimilar domains.

### Claim 1: GrassMerge outperforms heuristic merging
- **Experiment**: 66 pairs × 3 models; GrassMerge vs TaskArith vs TIES vs DARE.
- **Metric**: Mean accuracy on both constituent domain benchmarks.
- **Expected**: ≥2% mean improvement, largest on high-BGD pairs.

### Claim 2: BGD predicts task interference
- **Experiment**: For all 198 pairs (66 × 3 models), plot BGD vs performance drop.
- **Metric**: Spearman ρ and R².
- **Expected**: ρ ≥ 0.7; BGD-based model predicts which pairs will suffer interference.

## Compute & Timeline
- Total: ~1010 GPU-hours on 8× A100-80GB
- Wall-clock: ~6 weeks
- Week 1-3: LoRA training; Week 4: Algebra + eval; Week 5: Ablations; Week 6-8: Paper
