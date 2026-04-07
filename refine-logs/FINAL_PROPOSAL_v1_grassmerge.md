# Research Proposal: Grassmannian Composition — Geometry-Aware Merging of LoRA Adapters on Fixed-Rank Manifolds

## Problem Anchor

- **Bottom-line problem**: Current LoRA merging methods (Task Arithmetic, TIES, DARE) operate by naively combining parameter-space vectors, ignoring the geometric structure of the low-rank subspaces that LoRA adapters define. This leads to unprincipled composition with no guarantees on task performance preservation, no formal measure of inter-task interference, and no algebraic closure.

- **Must-solve bottleneck**: No mathematical framework treats LoRA adapters as geometric objects (subspaces) with well-defined operations that provably preserve task-relevant information. All existing merging methods are heuristic; their success or failure is empirically unpredictable.

- **Non-goals**: NOT building new LoRA training methods, architectures, or adapter designs.

- **Constraints**: NeurIPS 2026, 8× A100-80GB (~2000 GPU-hours), PEFT-compatible, theory + experiments paper.

- **Success condition**: Operator-distortion theorems + consistent empirical improvement + practical PEFT-compatible algorithms.

## Technical Gap

LoRA adapters ΔW = BA define rank-r operators. Existing merging methods treat these as flat parameter vectors, ignoring three geometric structures that carry task-specific information:

1. **Left subspace** (column space of B) — which output dimensions are adapted
2. **Right subspace** (row space of A) — which input features are used
3. **Spectral core** — the magnitude and importance distribution across rank components

Parameter-space averaging (Task Arithmetic, TIES, DARE) can:
- Produce an operator with rank > r, violating the low-rank inductive bias and requiring lossy truncation
- Rotate subspaces away from both originals, serving neither constituent task well
- Destructively interfere in the spectral domain when subspace alignment is poor

No prior work formulates LoRA composition as optimization on the product manifold G(r, d_out) × G(r, d_in) with explicit projected-core averaging.

## Method Thesis

LoRA adapters, represented as fixed-rank operators (U, Σ, V) via truncated SVD, can be composed by averaging their left and right subspaces on respective Grassmann manifolds and interpolating their spectral cores in the common subspace frame. This yields rank-r outputs by construction with provably smaller distortion from the ideal rank-r average than parameter-space methods.

**Smallest adequate intervention**: Zero new parameters. Only changes the post-training composition algorithm.

**Timeliness**: The adapter-zoo paradigm (HuggingFace PEFT hub, open-weight LoRA community) makes principled composition a pressing practical need. The field has exhausted heuristic merging approaches.

## Contribution Focus

- **Dominant contribution**: GrassMerge — a PEFT-compatible, closed-form algorithm for geometry-aware LoRA composition, with operator-distortion analysis connecting subspace geometry to merge quality.
- **Secondary contribution**: Bilateral Grassmann Distance (BGD) as a computable, theoretically motivated pre-merge interference diagnostic.
- **Non-contributions**: No new training methods, architectures, learned composition weights, or routing mechanisms.

## Proposed Method

### Complexity Budget
- **Frozen**: Any pre-trained LLM (tested on Qwen3.5-9B, Llama-3.1-8B, Mistral-7B-v0.3)
- **New trainable components**: NONE. All operations are post-training, closed-form.
- **Intentionally excluded**: Learned merge weights, meta-learning, routing, neural mergers.

### Algorithm 1: GrassMerge

**Input**: N rank-r LoRA adapters {(A_i, B_i)}_{i=1}^N, merge weights {w_i} with Σw_i = 1 (default: equal, w_i = 1/N)

**Per layer l** (independently for each LoRA-targeted layer):

```
1. Compute delta weights:
   ΔW_i = (α/r) · B_i · A_i    ∀i ∈ {1,...,N}

2. Truncated rank-r SVD:
   ΔW_i = U_i · diag(σ_i) · V_i^T
   where U_i ∈ R^{d_out × r}, V_i ∈ R^{d_in × r}, σ_i ∈ R^r_+

3. Karcher mean for left subspaces on G(r, d_out):
   Initialize: μ_U ← U_1
   For k = 1, ..., K:
     τ ← Σ_i w_i · Log_{μ_U}(U_i)      [tangent vector]
     μ_U ← Exp_{μ_U}(τ)                  [retraction]
     If ‖τ‖_F < ε: break
   U* ← μ_U

4. Karcher mean for right subspaces on G(r, d_in):
   (Same procedure as step 3 with V_i)
   V* ← μ_V

5. Projected-core averaging:
   For each adapter i:
     S_i = (U*)^T · ΔW_i · V*    ∈ R^{r × r}
   
   S_avg = Σ_i w_i · S_i          ∈ R^{r × r}
   
   Û · diag(σ*) · V̂^T = SVD(S_avg)    [r × r SVD, O(r³)]
   
   U_final = U* · Û
   V_final = V* · V̂

6. Reconstruct merged delta weight:
   ΔW* = U_final · diag(σ*) · V_final^T    [rank-r by construction]

7. Re-factorize to LoRA format:
   B* = U_final · diag(√σ*)
   A* = diag(√σ*) · V_final^T

8. Output: PEFT-compatible adapter (A*, B*) per layer
```

**Properties**:
- Output is rank-r by construction (no truncation needed)
- PEFT-compatible: saves as standard adapter_model.safetensors + adapter_config.json
- Projected-core form S_i = (U*)^T ΔW_i V* naturally handles sign and basis ambiguities
- Complexity: O(N · L · (d · r² + r³ · K)) per merge; typically 5-30 seconds for 9B model

**Merge weight policies** (main experiments use equal weights; alternatives ablated):
- Equal: w_i = 1/N
- Performance-proportional: w_i ∝ acc_i on calibration set
- Distance-aware: w_i ∝ exp(-λ · Σ_{j≠i} BGD(i,j))

**Adaptive merge rule**: If max_{i,j} BGD(i,j) < threshold τ_BGD, use simple parameter averaging (faster, equivalent when subspaces align). Otherwise use GrassMerge. Threshold τ_BGD is calibrated empirically.

### Theoretical Results

**Assumption A1 (Equal rank)**: All N adapters have the same LoRA rank r.
**Assumption A2 (Bounded spectra)**: ‖Σ_i‖ ≤ σ_max for all i.
**Assumption A3 (Bounded principal angles)**: The principal angles between any pair of left (resp. right) subspaces satisfy θ_k^{ij} ≤ θ_max < π/2.

**Theorem 1 (Rank-Preserving Distortion Bound)**:
Let ΔW_lin = Σ_i w_i ΔW_i be the weighted parameter average, [ΔW_lin]_r its best rank-r approximation, and ΔW_G = GrassMerge({ΔW_i}, {w_i}).

Under A1–A3, for the pairwise case (N=2, w_1=w_2=1/2):

  ‖ΔW_G - [ΔW_lin]_r‖_F ≤ (σ_max / 2) · (‖sin Θ_U‖_F + ‖sin Θ_V‖_F) + σ_{r+1}(ΔW_lin)

where Θ_U, Θ_V are the principal angle vectors between the left/right subspaces, and σ_{r+1}(ΔW_lin) is the (r+1)-th singular value of the linear average (the truncation loss).

Key insight: ΔW_G is rank-r by construction, while ΔW_lin generally has rank > r. The term σ_{r+1}(ΔW_lin) quantifies the information that linear averaging forces into rank > r and must discard. GrassMerge avoids this loss entirely by operating within the rank-r manifold.

When Θ_U = Θ_V = 0 (perfectly aligned subspaces), both methods produce identical results. The advantage of GrassMerge grows with subspace misalignment.

**Proof approach**: Decompose ΔW_G as the Karcher mean projected to the product manifold. Use the local quadratic approximation of geodesic distance on G(r,d) and Wedin's perturbation theorem for the SVD truncation. Full proof with explicit constants in Appendix A.

**Proposition 2 (Interference Upper Bound)**:
Under A2 and the additional assumption that L(θ + ΔW) is M-Lipschitz in ΔW within ‖ΔW‖_F ≤ R:

  |L(θ + ΔW_G) - L(θ + ΔW_i)| ≤ M · σ_max · h(BGD, N)

where h is an explicitly computable function. This provides a computable upper bound on per-task performance degradation from merging.

**Presentation note**: Proposition 2 is a theoretically motivated bound, not a tight prediction. We empirically measure the tightness ratio (predicted/actual interference) and report it alongside the correlation analysis.

### Bilateral Grassmann Distance (BGD) — Secondary Contribution

**Definition**: BGD(i, j) = √(d_G(U_i, U_j)² + d_G(V_i, V_j)²)

where d_G(U_i, U_j) = ‖Θ_U^{ij}‖_2 is the geodesic distance (norm of principal angles) on the Grassmannian.

**Computational cost**: O(d · r²) per pair (one SVD of r × r cross-Gram matrix per subspace).

**Use cases**:
- Pre-merge diagnostic: high BGD → high expected interference → consider separate deployment
- Merge partner selection: choose low-BGD pairs for multi-way composition
- Adaptive merge decision: BGD < τ → use cheap linear merge; BGD ≥ τ → use GrassMerge

BGD is explicitly secondary to GrassMerge. It provides practical utility for adapter management at scale.

### Integration

- All inputs: standard PEFT LoRA checkpoints (adapter_model.safetensors + adapter_config.json)
- All outputs: standard PEFT LoRA checkpoints
- Zero modifications to training or inference code
- Drop-in replacement for existing merge utilities

### Failure Modes and Diagnostics

| Failure Mode | Detection | Mitigation |
|-------------|-----------|------------|
| Mixed ranks | Adapters have different r | Pad smaller to r_max with zero singular values |
| Near-zero singular values | σ_k < ε · σ_1 | Threshold at ε = 1e-6; truncate effective rank |
| Highly dissimilar spectra | BGD > 2.0 | Warn user; recommend separate adapters or reduced weight |
| Karcher non-convergence | ‖τ‖ not decreasing | Fallback to single-step tangent mean (K=1) |
| Many adapters (N > 20) | Slow convergence | Use approximate Karcher via tangent PCA |
| Repeated singular values | Basis ambiguity in SVD | Projected-core form (U*)^T ΔW V* handles this naturally |

### Novelty Statement

Our contribution is not that Grassmannian geometry exists, but that we operationalize it into:
1. A PEFT-compatible, closed-form, post-training merge algorithm (GrassMerge) that preserves rank-r structure by construction
2. Operator-distortion bounds (Theorem 1) connecting subspace geometry to merge quality
3. A computable interference diagnostic (BGD) with theoretical motivation (Proposition 2)

**Comparison with concurrent/prior work**:

| Method | Manifold | Representation | Post-training? | Operator bounds? | PEFT-compatible? |
|--------|----------|----------------|---------------|------------------|-----------------|
| Fisher-Rao merging (2026) | Fisher-Rao | Predictive distributions | Yes | No (KL-based) | No (full-weight) |
| RiemannLoRA (2025) | Fixed-rank | LoRA parameters | No (training) | No | Yes |
| LoRA-S (ICLR 2026) | Quotient | LoRA factors | No (training) | No | Yes |
| TSPA (ICLR 2026) | Euclidean | Rotation alignment | Yes | No | Yes |
| **GrassMerge (ours)** | G(r,d)×G(r,d)×R^r | Full (U,Σ,V) | Yes | Yes | Yes |

To our knowledge, GrassMerge is the first post-training LoRA composition method that operates on the full fixed-rank operator representation with formal operator-distortion analysis.

## Claim-Driven Validation

### Ablation A0: Representation choice (decisive design ablation)
- **Question**: Does the full bi-Grassmann + projected-core representation outperform simpler alternatives?
- **Setup**: 66 domain pairs on Qwen3.5-9B. Compare:
  - (a) Parameter averaging (baseline)
  - (b) Column-only Grassmann (U-space only, ignore V and Σ)
  - (c) SVD-Procrustes alignment (align then average)
  - (d) GrassMerge (full algorithm)
- **Metric**: Mean accuracy on both constituent domain benchmarks per pair.
- **Expected**: (d) > (c) > (b) > (a), with largest gap on high-BGD pairs.

### Hypothesis 1: GrassMerge outperforms existing merging methods on geometrically diverse pairs
- **Setup**: 66 pairs on Qwen3.5-9B (primary). Compare:
  - GrassMerge (equal weights)
  - Task Arithmetic (scaling = 1.0)
  - TIES-Merging (density = 0.5)
  - DARE (drop_rate = 0.5)
  - SVD-Procrustes, KnOTS, TSPA
- **Metric**: Average accuracy on both constituent domain benchmarks.
- **Hypothesis**: GrassMerge advantage grows with subspace misalignment (high BGD); on well-aligned pairs, methods converge.

### Hypothesis 2: BGD predicts task interference better than simpler metrics
- **Setup**: All 66 pairs. Compute BGD, spectral-weighted BGD, cosine distance, Frobenius distance, and actual performance drop.
- **Metric**: Spearman ρ, calibration plot (predicted vs actual interference).
- **Hypothesis**: BGD correlates with merge degradation; spectral-weighted BGD improves upon unweighted BGD.

### Additional Ablations
- **Merge weights**: equal vs performance-proportional vs distance-aware
- **LoRA rank**: r ∈ {4, 8, 16, 32, 64} (composition quality as function of rank)
- **Number of adapters**: N ∈ {2, 3, 4, 6, 12} (scalability of multi-way merge)
- **Karcher iterations**: K ∈ {1, 5, 10, 20} (convergence speed vs quality)
- **BGD threshold calibration**: Optimal τ for adaptive merge rule

## Training Plan

| Stage | Description | GPU-hours | Timeline |
|-------|-------------|-----------|----------|
| 1 | Train 12 domain LoRAs × 3 models (36 total), r=16, α=32, 2 epochs SFT | ~600 | Week 1-3 |
| 2 | GrassMerge experiments: 66 pairs × 6 methods × 3 models | ~10 (CPU) | Week 4 |
| 3 | Downstream evaluation on domain benchmarks | ~300 | Week 4-5 |
| 4 | Ablation studies | ~100 | Week 5-6 |
| **Total** | | **~1010** | **6-8 weeks** |

## Compute & Timeline Summary
- **Estimated GPU-hours**: ~1010 on 8× A100-80GB
- **Data / annotation cost**: $0 (all public datasets)
- **Wall-clock**: ~6-8 weeks (training dominant)
- **Risk buffer**: 2 additional weeks for debugging, re-runs, and paper writing
