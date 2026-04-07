# Research Proposal: The Rank Bottleneck of Adapter Composition

## One-Sentence Summary

We prove that rank-constrained LoRA merging has a tight approximation-theoretic lower bound on composition loss determined by the "composition rank" of domain-conditioned adapter subspaces, introduce the Adapter Composition Trilemma (Compact + Static + Faithful: pick 2), and provide Bottleneck-Aware Composition (BAC) — a method that achieves the bound with minimal per-input overhead by routing only irreducible bottleneck directions.

## Problem Anchor

- **Bottom-line problem**: LoRA adapter merging methods (Task Arithmetic, TIES, DARE, Core Space, KnOTS, ESM) produce a single input-independent merged adapter, but Raffel et al. (2026) showed this often reduces to regularization rather than genuine knowledge transfer. No theory explains WHEN merging must fail or WHAT the minimum fix costs.
- **Must-solve bottleneck**: Formal characterization of the information-theoretic limits of input-independent, rank-constrained adapter composition.
- **Non-goals**: NOT proposing another merging heuristic. NOT improving LoRA training.
- **Constraints**: NeurIPS 2026, 1-4× H100 (~1000 GPU-hours), PEFT-compatible, theory + experiments.
- **Success condition**: Sharp rank bottleneck theorem + CRS diagnostic predicting merge quality + BAC method matching routing at <5% overhead.

## Contribution Focus

- **Dominant contribution**: The Rank Bottleneck Theory — formal lower bound on composition loss as a function of composition rank r_c, geometric characterization of r_c via principal angles, and the Adapter Composition Trilemma.
- **Secondary contribution**: Bottleneck-Aware Composition (BAC) — a method that bridges merging (k=0 overhead) and routing (full overhead) by routing only the k = r_c - r bottleneck directions.
- **Tertiary contribution**: Composition Rank Score (CRS) — a pre-merge diagnostic with structural interpretation, validated via phase diagram.
- **Non-contributions**: No new training methods, no hypernetworks, no architecture changes.

## Setup

Layer l has frozen base weight W^l ∈ R^{d_out × d_in}. N LoRA adapters: ΔW_i^l = (α/r) B_i^l A_i^l. Domain-specific input distributions D_1, ..., D_N with covariances Σ_i^l. Mixture weights π_i.

Weighted layerwise approximation error of candidate static merge M:

  E_l(M) = Σ_i π_i · tr((ΔW_i^l - M) Σ_i^l (ΔW_i^l - M)^T)

Whitened operators: G_i^l = ΔW_i^l (Σ_i^l)^{1/2}. Stacked: G^l = [√π_1 G_1^l, ..., √π_N G_N^l].

## Theorem 1: Rank Bottleneck Lower Bound

**Definition (Composition rank)**: r_c^l(ε) = min{rank(M) : E_l(M) ≤ ε}. Exact: r_c^l = r_c^l(0).

**Theorem 1**: For any rank-k matrix M:

  E_l(M) ≥ Σ_{j>k} σ_j(G^l)²

where σ_j(G^l) are singular values of the stacked whitened operator.

**Tightness**: Achieved by the best rank-k approximation of G^l.

**Key interpretation**: This is NOT plain truncated SVD of a single matrix. The object G^l encodes the CONDITIONAL operator family {ΔW_i^l} under domain-conditioned activation geometry {Σ_i^l}. The lower bound measures the spectral energy that cannot fit in any rank-k static representation of this family.

## Theorem 2: Geometric Characterization of r_c

Let Q_i^l be an orthonormal basis of col(G_i^l). Principal angles θ_{ij,m}^l between whitened adapter subspaces defined via cos θ_{ij,m}^l = σ_m((Q_i^l)^T Q_j^l).

**Theorem 2**: Under rank(G_i^l) = r for all i:

  r_c^l = dim(Σ_i U_i^{l,Σ}) = rank(K^l)

where K^l is the Nr × Nr block Gram matrix with blocks K_{ij}^l = (Q_i^l)^T Q_j^l.

**Corollaries**:
1. Free merge: r_c^l = r iff all whitened adapter subspaces are identical. Approximate: max_{i,j} sin θ_max ≤ δ implies E_l(M*) ≤ C_l δ² Σ_i π_i ||G_i^l||_F².
2. Maximum bottleneck: r_c^l = Nr iff subspaces are in direct sum (pairwise orthogonal).
3. Angle-controlled bounds: if μ_l = max_{i≠j} ||K_{ij}^l||_2, then Nr - N(N-1)r·μ_l²/(1+(N-1)μ_l) ≤ r_c^l ≤ Nr.

**Why this goes beyond truncated SVD**: r_c is characterized geometrically by domain-conditioned subspace incidence and principal angles, not by the spectrum of any single merged matrix.

## Theorem 3: Multi-Layer Perturbation Bound

Under local linearization of layer nonlinearities with Jacobian bound κ_l and Hessian bound β_l:

  E[||F_oracle(x) - F_merge(x)||_2] ≤ Σ_l Γ_l · ε_l + Σ_l Γ_l · (β_l/2) · ξ_l

where Γ_l = Π_{t=l+1}^L κ_t ρ_t is the downstream sensitivity factor, and ε_l is the layerwise merge mismatch.

**Implication**: Prioritize early/mid layers with large Γ_l for bottleneck routing.

## Theorem 4: Adapter Composition Trilemma

If r_c^l > r at any layer, no method can simultaneously achieve:
1. Low-rank budget (rank ≤ r)
2. Universal domain coverage
3. Zero excess approximation error

One must sacrifice at least one. BAC resolves by keeping static merge on non-bottleneck directions and routing only the k = r_c - r bottleneck directions per layer.

## Method: Bottleneck-Aware Composition (BAC)

### Algorithm 1: Estimate r_c per layer
1. Collect probe activations (256-2048 tokens/domain/layer)
2. Estimate covariance Σ̂_i^l, form whitened Ĝ_i^l
3. Stack Ĝ^l, compute singular values
4. r̂_c^l(ε) = min{k : Σ_{j>k} σ_j² ≤ ε}

### Algorithm 2: Identify bottleneck directions
1. Top k_l singular vectors = shared static subspace
2. Next b_l singular vectors = bottleneck basis (covering τ% of tail energy)
3. Optional: validate by interchange intervention

### Algorithm 3: BAC composition
1. Fit static merge S^l (rank k_l) minimizing weighted Frobenius error
2. Compute residuals R_i^l = ΔW_i^l - S^l
3. Factor residuals: R_i^l ≈ U_bot^l D_i^l V_bot^{lT}
4. Train router γ^l(x) = softmax(W_rt^l z^l(x) + b^l) with CE + residual distillation loss
5. Inference: ΔW_BAC^l(x) = S^l + U_bot^l (Σ_i γ_i^l(x) D_i^l) V_bot^{lT}

### Complexity
- Static merge: O((d_out + d_in)k) params, zero routing overhead
- BAC: O((d_out + d_in)(k+b) + Nb² + router) params, tiny MLP forward per layer
- Full routing: O(N(d_out + d_in)r) params, full router overhead

### Router-Oracle Gap (Proposition 5)
If router misclassification rate is η_l, then excess residual error ≤ C_l² η_l + ξ_l^soft. BAC only needs to select among low-dimensional bottleneck residuals, not full adapters.

## Diagnostic: Composition Rank Score (CRS)

CRS_l = 1 - (r_c^l - r)/((N-1)r) ∈ [0, 1]
- CRS = 1: free merge (r_c = r)
- CRS = 0: maximum bottleneck (r_c = Nr)

Global CRS = 1 - [Σ_l w_l (r_c^l - r)] / [(N-1)r Σ_l w_l] with w_l = Γ_l (sensitivity weights).

CRS predicts merge quality because it estimates the rank of the conditional operator family under actual activation geometry (Theorem 1 connection).

## Kill Criteria

Abandon if:
1. CRS does not predict actual merge quality (Spearman ρ < 0.5 on 15+ pairs)
2. BAC does not outperform best static merge on high-CRS pairs
3. Phase diagram shows no sharp transition
4. r_c estimation is unstable under probe set variation

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| Theorem dismissed as "repackaged SVD" | Theorem 2 gives geometric characterization beyond plain SVD |
| CRS unstable | Probe set robustness analysis (Block 5B) |
| BAC overhead too high | k typically 3-10; ablation shows saturation |
| Theory-method gap | Proposition 5 formally bounds router excess risk |
| Fast-moving field | Paper framing is theory-first; method validates theory |

## Mandatory Citations

- Lotfi et al. (Mar 2026): empirical merging-vs-routing trade-off
- ESM (Feb 2026): activation-shift PCA merging
- SMEAR (ICLR 2025): input-dependent soft merge weights
- "Why More Experts Fail" (May 2025): saturation bounds
- Unified Generalization Framework (Jan 2026): L2-stability bounds
- PaCA/CaLoRA (NeurIPS 2025): parameter-level causal LoRA analysis
- Raffel et al. (Feb 2026): merging ≈ regularization negative result
