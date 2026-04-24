# Proposal: Compose Features, Not Weights — Interference-Free Adapter Merging via Sparse Feature Decomposition

## One-Sentence Summary

We show that LoRA adapter effects are sparse in SAE feature space, prove that
feature-disjoint adapters compose without interference, and provide Sparse Feature
Composition (SFC) — a zero-hyperparameter method that replaces all weight-space
merging heuristics with one principled operation in feature space.

## Problem Statement

LoRA adapter merging operates in weight space: Task Arithmetic adds delta weights,
TIES resolves sign conflicts, DARE randomly drops parameters. All these heuristics
share a fundamental flaw: **they operate in the wrong space.** Weight matrices are
entangled representations where a single parameter encodes multiple semantic features.
Merging in this space inevitably creates cross-feature interference.

Recent work proves this is not merely an engineering problem:
- "Rethinking Inter-LoRA Orthogonality" (2025) showed weight-space orthogonality
  does NOT predict composability
- Raffel et al. (2026) showed merging often reduces to regularization rather than
  genuine knowledge transfer

Meanwhile, mechanistic interpretability has produced Sparse Autoencoders (SAEs) that
decompose neural activations into monosemantic features — single, interpretable
directions in representation space. SAILS (Jan 2026) proved that SAE features can
construct LoRA adapters for safety. FSRL (2025) proved that individual adapter effects
are functionally equivalent to sparse feature modifications.

**The gap:** Nobody has connected these two facts. If each adapter modifies a sparse
set of features, and these features are disentangled, then composition in feature
space should be interference-free. This is the insight we formalize and exploit.

## Thesis (Falsifiable)

> Decomposing LoRA adapter effects into SAE feature space reveals that each adapter
> modifies a sparse (<5%) set of monosemantic features. Composing adapters by
> feature-level operations (union/max) in this space provably eliminates interference
> on non-overlapping features, outperforms all weight-space merging methods, and
> explains why composability scales with model size.

## Contributions

### C1 — Sparse Feature Decomposition Theorem

**Theorem 1:** For a rank-r LoRA adapter ΔW and an SAE with D features and
coherence μ, the feature support |S(ΔW)| ≤ C · r · μ(SAE) with high probability
over a probe distribution. Typically |S|/D < 0.05 for task-specific adapters.

**Key insight:** Low-rank constraint → low-dimensional activation perturbation →
sparse feature modification. The sparsity scales inversely with D (more features =
sparser effects).

### C2 — Interference Localization Theorem

**Theorem 2:** For two adapters with feature supports S₁, S₂ and coefficients c₁, c₂:

  Interference(SFC) = Σ_{k ∈ S₁∩S₂} ψ(c₁ₖ, c₂ₖ) · ||dₖ||²

where ψ captures the max-pool residual and dₖ are SAE dictionary atoms.

**When S₁ ∩ S₂ = ∅, interference = 0 exactly.**

This is NOT an approximation — it follows directly from the SAE's linear decode.
All interference is localized to the feature overlap.

### C3 — Sparse Feature Composition (SFC)

**Algorithm:**
1. For each adapter: compute feature profile (support + coefficients) via SAE encode
   of activation differences on a probe set
2. Compose: for each feature k in the union of supports, take max coefficient
3. Reconstruct: decode composed features back to activations

**Two variants:**
- **SFC-Exact:** Hook-based inference, applies composed features at each layer.
  Provably optimal. Small inference overhead (SAE forward per layer).
- **SFC-Merge:** Pre-compute composed weight update via least-squares fit, deploy
  as standard LoRA adapter. Zero inference overhead. Approximate.

**Properties:** Zero hyperparameters. One operation (feature max-pool). Subsumes
TIES (approximate sign consensus ≈ feature selection), DARE (random sparsity ≈
random feature selection), and Task Arithmetic (addition ≈ feature union without
collision handling).

### C4 — Feature Disentanglement Score (FDS)

FDS = 1 - |S₁ ∩ S₂| / |S₁ ∪ S₂|  (Jaccard distance of feature supports)

**Properties:**
- Structurally interpretable: each overlapping feature is a named, monosemantic concept
- Predicts composition quality from adapter weights + small probe set
- Explains WHY weight-space orthogonality ≠ composability: orthogonal weights can
  still modify the same features (via different projections)

### C5 — Composability Scaling Law (Corollary)

Under a random feature model where each rank-r adapter activates features with
probability p ∝ r/D:

  E[overlap(N)] / E[union(N)] = p^N / (1 - (1-p)^N)

As model size grows → D grows → p shrinks → overlap shrinks → composability improves.

This gives the **first theoretical explanation** of the empirical scaling law
(arxiv 2509.24244): larger models merge better because they have more features
and sparser adapter effects.

## Experimental Plan

### Models & SAEs (zero SAE training cost)
- **Primary:** Gemma-2-9B + Gemma Scope SAEs (400+ pre-trained SAEs, all layers)
- **Secondary:** Llama-3.1-8B + Llama Scope SAEs (all layers)
- **Scaling:** Gemma-2-2B + Gemma Scope (smaller model for scaling analysis)

### Domains (8 task-specific LoRA adapters per model)
math (GSM8K), code (MBPP), medical (MedQA), science (ARC-Challenge),
history (MMLU-History), philosophy (MMLU-Philosophy), law (MMLU-Law),
reasoning (ARC-Easy)

### Experiments

| Block | Experiment | Purpose | GPU-hours |
|-------|-----------|---------|-----------|
| E0 | Sparsity verification | Kill criterion: is |S|/D < 0.1? | 8 |
| E1 | SFC vs baselines (28 pairs) | Core comparison | 40 |
| E2 | FDS phase diagram | Diagnostic validation | 8 |
| E3 | Multi-model replication | Generalization | 40 |
| E4 | N-way scaling (2,4,8 adapters) | Scaling behavior | 20 |
| E5 | Ablations (SAE size, threshold, static vs dynamic) | Robustness | 16 |
| E6 | Synthetic theorem verification | Theory validation | 4 |
| E7 | Scaling law (2B vs 9B) | C5 verification | 16 |
| **Total** | | | **~150 GPU-hours** |

### Baselines
Task Arithmetic, TIES (d=0.5), DARE (p=0.5), ESM (if available),
GrassMerge (our prior work), Individual LoRA, Base model

### Key Figures
1. **Feature support visualization**: heatmap of which SAE features each adapter modifies
2. **FDS vs composition quality**: scatter plot showing FDS predicts performance
3. **N-way scaling**: SFC degrades gracefully while baselines collapse
4. **Scaling law**: sparsity vs model size, confirming p ∝ 1/D

## Kill Criteria

1. LoRA effects not sparse in SAE space (|S|/D > 0.2) → framework fails
2. SFC-Exact not better than best baseline on >60% of pairs → method not useful
3. FDS Spearman ρ < 0.5 with composition quality → theory disconnected
4. Sparsity does not decrease with model size → scaling law wrong

## Risk Analysis

| Risk | Probability | Mitigation |
|------|------------|-----------|
| SAE reconstruction error dominates | Medium | Use high-quality SAEs (Gemma Scope); bound error contribution |
| Effects not sparse for some tasks | Low | FSRL already showed sparsity; test diverse tasks |
| SFC-Merge reconstruction poor | Medium | SFC-Exact as gold standard; SFC-Merge is bonus |
| Feature overlap too high for related tasks | Medium | This is expected; document the transition point |
| Reviewer: "just PCA with extra steps" | Low | SAE features are monosemantic (interpretable), PCA is not |

## Mandatory Citations

- SAILS (arxiv 2512.23260): SAE → safety LoRA
- FSRL (arxiv 2509.12934): adapter ≈ sparse features
- STF (EMNLP 2025): feature-space merging
- Rethinking Orthogonality (arxiv 2510.03262): negative result
- Model Merging Scaling Laws (arxiv 2509.24244): empirical law
- Conceptor Steering (ICML 2025): Boolean algebra for composition
- SASFT (ICLR 2026): SAE guides fine-tuning
- Gemma Scope (arxiv 2408.05147): SAE infrastructure
