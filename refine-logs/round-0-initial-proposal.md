# Research Proposal: LoRA Algebra — A Grassmann Algebraic Framework for Principled LoRA Composition

## Problem Anchor

- **Bottom-line problem**: Current LoRA merging methods (Task Arithmetic, TIES, DARE) operate by naively combining parameter-space vectors, ignoring the geometric structure of the low-rank subspaces that LoRA adapters define. This leads to unprincipled composition with no guarantees on task performance preservation, no formal measure of inter-task interference, and no algebraic closure.

- **Must-solve bottleneck**: There is no mathematical framework that treats LoRA adapters as geometric objects (subspaces) with well-defined algebraic operations that provably preserve task-relevant information. All existing merging methods are heuristic; their success or failure is empirically unpredictable.

- **Non-goals**:
  - We are NOT building a new LoRA training method or proposing a new adapter architecture.
  - We are NOT doing text-conditioned adapter generation.
  - We are NOT proposing a new backbone or fine-tuning strategy.

- **Constraints**:
  - Target venue: NeurIPS 2026
  - Compute: 8× A100-80GB, ~2000 GPU-hours budget
  - Must work with standard LoRA (PEFT-compatible), no custom hardware
  - Theory + experiments paper: needs both provable results and empirical validation

- **Success condition**: A formal algebraic framework on LoRA weight spaces with (1) theorems establishing error bounds for composition operations relative to Grassmann geometry, (2) empirical demonstration that Grassmann-aware operations consistently outperform heuristic merging across diverse domains, and (3) practical algorithms that practitioners can adopt.

## Technical Gap

### Why current methods fail

Existing LoRA merging methods operate in parameter space:
- **Task Arithmetic**: ΔW_merged = Σ λ_i · ΔW_i — ignores subspace alignment; destructive interference when task vectors point in incompatible directions.
- **TIES**: Trim + Elect Sign + Disjoint Merge — heuristic sparsification that discards potentially important off-diagonal structure.
- **DARE**: Random dropout + rescale — stochastic, no formal guarantees, performance is seed-dependent.

The fundamental issue: **LoRA adapters ΔW = BA define low-rank subspaces (the column space of B and the row space of A), and these subspaces carry the task-specific information. Linear operations in parameter space do not respect this subspace structure.**

### Why naive bigger systems are not enough

Simply increasing rank, training longer, or combining methods (TIES+DARE) does not address the core problem: operating in the wrong geometric space. The subspace structure is inherent to low-rank adaptation and cannot be fixed by parameter-level tricks.

### What mechanism is missing

A principled algebraic framework where:
1. Each LoRA adapter is represented by its column subspace on the Grassmann manifold G(r, d)
2. Composition, subtraction, interpolation, and projection have well-defined geometric meanings
3. Task interference is quantified by geodesic distance on the manifold
4. Provable bounds connect geometric operations to task performance preservation

## Method Thesis

- **One-sentence thesis**: LoRA adapters, when viewed as points on the Grassmann manifold through their column subspaces, admit a principled algebraic framework where geodesic operations provably bound task interference — enabling composition, decomposition, and interpolation with formal guarantees that parameter-space heuristics cannot provide.

- **Why this is the smallest adequate intervention**: We add no new trainable parameters, change no training procedures, and modify no architectures. We only change HOW existing LoRA adapters are combined post-training — replacing heuristic parameter-space operations with geometrically principled manifold operations.

- **Why this route is timely in the foundation-model era**: As the number of domain-specific LoRA adapters proliferates (adapter zoos, LoRA hubs), the need for principled composition becomes critical. The field has exhausted simple heuristic approaches; a theoretical foundation is overdue.

## Contribution Focus

- **Dominant contribution**: A Grassmann algebraic framework for LoRA weight spaces with formal theorems on composition error bounds and task interference quantification.

- **Supporting contribution**: Practical algorithms and empirical validation demonstrating consistent improvements over TIES/DARE/Task Arithmetic across 12 domains on 3 base models.

- **Explicit non-contributions**: We do not propose new LoRA training methods, new architectures, or new adapter designs.

## Proposed Method

### Complexity Budget
- **Frozen / reused backbone**: Any pre-trained LLM (Qwen3.5-9B, Llama-3.1-8B, Mistral-7B-v0.3)
- **New trainable components**: NONE. All operations are post-training, closed-form.
- **Tempting additions intentionally not used**: Learned composition weights, meta-learning over adapter combinations, neural composition networks.

### System Overview

```
Input: N trained LoRA adapters {(A_i, B_i)}_{i=1}^N with rank r
                    ↓
Step 1: Compute delta weights ΔW_i = B_i A_i (scaling by α/r)
                    ↓
Step 2: SVD projection: ΔW_i = U_i Σ_i V_i^T → extract U_i ∈ G(r, d) (Grassmann point)
                    ↓
Step 3: Algebraic operation on Grassmann manifold:
  • Compose: Karcher mean of {U_i} on G(r, d)
  • Subtract: Logarithmic map U_a → U_b, negate, exponential map
  • Interpolate: Geodesic γ(t) from U_a to U_b on G(r, d)
  • Project: Orthogonal projection onto span({U_i})
                    ↓
Step 4: Reconstruct ΔW from manifold point + scale matching
                    ↓
Step 5: Re-factorize into LoRA format (A', B') via SVD
                    ↓
Output: Composed LoRA adapter (A', B') with rank r
```

### Core Mechanism: Grassmann Algebraic Operations

**Definition 1 (LoRA Subspace Representation)**: Given a LoRA adapter with delta weight ΔW = BA ∈ R^{d×d}, its subspace representation is the r-dimensional column subspace U = col(ΔW) ∈ G(r, d), where G(r, d) is the Grassmann manifold of r-planes in R^d.

**Definition 2 (Grassmann Composition)**: The composition of adapters {ΔW_i}_{i=1}^N is the Karcher/Fréchet mean on G(r, d):

  U* = argmin_{U ∈ G(r,d)} Σ_i w_i · d_G(U, U_i)²

where d_G is the geodesic distance and w_i are task weights.

**Theorem 1 (Composition Error Bound)**: Let ΔW_1, ΔW_2 be two LoRA adapters with subspace representations U_1, U_2 ∈ G(r, d). For the Grassmann composition U* and the linear composition ΔW_lin = ΔW_1 + ΔW_2:

  ‖P_{U*} - P_{U_lin}‖_F ≤ C · sin(θ_max)

where θ_max is the maximum principal angle between U_1 and U_2, P_U is the orthogonal projector onto U, and C depends on the condition numbers of ΔW_1, ΔW_2. When θ_max → 0 (aligned tasks), both methods agree. When θ_max → π/2 (orthogonal tasks), linear composition loses up to r dimensions of task information while Grassmann composition preserves the full rank.

**Theorem 2 (Task Interference Bound)**: For a composed adapter evaluated on task i with performance metric L_i:

  |L_i(Grassmann) - L_i(individual)| ≤ α · Σ_{j≠i} d_G(U_i, U_j)

where α depends on the Lipschitz constant of the model's loss landscape. This gives a formal upper bound on task interference as a function of Grassmann distances.

**Theorem 3 (Geodesic Interpolation Monotonicity)**: For two adapters U_1, U_2 and the geodesic interpolant U(t) = Exp_{U_1}(t · Log_{U_1}(U_2)), the projection energy onto U_i is monotone:

  E_i(t) = ‖P_{U(t)} P_{U_i}‖_F² is monotone non-decreasing in t for i=2 and non-increasing for i=1.

Linear interpolation violates this monotonicity when principal angles exceed π/4.

**Input / Output**: Input: N PEFT-compatible LoRA checkpoints. Output: composed LoRA checkpoint in standard PEFT format.

**Training signal / loss**: None — all operations are closed-form post-training.

**Why this is the main novelty**: No prior work establishes formal algebraic operations on LoRA weight spaces with provable error bounds tied to Grassmann geometry. Existing methods operate heuristically in parameter space.

### Modern Primitive Usage

- **Which primitive**: Standard PyTorch SVD, QR, and matrix operations for Grassmann computations.
- **Exact role**: Truncated SVD maps delta weights to Grassmann manifold; QR decomposition used in log/exp maps; Karcher mean iteration for composition.
- **Why more natural than alternatives**: Grassmann geometry is the mathematically correct framework for reasoning about low-rank subspaces. Unlike learned approaches, it requires no additional training and provides formal guarantees.

### Integration into Base Generator / Downstream Pipeline

The framework operates entirely post-training:
1. Train domain-specific LoRAs using any standard method (SFT with PEFT)
2. Apply LoRA Algebra operations to compose, interpolate, or decompose
3. Output is a standard PEFT-compatible adapter — drop-in replacement

No changes to the base model, training pipeline, or inference code.

### Training Plan

**Stage 1: Domain LoRA Training** (~96 GPU-hours)
- Train 12 domain-specific LoRA adapters on 3 base models (36 total)
- Standard SFT with PEFT: r=16, α=32, 2 epochs, cosine LR
- Domains: math, code, medical, legal, finance, science, history, geography, philosophy, psychology, creative_writing, translation

**Stage 2: Grassmann Algebra Experiments** (~24 GPU-hours, mostly CPU)
- Compute all C(12,2)=66 domain pair operations per model
- 6 operations × 66 pairs × 3 models = 1188 experiments
- Compare: Grassmann compose vs Task Arithmetic vs TIES vs DARE

**Stage 3: Downstream Evaluation** (~200 GPU-hours)
- Evaluate individual, composed, and baseline-merged adapters on domain benchmarks
- 12 domains × {individual, 10 compositions, 8 baselines} × 3 models

**Stage 4: Ablation Studies** (~80 GPU-hours)
- Rank sensitivity: r ∈ {4, 8, 16, 32, 64}
- Grassmann dimension: k ∈ {4, 8, 16, 32}
- Number of composed domains: n ∈ {2, 3, 4, 6, 12}
- Interpolation type: linear vs geodesic vs chord

### Failure Modes and Diagnostics

- **Failure mode**: Grassmann composition is expensive for many adapters (Karcher mean iteration)
  - **Detection**: Wall-clock time comparison
  - **Mitigation**: Fast Karcher approximation via single-step tangent mean

- **Failure mode**: Task interference bound may be loose in practice
  - **Detection**: Compare predicted vs actual interference
  - **Mitigation**: Tighten bounds with task-specific Lipschitz estimation

- **Failure mode**: Benefits vanish when tasks are nearly orthogonal
  - **Detection**: Grassmann distance analysis
  - **Mitigation**: Report operating regime; recommend Grassmann for related tasks, standard merge for orthogonal tasks

### Novelty and Elegance Argument

**Closest work**:
1. Fisher-Rao manifold merging (March 2026) — operates on function space, not LoRA weight subspace; no algebraic structure
2. RiemannLoRA (2025) — Riemannian optimization for LoRA training, not composition
3. LoRA-S / quotient manifold (ICLR 2026) — learning efficiency, not post-training composition
4. TSPA (ICLR 2026) — rotation alignment heuristic, no formal Grassmann algebra

**Exact difference**: We are the first to (a) formalize LoRA adapters as Grassmann manifold points, (b) define a complete set of algebraic operations with provable bounds, (c) show that Grassmann geometry predicts composition quality, and (d) provide practical algorithms that are PEFT-compatible with zero additional training.

**Why focused**: One framework, one mathematical insight, one practical algorithm set. No additional modules, no training changes, no architecture modifications.

## Claim-Driven Validation Sketch

### Claim 1: Grassmann composition outperforms heuristic merging
- **Minimal experiment**: Compose all 66 domain pairs using {Grassmann, TaskArith, TIES, DARE} on 3 base models, evaluate on both constituent domain benchmarks
- **Baselines**: Task Arithmetic (λ=0.5, 1.0), TIES (density ∈ {0.3, 0.5, 0.7}), DARE (drop ∈ {0.3, 0.5, 0.7})
- **Metric**: Accuracy on domain benchmarks; report mean and per-pair improvement
- **Expected evidence**: Grassmann composition achieves ≥2% higher average accuracy, with largest gains on dissimilar domain pairs (high Grassmann distance)

### Claim 2: Grassmann distance predicts task interference
- **Minimal experiment**: For all 66 pairs, compute Grassmann distance and plot against performance drop from individual to composed
- **Metric**: Pearson/Spearman correlation between d_G and performance drop; R² of linear fit
- **Expected evidence**: Strong correlation (ρ ≥ 0.7), confirming that Grassmann distance is a principled interference predictor

### Claim 3: Geodesic interpolation preserves monotonicity (theory verification)
- **Minimal experiment**: For selected pairs, sweep t ∈ [0, 1] with geodesic vs linear interpolation, evaluate on both tasks at each t
- **Metric**: Check monotonicity of task performance curves
- **Expected evidence**: Geodesic curves are monotone; linear curves exhibit non-monotone artifacts when principal angles > π/4

## Experiment Handoff Inputs

- **Must-prove claims**: Grassmann > heuristic merging; Grassmann distance predicts interference; geodesic interpolation is smoother
- **Must-run ablations**: rank sensitivity, number of composed tasks, Grassmann dimension, scaling
- **Critical datasets/metrics**: GSM8K (math), HumanEval/MBPP (code), MedQA (medical), ARC-Challenge (science), MMLU subsets (history/geography/philosophy/psychology)
- **Highest-risk assumptions**: Theorem 1 may require tighter conditions; Grassmann benefits may diminish for very similar tasks

## Compute & Timeline Estimate

- **Estimated GPU-hours**: ~1200 total (36 LoRA trainings × ~25h + evaluation)
- **Data / annotation cost**: $0 (all public datasets)
- **Timeline**: 
  - Week 1-2: LoRA training on 3 models (Stage 1)
  - Week 3: Algebra experiments + baseline comparisons (Stage 2)
  - Week 4-5: Full evaluation (Stage 3)
  - Week 6: Ablations (Stage 4)
  - Week 7-8: Paper writing
