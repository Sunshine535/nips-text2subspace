# Execution Plan: LoRA Algebra

8-week stage-gated plan. Each stage has a **go/no-go** checkpoint.

---

## Timeline Overview

```
Week 1-2:  Stage 1 — Train Domain LoRA Zoo (10+ adapters)
Week 3:    Stage 2 — Grassmann Framework & Canonicalization
Week 4:    Stage 3 — Implement Algebraic Operations
Week 5-6:  Stage 4 — Full Evaluation vs Baselines
Week 7:    Stage 5 — Ablations & Theoretical Analysis
Week 8:    Stage 6 — Paper Writing & Figures
```

---

## Stage 1: Train Domain LoRA Zoo (Week 1-2)

### Objectives
- Train 12 high-quality domain-specific LoRA adapters on Qwen3.5-9B
- Verify each adapter achieves competitive single-task performance
- Establish individual baselines for later composition evaluation

### Tasks
1. Set up LoRA training pipeline with PEFT + DeepSpeed ZeRO-3
2. Prepare training data for 12 domains:
   - Math (GSM8K train), Code (CodeAlpaca), Medical (MedQA train),
     Legal (LegalBench), Creative (WritingPrompts), Science (SciQ+ARC),
     Finance (FinQA), Multilingual (MGSM), Safety (BeaverTails),
     Instruction (Alpaca), Reasoning (StrategyQA), Summarization (CNN/DM)
3. Train each LoRA with rank r=16, α=32, target modules: q_proj, k_proj, v_proj, o_proj
4. Evaluate each LoRA on its respective benchmark
5. Also train a subset at r=4, r=8, r=32, r=64 for rank ablations

### Deliverables
- [ ] 12 LoRA checkpoints (r=16) with individual evaluation scores
- [ ] 4 additional rank variants for 3 representative domains
- [ ] Training curves and hyperparameter logs

### Go/No-Go (end of Week 2)
- **Go:** >= 10 of 12 LoRAs achieve >= 90% of full fine-tune quality
- **Kill:** If LoRA training pipeline fails on Qwen3.5-9B (infra issue)

---

## Stage 2: Grassmann Framework & Canonicalization (Week 3)

### Objectives
- Implement Grassmann manifold operations (distance, geodesic, exp/log maps)
- Learn canonicalization map that resolves LoRA basis ambiguity
- Verify canonicalization on synthetic and real LoRAs

### Tasks
1. Implement Grassmann manifold class:
   - Distance: d(S₁, S₂) via principal angles (SVD of U₁ᵀU₂)
   - Geodesic: γ(t) = U₁V cos(Σt) + Q sin(Σt) where QΣVᵀ = (I - U₁U₁ᵀ)U₂
   - Exponential map: Exp_U(Δ) on the tangent space at U
   - Logarithmic map: Log_U₁(U₂) = tangent vector from U₁ to U₂
2. Implement SVD-based canonical embedding:
   - φ(A, B) = U[:, :r] from SVD(ABᵀ) = UΣVᵀ
3. Learn neural canonicalization map f_θ:
   - Input: flattened LoRA matrices (A, B)
   - Output: canonical orthonormal basis on Gr(r, d)
   - Training: contrastive loss with augmented LoRAs (random basis rotations)
4. Verify: randomly rotate a LoRA's basis → canonicalized version should match original

### Deliverables
- [ ] Grassmann manifold utility library with unit tests
- [ ] SVD-based canonicalization with verification on synthetic data
- [ ] Learned canonicalization map checkpoint
- [ ] Canonicalization consistency metrics (same-task LoRA → same canonical rep)

### Go/No-Go (end of Week 3)
- **Go:** Canonicalization maps same-task (rotated) LoRAs to distance < 0.1 on Gr(r,d)
- **Kill:** Canonicalization cannot distinguish same-task from different-task LoRAs

---

## Stage 3: Implement Algebraic Operations (Week 4)

### Objectives
- Implement all four algebraic operations on canonicalized LoRAs
- Verify mathematical properties (rank preservation, manifold constraints)
- Quick sanity checks on 2-3 domain pairs

### Tasks
1. **Composition ⊕**: Geodesic midpoint of φ(L₁) and φ(L₂) on Gr(r,d)
   - Weighted variant: geodesic at parameter t for asymmetric composition
   - Verify: output has rank exactly r
2. **Interpolation**: Sweep t ∈ [0, 0.1, ..., 1.0] along geodesic
   - Verify: all intermediate points lie on Gr(r,d)
   - Evaluate quality at each t on both source tasks
3. **Projection π_S**: Project LoRA onto complement of task subspace
   - Use: "remove" a capability (e.g., project math LoRA away from safety subspace)
   - Verify: projected LoRA has rank r and reduced performance on target task
4. **Subtraction ⊖**: Logarithmic map tangent vector difference
   - Use: extract "task direction" between two LoRAs
   - Verify: adding back the difference recovers original
5. Inverse map φ⁻¹: reconstruct LoRA (A, B) from Grassmann representative
6. Quick evaluation: compose math+code, math+medical, code+legal pairs

### Deliverables
- [ ] All 4 algebraic operations implemented and unit-tested
- [ ] Mathematical property verification (rank, manifold, invertibility)
- [ ] Sanity check results on 3 domain pairs

### Go/No-Go (end of Week 4)
- **Go:** Composition of math+code retains >= 95% of both individual scores
- **Kill:** Composition consistently worse than simple weight averaging

---

## Stage 4: Full Evaluation vs Baselines (Week 5-6)

### Objectives
- Systematic comparison against TIES, DARE, Task Arithmetic, linear averaging, Git Re-Basin
- Evaluate all 66 pairwise compositions from 12 domains
- Evaluate interpolation smoothness and projection precision

### Tasks
1. Run all baselines (TIES, DARE, Task Arithmetic, linear average) on all 66 pairs
2. Run our Grassmann algebra on all 66 pairs
3. Evaluate each merged adapter on BOTH source tasks
4. Compute composition quality metrics: min(acc_A, acc_B) / min(individual_acc_A, individual_acc_B)
5. Run interpolation sweeps on 10 representative pairs
6. Run projection experiments on 5 (task, removal_target) pairs
7. Statistical significance: paired t-test across all 66 pairs

### Deliverables
- [ ] Main results table: all methods × all 66 pairs (min-accuracy retention)
- [ ] Interpolation smoothness plots for 10 pairs
- [ ] Projection precision/recall table
- [ ] Statistical significance results

### Go/No-Go (end of Week 6)
- **Go:** Grassmann algebra significantly outperforms best baseline on >= 40/66 pairs
- **Conditional:** If only better on 20-40 pairs, analyze which domain pairs benefit and why

---

## Stage 5: Ablations & Theoretical Analysis (Week 7)

### Objectives
- Ablate key design choices
- Verify theoretical predictions empirically
- Prepare supplementary material

### Tasks
1. **Ablation: rank sensitivity** — test r ∈ {4, 8, 16, 32, 64}
   - Theory predicts: larger gains at lower rank (higher curvature on Gr(r,d))
2. **Ablation: canonicalization** — SVD-based vs learned vs no canonicalization
3. **Ablation: geodesic vs chord** — geodesic midpoint vs linear average then project
4. **Ablation: number of source domains** — compose 2, 3, 4, 5 LoRAs simultaneously
5. **Analysis: Grassmann distances** — do domain similarity patterns match intuition?
   (e.g., math closer to reasoning than to creative)
6. **Analysis: principal angles** — what do principal angles between domain LoRAs reveal?
7. **Theoretical verification** — verify rank preservation theorem empirically
8. **Scaling analysis** — how do results change with base model size (4B vs 9B)?

### Deliverables
- [ ] Rank sensitivity plot
- [ ] Canonicalization ablation table
- [ ] Multi-domain composition results (2/3/4/5-way)
- [ ] Grassmann distance heatmap across 12 domains
- [ ] Principal angle analysis

---

## Stage 6: Paper Writing (Week 8)

### Objectives
- Write NeurIPS 2026 submission (9 pages + references + appendix)

### Paper Outline
1. **Introduction** (1 page): LoRA merging problem, our algebraic approach, key results
2. **Preliminaries** (1 page): LoRA, Grassmann manifold, existing merging methods
3. **Method** (2 pages): Canonicalization, four algebraic operations, theoretical guarantees
4. **Theory** (1 page): Rank preservation theorem, distance-minimizing composition, interpolation smoothness
5. **Experiments** (2.5 pages): Setup, pairwise composition, interpolation, projection, ablations
6. **Analysis** (1 page): Grassmann distance structure, principal angles, failure cases
7. **Conclusion** (0.5 pages)
8. **Appendix**: Proofs, full 66-pair table, additional ablations

### Deliverables
- [ ] Complete LaTeX manuscript
- [ ] All figures in publication quality
- [ ] Code release preparation

---

## Resource Allocation

| Stage | GPUs | Duration | GPU-Hours |
|-------|------|----------|-----------|
| Stage 1: LoRA Zoo | 4× A100 | 10 days | 960 |
| Stage 2: Grassmann+Canon | 2× A100 | 5 days | 240 |
| Stage 3: Algebra Ops | 4× A100 | 5 days | 480 |
| Stage 4: Full Eval | 4× A100 | 10 days | 960 |
| Stage 5: Ablations | 8× A100 | 5 days | 960 |
| Stage 6: Writing | 0 | 5 days | 0 |
| **Total** | | **40 days** | **3600 GPU-hours** |
