# Experiment Plan: The Rank Bottleneck of Adapter Composition

**Target**: NeurIPS 2026
**Hardware**: 1-4× H100 80GB
**Compute budget**: ~700-1150 GPU-hours (lean P0: ~700, full: ~1150)
**Timeline**: 16 weeks

## Overview

| Block | Description | Priority | GPU-hours | Weeks |
|-------|-------------|----------|-----------|-------|
| 0 | Infrastructure (training, baselines) | P0 | ~260 | 1-4 |
| 1 | Synthetic theorem verification | P0 | ~15-35 | 3-5 |
| 2 | CRS computation + phase diagram | P0 | ~15 | 5-6 |
| 3 | BAC implementation + evaluation | P0 | ~210-250 | 5-9 |
| 4 | Scaling (N-way, Llama replication) | P0 | ~220-260 | 7-11 |
| 5 | Ablations | P0/P1 | ~115 | 9-12 |
| 6 | Strong baselines | P0/P1 | ~70-130 | 7-11 |
| 7 | Failure analysis + held-out domains | P0 | ~50 | 11-14 |
| — | Writing + figures | — | — | 13-16 |

## Statistical Protocol

- 3 seeds for every trainable artifact (LoRAs, BAC routers, MoLoRA, multitask)
- ≥500 examples/benchmark for main tables
- 95% bootstrap CIs over examples
- Paired bootstrap for method comparisons
- Spearman ρ + CI for CRS prediction validation

## Block 0: Infrastructure

### 0A. Train Missing Adapters

New artifacts needed:
- Llama-3.1-8B: 6 domain LoRAs × 3 seeds = 18 runs
- Qwen3-8B held-out: legal, finance × 3 seeds = 6 runs
- Llama-3.1-8B held-out: legal, finance × 3 seeds = 6 runs
- Qwen3-8B multitask LoRA × 3 seeds = 3 runs
- Llama-3.1-8B multitask LoRA × 3 seeds = 3 runs
- **Total: 36 training runs, ~180 GPU-hours**

Training recipe: LoRA r=16, alpha=32, q/k/v/o_proj, same optimizer/LR/epochs across backbones.

Held-out data: legal (LegalBench 10K), finance (FiQA 10K).
Multitask data: balanced union of 6 in-domain datasets.

### 0B. Routing Baseline Infrastructure (~80 GPU-hours)

- Oracle routing: per-example best-adapter selection by domain label / answer correctness
- MoLoRA-style learned router: top-1/top-2 gating, trained on mixed-domain data, 3 seeds
- Simple ensemble: logit averaging / calibrated selection

## Block 1: Synthetic Theorem Verification

### 1A. Exact Verification (~10 GPU-hours)

**Hypothesis**: For controlled adapter pairs with known r_c, merge loss matches spectral tail bound exactly.

Procedure:
1. Construct synthetic rank-r LoRA pairs with controlled principal angle overlap
2. Vary: shared subspace dimension, spectral profiles, overlap pattern
3. Compute best rank-r merge, measure loss vs Σ_{j>r} σ_j²
4. Verify across r ∈ {4, 8, 16, 32}
5. Lift to toy transformer (2-layer, inject synthetic LoRAs)

**Kill criterion**: If reconstruction loss does not track tail energy.

### 1B. Semi-Synthetic Real-Adapter Verification (~20 GPU-hours)

**Hypothesis**: Real Qwen LoRA layers exhibit same rank bottleneck pattern.

Procedure:
1. Extract ΔW per layer for all domain pairs
2. Compute full singular spectra of domain-weighted sums
3. Layer-ablation: replace merged layer with best rank-r approximation, measure performance drop
4. Correlate tail energy with actual performance degradation

## Block 2: CRS Computation + Phase Diagram

### 2A. Pairwise CRS Map (~10 GPU-hours)

Compute CRS for all 15 Qwen pairs + all 15 Llama pairs.
Compare against: weight cosine, principal angle overlap, BGD, delta norm ratio.
Response variable: best static merge performance, gap to oracle.
Report Spearman ρ, partial correlations.

### 2B. Canonical Phase Diagram (~5 GPU-hours)

**THE central figure**: CRS (x) vs performance retention (y).

Include: pairwise + N-way points, both backbones, all methods.
Fit: piecewise linear model, logistic transition, compare to single linear.
Show: breakpoint where static methods collapse, BAC degrades slowly.

## Block 3: BAC Implementation + Evaluation

### 3A. BAC Main Experiment (~160 GPU-hours)

**Hypothesis**: BAC matches routing quality with <5% overhead on high-CRS pairs.

Coverage:
- All 15 Qwen pairs × 3 seeds
- 6 representative Llama pairs × 3 seeds
- Selected N-way sets

Compare against: TA, TIES, DARE, KnOTS, TSPA, GrassMerge, MoLoRA, oracle, multitask.

Report: macro avg, worst-domain, faithfulness gap, gap to oracle, CIs.

### 3B. Overhead Analysis (~10 GPU-hours)

Measure: tokens/sec, P50/P95 latency, peak memory, params loaded.
Conditions: batch {1, 8, 32}, prompt length {128, 512, 2048}.
Compare: static merge, BAC (k=4/8/16), MoLoRA, oracle.
Target: BAC <5% overhead vs static merge.

## Block 4: Scaling

### 4A. N-Way Composition (~100-140 GPU-hours)

Composition sizes: 3, 4, 6.
- 6 representative triples (low/mid/high CRS)
- 4 representative 4-ways
- 1 full 6-way
All methods, 3 seeds, both backbones.

### 4B. Second Backbone Replication (~120 GPU-hours)

Llama-3.1-8B: full 15-pair CRS + static merges, BAC on 6 representative pairs.
Verify phase diagram threshold transfers.

## Block 5: Ablations

### 5A. Router Dimension k (P0, ~60 GPU-hours)

k ∈ {0, 2, 4, 8, 16, 32} on 6 representative pairs + 1 triple + 1 six-way.
Plot Pareto frontier: quality vs overhead.
Expected: saturation around k ≤ 8-16.

### 5B. Probe Set Robustness (P0, ~15 GPU-hours)

Probe sizes: {32, 64, 128, 256, 512}.
Compositions: balanced, domain-skewed, random mixed.
CRS rank correlation stability.

### 5C. Spectral vs Alternative Bottleneck ID (P1, ~30 GPU-hours)

Compare: spectral (ours), activation disagreement, gradient/Fisher, random.
Same router architecture, swap only identification module.

### 5D. Layer-wise CRS Analysis (P1, ~10 GPU-hours)

Per-layer CRS, correlation with layer ablation impact.
Show conflict concentrates in subset of layers.

## Block 6: Strong Baselines

### 6A. MoLoRA Full Comparison (P0, ~30 GPU-hours)

Full comparison on all selected settings. Latency/memory benchmark.

### 6B. Multitask LoRA (P0, ~30 GPU-hours)

Evaluate on all in-domain + held-out benchmarks.
Compare training cost and storage cost.

### 6C. Oracle Routing (P0, ~40 GPU-hours)

Domain oracle + answer oracle variants. Both backbones.
Report "% of oracle recovered" for all methods.

### 6D. ESM / SMEAR (P1/P2, ~40-60 GPU-hours)

Reimplement if feasible. Otherwise cite as future work.

## Block 7: Failure Analysis + Held-Out Domains

### 7A. Held-Out Domains (P0, ~40 GPU-hours)

Legal + finance LoRAs (trained AFTER freezing BAC/CRS design).
Compose with in-domain adapters. Test CRS threshold transfer.

### 7B. Failure Taxonomy (P0, ~10 GPU-hours)

Rank cases by CRS-prediction residual and BAC-oracle gap.
Manual inspection of 50-100 examples.
Categorize: domain ambiguity, output-format mismatch, sparse layer conflicts, router error.

## Main Paper Figure/Table Set (P0)

1. Pairwise composition table (Qwen, all methods, 15 pairs)
2. CRS vs merge quality scatter (phase diagram)
3. BAC vs static/routing Pareto (quality vs overhead)
4. N-way scaling plot (2/3/4/6 adapters)
5. Llama replication summary table
6. Held-out domain generalization table
7. Router dimension k ablation
8. Synthetic theorem verification figure
9. Failure analysis (3-4 representative cases)

## Execution Order

1. Block 0 (infrastructure)
2. Block 1 (synthetic verification — validate theory before building method)
3. Block 2 (CRS on existing Qwen results)
4. Block 3A (BAC on 3 representative pairs — smoke test)
5. Block 3A (BAC full Qwen sweep)
6. Block 6A-C (strong baselines)
7. Block 4 (scaling + replication)
8. Block 7A (held-out domains)
9. Block 5 (ablations)
10. Block 7B (failure analysis)
11. Writing + figures (weeks 13-16)
