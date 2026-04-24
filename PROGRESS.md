# Project Progress — LoRA Adapter Composition Research

**Last Updated**: 2026-04-24
**Status**: All 3 attempted methods produced negative results. Needs fundamental pivot.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Infrastructure](#infrastructure)
3. [Methods Attempted](#methods-attempted)
4. [Complete Results](#complete-results)
5. [Code Audit Findings](#code-audit-findings)
6. [Honest Assessment](#honest-assessment)

---

## Executive Summary

We attempted 3 novel methods for LoRA adapter composition, targeting NeurIPS best-paper level contribution. **All 3 methods failed to beat simple baselines (Task Arithmetic, TIES-Merging)** after comprehensive testing on Qwen3.5-9B with 6 domain adapters.

Moreover, post-hoc code audit revealed that the third method (BCFF) had a **tautological bug**: its target function guaranteed zero cross-term coefficients regardless of input, meaning the claimed "cross-factor learning" never actually occurred. BCFF reduced to a scaled variant of Task Arithmetic.

**Conclusion**: Current research direction unlikely to produce positive results. Fundamental pivot required.

---

## Infrastructure

- **Cluster**: renf25-k8s (renf namespace), kubectl-based access
- **Pod**: renf-text2subspace-text2subspace on node n94
- **GPUs**: 4× H200 (143GB each)
- **CPU RAM**: Originally 16Gi (caused OOM crashes), expanded to 122Gi
- **Storage**: `/root/` persistent GPFS for code, models, datasets

### Assets Trained/Deployed

- **Base model**: Qwen3.5-9B at `/root/models/Qwen3.5-9B`
- **LoRA adapters** (6 domains): math, code, medical, science, history, philosophy
  - rank=16, α=32, targets=q/k/v/o_proj, 8 layers (32 total modules)
  - Stored at `/root/nips-text2subspace/results/sfc_loras_test/`
- **SAEs** (3 layers): 7, 15, 23 on Qwen3.5-9B residual stream
  - 16384 features, TopK=163, 10M wikitext tokens each
  - Final recon loss: layer 7 ~95, layer 15 ~100, layer 23 ~1200
  - Stored at `/root/saes/qwen3.5-9b/layer_{7,15,23}/`
- **Datasets cached**: gsm8k, medmcqa, arc_challenge, mmlu, mbpp, wikitext at `/root/datasets/`

### Total GPU Hours Spent

- SAE training: ~10h (sequential, one layer at a time on GPU 0)
- SFC evaluation: ~6h
- FLC evaluation: ~4h (crashed partway)
- BCFF evaluation: ~6h
- Plus debugging/restart overhead

---

## Methods Attempted

### Method 1: SFC (Sparse Feature Composition)

**Core claim**: Decompose LoRA effects into SAE feature space, compose by max-absolute-magnitude pooling, inject via inference hooks.

**Intended advantage over TA/TIES**: Theoretically eliminates interference when adapter feature supports are disjoint.

**Implementation**: `src/sparse_feature_composition.py`, `src/sae_decomposition.py`, `scripts/run_sfc_pilot.py`, `scripts/eval_sfc_downstream.py`

**Key algorithm steps**:
1. Collect base model activations on probe texts
2. For each adapter: collect with-adapter activations, compute feature delta via SAE encode
3. Threshold features using 3σ absolute magnitude
4. Compose pairs: take coefficient with max absolute magnitude per feature
5. Apply via forward hooks that add composed features to layer activations

**Problems discovered**:
- Static steering: same feature offset added to every token
- Features heavily overlap (FDS 0.26–0.49, meaning 51–74% overlap)
- SAE reconstruction error dominates composition signal
- Layer-locality: profile at layer 15 includes effects from layer 7's adapter

### Method 2: FLC (Functional LoRA Composition)

**Core claim**: Find optimal merged delta_W via weighted least squares on calibration activations.

**Intended advantage**: Provably minimizes activation reconstruction error across domains.

**Implementation**: `src/functional_composition.py`, `scripts/eval_flc.py`

**Key algorithm**:
```
min_{ΔW} Σ_i w_i ||ΔW·X_i - ΔW_i·X_i||²_F
```
Solved via:
```
ΔW = (Σ_i w_i Y_i X_i^T) (Σ_i w_i X_i X_i^T + λI)^{-1}
```
Then truncated to rank-16 via SVD.

**Problems discovered**:
- Rank bottleneck: two rank-16 adapters cannot fit in rank-16 merged space
- Energy retention only 18–20% after SVD truncation
- Merged adapter catastrophically degrades model (medical 0.66→0.16, science 0.74→0.02)

### Method 3: BCFF (Bilinear Cross-Factor Fusion)

**Core claim**: Extend LoRA factor algebra with cross-terms B₁A₂, B₂A₁ and learn optimal coefficients.

**Intended advantage**: Captures positive transfer directions that diagonal merging cannot express.

**Implementation**: `src/cross_factor_fusion.py`, `scripts/eval_bcff.py`

**Key algorithm**:
```
ΔW = c₁·B₁A₁ + c₂·B₂A₂ + c₃·B₁A₂ + c₄·B₂A₁
```
With ridge regression on calibration inputs X:
```
Y_target = (B₁A₁)X + (B₂A₂)X
Solve: c = argmin ||c₁·y₁ + c₂·y₂ + c₃·y₁₂ + c₄·y₂₁ - Y_target||² + λ||c||²
```

**FATAL bug discovered post-hoc**: The target `Y_target = y_1 + y_2` is **mathematically identical** to setting c₁=c₂=1, c₃=c₄=0. Ridge regression was guaranteed to return these coefficients **regardless of input data**. The cross-terms (c₃, c₄) were never actually tested.

**What BCFF actually computed**: `ΔW = ΔW₁ + ΔW₂` followed by SVD truncation — i.e., Task Arithmetic with scale=1.0 instead of 0.5.

---

## Complete Results

### Base Model & Single Adapter Performance

| Domain | Base | Single Adapter | Delta |
|--------|------|----------------|-------|
| math (GSM8K) | 0.02 | 0.02 | +0.00 |
| medical (MedMCQA) | 0.66 | 0.66 | +0.00 |
| philosophy (MMLU) | 0.42 | 0.44 | +0.02 |
| science (ARC) | 0.74 | 0.82 | **+0.08** |

*Note: Math and medical adapters fail to improve over base, limiting all composition methods' ability to show meaningful differences.*

### Method Comparison Across All Pairs

#### Pair 1: math+medical

| Method | math | medical | Mean |
|--------|------|---------|------|
| Base | 0.02 | 0.66 | 0.34 |
| SFC | 0.00 | 0.66 | 0.33 |
| FLC | 0.00 | 0.16 | 0.08 |
| **BCFF** | 0.02 | 0.66 | 0.34 |
| **TA** | 0.02 | 0.66 | 0.34 |
| **TIES** | 0.00 | 0.66 | 0.33 |

*SFC's FDS = 0.26 (74% overlap), BCFF energy retained = 0.76*

#### Pair 2: math+science

| Method | math | science | Mean |
|--------|------|---------|------|
| Base | 0.02 | 0.74 | 0.38 |
| SFC | 0.00 | 0.76 | 0.38 |
| FLC | 0.00 | 0.02 | 0.01 |
| BCFF | 0.02 | 0.82 | 0.42 |
| TA | 0.00 | 0.82 | 0.41 |
| **TIES** | 0.00 | **0.86** | **0.43** |

*SFC's FDS = 0.49 (51% overlap), BCFF energy retained = 0.81*

#### Pair 3: science+philosophy

| Method | science | philosophy | Mean |
|--------|---------|------------|------|
| Base | 0.74 | 0.42 | 0.58 |
| SFC | 0.66 | 0.40 | 0.53 |
| BCFF | **0.88** | 0.34 | 0.61 |
| **TA** | 0.82 | **0.42** | **0.62** |
| **TIES** | 0.82 | **0.42** | **0.62** |

*BCFF's 0.88 on science is the highest single score, but philosophy drops to 0.34*

#### Pair 4: medical+science

| Method | medical | science | Mean |
|--------|---------|---------|------|
| Base | 0.64 | 0.72 | 0.68 |
| BCFF | 0.56 | 0.84 | 0.70 |
| **TA** | 0.64 | 0.84 | **0.74** |
| **TIES** | 0.62 | **0.86** | **0.74** |

#### Pair 5: medical+philosophy

| Method | medical | philosophy | Mean |
|--------|---------|------------|------|
| Base | 0.64 | 0.42 | 0.53 |
| BCFF | 0.66 | 0.46 | 0.56 |
| **TA** | 0.66 | **0.72** | **0.69** |
| **TIES** | 0.66 | **0.72** | **0.69** |

### Method Win/Loss Count

| Method | Best on | Tied | Worse | Notes |
|--------|---------|------|-------|-------|
| **TA** | 2 pairs | 3 pairs | 0 | Most balanced, never worst |
| **TIES** | 3 pairs | 1 pair | 1 pair | Best on science-heavy pairs |
| **BCFF** | 1 single score (sci=0.88) | 1 pair | 3 pairs | Amplifies dominant adapter |
| **SFC** | 0 | 0 | 3 pairs | Always worst or tied worst |
| **FLC** | 0 | 0 | 3 pairs | Catastrophic failure |

---

## Code Audit Findings

### Bug Discovery Timeline

**Round 1 Review (GPT-5.4 nightmare, score 2/10)** — fixed before running experiments:
1. FATAL: `mean_abs_delta` destroyed sign information in SFC
2. FATAL: `full_coefficients` not zeroed for inactive features
3. CRITICAL: Baseline merging averaged LoRA A/B factors instead of delta_W = B@A
4. CRITICAL: SAE TopK/ReLU encoder mismatch
5. CRITICAL: Interference metric ignored SAE reconstruction error

**Round 2 Review (GPT-5.4, score 3/10)** — confirmed fixes, recommended pivot from SFC.

**Post-hoc audit (2026-04-24)** — discovered BCFF tautology:

```python
Y_target = y_1 + y_2  # Line 148 of src/cross_factor_fusion.py
F = torch.stack([y_1, y_2, y_12, y_21], dim=2)  # y_1, y_2 ARE in candidates
```

Mathematical proof: If `y_1 ∈ F` and `y_2 ∈ F`, then the least-squares solution to `min ||Y_target - F·c||²` with `Y_target = y_1 + y_2` is **trivially** `c = [1, 1, 0, 0]`. This holds for ANY calibration data.

**Consequence**: BCFF's cross-factor coefficients were structurally zero, never determined by data. The "bilinear fusion" hypothesis was never actually tested.

**What BCFF was actually computing**:
- Observed: `ΔW = 1·B₁A₁ + 1·B₂A₂` (sum, not average)
- TA baseline: `ΔW = 0.5·B₁A₁ + 0.5·B₂A₂` (average)
- Both followed by rank-16 SVD truncation

The only difference is a scalar factor of 2. This is equivalent to Task Arithmetic with scaling coefficient α=1.0 (instead of α=0.5). **Not a new method**.

### Analysis: Why "science=0.88" Isn't Meaningful

On science+philosophy pair:
- BCFF (scale=1): science=0.88, philosophy=0.34
- TA (scale=0.5): science=0.82, philosophy=0.42

Interpretation:
- Philosophy adapter is weak (single=0.44)
- Science adapter is strong (single=0.82)
- Scale=1 amplifies the stronger adapter's directions at the expense of the weaker one
- This is a known trade-off, not a novel contribution

---

## Honest Assessment

### Research Quality Issues

1. **Experimental Setup Limitations**:
   - LoRA adapters too weak (math/medical don't improve over base)
   - 50-sample evaluation is noisy (standard error ~7% at accuracy=0.5)
   - First-character MCQ extraction is crude
   - Only 32 of potentially 128 LoRA modules (~25% coverage)
   - GSM8K base=0.02 has no discrimination power

2. **Method Design Flaws**:
   - SFC: Static steering fundamentally mismatched to input-dependent adapters
   - FLC: Rank bottleneck inherent to the formulation
   - BCFF: Target function made the novel claim vacuous

3. **Field Saturation**:
   - 60+ papers on LoRA adapter merging reviewed
   - Simple baselines (TA, TIES) already near-optimal in our setting
   - RegMean, Pico (2026) are direct competitors for calibration-based merging
   - Room for genuinely novel "beats SOTA" contribution is very limited

### What Would Be Needed for Positive Results

- **Stronger adapters** (more training data, larger rank, domain-specific pretraining)
- **Proper evaluation** (hundreds of samples, log-probability scoring, multiple seeds)
- **Full module coverage** (all layers, not just 8)
- **A fundamentally different problem framing** (not "better merging method")

### Alternative Directions to Consider

1. **Negative-results paper**: Document when/why adapter merging methods fail ("A critical look at LoRA adapter composition")
2. **Diagnostic paper**: Use SAE features as a diagnostic for predicting merge success (not for performing the merge)
3. **Completely different problem**: Abandon adapter merging; pick a different research question

### Reviewer Scores Summary

| Round | Score | Method | Verdict |
|-------|-------|--------|---------|
| 1 | 2/10 | SFC (pre-fix) | Prototype quality, theorem overclaims |
| 2 | 3/10 | SFC (post-fix) | Results falsify central mechanism — pivot |
| — | N/A | FLC | Catastrophic failure, pivot to BCFF |
| — | N/A | BCFF | Marginal differences, post-hoc audit revealed tautological bug |

---

## Files in Repository

### Core Code
- `src/sae_decomposition.py` — SAE loading and feature decomposition
- `src/sparse_feature_composition.py` — SFC composition + FDS + hooks
- `src/functional_composition.py` — FLC calibration-based least squares
- `src/cross_factor_fusion.py` — BCFF with tautological target (see audit)
- `src/lora_algebra.py` — TA/TIES/DARE baselines

### Scripts
- `scripts/train_sae.py` — Manual TopK SAE training
- `scripts/run_sfc_pilot.py` — SFC sparsity/FDS pilot
- `scripts/eval_sfc_downstream.py` — SFC vs baselines evaluation
- `scripts/eval_flc.py` — FLC vs baselines
- `scripts/eval_bcff.py` — BCFF vs baselines
- `scripts/run_eval_lean.py` — Memory-optimized pair-by-pair eval
- `scripts/eval_one_pair.py` — Single-pair subprocess-isolated eval

### Synced Results
- `results-synced/bcff_results.json` — Full BCFF evaluation (5 pairs × 3 methods)
- `results-synced/sfc_remaining.json` — SFC+TA+TIES partial results
- `results-synced/sfc_pilot.json` — Initial sparsity/FDS measurements
- `results-synced/sae_layer_{7,15,23}_config.json` — SAE training configs
- `results-synced/*_eval.log` — Full evaluation logs
- `results-synced/sae_training.log` — 10h SAE training log

### Review Documentation
- `review-stage/AUTO_REVIEW.md` — All review rounds with verbatim reviewer responses
- `review-stage/REVIEWER_MEMORY.md` — GPT-5.4 persistent reviewer memory
- `review-stage/REVIEW_STATE.json` — Current loop state
- `idea-stage/RESEARCH_BRIEF.md` — Research context for idea discovery

### Previous (Pre-SFC)
- `PROPOSAL.md` — Original SFC proposal (claims disputed by code audit)
- `PROOF_AUDIT.md` — Theoretical issues with SFC (1 FATAL + 3 CRITICAL)
