# Auto Review Loop — LoRA Adapter Composition

**Started**: 2026-04-23
**Difficulty**: nightmare
**Target**: NeurIPS Best Paper
**Model**: Qwen3.5-9B, 4×H200, renf25-k8s cluster

---

## Experiment Configuration

- **Base Model**: Qwen3.5-9B (instruct)
- **LoRA Adapters**: 6 domains (math, code, medical, science, history, philosophy), rank=16, α=32, targets=q/k/v/o_proj, 8 layers
- **SAEs**: 3 layers (7, 15, 23), 16384 features each, 10M tokens wikitext training
- **Evaluation**: MCQ accuracy (MedMCQA, ARC-Challenge, MMLU) + GSM8K generation, 50 samples/domain, first-char extraction
- **Calibration**: 200 texts per domain from training splits

## Base Model & Single Adapter Performance

| Domain | Base | Single Adapter | Delta |
|--------|------|----------------|-------|
| math (GSM8K) | 0.02 | 0.02 | +0.00 |
| medical (MedMCQA) | 0.66 | 0.66 | +0.00 |
| philosophy (MMLU) | 0.42 | 0.44 | +0.02 |
| science (ARC) | 0.74 | 0.82 | **+0.08** |

Note: Math and medical adapters show zero improvement over base, limiting ability to demonstrate merging methods' differences.

---

## Method 1: SFC (Sparse Feature Composition)

### Approach
Decompose LoRA adapter effects through SAE feature space, compose by max-absolute-magnitude pooling, inject via inference hooks.

### Round 1 Review (Score: 2/10)
GPT-5.4 nightmare review identified 2 FATAL + 3 CRITICAL bugs. All fixed in commit 5acedc8.

### Results (after bug fixes)

| Pair | Method | Domain A | Domain B | Mean |
|------|--------|----------|----------|------|
| math+medical (FDS=0.26) | Base | 0.02 | 0.66 | 0.34 |
| | **SFC** | 0.00 | 0.66 | 0.33 |
| | TA | 0.02 | 0.66 | 0.34 |
| | TIES | 0.02 | 0.66 | 0.34 |
| math+science (FDS=0.49) | Base | 0.02 | 0.74 | 0.38 |
| | **SFC** | 0.00 | 0.76 | 0.38 |
| | TA | 0.00 | 0.84 | 0.42 |
| | TIES | 0.00 | 0.86 | 0.43 |
| science+philosophy (FDS=0.30) | Base | 0.74 | 0.42 | 0.58 |
| | **SFC** | 0.66 | 0.40 | 0.53 |
| | TA | 0.82 | 0.42 | 0.62 |
| | TIES | 0.82 | 0.42 | 0.62 |

### Verdict: NEGATIVE
SFC underperforms TA/TIES on all pairs. On science+philosophy, SFC degrades below base model (0.66 vs 0.74).

### Root Cause
1. Static feature steering — same coefficients added to all tokens regardless of input
2. FDS too low (0.26–0.49) — features overlap 51–74%, violating disjointness assumption
3. SAE reconstruction error provides noise floor that dominates
4. Layer-locality bug: profiles at layer 15 include effects from layer 7

---

## Method 2: FLC (Functional LoRA Composition)

### Approach
Calibration-optimal weighted least squares in activation space. Collect input activations per module, solve for optimal merged delta_W, truncate to rank-r via SVD.

### Results

| Pair | FLC | TA | TIES |
|------|-----|-----|------|
| math+medical | math=0.00, med=**0.16** | 0.02, 0.66 | 0.02, 0.66 |
| math+science | math=0.00, sci=**0.02** | 0.00, 0.84 | 0.00, 0.86 |
| science+philosophy | sci=**0.66**, phil=0.40 | 0.82, 0.42 | 0.82, 0.42 |

### Verdict: CATASTROPHIC FAILURE
FLC energy retention only 18–20% at rank-16. Merged adapter severely corrupts model outputs.

### Root Cause
Two rank-16 adapters cannot fit in rank-16 merged space. Only 32 LoRA modules exist (not 128). The rank bottleneck is fundamental — need rank-32+ for adequate coverage.

---

## Method 3: BCFF (Bilinear Cross-Factor Fusion)

### Approach
Extend LoRA factor algebra with cross-terms. For two adapters (B₁A₁, B₂A₂), learn coefficients for all 4 terms {B₁A₁, B₂A₂, B₁A₂, B₂A₁} via ridge regression on calibration data.

### Key Finding
Cross-term coefficients consistently **zero** across all 5 pairs. Ridge regression always selects c=[1.0, 1.0, 0.0, 0.0] — pure sum, no cross-factor contribution. BCFF reduces to "additive Task Arithmetic" (sum instead of average).

Energy retention: 76–81% (much better than FLC's 18–20%).

### Results

| Pair | BCFF mean | TA mean | TIES mean | Best |
|------|-----------|---------|-----------|------|
| math+medical | **0.34** | **0.34** | 0.33 | BCFF=TA |
| math+science | 0.42 | 0.41 | **0.43** | TIES |
| science+philosophy | 0.61 | **0.62** | **0.62** | TA=TIES |
| medical+science | 0.70 | **0.74** | **0.74** | TA=TIES |
| medical+philosophy | 0.56 | **0.69** | **0.69** | TA=TIES |

### Notable Results
- **BCFF science+philosophy science = 0.88** — highest single-domain score across all experiments (base=0.74, single=0.82, TIES=0.82)
- BCFF never drops math to 0.00 (TA/TIES often do)
- But BCFF degrades philosophy (0.34 vs 0.42 base) and medical (0.56 vs 0.64 base)

### Verdict: MIXED — NOT consistently better than baselines
BCFF amplifies the dominant adapter at the expense of the weaker one. The sum-based formulation (c=[1,1,0,0]) boosts strong adapters but hurts weak ones. Cross-factor hypothesis did not activate.

---

## Summary Across All Methods

### Method Comparison (Mean accuracy across all tested pairs)

| Method | Wins | Ties | Losses | Pattern |
|--------|------|------|--------|---------|
| **TA** | 2 | 3 | 0 | Most balanced, never worst |
| **TIES** | 3 | 1 | 1 | Best on science-heavy pairs |
| **BCFF** | 1 | 1 | 3 | Best single scores but inconsistent |
| **SFC** | 0 | 0 | 3 | Always worst |
| **FLC** | 0 | 0 | 3 | Catastrophic failure |

### Key Insights

1. **Simple baselines are very strong**: TA (weighted average) and TIES (trim+sign+merge) are hard to beat
2. **Adapter quality limits all methods**: Math/medical adapters don't improve over base, providing no signal for merging methods
3. **Cross-factor terms provide zero benefit**: For these adapters, B₁A₂ and B₂A₁ add no value
4. **Rank bottleneck is real**: FLC's failure at rank-16 confirms that naive compression destroys information
5. **SAE features overlap heavily**: FDS 0.26–0.49 means adapters are not disentangled in feature space
6. **Evaluation noise**: 50-sample MCQ with first-char extraction has high variance

### Failed Hypotheses

1. ❌ "LoRA adapters are sparse in SAE feature space" — Features overlap 51–74%
2. ❌ "Feature-space max-pool eliminates interference" — Static steering ignores input dependence
3. ❌ "Calibration-optimal merging beats heuristic averaging" — Rank bottleneck makes it worse
4. ❌ "Cross-factor terms B₁A₂, B₂A₁ provide positive transfer" — Coefficients consistently zero

### What Worked (Partially)

1. ✓ SAE training pipeline: 3 layers × 10M tokens, recon loss ~90-100
2. ✓ Delta_W space baselines: Correct TA/TIES implementation with SVD refactoring
3. ✓ BCFF science=0.88 — suggests sum-based merging can sometimes outperform average-based on dominant adapters

---

## Reviewer Scores

| Round | Score | Method | Verdict |
|-------|-------|--------|---------|
| 1 | 2/10 | SFC (pre-fix) | Not ready — prototype quality |
| 2 | 3/10 | SFC (post-fix, with results) | Pivot — results falsify central mechanism |

---

## Infrastructure

- **Pod**: renf-text2subspace-text2subspace, node n94, 4×H200 (143GB each)
- **Memory**: Initially 16Gi (caused OOM), expanded to 122Gi
- **SAE Training**: ~10h total (3 layers × 3.3h sequential on GPU 0)
- **Evaluation**: ~2h per pair (model load + calibration + merge + eval × 3 methods)
- **Total GPU hours**: ~60h across all experiments
