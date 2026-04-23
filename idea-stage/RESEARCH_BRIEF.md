# Research Brief: LoRA Adapter Composition — Improving Beyond SOTA

## What We Tried and Failed

### Method 1: SFC (Sparse Feature Composition)
- Decompose LoRA effects into SAE feature space, compose by max-pool
- **Result**: Worse than TA/TIES on all pairs. science+philosophy SFC=0.66 vs TA=0.82 (below base 0.74)
- **Root cause**: Static feature steering ignores input dependence; features overlap 51-74%

### Method 2: FLC (Functional LoRA Composition) 
- Calibration-optimal weighted least squares in activation space
- **Result**: Catastrophic (medical=0.16 vs base=0.66). Energy retention only 18-20% at rank-16
- **Root cause**: Two rank-16 adapters need rank-32+ to avoid destructive compression

## What Works (Baselines)
- **Task Arithmetic**: Simple delta_W averaging. Preserves base performance, sometimes improves
- **TIES**: Trim + sign consensus + average. Gets 0.86 on science (beats single adapter 0.82!)
- These simple methods are VERY strong. Hard to beat

## Available Infrastructure
- Model: Qwen3.5-9B on 4×H200 (143GB each), 122Gi CPU RAM
- 6 LoRA adapters: math, code, medical, science, history, philosophy (rank-16, q/k/v/o_proj, 8 layers)
- 3 SAEs trained (layers 7,15,23, 16384 features, 10M tokens each)
- Evaluation: MCQ (MedMCQA, ARC, MMLU) + generation (GSM8K), 50 samples per domain
- Datasets cached locally: gsm8k, medmcqa, arc_challenge, mmlu, mbpp, wikitext

## Key Observations
1. Adapters only have 32 LoRA modules (8 layers × 4 targets), not all 32 layers
2. Math/medical adapters don't improve over base (0.02/0.66 unchanged)
3. Science adapter: +8% (0.74→0.82), Philosophy: +2% (0.42→0.44)
4. TIES sometimes EXCEEDS single adapter performance (science: 0.86 > 0.82)
5. FLC energy retention at rank-16 is only 18-20% — rank bottleneck is critical

## Constraints
- Must build on existing LoRA composition infrastructure
- Must beat TA and TIES empirically
- Must be genuinely novel (checked against 60+ papers)
- Target: NeurIPS 2025 best paper level

## 60+ Related Papers Already Surveyed
Key competitors: RegMean, Pico (Crowded-in-B-Space), LoRA-LEGO, LoRI, TIES, DARE, TA, AdaMerging, Model Soups, Ortho-LoRA, various MoE-LoRA methods
