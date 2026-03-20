# LoRA Algebra: Algebraic Operations on LoRA Weight Spaces

## Overview

Model merging methods (TIES, DARE, Task Arithmetic) combine LoRA adapters via
**heuristic** operations — magnitude pruning, random dropping, linear arithmetic.
These lack theoretical guarantees and fail unpredictably when adapter subspaces
are misaligned.

LoRA Algebra defines **principled algebraic operations** (compose, interpolate,
project, subtract) on LoRA weight spaces by mapping adapters to points on a
**Grassmann manifold**, where geometric operations have provable properties.

**This is NOT LoRA generation** (that's Text-to-LoRA / LoRAGen). We define the
**algebra** for manipulating existing LoRA adapters with theoretical guarantees.

**Target venue:** NeurIPS 2026
**Status:** Rewrite of pilot → full algebraic framework

## Core Contribution

```
Existing LoRA Adapters (10+ domains)
    │
    ▼
┌──────────────────────────────┐
│  Canonicalization Map φ       │
│  LoRA → Grassmann Manifold   │
│  Gr(r, d) where r=rank, d=dim│
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Algebraic Operations        │
│                              │
│  ⊕ Compose: φ(A) ⊕ φ(B)    │  ← Geodesic midpoint on Gr(r,d)
│  ⊖ Subtract: φ(A) ⊖ φ(B)   │  ← Logarithmic map difference
│  ⊗ Interpolate: t·φ(A)+(1-t)·φ(B) │  ← Geodesic interpolation
│  π Project: π_S(φ(A))       │  ← Projection onto task subspace S
│                              │
│  Guarantees:                 │
│  • Composition preserves rank│
│  • Interpolation stays on    │
│    manifold (no degeneracy)  │
│  • Projection minimizes      │
│    Grassmann distance        │
└──────────┬───────────────────┘
           │
           ▼
    Merged / Composed LoRA
    with theoretical guarantees
```

## Key Differentiators

| Aspect | TIES / DARE | Task Arithmetic | Text-to-LoRA / LoRAGen | **Ours** |
|--------|-------------|-----------------|------------------------|----------|
| What it does | Merge heuristically | Add/subtract task vectors | Generate new LoRA from text | Algebra on LoRA space |
| Theoretical guarantee | None | None (linear approx) | None | Grassmann manifold geometry |
| Handles basis misalignment | No | No | N/A | Yes (canonicalization) |
| Rank preservation | Not guaranteed | Not guaranteed | N/A | Guaranteed |
| Interpolation quality | Degrades off-manifold | Degrades off-manifold | N/A | Stays on manifold |

## Models & Hardware

| Component | Model | Role |
|-----------|-------|------|
| Base LLM | Qwen3.5-9B | Base model for all LoRA adapters |
| LoRA training | Qwen3.5-4B / 9B | Train 10+ domain-specific LoRA adapters |
| Canonicalization | Learned map | Map LoRA matrices to Grassmann manifold |
| **Hardware** | **8× A100-80GB** | Training domain LoRAs and learning canonical map |

## Domain LoRA Zoo (10+ adapters)

| Domain | Dataset | LoRA Rank | Expected Quality |
|--------|---------|-----------|------------------|
| Math | GSM8K + MATH | r=16 | ~70% acc on GSM8K |
| Code | CodeAlpaca + MBPP | r=16 | ~40% pass@1 on MBPP |
| Medical | MedQA + PubMedQA | r=16 | ~55% acc on MedQA |
| Legal | LegalBench | r=16 | ~60% acc |
| Creative | WritingPrompts | r=16 | High MAUVE score |
| Science | SciQ + ARC | r=16 | ~75% acc on ARC |
| Finance | FinQA + ConvFinQA | r=16 | ~55% acc |
| Multilingual | MGSM (zh, ja, de) | r=16 | ~60% acc |
| Safety | BeaverTails | r=16 | Low harmful rate |
| Instruction | Alpaca + ShareGPT | r=16 | High MT-Bench |
| Reasoning | StrategyQA + BBH | r=16 | ~65% acc |
| Summarization | CNN/DailyMail | r=16 | High ROUGE |

## Repository Structure

```
nips-text2subspace/
├── README.md              # This file
├── PROPOSAL.md            # Falsifiable thesis and success criteria
├── PLAN.md                # Stage-gate execution plan (8 weeks)
├── PAPERS.md              # Core references with URLs and annotations
├── EXPERIMENTS.md         # Evaluation protocol and results
├── environment.yml        # Conda environment spec
├── requirements.txt       # Pip dependencies
├── scripts/
│   ├── train_domain_loras.py      # Train 10+ domain LoRA adapters
│   ├── learn_canonicalization.py  # Learn Grassmann canonicalization map
│   ├── eval_algebra_ops.py        # Evaluate algebraic operations
│   └── run_text2subspace_pilot.py # Legacy pilot script
├── src/
│   ├── algebra/                   # Core algebraic operations
│   │   ├── grassmann.py           # Grassmann manifold utilities
│   │   ├── canonicalize.py        # Canonicalization map φ
│   │   ├── compose.py             # Composition ⊕
│   │   ├── interpolate.py         # Geodesic interpolation
│   │   ├── project.py             # Subspace projection π
│   │   └── subtract.py            # Logarithmic map subtraction ⊖
│   ├── lora/                      # LoRA training and loading
│   │   ├── train_lora.py          # Domain LoRA training loop
│   │   └── lora_zoo.py            # LoRA zoo management
│   └── eval/                      # Evaluation suite
│       ├── merge_eval.py          # Evaluate merged adapters
│       └── baselines.py           # TIES, DARE, Task Arithmetic baselines
├── configs/
│   ├── lora_training.yaml         # LoRA hyperparameters
│   ├── canonicalization.yaml      # Grassmann map config
│   └── eval_config.yaml           # Evaluation settings
└── results/
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate nips_lora_algebra

# Stage 1: Train domain LoRAs
python scripts/train_domain_loras.py --domains math,code,medical,legal,creative

# Stage 2: Learn canonicalization map
python scripts/learn_canonicalization.py --lora-zoo results/lora_zoo/

# Stage 3: Evaluate algebraic operations
python scripts/eval_algebra_ops.py --ops compose,interpolate,project,subtract
```

## Evaluation Protocol

### Composition Quality
For each pair of domains (A, B):
1. Train individual LoRAs for A and B
2. Compose via our algebra: φ⁻¹(φ(A) ⊕ φ(B))
3. Compose via TIES, DARE, Task Arithmetic
4. Evaluate on A's benchmark AND B's benchmark
5. Metric: **minimum degradation** = min(acc_A, acc_B) relative to individual LoRAs

### Interpolation Smoothness
For domains A and B, sweep t ∈ [0, 0.1, ..., 1.0]:
1. Interpolate: φ⁻¹(geodesic(φ(A), φ(B), t))
2. Evaluate on both benchmarks at each t
3. Metric: **smoothness** = max variation between adjacent t values
4. Compare with linear interpolation in weight space

### Projection for Task Removal
Given a multi-task LoRA and an unwanted capability:
1. Project LoRA onto complement of unwanted task subspace
2. Verify unwanted capability is removed
3. Verify other capabilities are preserved
4. Metric: **precision** of removal (target task acc drop) and **recall** of preservation (other task acc retention)

## Success Criteria

- **Primary:** Composition via Grassmann algebra degrades <= 2pt accuracy vs
  individual LoRAs, while TIES/DARE degrade >= 5pt (i.e., >= 3pt advantage)
- **Secondary:** Interpolation is monotonically smooth on Grassmann geodesic
  (no quality collapse at intermediate t)
- **Tertiary:** Projection achieves >= 90% removal of target capability with
  <= 2pt degradation on preserved capabilities

## Baselines

1. **TIES** (NeurIPS 2023) — Trim, Elect, Merge with sign consensus
2. **DARE** (ICML 2024) — Drop And REscale for merging
3. **Task Arithmetic** (ICLR 2023) — Linear add/subtract of task vectors
4. **Linear Interpolation** — Naive weight-space averaging
5. **Git Re-Basin** (ICLR 2023) — Permutation alignment before merging

## Key References

- Text-to-LoRA (ICLR 2025) — generates LoRA from text (different goal)
- LoRAGen (ICLR 2026) — generates LoRA for new tasks (different goal)
- TIES (NeurIPS 2023) — heuristic merging baseline
- DARE (ICML 2024) — heuristic merging baseline
- Task Arithmetic (ICLR 2023) — linear task vector arithmetic

See [PAPERS.md](PAPERS.md) for full annotated reference list.

## License

Research code for academic use.
