# Text2Subspace: Canonicalized Diffusion Hypernetwork for Instant LoRA Generation

## Overview

This project explores **text-conditioned adapter generation** in canonicalized low-rank subspaces. Instead of fine-tuning LoRA adapters from scratch for each new task, Text2Subspace learns a conditional generator that produces LoRA weights directly from a task description and a few examples—achieving near-LoRA quality with orders of magnitude less adaptation time.

**Target venue:** NeurIPS 2026

**Status:** ~30% complete (pilot stage)

## Research Questions

1. Can text-conditioned adapter generation approach gradient LoRA quality with >= 10x lower adaptation latency?
2. Does canonicalization improve unseen-task transfer?
3. Does merge-aware generation reduce interference in multi-skill composition?

## Core Idea

```
Task Description + Few Examples
          │
    ┌─────┴─────┐
    │ Encoder   │
    │ (text +   │
    │ exemplar) │
    └─────┬─────┘
          │
    ┌─────┴─────────────┐
    │ Conditional       │
    │ Generator         │
    │ (Diffusion /      │
    │  Hypernetwork)    │
    └─────┬─────────────┘
          │
    ┌─────┴─────────────┐
    │ Canonicalized     │   ← Handles permutation symmetry
    │ LoRA Subspace     │     and basis alignment
    └─────┬─────────────┘
          │
    Generated LoRA Weights → Plug into base LLM → Evaluate
```

## Method

1. **Adapter Zoo:** Collect diverse task-specific LoRA adapters
2. **Canonicalization:** Align adapter basis vectors to handle permutation symmetry
3. **Conditional Generator:** Train `G(task_text, exemplars) → LoRA weights` using diffusion or hypernetwork
4. **Merge-Aware Training:** Add conflict penalty for multi-skill adapter composition

## Current Results (Pilot)

Low-rank vs full-rank policy-head proxy experiment. Low-rank rank-8 matches full-rank (acc=0.40, utility=0.3848). Not yet real adapter generation.

## Repository Structure

```
nips-text2subspace/
├── README.md              # This file
├── PROPOSAL.md            # Falsifiable thesis and success criteria
├── PLAN.md                # Stage-gate execution plan
├── EXPERIMENTS.md          # Evaluation protocol and results
├── PAPERS.md              # Core references with URLs
├── README_RUN.md          # Runbook
├── environment.yml        # Conda environment spec
├── scripts/
│   └── run_text2subspace_pilot.py   # Pilot experiment script
└── results/
    ├── text2subspace_pilot_20260227_150038.json
    └── text2subspace_pilot_20260227_151439.json
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate nips_text2subspace
python scripts/run_text2subspace_pilot.py
```

## Quantitative Success Criteria

- **Primary:** Achieve >= 90% of LoRA quality with >= 10x lower adaptation latency
- **Secondary:** Unseen-family transfer >= +2 absolute over non-canonical baseline

## Key References

- D2NWG (ICLR 2025)
- Text-to-LoRA (ICML 2025)
- Twin-Merging (NeurIPS 2024)
- Git Re-Basin (ICLR 2023)
- HINT (ACL 2023)

See [PAPERS.md](PAPERS.md) for full list with direct URLs.

## Remaining Work

1. Build adapter zoo from diverse task LoRAs
2. Implement canonicalization algorithm
3. Train conditional generator (diffusion-based)
4. Evaluate zero-shot/few-shot on unseen tasks
5. Compare against standard LoRA, Text-to-LoRA, random init

## License

Research code for academic use.
