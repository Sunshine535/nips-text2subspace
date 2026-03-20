# LoRA Algebra: Algebraic Operations on LoRA Weight Spaces via Grassmann Manifolds

> **NeurIPS 2026 Submission**

## Abstract

Low-Rank Adaptation (LoRA) modules encode domain-specific knowledge as low-rank perturbations, but combining multiple LoRA adapters remains ad-hoc and theoretically underexplored. We introduce **LoRA Algebra**, a principled framework that treats LoRA weight matrices as points on Grassmann manifolds and defines six algebraic operations—addition, subtraction, interpolation, projection, intersection, and negation—with formal guarantees on subspace geometry. Applied to 12 domain-specific LoRA adapters trained on Qwen3.5-9B, our algebra produces composed adapters that outperform TIES-Merging, DARE, and Task Arithmetic baselines by 2–5% on domain benchmarks while enabling novel capabilities like knowledge transfer via subspace projection.

## Quick Start

```bash
git clone https://github.com/<org>/nips-text2subspace.git
cd nips-text2subspace
bash setup.sh
bash scripts/run_all_experiments.sh
```

## Hardware Requirements

| Resource | Specification |
|----------|--------------|
| GPUs | 4–8× NVIDIA A100 80GB (auto-detected) |
| RAM | ≥ 128 GB |
| Disk | ≥ 400 GB (12 LoRA checkpoints + algebra outputs) |
| CUDA | ≥ 12.1 |

GPU count is automatically detected via `scripts/gpu_utils.sh`. The pipeline adapts batch sizes and parallelism accordingly.

## Project Structure

```
nips-text2subspace/
├── README.md
├── LICENSE                            # MIT License
├── setup.sh                           # One-command environment setup
├── requirements.txt                   # Pinned dependencies
├── configs/
│   └── domains.yaml                   # 12-domain configuration
├── scripts/
│   ├── gpu_utils.sh                   # Shared GPU auto-detection
│   ├── run_all_experiments.sh         # Master pipeline (4 stages)
│   ├── train_domain_loras.py          # Stage 1: 12 domain LoRA training
│   ├── run_algebra_experiments.py     # Stage 2: 66-pair algebra ops
│   ├── eval_domain_accuracy.py        # Stage 3: Domain evaluation
│   ├── run_ablations.py              # Stage 4: Rank/operation ablations
│   └── lora_algebra_ops.py           # Core algebra operations library
├── src/                               # Grassmann manifold utilities
├── results/                           # Experiment outputs
├── logs/                              # Training logs
└── docs/                              # Additional documentation
```

## Experiments

| # | Stage | Description | Est. Time (8×A100) |
|---|-------|-------------|-------------------|
| 1 | Domain LoRA Training | Train 12 domain-specific LoRA adapters (r=16, α=32, 2 epochs each) | ~96 hrs |
| 2 | Algebra Experiments | All 6 operations on C(12,2)=66 domain pairs + TIES/DARE/TaskArith baselines | ~120 hrs |
| 3 | Domain Evaluation | Evaluate original + composed adapters on domain benchmarks | ~48 hrs |
| 4 | Ablation Studies | Rank sensitivity (r ∈ {4,8,16,32,64}), operation-type ablation, manifold distance analysis | ~36 hrs |

## Timeline & GPU Hours

- **Model**: Qwen/Qwen3.5-9B
- **Total estimated GPU-hours**: ~3600 (8× A100-80GB)
- **Wall-clock time**: ~18–20 days on 8× A100

## Citation

```bibtex
@inproceedings{loraalgebra2026neurips,
  title     = {{LoRA} Algebra: Algebraic Operations on {LoRA} Weight Spaces via Grassmann Manifolds},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
