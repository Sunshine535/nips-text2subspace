# GrassMerge: Geometry-Aware LoRA Composition on Grassmann Manifolds

Compose multiple LoRA adapters by treating their column/row spans as points on Grassmann manifolds and averaging in the intrinsic geometry. The merged adapter is rank-*r* by construction, PEFT-compatible, and requires zero additional parameters.

## Method

GrassMerge decomposes each LoRA delta **ΔW = BA · (α/r)** into a triple **(U, Σ, V)** via truncated SVD, then:

1. Computes the **Karcher mean** on G(*r*, *d_out*) × G(*r*, *d_in*) for left/right subspaces
2. Projects each core via **S_i = U\*ᵀ ΔW_i V\*** → averages → rank-*r* SVD
3. Reconstructs the merged delta and refactorizes into PEFT-compatible **(B\*, A\*)** factors

A **Bilateral Grassmann Distance (BGD)** diagnostic predicts interference *before* merging. We also provide spectral-weighted BGD, cosine, and Frobenius variants for correlation analysis.

## Quick Start

```bash
git clone https://github.com/Sunshine535/nips-text2subspace.git
cd nips-text2subspace

# Environment setup (Python 3.10+, PyTorch, CUDA)
bash setup.sh

# Run full 5-stage pipeline
bash scripts/run_production.sh

# Or run in background
nohup bash scripts/run_production.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done
ls results/.phase_markers/
```

### Resume After Interruption

Re-run `bash scripts/run_production.sh` — completed phases are automatically skipped via phase markers.
To force re-run all stages: `FORCE_RERUN=1 bash scripts/run_production.sh`

### Checkpoint Resume for Training

Each domain LoRA training supports checkpoint resume:

```bash
python scripts/train_domain_lora.py \
    --domain math \
    --resume_from_checkpoint results/domain_loras/math/checkpoint-500
```

## Experiment Pipeline (5 Stages)

| Stage | Description | Script | GPU |
|-------|-------------|--------|-----|
| 1 | Train 6 core domain LoRAs (math, code, medical, science, history, philosophy) | `train_domain_loras.py` | Multi-GPU (torchrun) |
| 2 | GrassMerge composition + pairwise baselines + BGD analysis | `run_algebra_experiments.py` | CPU/single GPU |
| 3 | Domain evaluation on standardized benchmarks | `eval_domain_accuracy.py` | Single GPU |
| 4 | Ablation studies (rank, normalization, interpolation, N-way) | `run_ablations.py` | CPU/single GPU |
| 5 | BGD-performance correlation analysis | `analyze_bgd_correlation.py` | CPU |

### Evaluation Benchmarks

| Domain | Benchmark | Metric |
|--------|-----------|--------|
| Math | GSM8K | Accuracy (exact match) |
| Code | MBPP (sanitized) | Accuracy (exact match) |
| Medical | MedQA (USMLE) | Accuracy (multiple choice) |
| Science | ARC-Challenge | Accuracy (multiple choice) |
| History | MMLU (high_school_us_history) | Accuracy (multiple choice) |
| Philosophy | MMLU (philosophy) | Accuracy (multiple choice) |

### Baselines

- **Task Arithmetic** — weighted sum of delta weights
- **TIES-Merging** — trim, elect sign, disjoint merge
- **DARE** — random drop + rescale before merge
- **KnOTS** — orthogonal Procrustes alignment
- **TSPA** — task-specific parameter alignment (per-component sign correction + permutation)
- **SVD-Procrustes** — align left singular vectors via Procrustes

## Multi-GPU Training

Stage 1 uses `torchrun` for data-parallel training across all available GPUs:

```bash
# Auto-detected GPUs
bash scripts/run_production.sh

# Specify GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_production.sh
```

## Project Structure

```
nips-text2subspace/
├── src/
│   └── lora_algebra.py              # GrassMerge, baselines (KnOTS, TSPA, TIES, DARE), BGD
├── configs/
│   └── domains.yaml                 # Domain datasets, LoRA hyperparams, training config
├── scripts/
│   ├── run_production.sh            # Production pipeline (5 stages, phase resume)
│   ├── run_all_experiments.sh       # Full pipeline (all 12 domains)
│   ├── train_domain_lora.py         # Single domain LoRA training (TRL SFTTrainer)
│   ├── train_domain_loras.py        # Orchestrator: sequential multi-domain training
│   ├── run_algebra_experiments.py   # GrassMerge + baselines + BGD analysis
│   ├── eval_domain_accuracy.py      # Benchmark evaluation
│   ├── run_ablations.py             # Ablation studies
│   ├── analyze_bgd_correlation.py   # BGD vs degradation correlation
│   ├── collect_results.py           # Aggregate results into summary report
│   └── gpu_utils.sh                 # GPU detection, torchrun helpers
├── setup.sh                         # Environment setup (venv + PyTorch + CUDA)
├── requirements.txt                 # Python dependencies
├── refine-logs/                     # Research proposal refinement history
└── results/                         # Experiment outputs (gitignored)
```

## Base Model

- **Qwen/Qwen3.5-9B** — primary model for all experiments
- LoRA config: r=16, α=32, dropout=0.05, targets=q/k/v/o/gate/up/down projections

## Results Collection

After the pipeline completes, generate a summary report:

```bash
python scripts/collect_results.py --results_dir results --output results/RESULTS_SUMMARY.md
```

## Citation

```bibtex
@inproceedings{grassmerge2026neurips,
  title     = {Grassmannian Composition: Geometry-Aware Merging of {LoRA} Adapters on Fixed-Rank Manifolds},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
