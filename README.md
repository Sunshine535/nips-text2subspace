# LoRA Algebra: Algebraic Operations on LoRA Weight Spaces via Grassmann Manifolds

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-text2subspace.git
cd nips-text2subspace

# 2. Install dependencies
bash setup.sh

# 3. Run all experiments
bash run.sh

# 4. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-text2subspace_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`

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
