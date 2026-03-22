# LoRA Algebra: Algebraic Operations on LoRA Weight Spaces via Grassmann Manifolds

---

## How to Run (Complete Guide)

### Requirements

- Linux server with NVIDIA GPU (4-8x A100 80GB recommended)
- CUDA 12.8 compatible driver
- `git`, `curl` installed
- ~200GB disk space (model weights + checkpoints)

### Step 1: Clone and Run (One Command)

```bash
git clone https://github.com/Sunshine535/nips-text2subspace.git
cd nips-text2subspace
bash run.sh
```

`run.sh` will automatically:
1. Install `uv` package manager (if not present)
2. Create Python 3.10 virtual environment
3. Install PyTorch 2.10 + CUDA 12.8
4. Install all dependencies
5. Run **all experiments** in full production mode
6. Display real-time progress in terminal and save to `run.log`

### Step 2: Monitor Progress

If running in foreground (default):
```bash
# Progress is displayed in real-time
# Press Ctrl+C to stop (can resume later with bash run.sh)
```

If running in background (recommended for long experiments):
```bash
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Step 3: Check Completion

```bash
cat results/.pipeline_done
# If this file exists and shows "PIPELINE_COMPLETE", all experiments finished successfully
```

### Step 4: Package and Send Results

```bash
# Option A: Push to GitHub (recommended)
git add results/ logs/
git commit -m "Experiment results $(date +%Y%m%d)"
git push origin main

# Option B: Create tarball for manual transfer
bash collect_results.sh
# Creates: results_archive/nips-text2subspace_results_YYYYMMDD_HHMMSS.tar.gz
# Send this file via scp/email/cloud drive
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Experiment interrupted | Re-run `bash run.sh` — completed phases are automatically skipped |
| Want to re-run everything from scratch | `FORCE_RERUN=1 bash run.sh` |
| GPU out of memory | The script auto-detects GPUs; ensure CUDA drivers are installed |
| Network issues downloading models | Set `HF_ENDPOINT=https://hf-mirror.com` before running |
| Check which phases completed | `ls results/.phase_markers/` |

---


## How to Run (Complete Guide)

### Requirements

- Linux server with NVIDIA GPU (4-8x A100 80GB recommended)
- CUDA 12.8 compatible driver
- `git`, `curl` installed
- ~200GB disk space (model weights + checkpoints)

### Step 1: Clone and Run (One Command)

```bash
git clone https://github.com/Sunshine535/nips-text2subspace.git
cd nips-text2subspace
bash run.sh
```

`run.sh` will automatically:
1. Install `uv` package manager (if not present)
2. Create Python 3.10 virtual environment
3. Install PyTorch 2.10 + CUDA 12.8
4. Install all dependencies
5. Run **all experiments** in full production mode
6. Display real-time progress in terminal and save to `run.log`

### Step 2: Monitor Progress

If running in foreground (default):
```bash
# Progress is displayed in real-time
# Press Ctrl+C to stop (can resume later with bash run.sh)
```

If running in background (recommended for long experiments):
```bash
nohup bash run.sh > run.log 2>&1 &
tail -f run.log          # Watch progress
```

### Step 3: Check Completion

```bash
cat results/.pipeline_done
# If this file exists and shows "PIPELINE_COMPLETE", all experiments finished successfully
```

### Step 4: Package and Send Results

```bash
# Option A: Push to GitHub (recommended)
git add results/ logs/
git commit -m "Experiment results $(date +%Y%m%d)"
git push origin main

# Option B: Create tarball for manual transfer
bash collect_results.sh
# Creates: results_archive/nips-text2subspace_results_YYYYMMDD_HHMMSS.tar.gz
# Send this file via scp/email/cloud drive
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Experiment interrupted | Re-run `bash run.sh` — completed phases are automatically skipped |
| Want to re-run everything from scratch | `FORCE_RERUN=1 bash run.sh` |
| GPU out of memory | The script auto-detects GPUs; ensure CUDA drivers are installed |
| Network issues downloading models | Set `HF_ENDPOINT=https://hf-mirror.com` before running |
| Check which phases completed | `ls results/.phase_markers/` |

### Output Structure

After completion, key results are in:

```
nips-text2subspace/
├── results/              # All experiment outputs (JSON, figures, metrics)
│   └── .pipeline_done    # Completion marker
├── logs/                 # Per-phase log files
├── run.log               # Full pipeline log
└── results_archive/      # Packaged tarballs (after collect_results.sh)
```

---

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
