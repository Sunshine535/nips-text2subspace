#!/bin/bash
# =============================================================================
# Rank Bottleneck of Adapter Composition — Full Pipeline
# Auto-detects GPUs. Resumes from last completed stage.
# Usage: bash run.sh
# =============================================================================
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
if [ -f "$PROJ_DIR/.venv/bin/activate" ]; then
    source "$PROJ_DIR/.venv/bin/activate"
elif [ -d "$PROJ_DIR/.venv/conda-meta" ] && command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$PROJ_DIR/.venv"
fi

python3 -c "import torch, transformers, datasets" 2>/dev/null || {
    echo "[ERROR] Missing dependencies. Run: bash setup.sh"; exit 1
}

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No GPUs detected."; exit 1
fi

CONFIG=configs/domains.yaml
LORA_DIR=results/domain_loras
SEED=${SEED:-42}
MAX_TRAIN=${MAX_TRAIN_SAMPLES:-10000}
MAX_EVAL=${MAX_EVAL_SAMPLES:-500}

echo "============================================"
echo "  Rank Bottleneck — Full Pipeline"
echo "  GPUs: $NUM_GPUS (auto-detected)"
echo "  Seed: $SEED"
echo "============================================"

# ---------------------------------------------------------------------------
# Stage 1: Train domain LoRAs (torchrun DDP, auto multi-GPU)
# ---------------------------------------------------------------------------
echo ""; echo "=== Stage 1: Train Domain LoRAs ==="
python3 scripts/train_domain_loras.py \
    --config "$CONFIG" \
    --output_root "$LORA_DIR" \
    --domains math code medical science history philosophy \
    --seed "$SEED"
# --num_gpus auto-detects (default=0 → torch.cuda.device_count())

# ---------------------------------------------------------------------------
# Stage 2: Composition rank analysis (CRS + phase diagram data)
# ---------------------------------------------------------------------------
echo ""; echo "=== Stage 2: Composition Rank Analysis ==="
python3 scripts/compute_crs.py \
    --lora_dir "$LORA_DIR" \
    --output results/crs

# ---------------------------------------------------------------------------
# Stage 3: Merge baselines + BAC
# ---------------------------------------------------------------------------
echo ""; echo "=== Stage 3: Merge Experiments ==="
python3 scripts/run_algebra_experiments.py \
    --config "$CONFIG" \
    --lora_dir "$LORA_DIR" \
    --output_dir results/algebra

# ---------------------------------------------------------------------------
# Stage 4: Evaluate (parallel across GPUs)
# ---------------------------------------------------------------------------
echo ""; echo "=== Stage 4: Evaluation ($NUM_GPUS GPUs parallel) ==="
python3 scripts/run_parallel_eval.py \
    --num_gpus "$NUM_GPUS" \
    --lora_dir "$LORA_DIR" \
    --output_dir results/eval_v3 \
    --max_samples "$MAX_EVAL"

# ---------------------------------------------------------------------------
# Stage 5: Collect results
# ---------------------------------------------------------------------------
echo ""; echo "=== Stage 5: Collect Results ==="
python3 scripts/collect_results.py 2>&1 || true

echo ""
echo "=== Pipeline Complete ==="
echo "Results: $PROJ_DIR/results/"
