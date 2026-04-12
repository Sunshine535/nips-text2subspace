#!/bin/bash
# =============================================================================
# Sparse Feature Composition (SFC) — Full Pipeline
# Auto-detects GPUs. One command: bash run.sh
# =============================================================================
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

# --- HF cache ---
export HF_HOME="${HF_HOME:-$(dirname "$PROJ_DIR")/.cache/hf}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

# --- Model config ---
MODEL="${MODEL:-Qwen/Qwen3.5-9B-Base}"
SAE_DIR="${SAE_DIR:-saes/qwen3.5-9b}"
SAE_LAYERS="${SAE_LAYERS:-8,12,16,24,32}"
SAE_FEATURES="${SAE_FEATURES:-16384}"
SAE_TOKENS="${SAE_TOKENS:-100000000}"
LORA_DIR="${LORA_DIR:-results/sfc_loras}"

# --- Activate venv ---
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

echo "============================================"
echo "  SFC — Full Pipeline"
echo "  Model:  $MODEL"
echo "  GPUs:   $NUM_GPUS"
echo "  SAE:    $SAE_DIR ($SAE_LAYERS)"
echo "  HF_HOME: $HF_HOME"
echo "============================================"

# ---------------------------------------------------------------------------
# Stage 0: Synthetic theorem verification (CPU, ~2min)
# ---------------------------------------------------------------------------
echo ""; echo "=== Stage 0: Synthetic Verification ==="
if [ -f results/sfc_synthetic.json ]; then
    echo "  [SKIP] results/sfc_synthetic.json exists"
else
    python3 scripts/verify_sfc_synthetic.py --output results/sfc_synthetic.json
fi

# ---------------------------------------------------------------------------
# Stage 1: Train SAEs (parallel across GPUs, ~3-4h)
# ---------------------------------------------------------------------------
echo ""; echo "=== Stage 1: Train SAEs ==="
NEED_SAE=0
IFS=',' read -ra LAYERS <<< "$SAE_LAYERS"
for L in "${LAYERS[@]}"; do
    if [ ! -f "$SAE_DIR/layer_${L}/sae_weights.safetensors" ]; then
        NEED_SAE=1; break
    fi
done

if [ "$NEED_SAE" -eq 0 ]; then
    echo "  [SKIP] All SAEs already trained in $SAE_DIR"
else
    python3 scripts/train_sae.py \
        --model "$MODEL" \
        --layers "$SAE_LAYERS" \
        --output "$SAE_DIR" \
        --n-features "$SAE_FEATURES" \
        --training-tokens "$SAE_TOKENS"
fi

# ---------------------------------------------------------------------------
# Stage 2: Train domain LoRA adapters (8 domains, parallel across GPUs, ~30min)
# ---------------------------------------------------------------------------
echo ""; echo "=== Stage 2: Train LoRA Adapters ==="
bash scripts/train_sfc_adapters.sh "$MODEL" "$LORA_DIR"

# ---------------------------------------------------------------------------
# Stage 3: E0 Pilot — sparsity verification (kill criterion)
# ---------------------------------------------------------------------------
echo ""; echo "=== Stage 3: E0 Pilot ==="
if [ -f results/sfc_pilot.json ]; then
    echo "  [SKIP] results/sfc_pilot.json exists"
else
    python3 scripts/run_sfc_pilot.py \
        --model "$MODEL" \
        --sae-repo "$SAE_DIR" \
        --layers "$SAE_LAYERS" \
        --adapter-dir "$LORA_DIR" \
        --probe-size 512 \
        --batch-size 4 \
        --output results/sfc_pilot.json
fi

# Check kill criterion
python3 -c "
import json
r = json.load(open('results/sfc_pilot.json'))
s = r.get('summary', {})
v = s.get('kill_criterion_verdict', 'UNKNOWN')
print(f'  Kill criterion: {v} (max sparsity: {s.get(\"max_sparsity\", \"?\"):.4f})')
if v == 'FAIL':
    print('  [ABORT] Sparsity > 20%%. SFC framework not viable.')
    exit(1)
"

# ---------------------------------------------------------------------------
# Stage 4: E1-E5 Full evaluation
# ---------------------------------------------------------------------------
echo ""; echo "=== Stage 4: Full Evaluation ==="
python3 scripts/run_sfc_full.py \
    --experiment all \
    --model "$MODEL" \
    --sae-repo "$SAE_DIR" \
    --layers "$SAE_LAYERS" \
    --adapter-dir "$LORA_DIR" \
    --output results/sfc_full/

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  SFC Pipeline Complete!"
echo "  Results: $PROJ_DIR/results/"
echo "============================================"
echo "Key files:"
echo "  results/sfc_synthetic.json  — theorem verification"
echo "  results/sfc_pilot.json      — sparsity + FDS"
echo "  results/sfc_full/           — full comparison"
