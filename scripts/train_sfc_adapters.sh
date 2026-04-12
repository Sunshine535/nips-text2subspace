#!/usr/bin/env bash
# Train domain-specific LoRA adapters for SFC experiments.
# Auto-detects GPU count and distributes training accordingly.
#
# Usage:
#   bash scripts/train_sfc_adapters.sh [--model MODEL] [--output DIR]
#
# Supports: 1-8 GPUs, auto-adapts batch size and parallelism.

set -euo pipefail

MODEL="${1:-Qwen/Qwen3.5-9B}"
OUTPUT_DIR="${2:-results/sfc_loras}"
SAMPLES_PER_DOMAIN=5000
RANK=16
ALPHA=32
EPOCHS=2

# --- GPU auto-detection ---
if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
else
    NUM_GPUS=0
    GPU_MEM=0
fi

echo "=== SFC Adapter Training ==="
echo "Model:    $MODEL"
echo "Output:   $OUTPUT_DIR"
echo "GPUs:     $NUM_GPUS (${GPU_MEM}MB each)"
echo "Domains:  math, code, medical, science, history, philosophy, law, reasoning"
echo ""

# Adapt batch size to GPU memory
if [ "${GPU_MEM:-0}" -ge 70000 ]; then
    BATCH_SIZE=8
    GRAD_ACCUM=2
elif [ "${GPU_MEM:-0}" -ge 40000 ]; then
    BATCH_SIZE=4
    GRAD_ACCUM=4
elif [ "${GPU_MEM:-0}" -ge 20000 ]; then
    BATCH_SIZE=2
    GRAD_ACCUM=8
else
    BATCH_SIZE=1
    GRAD_ACCUM=16
fi

DOMAINS=("math" "code" "medical" "science" "history" "philosophy" "law" "reasoning")

mkdir -p "$OUTPUT_DIR"

# --- Training function ---
train_domain() {
    local domain=$1
    local gpu_id=$2
    local domain_dir="$OUTPUT_DIR/$domain"

    if [ -f "$domain_dir/adapter_config.json" ] && [ -f "$domain_dir/adapter_model.safetensors" ]; then
        echo "[SKIP] $domain — adapter already exists"
        return 0
    fi

    echo "[TRAIN] $domain on GPU $gpu_id (batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM)"

    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/run_sfc_pilot.py \
        --model "$MODEL" \
        --train-adapters \
        --train-samples "$SAMPLES_PER_DOMAIN" \
        --lora-rank "$RANK" \
        --lora-alpha "$ALPHA" \
        --train-epochs "$EPOCHS" \
        --adapter-dir "$OUTPUT_DIR" \
        --device cuda \
        --output /dev/null 2>&1 | while IFS= read -r line; do echo "  [$domain] $line"; done

    echo "[DONE] $domain"
}

# --- Multi-GPU parallel training ---
if [ "$NUM_GPUS" -ge 2 ]; then
    echo "Running ${#DOMAINS[@]} domains across $NUM_GPUS GPUs in parallel..."
    echo ""

    # Use GNU parallel or background jobs
    pids=()
    for i in "${!DOMAINS[@]}"; do
        gpu_id=$((i % NUM_GPUS))
        domain="${DOMAINS[$i]}"

        train_domain "$domain" "$gpu_id" &
        pids+=($!)

        # If we've filled all GPUs, wait for one to finish
        if [ ${#pids[@]} -ge "$NUM_GPUS" ]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done

    # Wait for remaining
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
else
    echo "Single GPU training (sequential)..."
    echo ""
    for domain in "${DOMAINS[@]}"; do
        train_domain "$domain" "0"
    done
fi

echo ""
echo "=== Training Complete ==="
echo "Adapters saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*/adapter_config.json 2>/dev/null | wc -l
echo "adapters ready."
