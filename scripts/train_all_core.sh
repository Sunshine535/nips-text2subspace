#!/bin/bash
# Train all 6 core domain LoRAs — auto-detects GPU count
# Usage: bash scripts/train_all_core.sh [max_samples] [num_gpus]
set -e

cd "$(dirname "$0")/.."

MAX_SAMPLES=${1:-10000}
NUM_GPUS=${2:-0}  # 0 = auto-detect
SEED=42
CONFIG=configs/domains.yaml
OUTPUT_ROOT=results/domain_loras
CORE_DOMAINS="math code medical science history philosophy"

# Auto-detect GPUs if not specified
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi

echo "=== Training 6 Core Domain LoRAs ==="
echo "Max samples per domain: $MAX_SAMPLES"
echo "GPUs: $NUM_GPUS (per domain)"
echo "Seed: $SEED"
echo "Start time: $(date)"
echo ""

for domain in $CORE_DOMAINS; do
    echo "-------------------------------------------"
    echo "Training domain: $domain"
    echo "Start: $(date)"

    if [ -f "$OUTPUT_ROOT/$domain/adapter_model.safetensors" ]; then
        echo "  Already trained, skipping. (delete to retrain)"
        continue
    fi

    python3 scripts/train_domain_lora.py \
        --config "$CONFIG" \
        --domain "$domain" \
        --output_dir "$OUTPUT_ROOT/$domain" \
        --max_samples "$MAX_SAMPLES" \
        --num_gpus "$NUM_GPUS" \
        --seed "$SEED"

    echo "Finished: $domain at $(date)"
    echo ""
done

echo "=== All core domains trained ==="
echo "End time: $(date)"
ls -la $OUTPUT_ROOT/*/adapter_model.safetensors 2>/dev/null | wc -l
echo "domains trained successfully"
