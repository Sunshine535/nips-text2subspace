#!/bin/bash
# Train all 6 core domain LoRAs sequentially on single GPU
# Usage: bash scripts/train_all_core.sh [max_samples]
set -e

export PATH="/home/claude/.local/bin:$PATH"
cd /home/claude/nips-text2subspace

MAX_SAMPLES=${1:-10000}
SEED=42
CONFIG=configs/domains.yaml
OUTPUT_ROOT=results/domain_loras
CORE_DOMAINS="math code medical science history philosophy"

echo "=== Training 6 Core Domain LoRAs ==="
echo "Max samples per domain: $MAX_SAMPLES"
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
        --num_gpus 1 \
        --seed "$SEED"

    echo "Finished: $domain at $(date)"
    echo ""
done

echo "=== All core domains trained ==="
echo "End time: $(date)"
ls -la $OUTPUT_ROOT/*/adapter_model.safetensors 2>/dev/null | wc -l
echo "domains trained successfully"
