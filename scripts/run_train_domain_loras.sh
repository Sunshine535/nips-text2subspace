#!/usr/bin/env bash
# DEPRECATED: Use scripts/run_production.sh as the canonical entry point.
# This script is kept for reference only.
set -euo pipefail
_PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_PROJ_ROOT/.venv/bin/activate" ]; then source "$_PROJ_ROOT/.venv/bin/activate"; fi
export PATH="$HOME/.local/bin:$PATH"

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/domains.yaml"
OUTPUT_ROOT="${PROJECT_DIR}/outputs/domain_loras"
NUM_GPUS=8

DOMAINS=(math code medical legal creative science history finance philosophy multilingual)

cd "$PROJECT_DIR"

echo "========================================"
echo "  Text2Subspace: Train 10 Domain LoRAs"
echo "  Base model: Qwen/Qwen3.5-9B"
echo "  GPUs: ${NUM_GPUS}"
echo "========================================"

for domain in "${DOMAINS[@]}"; do
    echo ""
    echo "========== Training domain: ${domain} =========="
    DOMAIN_OUTPUT="${OUTPUT_ROOT}/${domain}"
    mkdir -p "$DOMAIN_OUTPUT"

    if [ -f "${DOMAIN_OUTPUT}/adapter_config.json" ]; then
        echo "  [SKIP] ${domain} already trained at ${DOMAIN_OUTPUT}"
        continue
    fi

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=$((29500 + RANDOM % 100)) \
        scripts/train_domain_lora.py \
            --config "$CONFIG" \
            --domain "$domain" \
            --output_dir "$DOMAIN_OUTPUT" \
            --num_gpus ${NUM_GPUS} \
        2>&1 | tee "${DOMAIN_OUTPUT}/train.log"

    echo "  [DONE] ${domain} → ${DOMAIN_OUTPUT}"
done

echo ""
echo "========================================"
echo "  All domain LoRAs trained."
echo "  Running LoRA algebra operations..."
echo "========================================"

python scripts/lora_algebra_ops.py \
    --config "$CONFIG" \
    --lora_dir "$OUTPUT_ROOT" \
    --output_dir "${PROJECT_DIR}/outputs/algebra_results" \
    2>&1 | tee "${PROJECT_DIR}/outputs/algebra_results/algebra.log"

echo ""
echo "========================================"
echo "  Evaluating composed LoRAs..."
echo "========================================"

python scripts/eval_lora_algebra.py \
    --config "$CONFIG" \
    --lora_dir "$OUTPUT_ROOT" \
    --algebra_dir "${PROJECT_DIR}/outputs/algebra_results" \
    --output_dir "${PROJECT_DIR}/outputs/eval_results" \
    2>&1 | tee "${PROJECT_DIR}/outputs/eval_results/eval.log"

echo ""
echo "=== Pipeline complete ==="
