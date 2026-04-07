#!/bin/bash
# Complete evaluation pipeline: all methods, all domains, 200 samples
# GPU runs eval, CPU runs ablations in parallel
set -e
cd "$(dirname "$0")/.."

CONFIG=configs/domains.yaml
LORA_DIR=results/domain_loras
ALGEBRA_DIR=results/algebra
EVAL_DIR=results/eval_v2
SEED=42
MAX_SAMPLES=${1:-200}

echo "=========================================="
echo " Complete Evaluation Pipeline"
echo " Samples: $MAX_SAMPLES per benchmark"
echo " Start: $(date)"
echo "=========================================="

mkdir -p $EVAL_DIR logs

# Phase 1: Full evaluation (GPU) - base + individual + GrassMerge + Task Arithmetic
echo ""
echo "=== Phase 1: Full Evaluation (GPU) ==="
python3 -u scripts/eval_domain_accuracy.py \
    --config $CONFIG \
    --lora_dir $LORA_DIR \
    --algebra_dir $ALGEBRA_DIR \
    --output_dir $EVAL_DIR \
    --domains math code medical science history philosophy \
    --max_samples $MAX_SAMPLES \
    --seed $SEED 2>&1 | tee logs/eval_v2.log

echo ""
echo "=== Evaluation Complete: $(date) ==="
echo "Results: $EVAL_DIR/eval_results.json"

# Phase 2: Generate paper results
echo ""
echo "=== Generating Paper Results ==="
python3 -u scripts/generate_paper_results.py 2>&1

echo ""
echo "=========================================="
echo " ALL COMPLETE: $(date)"
echo "=========================================="
