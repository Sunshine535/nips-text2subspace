#!/bin/bash
# Full pipeline: Merging → Evaluation → Ablation → Analysis
# Run AFTER domain LoRA training is complete
set -e

export PATH="/home/claude/.local/bin:$PATH"
cd /home/claude/nips-text2subspace

SEED=42
CONFIG=configs/domains.yaml
LORA_DIR=results/domain_loras
ALGEBRA_DIR=results/algebra
EVAL_DIR=results/eval
ABLATION_DIR=results/ablations
MAX_EVAL_SAMPLES=${1:-200}  # Override for quick runs

echo "=========================================="
echo " GrassMerge Full Experiment Pipeline"
echo "=========================================="
echo "Start: $(date)"
echo ""

# Verify LoRAs exist
TRAINED=0
for d in math code medical science history philosophy; do
    if [ -f "$LORA_DIR/$d/adapter_model.safetensors" ]; then
        TRAINED=$((TRAINED + 1))
    else
        echo "WARNING: Missing LoRA for domain: $d"
    fi
done
echo "Trained LoRAs: $TRAINED/6"
if [ "$TRAINED" -lt 2 ]; then
    echo "ERROR: Need at least 2 trained LoRAs"
    exit 1
fi

# ===== Stage 2: Merging Experiments =====
echo ""
echo "=========================================="
echo " Stage 2: GrassMerge + Baselines"
echo "=========================================="
python3 scripts/run_algebra_experiments.py \
    --config "$CONFIG" \
    --lora_dir "$LORA_DIR" \
    --output_dir "$ALGEBRA_DIR" \
    --seed "$SEED"

echo "Merging complete: $(date)"

# ===== Stage 3: Evaluation =====
echo ""
echo "=========================================="
echo " Stage 3: Domain Evaluation"
echo "=========================================="
python3 scripts/eval_domain_accuracy.py \
    --config "$CONFIG" \
    --lora_dir "$LORA_DIR" \
    --algebra_dir "$ALGEBRA_DIR" \
    --output_dir "$EVAL_DIR" \
    --domains math code medical science history philosophy \
    --max_samples "$MAX_EVAL_SAMPLES" \
    --seed "$SEED"

echo "Evaluation complete: $(date)"

# ===== Stage 4: Ablation Studies =====
echo ""
echo "=========================================="
echo " Stage 4: Ablations"
echo "=========================================="
python3 scripts/run_ablations.py \
    --config "$CONFIG" \
    --lora_dir "$LORA_DIR" \
    --output_dir "$ABLATION_DIR" \
    --seed "$SEED"

echo "Ablations complete: $(date)"

# ===== Stage 5: BGD Correlation Analysis =====
echo ""
echo "=========================================="
echo " Stage 5: BGD Correlation"
echo "=========================================="
python3 scripts/analyze_bgd_correlation.py \
    --interference_file "$ALGEBRA_DIR/interference_metrics.json" \
    --eval_file "$EVAL_DIR/eval_results.json" \
    --output_dir results/analysis

echo "Analysis complete: $(date)"

# ===== Summary =====
echo ""
echo "=========================================="
echo " Pipeline Complete"
echo "=========================================="
echo "End: $(date)"
echo ""
echo "Results:"
echo "  Merging:    $ALGEBRA_DIR/all_algebra_results.json"
echo "  Evaluation: $EVAL_DIR/eval_results.json"
echo "  Ablations:  $ABLATION_DIR/"
echo "  Analysis:   results/analysis/"
