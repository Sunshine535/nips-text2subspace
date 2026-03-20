#!/usr/bin/env bash
set -euo pipefail

#####################################################################
#  LoRA Algebra: Full Experiment Pipeline
#
#  Pipeline: train_domain_loras → algebra_experiments → eval → ablations
#
#  Hardware: 4–8× A100-80GB (auto-detected)
#  Estimated time: ~24-36 hours total
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=gpu_utils.sh
source "${SCRIPT_DIR}/gpu_utils.sh"
auto_setup

CONFIG="${PROJECT_DIR}/configs/domains.yaml"
LORA_DIR="${PROJECT_DIR}/results/domain_loras"
ALGEBRA_DIR="${PROJECT_DIR}/results/algebra"
EVAL_DIR="${PROJECT_DIR}/results/eval"
ABLATION_DIR="${PROJECT_DIR}/results/ablations"

cd "$PROJECT_DIR"
mkdir -p results

echo "================================================================"
echo "  LoRA Algebra — Full Experiment Pipeline"
echo "  Base model : Qwen/Qwen3.5-9B"
echo "  Domains    : 12"
echo "  LoRA       : r=16, alpha=32, 2 epochs"
echo "  GPUs       : ${NUM_GPUS}"
echo "  Config     : ${CONFIG}"
echo "  Started    : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

# ==========================================
#  STAGE 1: Train 12 Domain LoRAs
# ==========================================
echo ""
echo "========== STAGE 1: Train Domain LoRAs =========="
python scripts/train_domain_loras.py \
    --config "$CONFIG" \
    --output_root "$LORA_DIR" \
    --num_gpus ${NUM_GPUS} \
    2>&1 | tee results/stage1_train.log

echo "  [DONE] Stage 1 — Domain LoRA training complete"

# ==========================================
#  STAGE 2: LoRA Algebra Experiments
# ==========================================
echo ""
echo "========== STAGE 2: Algebra Experiments =========="
python scripts/run_algebra_experiments.py \
    --config "$CONFIG" \
    --lora_dir "$LORA_DIR" \
    --output_dir "$ALGEBRA_DIR" \
    2>&1 | tee results/stage2_algebra.log

echo "  [DONE] Stage 2 — Algebra experiments complete"

# ==========================================
#  STAGE 3: Domain-Specific Evaluation
# ==========================================
echo ""
echo "========== STAGE 3: Domain Evaluation =========="
python scripts/eval_domain_accuracy.py \
    --config "$CONFIG" \
    --lora_dir "$LORA_DIR" \
    --algebra_dir "$ALGEBRA_DIR" \
    --output_dir "$EVAL_DIR" \
    2>&1 | tee results/stage3_eval.log

echo "  [DONE] Stage 3 — Evaluation complete"

# ==========================================
#  STAGE 4: Ablation Study
# ==========================================
echo ""
echo "========== STAGE 4: Ablation Study =========="
python scripts/run_ablations.py \
    --config "$CONFIG" \
    --lora_dir "$LORA_DIR" \
    --output_dir "$ABLATION_DIR" \
    2>&1 | tee results/stage4_ablations.log

echo "  [DONE] Stage 4 — Ablations complete"

# ==========================================
#  Summary
# ==========================================
echo ""
echo "================================================================"
echo "  Pipeline Complete — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
echo ""
echo "  Results:"
echo "    Domain LoRAs  : ${LORA_DIR}/"
echo "    Algebra       : ${ALGEBRA_DIR}/all_algebra_results.json"
echo "    Evaluation    : ${EVAL_DIR}/eval_results.json"
echo "    Ablations     : ${ABLATION_DIR}/ablation_results.json"
echo "    Comparison    : ${EVAL_DIR}/comparison_table.md"
echo ""
echo "================================================================"
