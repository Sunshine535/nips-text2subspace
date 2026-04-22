#!/usr/bin/env bash
# =============================================================================
# Comprehensive Experiment Pipeline for NeurIPS Best Paper
# Addresses ALL reviewer concerns from Round 7 review
# =============================================================================
set -euo pipefail

export HF_HOME="${HF_HOME:-/openbayes/input/input0}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/gpu_utils.sh"
auto_setup

cd "$PROJECT_DIR"

CONFIG="${PROJECT_DIR}/configs/domains.yaml"
LORA_DIR="${PROJECT_DIR}/results/domain_loras"
ALGEBRA_DIR="${PROJECT_DIR}/results/algebra"
EVAL_DIR="${PROJECT_DIR}/results/eval"
ABLATION_DIR="${PROJECT_DIR}/results/ablations"
ANALYSIS_DIR="${PROJECT_DIR}/results/analysis"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$EVAL_DIR" "$ALGEBRA_DIR" "$ABLATION_DIR" "$ANALYSIS_DIR"

SEED="${SEED:-42}"
CORE_DOMAINS="math code medical science history philosophy"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"

echo "================================================================"
echo "  GrassMerge — Comprehensive NeurIPS Pipeline"
echo "  Model: Qwen/Qwen3-8B | GPUs: ${NUM_GPUS} | Seed: ${SEED}"
echo "  Eval samples: ${EVAL_SAMPLES} per benchmark"
echo "  $(date)"
echo "================================================================"

# Phase marker system
PHASE_DIR="$PROJECT_DIR/results/.comprehensive_phases"
mkdir -p "$PHASE_DIR"
FORCE="${FORCE_RERUN:-0}"
phase_done() { touch "$PHASE_DIR/phase_${1}.done"; echo "[PHASE $1] Done at $(date)"; }
is_done() {
    [[ "$FORCE" == "1" ]] && return 1
    [[ -f "$PHASE_DIR/phase_${1}.done" ]] && echo "[PHASE $1] Already done. Skipping." && return 0
    return 1
}

# ============================================================
# STEP 1: Verify domain LoRAs exist
# ============================================================
echo "========== STEP 1: Verify Domain LoRAs =========="
for d in $CORE_DOMAINS; do
    if [ ! -f "$LORA_DIR/$d/adapter_model.safetensors" ]; then
        echo "[ERROR] Missing adapter for domain: $d"
        echo "  Run training first: python scripts/train_domain_loras.py --config $CONFIG --output_root $LORA_DIR --domains $CORE_DOMAINS"
        exit 1
    fi
done
echo "  All 6 domain LoRAs verified."

# ============================================================
# STEP 2: Run ALL algebra experiments (GrassMerge + 5 baselines)
# ============================================================
if ! is_done "2_algebra"; then
    echo "========== STEP 2: Algebra Experiments (all baselines) =========="
    python scripts/run_algebra_experiments.py \
        --config "$CONFIG" --lora_dir "$LORA_DIR" --output_dir "$ALGEBRA_DIR" \
        --seed ${SEED} \
        2>&1 | tee "$LOG_DIR/step2_algebra_${TIMESTAMP}.log"
    phase_done "2_algebra"
fi

# ============================================================
# STEP 3: Full evaluation — base model + individual LoRAs + GrassMerge + ALL baselines
# ============================================================
if ! is_done "3_eval"; then
    echo "========== STEP 3: Comprehensive Evaluation =========="
    python scripts/eval_domain_accuracy.py \
        --config "$CONFIG" --lora_dir "$LORA_DIR" --algebra_dir "$ALGEBRA_DIR" \
        --output_dir "$EVAL_DIR" \
        --domains $CORE_DOMAINS \
        --max_samples ${EVAL_SAMPLES} \
        --seed ${SEED} \
        2>&1 | tee "$LOG_DIR/step3_eval_${TIMESTAMP}.log"
    phase_done "3_eval"
fi

# ============================================================
# STEP 4: Cross-domain interference matrix
# Evaluate each merged adapter on ALL 6 domains (not just its 2 constituents)
# ============================================================
if ! is_done "4_interference"; then
    echo "========== STEP 4: Cross-Domain Interference Matrix =========="
    python scripts/eval_interference_matrix.py \
        --config "$CONFIG" --lora_dir "$LORA_DIR" --algebra_dir "$ALGEBRA_DIR" \
        --output_dir "$EVAL_DIR" \
        --domains $CORE_DOMAINS \
        --max_samples ${EVAL_SAMPLES} \
        --seed ${SEED} \
        2>&1 | tee "$LOG_DIR/step4_interference_${TIMESTAMP}.log"
    phase_done "4_interference"
fi

# ============================================================
# STEP 5: Multitask union baseline (train LoRA on union of domain pairs)
# ============================================================
if ! is_done "5_multitask"; then
    echo "========== STEP 5: Multitask Union Baselines =========="
    python scripts/train_multitask_baselines.py \
        --config "$CONFIG" --output_dir "${PROJECT_DIR}/results/multitask_loras" \
        --pairs "history+philosophy,history+science,science+philosophy" \
        --seed ${SEED} \
        2>&1 | tee "$LOG_DIR/step5_multitask_${TIMESTAMP}.log"
    phase_done "5_multitask"
fi

# ============================================================
# STEP 6: Evaluate multitask baselines
# ============================================================
if ! is_done "6_eval_multitask"; then
    echo "========== STEP 6: Evaluate Multitask Baselines =========="
    python scripts/eval_multitask_baselines.py \
        --config "$CONFIG" --multitask_dir "${PROJECT_DIR}/results/multitask_loras" \
        --output_dir "$EVAL_DIR" \
        --max_samples ${EVAL_SAMPLES} \
        --seed ${SEED} \
        2>&1 | tee "$LOG_DIR/step6_eval_multitask_${TIMESTAMP}.log"
    phase_done "6_eval_multitask"
fi

# ============================================================
# STEP 7: Ablations with downstream accuracy
# ============================================================
if ! is_done "7_ablations"; then
    echo "========== STEP 7: Downstream Ablations =========="
    python scripts/run_ablations_downstream.py \
        --config "$CONFIG" --lora_dir "$LORA_DIR" --algebra_dir "$ALGEBRA_DIR" \
        --output_dir "$ABLATION_DIR" \
        --max_samples ${EVAL_SAMPLES} \
        --seed ${SEED} \
        2>&1 | tee "$LOG_DIR/step7_ablations_${TIMESTAMP}.log"
    phase_done "7_ablations"
fi

# ============================================================
# STEP 8: BGD correlation analysis + comparison with simpler predictors
# ============================================================
if ! is_done "8_analysis"; then
    echo "========== STEP 8: BGD Correlation Analysis =========="
    python scripts/analyze_bgd_correlation.py \
        --interference_file "$ALGEBRA_DIR/interference_metrics.json" \
        --eval_file "$EVAL_DIR/eval_results.json" \
        --output_dir "$ANALYSIS_DIR" \
        2>&1 | tee "$LOG_DIR/step8_analysis_${TIMESTAMP}.log"
    phase_done "8_analysis"
fi

# ============================================================
# STEP 9: Runtime/memory profiling
# ============================================================
if ! is_done "9_profile"; then
    echo "========== STEP 9: Runtime & Memory Profiling =========="
    python scripts/profile_methods.py \
        --config "$CONFIG" --lora_dir "$LORA_DIR" \
        --output_dir "$ANALYSIS_DIR" \
        2>&1 | tee "$LOG_DIR/step9_profile_${TIMESTAMP}.log"
    phase_done "9_profile"
fi

echo "================================================================"
echo "  Comprehensive Pipeline Complete — $(date)"
echo "  Results in: $PROJECT_DIR/results/"
echo "================================================================"
