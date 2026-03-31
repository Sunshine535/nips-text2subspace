#!/usr/bin/env bash
# DEPRECATED: Use scripts/run_production.sh as the canonical entry point.
# This script is kept for reference only.
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/gpu_utils.sh"
auto_setup
TORCHRUN=$(get_torchrun_cmd)
export TORCHRUN

PROJ_DIR_ROOT="$PROJECT_DIR"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

# --- Phase resume ---
PHASE_MARKER_DIR="$PROJECT_DIR/results/.phase_markers"
mkdir -p "$PHASE_MARKER_DIR"
FORCE_RERUN="${FORCE_RERUN:-0}"
phase_done() { touch "$PHASE_MARKER_DIR/phase_${1}.done"; echo "[PHASE $1] Completed at $(date)"; }
is_phase_done() {
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${1}.done" ]] && echo "[PHASE $1] Already completed. Skipping." && return 0
    return 1
}

CONFIG="${PROJECT_DIR}/configs/domains.yaml"
LORA_DIR="${PROJECT_DIR}/results/domain_loras"
ALGEBRA_DIR="${PROJECT_DIR}/results/algebra"
EVAL_DIR="${PROJECT_DIR}/results/eval"
ABLATION_DIR="${PROJECT_DIR}/results/ablations"
LOG_DIR="${PROJECT_DIR}/logs"

cd "$PROJECT_DIR"
mkdir -p results "$LOG_DIR"

echo "================================================================"
echo "  LoRA Algebra — Full Experiment Pipeline"
echo "  Base model: Qwen/Qwen3.5-9B | Domains: 12 | GPUs: ${NUM_GPUS}"
echo "================================================================"

if ! is_phase_done 1; then
    echo "========== STAGE 1: Train Domain LoRAs =========="
    python scripts/train_domain_loras.py \
        --config "$CONFIG" --output_root "$LORA_DIR" --num_gpus ${NUM_GPUS} \
        2>&1 | tee "$LOG_DIR/stage1_train.log"
    phase_done 1
fi

if ! is_phase_done 2; then
    echo "========== STAGE 2: Algebra Experiments =========="
    python scripts/run_algebra_experiments.py \
        --config "$CONFIG" --lora_dir "$LORA_DIR" --output_dir "$ALGEBRA_DIR" \
        2>&1 | tee "$LOG_DIR/stage2_algebra.log"
    phase_done 2
fi

if ! is_phase_done 3; then
    echo "========== STAGE 3: Domain Evaluation =========="
    python scripts/eval_domain_accuracy.py \
        --config "$CONFIG" --lora_dir "$LORA_DIR" --algebra_dir "$ALGEBRA_DIR" \
        --output_dir "$EVAL_DIR" \
        2>&1 | tee "$LOG_DIR/stage3_eval.log"
    phase_done 3
fi

if ! is_phase_done 4; then
    echo "========== STAGE 4: Ablation Study =========="
    python scripts/run_ablations.py \
        --config "$CONFIG" --lora_dir "$LORA_DIR" --output_dir "$ABLATION_DIR" \
        2>&1 | tee "$LOG_DIR/stage4_ablations.log"
    phase_done 4
fi

if ! is_phase_done 5; then
    echo "========== STAGE 5: BGD Correlation Analysis =========="
    python scripts/analyze_bgd_correlation.py \
        --interference_file "$ALGEBRA_DIR/interference_metrics.json" \
        --eval_file "$EVAL_DIR/eval_results.json" \
        --output_dir "${PROJECT_DIR}/results/analysis" \
        2>&1 | tee "$LOG_DIR/stage5_analysis.log"
    phase_done 5
fi

echo "================================================================"
echo "  Pipeline Complete — $(date)"
echo "================================================================"

DONE_FILE="$PROJECT_DIR/results/.pipeline_done"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "nips-text2subspace",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo "[PIPELINE_COMPLETE] Run 'bash collect_results.sh' to package results."
