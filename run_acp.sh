#!/bin/bash
# =============================================================================
# SenseCore ACP launch script for nips-text2subspace (GrassMerge)
#
# Expects Docker image with: PyTorch 2.10, TRL, PEFT, DeepSpeed, Accelerate
#
# Usage:
#   bash run_acp.sh                   # full pipeline from Phase 1
#   bash run_acp.sh --from-phase 3    # resume from Phase 3
#   bash run_acp.sh --only-phase 1    # run single phase
#   FORCE_RERUN=1 bash run_acp.sh     # ignore phase markers
# =============================================================================
set -euo pipefail

# === Paths ===
PROJECT_DIR="/data/szs/250010072/nwh/nips-text2subspace"
DATA_DIR="/data/szs/share/text2subspace"
MODEL_9B="/data/szs/share/Qwen3.5-9B"

# === Environment ===
export HF_HOME="${DATA_DIR}/hf_cache"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8

mkdir -p "$HF_HOME"

# === GPU detection ===
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi not found"
    exit 1
fi
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No GPUs detected"
    exit 1
fi

GPU_IDS=""
for ((i=0; i<NUM_GPUS; i++)); do
    [ -n "$GPU_IDS" ] && GPU_IDS="${GPU_IDS},"
    GPU_IDS="${GPU_IDS}${i}"
done
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')

python3 -c "
import torch
n = torch.cuda.device_count()
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.version.cuda}')
print(f'GPUs:    {n}')
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    mem_gib = props.total_memory / (1024**3)
    print(f'  GPU {i}: {props.name}  {mem_gib:.1f} GiB')
"

echo "============================================================"
echo " SenseCore ACP — GrassMerge (nips-text2subspace)"
echo " GPUs: ${NUM_GPUS} × ${GPU_MEM_MIB} MiB"
echo " PROJECT_DIR: ${PROJECT_DIR}"
echo " DATA_DIR:    ${DATA_DIR}"
echo " MODEL_9B:    ${MODEL_9B}"
echo " HF_HOME:     ${HF_HOME}"
echo "============================================================"

# === Validate local model ===
if [ ! -d "$MODEL_9B" ]; then
    echo "[ERROR] Model dir not found: $MODEL_9B"
    exit 1
fi
echo "[OK] Model found: $MODEL_9B"

# === Project setup ===
cd "$PROJECT_DIR"

# Symlink results/ and logs/ to shared storage for persistence
SHARED_RESULTS="${DATA_DIR}/results"
SHARED_LOGS="${DATA_DIR}/logs"
mkdir -p "$SHARED_RESULTS" "$SHARED_LOGS"

if [ ! -L "$PROJECT_DIR/results" ]; then
    if [ -d "$PROJECT_DIR/results" ]; then
        cp -rn "$PROJECT_DIR/results/"* "$SHARED_RESULTS/" 2>/dev/null || true
        rm -rf "$PROJECT_DIR/results"
    fi
    ln -sf "$SHARED_RESULTS" "$PROJECT_DIR/results"
    echo "[SYMLINK] results/ -> $SHARED_RESULTS"
fi

if [ ! -L "$PROJECT_DIR/logs" ]; then
    if [ -d "$PROJECT_DIR/logs" ]; then
        cp -rn "$PROJECT_DIR/logs/"* "$SHARED_LOGS/" 2>/dev/null || true
        rm -rf "$PROJECT_DIR/logs"
    fi
    ln -sf "$SHARED_LOGS" "$PROJECT_DIR/logs"
    echo "[SYMLINK] logs/ -> $SHARED_LOGS"
fi

mkdir -p "$PROJECT_DIR/results" "$PROJECT_DIR/logs"

# === Install missing deps (Docker should have most) ===
pip install --quiet datasets scipy matplotlib pandas huggingface_hub tqdm pyyaml scikit-learn evaluate 2>/dev/null || true

# === Override base_model in config to use local path ===
CONFIG="${PROJECT_DIR}/configs/domains.yaml"
if ! grep -q "$MODEL_9B" "$CONFIG" 2>/dev/null; then
    echo "[INFO] Patching domains.yaml base_model to local path: $MODEL_9B"
    sed -i "s|^base_model:.*|base_model: \"${MODEL_9B}\"|" "$CONFIG"
fi

# === Phase markers for resume ===
RESULTS_DIR="results"
PHASE_MARKERS="${RESULTS_DIR}/.phase_markers"
mkdir -p "$PHASE_MARKERS"

phase_done() {
    local phase=$1
    [ -f "$PHASE_MARKERS/phase${phase}.done" ] && [ "${FORCE_RERUN:-0}" != "1" ]
}

mark_phase_done() {
    local phase=$1
    echo "{\"phase\":$phase,\"completed\":\"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\",\"hostname\":\"$(hostname)\"}" \
        > "$PHASE_MARKERS/phase${phase}.done"
}

get_torchrun_cmd() {
    local nproc="${1:-$NUM_GPUS}"
    echo "torchrun --nproc_per_node=$nproc --master_port=$(( RANDOM % 10000 + 20000 ))"
}

# === Parse arguments ===
FROM_PHASE=1
ONLY_PHASE=-1

while [[ $# -gt 0 ]]; do
    case $1 in
        --from-phase) FROM_PHASE="$2"; shift 2 ;;
        --only-phase) ONLY_PHASE="$2"; shift 2 ;;
        *) echo "WARNING: Unknown arg: $1 (ignored)"; shift ;;
    esac
done

should_run() {
    local phase=$1
    if phase_done "$phase"; then
        echo "[SKIP] Phase $phase already completed (marker: $PHASE_MARKERS/phase${phase}.done)"
        return 1
    fi
    if [[ $ONLY_PHASE -ge 0 ]]; then [[ $phase -eq $ONLY_PHASE ]]; else [[ $phase -ge $FROM_PHASE ]]; fi
}

log_phase() { echo ""; echo "=== Phase $1: $2 === [$(date '+%Y-%m-%d %H:%M:%S')]"; echo ""; }

LORA_DIR="${RESULTS_DIR}/domain_loras"
ALGEBRA_DIR="${RESULTS_DIR}/algebra"
EVAL_DIR="${RESULTS_DIR}/eval"
ABLATION_DIR="${RESULTS_DIR}/ablations"
ANALYSIS_DIR="${RESULTS_DIR}/analysis"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

CORE_DOMAINS="math code medical science history philosophy"

echo ""
echo "============================================================"
echo " Pipeline starting: $(date)"
echo " FROM_PHASE=$FROM_PHASE  ONLY_PHASE=$ONLY_PHASE"
echo " FORCE_RERUN=${FORCE_RERUN:-0}"
echo " Domains: $CORE_DOMAINS"
echo "============================================================"

# Phase 1: Train domain LoRAs (torchrun multi-GPU, auto-resume from checkpoints)
if should_run 1; then
    log_phase 1 "Train Core Domain LoRAs (${NUM_GPUS} GPUs)"
    python scripts/train_domain_loras.py \
        --config "$CONFIG" \
        --output_root "$LORA_DIR" \
        --domains $CORE_DOMAINS \
        --num_gpus ${NUM_GPUS} \
        2>&1 | tee "logs/phase1_train_${TIMESTAMP}.log"
    mark_phase_done 1
fi

# Phase 2: GrassMerge + baselines + BGD analysis
if should_run 2; then
    log_phase 2 "Algebra Experiments (GrassMerge + baselines + BGD)"
    python scripts/run_algebra_experiments.py \
        --config "$CONFIG" --lora_dir "$LORA_DIR" --output_dir "$ALGEBRA_DIR" \
        2>&1 | tee "logs/phase2_algebra_${TIMESTAMP}.log"
    mark_phase_done 2
fi

# Phase 3: Domain evaluation
if should_run 3; then
    log_phase 3 "Domain Evaluation"
    python scripts/eval_domain_accuracy.py \
        --config "$CONFIG" --lora_dir "$LORA_DIR" --algebra_dir "$ALGEBRA_DIR" \
        --output_dir "$EVAL_DIR" --domains $CORE_DOMAINS \
        2>&1 | tee "logs/phase3_eval_${TIMESTAMP}.log"
    mark_phase_done 3
fi

# Phase 4: Ablation studies (rank, weight, K)
if should_run 4; then
    log_phase 4 "Ablation Study"
    python scripts/run_ablations.py \
        --config "$CONFIG" --lora_dir "$LORA_DIR" --output_dir "$ABLATION_DIR" \
        2>&1 | tee "logs/phase4_ablations_${TIMESTAMP}.log"
    mark_phase_done 4
fi

# Phase 5: BGD correlation analysis
if should_run 5; then
    log_phase 5 "BGD Correlation Analysis"
    python scripts/analyze_bgd_correlation.py \
        --interference_file "$ALGEBRA_DIR/interference_metrics.json" \
        --eval_file "$EVAL_DIR/eval_results.json" \
        --output_dir "$ANALYSIS_DIR" \
        2>&1 | tee "logs/phase5_analysis_${TIMESTAMP}.log"
    mark_phase_done 5
fi

# === Done ===
echo ""
echo "============================================================"
echo " GrassMerge — Pipeline Complete [$(date)]"
echo " Results: ${RESULTS_DIR}/ -> ${SHARED_RESULTS}"
echo " Logs:    logs/ -> ${SHARED_LOGS}"
echo "============================================================"

cat > "${RESULTS_DIR}/.pipeline_done" << DONEEOF
{
  "project": "grassmerge",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS}",
  "model_9b": "${MODEL_9B}",
  "core_domains": "$CORE_DOMAINS",
  "stages_completed": 5,
  "results_dirs": {
    "loras": "$LORA_DIR",
    "algebra": "$ALGEBRA_DIR",
    "eval": "$EVAL_DIR",
    "ablations": "$ABLATION_DIR",
    "analysis": "$ANALYSIS_DIR"
  },
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo "[PIPELINE_COMPLETE] Results in: ${SHARED_RESULTS}/"
