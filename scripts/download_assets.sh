#!/bin/bash
# =============================================================================
# Download all assets needed for SFC experiments
#
# Run on a machine WITH internet (e.g., download container at /openbayes/input/input0),
# then scp the resulting directories to the offline GPU server.
#
# Usage:
#   bash scripts/download_assets.sh
#   bash scripts/download_assets.sh /custom/output/path
#
# What it downloads:
#   - Qwen/Qwen3.5-9B (base model, ~18GB)
#   - Training datasets: gsm8k, mbpp, medqa, arc, mmlu
#   - SAE training corpus: wikitext (small, ~500MB)
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${1:-$(pwd)}"

echo "============================================"
echo "  SFC Asset Downloader"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

# Set HF cache to output directory so everything lands together
export HF_HOME="$OUTPUT_DIR/hf_cache"
mkdir -p "$HF_HOME"

# --- Install dependencies if needed ---
python3 -c "import huggingface_hub" 2>/dev/null || pip install -U huggingface_hub datasets

# --- 1. Download the model ---
MODEL="Qwen/Qwen3.5-9B"
MODEL_DIR="$OUTPUT_DIR/models/Qwen3.5-9B"
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "[SKIP] Model already downloaded: $MODEL_DIR"
else
    echo "[1/3] Downloading model: $MODEL (~18GB)"
    huggingface-cli download "$MODEL" --local-dir "$MODEL_DIR" \
        --exclude "*.msgpack" "*.h5" "*.ot" "*.bin"
fi

# --- 2. Download datasets ---
DATASETS_DIR="$OUTPUT_DIR/datasets"
mkdir -p "$DATASETS_DIR"

download_dataset() {
    local repo="$1"
    local config="$2"
    local local_name="${3:-${repo//\//_}}"
    local target="$DATASETS_DIR/$local_name"

    if [ -d "$target" ] && [ "$(ls -A "$target" 2>/dev/null)" ]; then
        echo "  [SKIP] $repo → $target"
        return
    fi

    echo "  [GET] $repo ($config) → $target"
    python3 -c "
from datasets import load_dataset
ds = load_dataset('$repo', '$config' if '$config' else None)
ds.save_to_disk('$target')
print(f'  saved: {\"$target\"}')
" || echo "  [WARN] Failed to download $repo"
}

echo "[2/3] Downloading datasets"
download_dataset "openai/gsm8k" "main" "gsm8k"
download_dataset "google-research-datasets/mbpp" "full" "mbpp"
download_dataset "bigbio/med_qa" "med_qa_en_4options_source" "med_qa"
download_dataset "allenai/ai2_arc" "ARC-Challenge" "arc_challenge"
download_dataset "allenai/ai2_arc" "ARC-Easy" "arc_easy"
download_dataset "cais/mmlu" "all" "mmlu"
download_dataset "EleutherAI/hendrycks_math" "" "hendrycks_math"

# --- 3. Download SAE training corpus ---
echo "[3/3] Downloading SAE training corpus (wikitext)"
download_dataset "Salesforce/wikitext" "wikitext-103-raw-v1" "wikitext"

# --- Summary ---
echo ""
echo "============================================"
echo "  Download Complete"
echo "============================================"
echo "Model:    $MODEL_DIR ($(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1))"
echo "Datasets: $DATASETS_DIR ($(du -sh "$DATASETS_DIR" 2>/dev/null | cut -f1))"
echo "HF cache: $HF_HOME ($(du -sh "$HF_HOME" 2>/dev/null | cut -f1))"
echo ""
echo "Transfer to GPU server:"
echo "  scp -r $MODEL_DIR user@tju-hpc:/path/to/models/"
echo "  scp -r $DATASETS_DIR user@tju-hpc:/path/to/datasets/"
echo ""
echo "Or use rsync (resumable):"
echo "  rsync -avP $OUTPUT_DIR/ user@tju-hpc:/path/to/sfc_assets/"
