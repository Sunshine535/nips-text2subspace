#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="nips-text2subspace"

echo "============================================"
echo " Environment Setup (venv + pip + PyTorch 2.10 + CUDA 12.8)"
echo "============================================"

PYTHON_CMD=""
for try in python3.10 python3.11 python3.12 python3; do
    if command -v "$try" &>/dev/null; then
        PYTHON_CMD="$try"
        break
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Need python3.10+."
    exit 1
fi
echo "[1/5] Using: $($PYTHON_CMD --version)"

VENV_DIR="$PROJ_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/5] Creating venv ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    echo "[2/5] Venv exists: $VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-600}"

echo "[3/5] Upgrading pip ..."
python -m pip install -U pip setuptools wheel

REQ_TMP="$(mktemp)"
trap 'rm -f "$REQ_TMP"' EXIT
grep -v '^flash-attn' "$PROJ_DIR/requirements.txt" > "$REQ_TMP"

echo "[4/5] Installing PyTorch 2.10.0 + CUDA 12.8 + project deps ..."
python -m pip install \
    "torch==2.10.0" "torchvision" "torchaudio" \
    -r "$REQ_TMP" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple

echo "[5/5] Installing flash-attn (optional) ..."
if [ -z "$CUDA_HOME" ]; then
    for p in /usr/local/cuda-12.8 /usr/local/cuda-12 /usr/local/cuda; do
        if [ -f "$p/bin/nvcc" ]; then export CUDA_HOME="$p"; break; fi
    done
fi
_FA_MARKER="$VENV_DIR/.flash_attn_attempted"
if [ ! -f "$_FA_MARKER" ] && [ -n "$CUDA_HOME" ]; then
    echo "  CUDA_HOME=$CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    python -m pip install flash-attn --no-build-isolation 2>&1 || echo "  flash-attn build failed (optional)"
    touch "$_FA_MARKER"
elif [ -f "$_FA_MARKER" ]; then
    echo "  Flash-attn already attempted (skip rebuild)"
else
    echo "  CUDA toolkit not found, skipping flash-attn"
fi

echo ""
echo "============================================"
python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
print(f'  GPUs     : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo "============================================"
echo ""
echo "Setup complete!"
echo "  Activate:  source $VENV_DIR/bin/activate"
echo "  Run:       bash scripts/run_all_experiments.sh"
