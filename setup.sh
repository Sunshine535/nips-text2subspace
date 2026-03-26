#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="nips-text2subspace"

echo "============================================"
echo " Environment Setup (uv + PyTorch 2.10 + CUDA 12.8)"
echo "============================================"

# --- Install uv if missing ---
if ! command -v uv &>/dev/null; then
    echo "[1/5] Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/5] uv already installed: $(uv --version)"
fi

# --- Create venv ---
VENV_DIR="$PROJ_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/5] Creating Python 3.10 venv ..."
    uv venv "$VENV_DIR" --python 3.10 2>/dev/null || uv venv "$VENV_DIR"
else
    echo "[2/5] Venv exists: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- Install PyTorch CUDA + project dependencies (single resolve) ---
echo "[3/5] Installing PyTorch 2.10.0 + CUDA 12.8 + project deps ..."
grep -v '^flash-attn' "$PROJ_DIR/requirements.txt" > /tmp/_req_no_flash.txt
uv pip install "torch==2.10.0" "torchvision" "torchaudio" \
    -r /tmp/_req_no_flash.txt \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
    --index-strategy unsafe-best-match
rm -f /tmp/_req_no_flash.txt

# --- Optional: flash-attention ---
echo "[5/5] Installing flash-attn (optional) ..."
if [ -z "$CUDA_HOME" ]; then
    for p in /usr/local/cuda-12.8 /usr/local/cuda-12 /usr/local/cuda; do
        if [ -f "$p/bin/nvcc" ]; then export CUDA_HOME="$p"; break; fi
    done
fi
if [ -n "$CUDA_HOME" ]; then
    echo "  CUDA_HOME=$CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    uv pip install flash-attn --no-build-isolation 2>&1 || echo "  flash-attn build failed (optional, falling back to sdpa)"
else
    echo "  CUDA toolkit not found, skipping flash-attn (will use sdpa)"
fi

# --- Verify ---
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
