#!/bin/bash
# Watch for training completion and auto-launch downstream experiments
set -e
cd "$(dirname "$0")/.."

LORA_DIR=results/domain_loras
REQUIRED_DOMAINS="math code medical science history philosophy"
CHECK_INTERVAL=60

echo "Watching for LoRA training completion..."

while true; do
    READY=0
    for d in $REQUIRED_DOMAINS; do
        if [ -f "$LORA_DIR/$d/adapter_model.safetensors" ]; then
            READY=$((READY + 1))
        fi
    done

    echo "$(date): $READY/6 domains trained"

    if [ "$READY" -ge 6 ]; then
        echo "All domains trained! Launching experiments..."
        break
    fi

    # If at least 2 domains are ready and training seems stuck, proceed anyway
    if [ "$READY" -ge 2 ]; then
        # Check if training process is still running
        if ! pgrep -f "train_domain_lora" > /dev/null 2>&1; then
            echo "Training process not found but $READY domains ready. Proceeding..."
            break
        fi
    fi

    sleep $CHECK_INTERVAL
done

echo ""
echo "=========================================="
echo " Starting downstream experiments"
echo "=========================================="
echo ""

# Stage 2: Merging (fast, CPU/GPU)
echo "--- Stage 2: Merging Experiments ---"
python3 scripts/run_algebra_experiments.py \
    --config configs/domains.yaml \
    --lora_dir "$LORA_DIR" \
    --output_dir results/algebra \
    --seed 42 2>&1 | tee logs/merging.log

# Stage 3: Evaluation (first pass: skip baselines, 100 samples)
echo "--- Stage 3: Evaluation (individual + GrassMerge) ---"
python3 scripts/eval_domain_accuracy.py \
    --config configs/domains.yaml \
    --lora_dir "$LORA_DIR" \
    --algebra_dir results/algebra \
    --output_dir results/eval \
    --domains math code medical science history philosophy \
    --max_samples 100 \
    --skip_baselines \
    --seed 42 2>&1 | tee logs/eval.log

# Stage 4: Ablations
echo "--- Stage 4: Ablation Studies ---"
python3 scripts/run_ablations.py \
    --config configs/domains.yaml \
    --lora_dir "$LORA_DIR" \
    --output_dir results/ablations \
    --seed 42 2>&1 | tee logs/ablations.log

# Stage 5: Analysis
echo "--- Stage 5: Analysis ---"
python3 scripts/analyze_bgd_correlation.py \
    --interference_file results/algebra/interference_metrics.json \
    --eval_file results/eval/eval_results.json \
    --output_dir results/analysis 2>&1 | tee logs/analysis.log

# Generate paper results
echo "--- Generating paper results ---"
python3 scripts/generate_paper_results.py 2>&1

echo ""
echo "=========================================="
echo " ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo "$(date)"
echo "Results in results/paper_results/"
