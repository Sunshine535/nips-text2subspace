#!/bin/bash
# Run evaluations sequentially as separate processes to avoid OOM
set -e
cd /root/nips-text2subspace
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUTDIR=/root/nips-text2subspace/results/eval_parts
mkdir -p $OUTDIR

PAIRS=("math medical" "math science" "science philosophy")

for pair in "${PAIRS[@]}"; do
    d1=$(echo $pair | cut -d' ' -f1)
    d2=$(echo $pair | cut -d' ' -f2)
    echo "$(date) === Pair: $d1+$d2 ==="

    for method in sfc ta ties; do
        out="$OUTDIR/${d1}_${d2}_${method}.json"
        if [ -f "$out" ]; then
            echo "$(date) SKIP $method (already done)"
            continue
        fi
        echo "$(date) Running $method for $d1+$d2..."
        python scripts/eval_one_pair.py --method $method --d1 $d1 --d2 $d2 --output "$out" 2>&1 | tail -20
        echo "$(date) $method DONE"
    done
done

# Combine results
echo "$(date) Combining results..."
python3 -c "
import json, glob, os
base = {'math': {'accuracy': 0.02}, 'medical': {'accuracy': 0.66}, 'philosophy': {'accuracy': 0.42}, 'science': {'accuracy': 0.74}}
parts = glob.glob('$OUTDIR/*.json')
pairs = {}
for p in sorted(parts):
    r = json.load(open(p))
    key = r['pair']
    if key not in pairs: pairs[key] = {}
    pairs[key][r['method']] = r
result = {'base_model': base, 'pairs': []}
for pair_name, methods in pairs.items():
    entry = {'pair': pair_name}
    for m, r in methods.items():
        entry[m] = r.get('scores', {})
        if 'fds' in r: entry['fds'] = r['fds']
        if 'sparsity' in r: entry['sparsity'] = r['sparsity']
    result['pairs'].append(entry)
with open('/root/nips-text2subspace/results/sfc_downstream.json', 'w') as f:
    json.dump(result, f, indent=2)
print('Combined results saved!')
"
echo "$(date) ALL DONE"
