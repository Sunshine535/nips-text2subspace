#!/usr/bin/env python3
"""Analyze correlation between interference metrics (BGD, spectral-BGD, cosine, Frobenius)
and actual performance degradation from merging.

Reads: results/algebra/interference_metrics.json + results/eval/eval_results.json
Outputs: results/analysis/bgd_correlation.json with Spearman/Pearson correlations
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_degradation(eval_results: dict) -> dict:
    """Compute per-pair performance degradation from merging.
    
    degradation(d1+d2) = avg(individual_d1_acc, individual_d2_acc) - avg(merged_on_d1_acc, merged_on_d2_acc)
    Higher degradation = more interference.
    """
    individual = eval_results.get("individual_loras", {})
    grassmerge = eval_results.get("grassmerge", {})

    pair_degradation = {}
    for key, merged_metrics in grassmerge.items():
        parts = key.split("_on_")
        if len(parts) < 2:
            continue
        pair_name = parts[0]
        domains = pair_name.split("+")
        if len(domains) != 2:
            continue

        d1, d2 = domains
        merged_acc = merged_metrics.get("accuracy", -1)
        if merged_acc < 0:
            continue

        d1_accs = []
        for bench_name, bench_data in individual.get(d1, {}).items():
            if isinstance(bench_data, dict) and "accuracy" in bench_data and bench_data["accuracy"] >= 0:
                d1_accs.append(bench_data["accuracy"])

        d2_accs = []
        for bench_name, bench_data in individual.get(d2, {}).items():
            if isinstance(bench_data, dict) and "accuracy" in bench_data and bench_data["accuracy"] >= 0:
                d2_accs.append(bench_data["accuracy"])

        if not d1_accs or not d2_accs:
            continue

        target_domain = parts[1].split("_")[0]
        if target_domain == d1:
            individual_acc = np.mean(d1_accs)
        elif target_domain == d2:
            individual_acc = np.mean(d2_accs)
        else:
            continue

        degradation = individual_acc - merged_acc
        if pair_name not in pair_degradation:
            pair_degradation[pair_name] = []
        pair_degradation[pair_name].append(degradation)

    return {k: float(np.mean(v)) for k, v in pair_degradation.items()}


def spearman_rho(x, y):
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def pearson_r(x, y):
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)
    return float(r), float(p)


def main():
    parser = argparse.ArgumentParser(description="BGD vs performance degradation correlation")
    parser.add_argument("--interference_file", type=str,
                        default="results/algebra/interference_metrics.json")
    parser.add_argument("--eval_file", type=str, default="results/eval/eval_results.json")
    parser.add_argument("--output_dir", type=str, default="results/analysis")
    args = parser.parse_args()

    if not os.path.exists(args.interference_file):
        logger.error("Interference metrics not found: %s", args.interference_file)
        sys.exit(1)
    if not os.path.exists(args.eval_file):
        logger.error("Eval results not found: %s", args.eval_file)
        sys.exit(1)

    with open(args.interference_file) as f:
        interference = json.load(f)
    with open(args.eval_file) as f:
        eval_results = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    names = interference["domain_names"]
    N = len(names)

    pair_degradation = compute_degradation(eval_results)
    if not pair_degradation:
        logger.error("No degradation data computed. Ensure eval results contain both individual and grassmerge data.")
        sys.exit(1)

    metrics = {
        "bgd": np.array(interference["bgd_matrix"]),
        "spectral_bgd": np.array(interference["spectral_bgd_matrix"]),
        "cosine": np.array(interference["cosine_interference_matrix"]),
        "frobenius": np.array(interference["frobenius_interference_matrix"]),
    }

    results = {}
    for metric_name, matrix in metrics.items():
        metric_vals = []
        degrad_vals = []
        for i in range(N):
            for j in range(i + 1, N):
                pair_name = f"{names[i]}+{names[j]}"
                if pair_name in pair_degradation:
                    metric_vals.append(matrix[i][j])
                    degrad_vals.append(pair_degradation[pair_name])
                pair_name_rev = f"{names[j]}+{names[i]}"
                if pair_name_rev in pair_degradation:
                    metric_vals.append(matrix[i][j])
                    degrad_vals.append(pair_degradation[pair_name_rev])

        if len(metric_vals) < 3:
            logger.warning("Too few data points for %s (%d)", metric_name, len(metric_vals))
            results[metric_name] = {"n": len(metric_vals), "error": "too few data points"}
            continue

        rho, p_rho = spearman_rho(metric_vals, degrad_vals)
        r, p_r = pearson_r(metric_vals, degrad_vals)
        results[metric_name] = {
            "n": len(metric_vals),
            "spearman_rho": round(rho, 4),
            "spearman_p": round(p_rho, 6),
            "pearson_r": round(r, 4),
            "pearson_p": round(p_r, 6),
        }
        logger.info("  %s: Spearman=%.4f (p=%.4f), Pearson=%.4f (p=%.4f), n=%d",
                     metric_name, rho, p_rho, r, p_r, len(metric_vals))

    output_path = os.path.join(args.output_dir, "bgd_correlation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
