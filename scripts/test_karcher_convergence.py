#!/usr/bin/env python3
"""Karcher mean convergence analysis.

Tests convergence speed and stability of the iterative Karcher mean
algorithm on Grassmann manifolds across different numbers of LoRAs.
"""

import json
import logging
import os
import sys
import itertools
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lora_algebra import GrassMerge, GrassmannOps, LoRAWeights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORE_DOMAINS = ["math", "code", "medical", "science", "history", "philosophy"]


def main():
    lora_dir = "results/domain_loras"
    output_path = "results/eval_comprehensive/karcher_convergence.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load LoRAs and compute fast SVD
    loras = {}
    all_svds = {}
    for d in CORE_DOMAINS:
        path = os.path.join(lora_dir, d)
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            logger.info("Loading LoRA: %s", d)
            loras[d] = LoRAWeights.from_peft_dir(d, path)
            all_svds[d] = loras[d].fast_svd()

    domains = sorted(loras.keys())
    results = []

    for n in [2, 3, 4, 6]:
        combos = list(itertools.combinations(domains, n))
        for combo in combos[:5]:
            name = "+".join(combo)
            layer_keys = sorted(all_svds[combo[0]].keys())

            # Test on 3 representative layers: early, middle, late
            test_layers = [layer_keys[0], layer_keys[len(layer_keys)//2], layer_keys[-1]]

            for test_key in test_layers:
                Us = []
                for d in combo:
                    if test_key in all_svds[d]:
                        Us.append(all_svds[d][test_key][0])  # U matrix

                if len(Us) != n:
                    continue

                # Track convergence by running with increasing max_iter
                convergence = []
                prev_dist = None
                for max_iter in [1, 2, 3, 5, 10, 20, 50]:
                    t0 = time.time()
                    mean_U = GrassmannOps.karcher_mean(Us, max_iter=max_iter)
                    elapsed = time.time() - t0

                    # Sum of squared geodesic distances to mean
                    total_dist_sq = 0.0
                    for U in Us:
                        d_val = GrassmannOps.geodesic_distance(mean_U, U)
                        total_dist_sq += d_val ** 2

                    convergence.append({
                        "max_iter": max_iter,
                        "frechet_variance": round(total_dist_sq, 6),
                        "time_s": round(elapsed, 4),
                    })

                # Determine convergence point
                converged_at = 50
                for i in range(1, len(convergence)):
                    if convergence[i-1]["frechet_variance"] > 1e-10:
                        rel_change = abs(convergence[i]["frechet_variance"] - convergence[i-1]["frechet_variance"]) / convergence[i-1]["frechet_variance"]
                        if rel_change < 0.001:
                            converged_at = convergence[i]["max_iter"]
                            break

                results.append({
                    "combo": name,
                    "n": n,
                    "layer": test_key,
                    "convergence": convergence,
                    "converged_at": converged_at,
                    "final_frechet_variance": convergence[-1]["frechet_variance"],
                })

            logger.info("  %d-way %s: done (3 layers)", n, name)

    # Summary by N
    summary = {}
    for n in [2, 3, 4, 6]:
        n_results = [r for r in results if r["n"] == n]
        if n_results:
            avg_converge = np.mean([r["converged_at"] for r in n_results])
            avg_variance = np.mean([r["final_frechet_variance"] for r in n_results])
            summary[f"{n}way"] = {
                "avg_convergence_iter": round(avg_converge, 1),
                "avg_final_frechet_variance": round(avg_variance, 6),
                "num_tests": len(n_results),
            }
            logger.info("  %d-way: avg convergence at iter %.1f, avg variance %.6f",
                        n, avg_converge, avg_variance)

    output = {"summary": summary, "results": results}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
