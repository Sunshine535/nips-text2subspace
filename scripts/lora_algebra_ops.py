#!/usr/bin/env python3
"""Implement and test LoRA algebraic operations: compose, subtract, interpolate, project.
Compare with TIES, DARE, Task Arithmetic baselines."""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.lora_algebra import (
    GrassmannProjector,
    LoRAAlgebra,
    LoRAWeights,
    MergingBaselines,
    compute_similarity_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_all_loras(lora_dir: str, domains: list) -> dict:
    """Load all trained domain LoRAs."""
    loras = {}
    for domain in domains:
        path = os.path.join(lora_dir, domain)
        if not os.path.exists(path):
            logger.warning("LoRA for domain '%s' not found at %s, skipping", domain, path)
            continue
        logger.info("Loading LoRA: %s", domain)
        loras[domain] = LoRAWeights.from_peft_dir(domain, path)
    return loras


def run_compose_experiments(algebra: LoRAAlgebra, loras: dict, output_dir: str):
    """Test pairwise composition (A + B) for all domain pairs."""
    results = {}
    domains = sorted(loras.keys())
    compose_dir = os.path.join(output_dir, "composed")
    os.makedirs(compose_dir, exist_ok=True)

    for i, d1 in enumerate(domains):
        for d2 in domains[i + 1 :]:
            name = f"{d1}+{d2}"
            logger.info("Composing: %s", name)
            t0 = time.time()
            composed = algebra.compose(loras[d1], loras[d2], name=name)
            elapsed = time.time() - t0
            sd = composed.to_state_dict()
            torch.save(sd, os.path.join(compose_dir, f"{name}.pt"))
            results[name] = {
                "operation": "compose",
                "domains": [d1, d2],
                "rank": composed.rank,
                "num_params": sum(v.numel() for v in sd.values()),
                "time_seconds": round(elapsed, 2),
            }
    return results


def run_subtract_experiments(algebra: LoRAAlgebra, loras: dict, output_dir: str):
    """Test pairwise subtraction (A - B) for selected domain pairs."""
    results = {}
    subtract_dir = os.path.join(output_dir, "subtracted")
    os.makedirs(subtract_dir, exist_ok=True)

    interesting_pairs = [
        ("math", "code"),
        ("medical", "legal"),
        ("creative", "science"),
        ("finance", "philosophy"),
    ]
    domains = sorted(loras.keys())
    for d1, d2 in interesting_pairs:
        if d1 not in loras or d2 not in loras:
            continue
        name = f"{d1}-{d2}"
        logger.info("Subtracting: %s", name)
        t0 = time.time()
        subtracted = algebra.subtract(loras[d1], loras[d2], name=name)
        elapsed = time.time() - t0
        sd = subtracted.to_state_dict()
        torch.save(sd, os.path.join(subtract_dir, f"{name}.pt"))
        results[name] = {
            "operation": "subtract",
            "domains": [d1, d2],
            "rank": subtracted.rank,
            "time_seconds": round(elapsed, 2),
        }
    return results


def run_interpolation_experiments(algebra: LoRAAlgebra, loras: dict, alphas: list, output_dir: str):
    """Test interpolation at various alpha values."""
    results = {}
    interp_dir = os.path.join(output_dir, "interpolated")
    os.makedirs(interp_dir, exist_ok=True)

    pairs = [("math", "code"), ("medical", "science"), ("creative", "philosophy")]
    for d1, d2 in pairs:
        if d1 not in loras or d2 not in loras:
            continue
        for alpha in alphas:
            name = f"{d1}_{d2}_a{alpha:.1f}"
            logger.info("Interpolating: %s (alpha=%.2f)", name, alpha)
            interp = algebra.interpolate(loras[d1], loras[d2], alpha=alpha, name=name)
            sd = interp.to_state_dict()
            torch.save(sd, os.path.join(interp_dir, f"{name}.pt"))
            results[name] = {
                "operation": "interpolate",
                "domains": [d1, d2],
                "alpha": alpha,
                "rank": interp.rank,
            }
    return results


def run_grassmann_analysis(loras: dict, config: dict, output_dir: str):
    """Compute Grassmann manifold distances and projections."""
    projector = GrassmannProjector(svd_rank=config.get("algebra", {}).get("svd_rank", 32))
    lora_list = [loras[d] for d in sorted(loras.keys())]
    names = sorted(loras.keys())

    logger.info("Computing Grassmann distance matrix...")
    dist_matrix = compute_similarity_matrix(lora_list, projector)

    results = {"domain_names": names, "distance_matrix": dist_matrix.tolist()}
    with open(os.path.join(output_dir, "grassmann_distances.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Grassmann distance matrix:\n%s", np.array2string(dist_matrix, precision=3))
    return results


def run_baseline_comparisons(loras: dict, output_dir: str):
    """Compare with TIES, DARE, Task Arithmetic baselines."""
    results = {}
    baseline_dir = os.path.join(output_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)
    lora_list = list(loras.values())

    for method_name, method_fn, kwargs in [
        ("task_arithmetic", MergingBaselines.task_arithmetic, {"scaling": 0.5}),
        ("ties_0.5", MergingBaselines.ties_merging, {"density": 0.5, "scaling": 0.5}),
        ("ties_0.3", MergingBaselines.ties_merging, {"density": 0.3, "scaling": 0.5}),
        ("dare_0.5", MergingBaselines.dare_merging, {"drop_rate": 0.5, "scaling": 0.5}),
        ("dare_0.7", MergingBaselines.dare_merging, {"drop_rate": 0.7, "scaling": 0.5}),
    ]:
        logger.info("Running baseline: %s", method_name)
        t0 = time.time()
        merged_deltas = method_fn(lora_list, **kwargs)
        elapsed = time.time() - t0
        torch.save(merged_deltas, os.path.join(baseline_dir, f"{method_name}.pt"))
        results[method_name] = {
            "method": method_name,
            "num_layers": len(merged_deltas),
            "time_seconds": round(elapsed, 2),
            "total_params": sum(v.numel() for v in merged_deltas.values()),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="LoRA Algebra Operations")
    parser.add_argument("--config", type=str, default="configs/domains.yaml")
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory with trained domain LoRAs")
    parser.add_argument("--output_dir", type=str, default="outputs/algebra_results")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    domains = list(config["domains"].keys())
    logger.info("Loading %d domain LoRAs from %s", len(domains), args.lora_dir)
    loras = load_all_loras(args.lora_dir, domains)

    if len(loras) < 2:
        logger.error("Need at least 2 trained LoRAs, found %d", len(loras))
        sys.exit(1)

    algebra = LoRAAlgebra(grassmann_rank=config.get("algebra", {}).get("svd_rank", 32))
    all_results = {}

    logger.info("=== Running compose experiments ===")
    all_results["compose"] = run_compose_experiments(algebra, loras, args.output_dir)

    logger.info("=== Running subtract experiments ===")
    all_results["subtract"] = run_subtract_experiments(algebra, loras, args.output_dir)

    logger.info("=== Running interpolation experiments ===")
    alphas = config.get("algebra", {}).get("interpolation_alphas", [0.0, 0.25, 0.5, 0.75, 1.0])
    all_results["interpolation"] = run_interpolation_experiments(algebra, loras, alphas, args.output_dir)

    logger.info("=== Running Grassmann manifold analysis ===")
    all_results["grassmann"] = run_grassmann_analysis(loras, config, args.output_dir)

    logger.info("=== Running baseline comparisons ===")
    all_results["baselines"] = run_baseline_comparisons(loras, args.output_dir)

    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=== All algebra operations complete. Results saved to %s ===", args.output_dir)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
