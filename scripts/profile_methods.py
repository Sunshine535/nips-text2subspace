#!/usr/bin/env python3
"""Runtime and memory profiling for all merge methods.

Profiles each method (GrassMerge, Task Arithmetic, TIES, DARE, KnOTS, TSPA,
parameter averaging) across different adapter ranks (4, 8, 16, 32, 64).

Outputs: profile_results.json with timing and peak GPU memory per method x rank.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lora_algebra import (
    ColumnOnlyGrassmannMerge,
    GrassMerge,
    KnOTSMerge,
    LoRAWeights,
    MergingBaselines,
    SVDProcrustesMerge,
    TSPAMerge,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_all_loras(lora_dir: str, domains: list) -> dict:
    loras = {}
    for domain in domains:
        path = os.path.join(lora_dir, domain)
        if not os.path.exists(path):
            continue
        loras[domain] = LoRAWeights.from_peft_dir(domain, path)
    return loras


def refactorize_at_rank(lora: LoRAWeights, target_rank: int) -> LoRAWeights:
    """Re-factorize a LoRA at a different rank using fast SVD."""
    svd_data = lora.fast_svd()
    new_A, new_B = {}, {}
    for key, (U, S, Vh) in svd_data.items():
        r = min(target_rank, len(S))
        sqrt_S = torch.sqrt(S[:r].clamp(min=0))
        new_B[key] = U[:, :r] @ torch.diag(sqrt_S)
        new_A[key] = torch.diag(sqrt_S) @ Vh[:r, :]
    return LoRAWeights(
        name=f"{lora.name}_r{target_rank}",
        lora_A=new_A, lora_B=new_B, rank=target_rank, alpha=1.0,
    )


def count_parameters(lora: LoRAWeights) -> int:
    """Count total parameters in a LoRAWeights object."""
    total = 0
    for key in lora.lora_A:
        total += lora.lora_A[key].numel()
    for key in lora.lora_B:
        total += lora.lora_B[key].numel()
    return total


def count_delta_parameters(delta: dict) -> int:
    """Count total parameters in a delta-weight dict."""
    return sum(v.numel() for v in delta.values())


def reset_memory_stats():
    """Reset GPU memory tracking if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_peak_memory_mb() -> float:
    """Get peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def profile_merge_method(method_fn, lora_list, num_runs: int = 3) -> dict:
    """Profile a merge method: timing (avg over num_runs) and peak GPU memory.

    method_fn: callable that takes lora_list and returns either LoRAWeights or dict.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Warmup run
    try:
        _ = method_fn(lora_list)
    except Exception as e:
        return {"error": str(e), "time_seconds": -1, "peak_memory_mb": -1, "num_params": -1}

    # Timed runs
    times = []
    peak_mem = 0.0
    result = None
    for _ in range(num_runs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reset_memory_stats()

        t0 = time.perf_counter()
        result = method_fn(lora_list)
        elapsed = time.perf_counter() - t0

        mem = get_peak_memory_mb()
        peak_mem = max(peak_mem, mem)
        times.append(elapsed)

    # Count output parameters
    if isinstance(result, LoRAWeights):
        num_params = count_parameters(result)
    elif isinstance(result, dict):
        num_params = count_delta_parameters(result)
    else:
        num_params = -1

    return {
        "time_seconds": round(float(np.mean(times)), 6),
        "time_std": round(float(np.std(times)), 6),
        "peak_memory_mb": round(peak_mem, 2),
        "num_params": num_params,
        "num_runs": num_runs,
    }


def define_methods() -> dict:
    """Return a dict of method_name -> merge callable."""
    grassmerge = GrassMerge()
    col_grassmann = ColumnOnlyGrassmannMerge()

    def grassmerge_fn(loras):
        return grassmerge.merge(loras, name="grassmerge")

    def task_arithmetic_fn(loras):
        return MergingBaselines.task_arithmetic(loras, scaling=1.0)

    def ties_fn(loras):
        return MergingBaselines.ties_merging(loras, density=0.5, scaling=1.0)

    def dare_fn(loras):
        return MergingBaselines.dare_merging(loras, drop_rate=0.5, scaling=1.0)

    def knots_fn(loras):
        return KnOTSMerge.merge(loras, name="knots")

    def tspa_fn(loras):
        return TSPAMerge.merge(loras, name="tspa")

    def param_avg_fn(loras):
        """Parameter averaging: average lora_A and lora_B factors directly."""
        common_keys = set(loras[0].lora_A.keys())
        for lora in loras[1:]:
            common_keys &= set(lora.lora_A.keys())
        common_keys &= set(loras[0].lora_B.keys())
        for lora in loras[1:]:
            common_keys &= set(lora.lora_B.keys())
        avg_A, avg_B = {}, {}
        n = len(loras)
        for key in sorted(common_keys):
            avg_A[key] = sum(lora.lora_A[key].float() for lora in loras) / n
            avg_B[key] = sum(lora.lora_B[key].float() for lora in loras) / n
        rank = max(lora.rank for lora in loras)
        return LoRAWeights(name="param_avg", lora_A=avg_A, lora_B=avg_B, rank=rank, alpha=1.0)

    return {
        "GrassMerge": grassmerge_fn,
        "TaskArithmetic": task_arithmetic_fn,
        "TIES": ties_fn,
        "DARE": dare_fn,
        "KnOTS": knots_fn,
        "TSPA": tspa_fn,
        "ParameterAveraging": param_avg_fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Runtime and memory profiling for merge methods")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "domains.yaml"))
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory with trained domain LoRAs")
    parser.add_argument("--output_dir", type=str, default="results/profiling")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    domains = list(config["domains"].keys())
    logger.info("Loading LoRAs from %s", args.lora_dir)
    loras = load_all_loras(args.lora_dir, domains)

    if len(loras) < 2:
        logger.error("Need at least 2 trained LoRAs, found %d", len(loras))
        sys.exit(1)
    logger.info("Loaded %d LoRAs: %s", len(loras), sorted(loras.keys()))

    # Use a representative pair for profiling
    pair_names = sorted(loras.keys())[:2]
    base_loras = [loras[pair_names[0]], loras[pair_names[1]]]

    methods = define_methods()
    ranks = [4, 8, 16, 32, 64]

    all_results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair_names,
            "ranks": ranks,
            "methods": list(methods.keys()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    }

    for rank in ranks:
        logger.info("=== Profiling at rank=%d ===", rank)
        # Refactorize both LoRAs to the target rank
        lora1_r = refactorize_at_rank(base_loras[0], rank)
        lora2_r = refactorize_at_rank(base_loras[1], rank)
        lora_list = [lora1_r, lora2_r]

        rank_results = {}
        for method_name, method_fn in methods.items():
            logger.info("  Method: %s, rank=%d", method_name, rank)
            profile = profile_merge_method(method_fn, lora_list, num_runs=3)
            rank_results[method_name] = profile
            logger.info("    time=%.4fs (std=%.4f), mem=%.1fMB, params=%s",
                        profile.get("time_seconds", -1),
                        profile.get("time_std", -1),
                        profile.get("peak_memory_mb", -1),
                        profile.get("num_params", -1))

        all_results[f"rank_{rank}"] = rank_results

    results_path = os.path.join(args.output_dir, "profile_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    logger.info("=" * 80)
    logger.info("  Profiling Summary")
    logger.info("=" * 80)
    header = f"{'Method':<22}" + "".join(f"{'r=' + str(r):>12}" for r in ranks)
    logger.info("  Time (seconds):")
    logger.info("  %s", header)
    for method_name in methods:
        row = f"  {method_name:<22}"
        for rank in ranks:
            key = f"rank_{rank}"
            t = all_results.get(key, {}).get(method_name, {}).get("time_seconds", -1)
            row += f"{t:>12.4f}"
        logger.info("%s", row)

    logger.info("")
    logger.info("  Peak GPU Memory (MB):")
    logger.info("  %s", header)
    for method_name in methods:
        row = f"  {method_name:<22}"
        for rank in ranks:
            key = f"rank_{rank}"
            m = all_results.get(key, {}).get(method_name, {}).get("peak_memory_mb", -1)
            row += f"{m:>12.1f}"
        logger.info("%s", row)

    logger.info("")
    logger.info("  Results saved to: %s", results_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
