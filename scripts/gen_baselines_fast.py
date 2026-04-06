#!/usr/bin/env python3
"""Fast baseline generation: compute delta weights ONCE, reuse for all methods."""

import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lora_algebra import (
    GrassMerge,
    KnOTSMerge,
    LoRAWeights,
    MergingBaselines,
    TSPAMerge,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    config_path = "configs/domains.yaml"
    lora_dir = "results/domain_loras"
    output_dir = "results/algebra/baselines"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load all LoRAs
    domains = ["code", "history", "math", "medical", "philosophy", "science"]
    loras = {}
    for d in domains:
        path = os.path.join(lora_dir, d)
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            logger.info("Loading LoRA: %s", d)
            loras[d] = LoRAWeights.from_peft_dir(d, path)

    logger.info("Loaded %d LoRAs: %s", len(loras), sorted(loras.keys()))

    # Pre-compute ALL delta weights ONCE
    logger.info("Pre-computing delta weights for all domains...")
    t0 = time.time()
    delta_cache = {}
    for name, lora in loras.items():
        delta_cache[name] = lora.to_delta_weight()
    logger.info("Delta weights computed in %.1fs", time.time() - t0)

    pairs = list(itertools.combinations(sorted(loras.keys()), 2))
    logger.info("Will process %d pairs", len(pairs))

    # --- Task Arithmetic ---
    _run_simple_baseline("task_arithmetic", pairs, delta_cache, output_dir,
                         lambda deltas: {k: torch.stack([d[k] for d in deltas if k in d]).mean(dim=0)
                                         for k in set().union(*(d.keys() for d in deltas))})

    # --- TIES (density=0.5) ---
    _run_method_baseline("ties_d0.5", pairs, loras, output_dir,
                         lambda pair: MergingBaselines.ties_merging(pair, density=0.5, scaling=1.0))

    # --- DARE (drop_rate=0.5) ---
    _run_method_baseline("dare_p0.5", pairs, loras, output_dir,
                         lambda pair: MergingBaselines.dare_merging(pair, drop_rate=0.5, scaling=1.0))

    # --- KnOTS ---
    _run_lora_baseline("knots", pairs, loras, output_dir,
                       lambda pair: KnOTSMerge.merge(pair))

    # --- TSPA ---
    _run_lora_baseline("tspa", pairs, loras, output_dir,
                       lambda pair: TSPAMerge.merge(pair))

    logger.info("All baselines complete!")


def _run_simple_baseline(name, pairs, delta_cache, output_dir, merge_fn):
    """For baselines that operate on pre-computed delta dicts."""
    method_dir = os.path.join(output_dir, name)
    os.makedirs(method_dir, exist_ok=True)
    logger.info("Baseline: %s (%d pairs)", name, len(pairs))
    for d1, d2 in pairs:
        pair_name = f"{d1}+{d2}"
        t0 = time.time()
        merged = merge_fn([delta_cache[d1], delta_cache[d2]])
        elapsed = time.time() - t0
        torch.save(merged, os.path.join(method_dir, f"{pair_name}.pt"))
        logger.info("  %s: %.1fs", pair_name, elapsed)


def _run_method_baseline(name, pairs, loras, output_dir, merge_fn):
    """For baselines that take LoRAWeights and return delta dicts."""
    method_dir = os.path.join(output_dir, name)
    os.makedirs(method_dir, exist_ok=True)
    logger.info("Baseline: %s (%d pairs)", name, len(pairs))
    for d1, d2 in pairs:
        pair_name = f"{d1}+{d2}"
        t0 = time.time()
        merged = merge_fn([loras[d1], loras[d2]])
        elapsed = time.time() - t0
        torch.save(merged, os.path.join(method_dir, f"{pair_name}.pt"))
        logger.info("  %s: %.1fs", pair_name, elapsed)


def _run_lora_baseline(name, pairs, loras, output_dir, merge_fn):
    """For baselines that return LoRAWeights."""
    method_dir = os.path.join(output_dir, name)
    os.makedirs(method_dir, exist_ok=True)
    logger.info("Baseline: %s (%d pairs)", name, len(pairs))
    for d1, d2 in pairs:
        pair_name = f"{d1}+{d2}"
        t0 = time.time()
        merged = merge_fn([loras[d1], loras[d2]])
        elapsed = time.time() - t0
        delta = merged.to_delta_weight()
        torch.save(delta, os.path.join(method_dir, f"{pair_name}.pt"))
        logger.info("  %s: %.1fs", pair_name, elapsed)


if __name__ == "__main__":
    main()
