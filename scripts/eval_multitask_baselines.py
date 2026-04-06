#!/usr/bin/env python3
"""Evaluate multitask LoRA baselines on their constituent domains.

For each multitask adapter (e.g., history+philosophy), loads the base model with
the adapter and evaluates on all constituent domain benchmarks. Results are saved
as a structured JSON file.
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_domain_accuracy import (
    DOMAIN_BENCHMARKS,
    evaluate_on_benchmark,
    load_model_with_adapter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def discover_multitask_adapters(multitask_dir: str) -> list[str]:
    """Find all multitask adapter directories (contain adapter_config.json)."""
    adapters = []
    if not os.path.isdir(multitask_dir):
        logger.warning("Multitask directory does not exist: %s", multitask_dir)
        return adapters
    for entry in sorted(os.listdir(multitask_dir)):
        entry_path = os.path.join(multitask_dir, entry)
        if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "adapter_config.json")):
            adapters.append(entry)
    return adapters


def main():
    parser = argparse.ArgumentParser(description="Evaluate multitask LoRA baselines")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "domains.yaml"))
    parser.add_argument("--multitask_dir", type=str, default="results/multitask_loras",
                        help="Directory containing multitask adapter subdirs (e.g., history+philosophy/)")
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    base_model_name = config["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Discover multitask adapters
    adapter_names = discover_multitask_adapters(args.multitask_dir)
    if not adapter_names:
        logger.error("No multitask adapters found in %s", args.multitask_dir)
        sys.exit(1)

    logger.info("Found %d multitask adapter(s): %s", len(adapter_names), adapter_names)

    all_results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_model": base_model_name,
            "multitask_dir": args.multitask_dir,
            "adapters": adapter_names,
        },
        "multitask_baselines": {},
    }

    for adapter_name in adapter_names:
        adapter_path = os.path.join(args.multitask_dir, adapter_name)
        constituent_domains = adapter_name.split("+")

        logger.info("=" * 60)
        logger.info("  Evaluating multitask adapter: %s", adapter_name)
        logger.info("  Constituent domains: %s", constituent_domains)
        logger.info("=" * 60)

        model = load_model_with_adapter(base_model_name, adapter_path)
        pair_results = {}

        for domain in constituent_domains:
            if domain not in DOMAIN_BENCHMARKS:
                logger.warning("No benchmark defined for domain '%s', skipping", domain)
                continue

            domain_results = {}
            for bench_name, bench_cfg in DOMAIN_BENCHMARKS[domain].items():
                if args.max_samples:
                    bench_cfg = {**bench_cfg, "max_samples": args.max_samples}
                logger.info("  %s → %s/%s", adapter_name, domain, bench_name)
                domain_results[bench_name] = evaluate_on_benchmark(model, tokenizer, bench_cfg, domain)
                logger.info("    accuracy: %.4f", domain_results[bench_name].get("accuracy", -1))
            pair_results[domain] = domain_results

        all_results["multitask_baselines"][adapter_name] = pair_results

        del model
        torch.cuda.empty_cache()

    # Save results
    results_path = os.path.join(args.output_dir, "multitask_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info("=" * 60)
    logger.info("  Multitask Baseline Evaluation Summary")
    logger.info("=" * 60)
    for adapter_name, pair_results in all_results["multitask_baselines"].items():
        for domain, domain_results in pair_results.items():
            for bench_name, metrics in domain_results.items():
                acc = metrics.get("accuracy", -1)
                logger.info("  %s → %s/%s: %.4f", adapter_name, domain, bench_name, acc)

    logger.info("Results saved to: %s", results_path)


if __name__ == "__main__":
    main()
