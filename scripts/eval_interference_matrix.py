#!/usr/bin/env python3
"""Cross-domain interference evaluation for merged adapters.

For each merged adapter (e.g., math+code), evaluates on ALL 6 core domains,
not just the 2 constituent domains.  This answers the reviewer question:
"does merging domain A+B harm performance on unrelated domain C?"

Produces an interference matrix saved as interference_matrix.json.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

# Import evaluation utilities from the existing eval_domain_accuracy module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_domain_accuracy import (
    DOMAIN_BENCHMARKS,
    apply_delta_weights,
    evaluate_on_benchmark,
    load_model_with_adapter,
)

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORE_DOMAINS = ["math", "code", "medical", "science", "history", "philosophy"]


def discover_merged_pairs(directory: str) -> list[tuple[str, str]]:
    """Find all merged adapter pairs in a directory.

    Returns list of (pair_name, path) where path is either the PEFT dir
    or the .pt delta file.
    """
    if not os.path.isdir(directory):
        return []
    pairs = []
    pt_files = sorted(f for f in os.listdir(directory) if f.endswith(".pt"))
    for fname in pt_files:
        name = fname.replace(".pt", "")
        peft_dir = os.path.join(directory, name)
        if os.path.isdir(peft_dir) and os.path.exists(
            os.path.join(peft_dir, "adapter_config.json")
        ):
            pairs.append((name, peft_dir))
        else:
            pairs.append((name, os.path.join(directory, fname)))
    return pairs


def evaluate_adapter_on_all_domains(
    base_model_name: str,
    adapter_path: str,
    tokenizer,
    eval_domains: list[str],
    max_samples: int | None,
) -> dict:
    """Load one adapter and evaluate it on every domain.

    Returns {domain: {benchmark_name: {accuracy, ...}}}.
    """
    # Load model with adapter (PEFT dir) or delta weights (.pt file)
    if adapter_path.endswith(".pt"):
        logger.info("    Loading delta weights: %s", adapter_path)
        model = apply_delta_weights(base_model_name, adapter_path)
    else:
        logger.info("    Loading PEFT adapter: %s", adapter_path)
        model = load_model_with_adapter(base_model_name, adapter_path)

    domain_results = {}
    for domain in eval_domains:
        if domain not in DOMAIN_BENCHMARKS:
            logger.warning("    Domain '%s' not in DOMAIN_BENCHMARKS, skipping", domain)
            continue
        domain_results[domain] = {}
        for bench_name, bench_cfg in DOMAIN_BENCHMARKS[domain].items():
            if bench_cfg.get("synthetic"):
                logger.info("    Skipping synthetic benchmark %s/%s", domain, bench_name)
                continue
            cfg = {**bench_cfg}
            if max_samples is not None:
                cfg["max_samples"] = max_samples
            logger.info("    Evaluating on %s/%s", domain, bench_name)
            domain_results[domain][bench_name] = evaluate_on_benchmark(
                model, tokenizer, cfg, domain
            )

    del model
    torch.cuda.empty_cache()
    return domain_results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-domain interference evaluation for merged adapters"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "domains.yaml"),
        help="Path to domains.yaml config",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="Directory with trained domain LoRAs (unused here, kept for CLI compat)",
    )
    parser.add_argument(
        "--algebra_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "results" / "algebra"),
        help="Directory with algebra experiment outputs (contains grassmerge/ and baselines/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/eval",
        help="Output directory for interference_matrix.json",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Domains to evaluate on (default: 6 core domains)",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Cap samples per benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    base_model_name = config["base_model"]
    eval_domains = args.domains or CORE_DOMAINS
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_model": base_model_name,
            "eval_domains": eval_domains,
            "description": (
                "Cross-domain interference matrix. Each merged adapter is evaluated "
                "on ALL eval domains, not just its constituent domains."
            ),
        },
        "grassmerge": {},
        "task_arithmetic": {},
    }

    # --- GrassMerge merged pairs ---
    grassmerge_dir = os.path.join(args.algebra_dir, "grassmerge")
    pairs = discover_merged_pairs(grassmerge_dir)
    if not pairs:
        logger.warning("No GrassMerge pairs found in %s", grassmerge_dir)
    else:
        logger.info("=" * 60)
        logger.info("  GRASSMERGE: evaluating %d pairs on %d domains", len(pairs), len(eval_domains))
        logger.info("=" * 60)
        for pair_name, adapter_path in pairs:
            logger.info("  Pair: %s", pair_name)
            t0 = time.time()
            results["grassmerge"][pair_name] = evaluate_adapter_on_all_domains(
                base_model_name, adapter_path, tokenizer, eval_domains, args.max_samples,
            )
            elapsed = time.time() - t0
            logger.info("  Pair %s done in %.1fs", pair_name, elapsed)

    # --- Task Arithmetic baseline (best-performing baseline for comparison) ---
    ta_dir = os.path.join(args.algebra_dir, "baselines", "task_arithmetic")
    ta_pairs = discover_merged_pairs(ta_dir)
    if not ta_pairs:
        # Also check top-level task_arithmetic dir
        ta_dir = os.path.join(args.algebra_dir, "task_arithmetic")
        ta_pairs = discover_merged_pairs(ta_dir)

    if not ta_pairs:
        logger.warning("No task_arithmetic pairs found")
    else:
        logger.info("=" * 60)
        logger.info("  TASK ARITHMETIC: evaluating %d pairs on %d domains", len(ta_pairs), len(eval_domains))
        logger.info("=" * 60)
        for pair_name, adapter_path in ta_pairs:
            logger.info("  Pair: %s", pair_name)
            t0 = time.time()
            results["task_arithmetic"][pair_name] = evaluate_adapter_on_all_domains(
                base_model_name, adapter_path, tokenizer, eval_domains, args.max_samples,
            )
            elapsed = time.time() - t0
            logger.info("  Pair %s done in %.1fs", pair_name, elapsed)

    # --- Save results ---
    output_path = os.path.join(args.output_dir, "interference_matrix.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Interference matrix saved to %s", output_path)

    # --- Print summary ---
    logger.info("=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)
    for method in ["grassmerge", "task_arithmetic"]:
        method_data = results.get(method, {})
        if not method_data:
            continue
        logger.info("  %s:", method)
        for pair_name, domain_data in method_data.items():
            constituent = set(pair_name.split("+"))
            for domain, benchmarks in domain_data.items():
                tag = "IN-DOMAIN" if domain in constituent else "CROSS"
                for bench_name, metrics in benchmarks.items():
                    acc = metrics.get("accuracy", -1)
                    logger.info(
                        "    %s | %-12s | %-20s | acc=%.4f  [%s]",
                        pair_name, domain, bench_name, acc, tag,
                    )

    logger.info("Done. Output: %s", output_path)


if __name__ == "__main__":
    main()
