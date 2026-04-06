#!/usr/bin/env python3
"""Comprehensive evaluation: base model, individual LoRAs, all merge methods.

Loads the base model ONCE, swaps PEFT adapters efficiently.
Evaluates on all 6 trained domains with configurable sample size.
"""

import argparse
import itertools
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
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import eval helpers from existing script
from scripts.eval_domain_accuracy import (
    DOMAIN_BENCHMARKS,
    decode_answer,
    evaluate_code_execution,
    evaluate_on_benchmark,
    extract_answer,
    format_mmlu_question,
    generate_response,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRAINED_DOMAINS = ["math", "code", "medical", "science", "history", "philosophy"]


def eval_model_on_domain(model, tokenizer, domain, max_samples=200):
    """Evaluate a model on a single domain's benchmark."""
    if domain not in DOMAIN_BENCHMARKS:
        return {}
    results = {}
    for bench_name, bench_cfg in DOMAIN_BENCHMARKS[domain].items():
        if bench_cfg.get("synthetic"):
            continue
        cfg = {**bench_cfg, "max_samples": max_samples}
        results[bench_name] = evaluate_on_benchmark(model, tokenizer, cfg, domain)
    return results


def load_base_model(model_name):
    """Load base model and tokenizer."""
    logger.info("Loading base model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
    )
    model.eval()
    return model, tokenizer


def apply_peft_adapter(base_model_name, adapter_path, device_map="auto"):
    """Load base model with a PEFT adapter."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device_map,
    )
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def apply_delta_weights(base_model_name, delta_path, device_map="auto"):
    """Load base model and apply delta weights from .pt file."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device_map,
    )
    delta = torch.load(delta_path, map_location="cpu")
    state = model.state_dict()
    for key, d in delta.items():
        for suffix in ["", ".weight"]:
            full_key = key + suffix
            if full_key in state:
                state[full_key] = state[full_key].float() + d.float()
                state[full_key] = state[full_key].to(torch.bfloat16)
                break
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/domains.yaml")
    parser.add_argument("--lora_dir", default="results/domain_loras")
    parser.add_argument("--algebra_dir", default="results/algebra")
    parser.add_argument("--output_dir", default="results/eval_v2")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_base", action="store_true")
    parser.add_argument("--skip_individual", action="store_true")
    parser.add_argument("--skip_grassmerge", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--domains", nargs="+", default=TRAINED_DOMAINS)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Baseline methods to eval (default: all found)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    base_model_name = config["base_model"]
    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_model": base_model_name,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "domains": args.domains,
        }
    }

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Phase 1: Base model ---
    if not args.skip_base:
        logger.info("=" * 60)
        logger.info("  PHASE 1: Base Model (no adapter)")
        logger.info("=" * 60)
        model, _ = load_base_model(base_model_name)
        base_results = {}
        for domain in args.domains:
            logger.info("  Base → %s", domain)
            base_results[domain] = eval_model_on_domain(model, tokenizer, domain, args.max_samples)
        results["base_model"] = base_results
        del model
        torch.cuda.empty_cache()
        # Save incrementally
        _save(results, args.output_dir)

    # --- Phase 2: Individual LoRAs ---
    if not args.skip_individual:
        logger.info("=" * 60)
        logger.info("  PHASE 2: Individual Domain LoRAs")
        logger.info("=" * 60)
        individual_results = {}
        for domain in args.domains:
            adapter_path = os.path.join(args.lora_dir, domain)
            if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                logger.warning("  No adapter for %s, skip", domain)
                continue
            logger.info("  Individual LoRA → %s", domain)
            model = apply_peft_adapter(base_model_name, adapter_path)
            individual_results[domain] = eval_model_on_domain(model, tokenizer, domain, args.max_samples)
            del model
            torch.cuda.empty_cache()
        results["individual_loras"] = individual_results
        _save(results, args.output_dir)

    # --- Phase 3: GrassMerge pairwise ---
    if not args.skip_grassmerge:
        logger.info("=" * 60)
        logger.info("  PHASE 3: GrassMerge Pairwise")
        logger.info("=" * 60)
        gm_results = {}
        gm_dir = os.path.join(args.algebra_dir, "grassmerge")
        pairs = list(itertools.combinations(args.domains, 2))
        for idx, (d1, d2) in enumerate(pairs):
            pair_name = f"{d1}+{d2}"
            peft_dir = os.path.join(gm_dir, pair_name)
            pt_path = os.path.join(gm_dir, f"{pair_name}.pt")
            if os.path.isdir(peft_dir) and os.path.exists(os.path.join(peft_dir, "adapter_config.json")):
                logger.info("  [%d/%d] GrassMerge %s (PEFT)", idx+1, len(pairs), pair_name)
                model = apply_peft_adapter(base_model_name, peft_dir)
            elif os.path.exists(pt_path):
                logger.info("  [%d/%d] GrassMerge %s (delta)", idx+1, len(pairs), pair_name)
                model = apply_delta_weights(base_model_name, pt_path)
            else:
                logger.warning("  No GrassMerge output for %s", pair_name)
                continue
            for d in [d1, d2]:
                key = f"{pair_name}_on_{d}"
                logger.info("    eval %s", key)
                gm_results[key] = eval_model_on_domain(model, tokenizer, d, args.max_samples)
            del model
            torch.cuda.empty_cache()
        results["grassmerge"] = gm_results
        _save(results, args.output_dir)

    # --- Phase 4: Baseline methods ---
    if not args.skip_baselines:
        logger.info("=" * 60)
        logger.info("  PHASE 4: Baseline Methods")
        logger.info("=" * 60)
        baseline_dir = os.path.join(args.algebra_dir, "baselines")
        if os.path.isdir(baseline_dir):
            methods = args.methods or sorted(os.listdir(baseline_dir))
            for method_name in methods:
                method_dir = os.path.join(baseline_dir, method_name)
                if not os.path.isdir(method_dir):
                    continue
                logger.info("  Baseline: %s", method_name)
                method_results = {}
                pairs = list(itertools.combinations(args.domains, 2))
                for idx, (d1, d2) in enumerate(pairs):
                    pair_name = f"{d1}+{d2}"
                    peft_dir = os.path.join(method_dir, pair_name)
                    pt_path = os.path.join(method_dir, f"{pair_name}.pt")
                    if os.path.isdir(peft_dir) and os.path.exists(os.path.join(peft_dir, "adapter_config.json")):
                        logger.info("    [%d/%d] %s %s (PEFT)", idx+1, len(pairs), method_name, pair_name)
                        model = apply_peft_adapter(base_model_name, peft_dir)
                    elif os.path.exists(pt_path):
                        logger.info("    [%d/%d] %s %s (delta)", idx+1, len(pairs), method_name, pair_name)
                        model = apply_delta_weights(base_model_name, pt_path)
                    else:
                        logger.warning("    No output for %s/%s", method_name, pair_name)
                        continue
                    for d in [d1, d2]:
                        key = f"{pair_name}_on_{d}"
                        logger.info("      eval %s", key)
                        method_results[key] = eval_model_on_domain(model, tokenizer, d, args.max_samples)
                    del model
                    torch.cuda.empty_cache()
                results[f"baseline_{method_name}"] = method_results
                _save(results, args.output_dir)

    _save(results, args.output_dir)
    logger.info("=" * 60)
    logger.info("  Comprehensive evaluation complete")
    logger.info("  Results: %s", os.path.join(args.output_dir, "eval_results.json"))
    logger.info("=" * 60)


def _save(results, output_dir):
    """Incremental save after each phase."""
    path = os.path.join(output_dir, "eval_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
