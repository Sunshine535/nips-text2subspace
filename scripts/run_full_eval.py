#!/usr/bin/env python3
"""Integrated merge + evaluation pipeline.

Generates baselines on-the-fly and evaluates immediately, avoiding 28GB delta files.
Loads base model ONCE, applies adapters/deltas in-place, evaluates, then resets.

Usage:
  python3 scripts/run_full_eval.py --max_samples 200 --seed 42
  python3 scripts/run_full_eval.py --phase base       # base model only
  python3 scripts/run_full_eval.py --phase individual  # individual LoRAs
  python3 scripts/run_full_eval.py --phase grassmerge  # pairwise GrassMerge
  python3 scripts/run_full_eval.py --phase baselines   # all baselines
  python3 scripts/run_full_eval.py --phase nway        # N-way merges
  python3 scripts/run_full_eval.py --phase all         # everything
"""

import argparse
import gc
import itertools
import json
import logging
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.eval_domain_accuracy as eval_module
from scripts.eval_domain_accuracy import (
    DOMAIN_BENCHMARKS,
    evaluate_code_execution,
    evaluate_on_benchmark,
)
from src.lora_algebra import (
    ColumnOnlyGrassmannMerge,
    GrassMerge,
    KnOTSMerge,
    LoRAWeights,
    MergingBaselines,
    SVDProcrustesMerge,
    TSPAMerge,
    bilateral_grassmann_distance,
    cosine_interference,
    frobenius_interference,
    spectral_weighted_bgd,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORE_DOMAINS = ["math", "code", "medical", "science", "history", "philosophy"]
RESULTS_FILE = "results/eval_comprehensive/eval_results.json"


def load_config(path="configs/domains.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_peft_mode(is_peft):
    """Toggle whether generate_response uses PEFT-compatible format (no thinking prefix)."""
    eval_module._IS_PEFT_MODEL = is_peft


def eval_on_domain(model, tokenizer, domain, max_samples=200):
    """Evaluate model on a single domain's benchmark. Returns dict of bench->metrics."""
    if domain not in DOMAIN_BENCHMARKS:
        return {}
    results = {}
    for bench_name, bench_cfg in DOMAIN_BENCHMARKS[domain].items():
        if bench_cfg.get("synthetic"):
            continue
        cfg = {**bench_cfg, "max_samples": max_samples}
        if domain == "code" and "mbpp" in bench_cfg.get("dataset_id", ""):
            results[bench_name] = evaluate_code_execution(model, tokenizer, cfg)
        else:
            results[bench_name] = evaluate_on_benchmark(model, tokenizer, cfg, domain)
    return results


def get_primary_accuracy(domain_results):
    """Extract the primary accuracy from domain results."""
    for bench_name, metrics in domain_results.items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            return metrics["accuracy"]
    return None


def apply_delta_to_model(model, delta_dict):
    """Apply delta weights to model in-place. Returns list of modified keys for rollback."""
    state = model.state_dict()
    modified_keys = []
    original_values = {}
    for key, d in delta_dict.items():
        for suffix in ["", ".weight"]:
            full_key = key + suffix
            if full_key in state:
                original_values[full_key] = state[full_key].clone()
                # Move delta to same device as model weights
                device = state[full_key].device
                state[full_key] = (state[full_key].float() + d.float().to(device)).to(torch.bfloat16)
                modified_keys.append(full_key)
                break
    model.load_state_dict(state)
    return original_values


def rollback_model(model, original_values):
    """Restore original model weights."""
    state = model.state_dict()
    for key, val in original_values.items():
        state[key] = val
    model.load_state_dict(state)


def load_results(path):
    """Load existing results file or return empty dict."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(results, path):
    """Save results incrementally."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def merge_on_the_fly(lora_list, method_name, method_kwargs=None):
    """Compute merged delta dict without saving to disk."""
    if method_kwargs is None:
        method_kwargs = {}

    if method_name == "task_arithmetic":
        return MergingBaselines.task_arithmetic_avg(lora_list, scaling=method_kwargs.get("scaling", 1.0))
    elif method_name == "ties":
        return MergingBaselines.ties_merging(lora_list, density=method_kwargs.get("density", 0.5), scaling=1.0)
    elif method_name == "dare":
        return MergingBaselines.dare_merging(lora_list, drop_rate=method_kwargs.get("drop_rate", 0.5), scaling=1.0)
    elif method_name == "knots":
        composed = KnOTSMerge.merge(lora_list)
        return composed.to_delta_weight()
    elif method_name == "tspa":
        composed = TSPAMerge.merge(lora_list)
        return composed.to_delta_weight()
    elif method_name == "svd_procrustes":
        composed = SVDProcrustesMerge.merge(lora_list)
        return composed.to_delta_weight()
    elif method_name == "col_grassmann":
        composed = ColumnOnlyGrassmannMerge().merge(lora_list)
        return composed.to_delta_weight()
    elif method_name == "grassmerge":
        composed = GrassMerge().merge(lora_list)
        return composed.to_delta_weight()
    else:
        raise ValueError(f"Unknown method: {method_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", default="all",
                        choices=["all", "base", "individual", "grassmerge", "baselines",
                                 "nway", "profiling", "bgd"])
    parser.add_argument("--output", default=RESULTS_FILE)
    parser.add_argument("--lora_dir", default="results/domain_loras")
    parser.add_argument("--algebra_dir", default="results/algebra")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config()
    base_model_name = config["base_model"]

    results = load_results(args.output)
    results["meta"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model_name,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "domains": CORE_DOMAINS,
    }

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    phases = [args.phase] if args.phase != "all" else [
        "base", "individual", "grassmerge", "baselines", "nway", "profiling", "bgd"
    ]

    # =========================================================================
    # PHASE: Base model evaluation
    # =========================================================================
    if "base" in phases:
        if "base_model" in results and len(results["base_model"]) >= len(CORE_DOMAINS):
            logger.info("Base model results already exist, skipping")
        else:
            logger.info("=" * 60)
            logger.info("  PHASE: Base Model Evaluation")
            logger.info("=" * 60)
            set_peft_mode(False)  # Base model: use enable_thinking=False
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
            model.eval()
            base_results = results.get("base_model", {})
            for domain in CORE_DOMAINS:
                if domain in base_results:
                    logger.info("  Base → %s (cached)", domain)
                    continue
                logger.info("  Base → %s", domain)
                base_results[domain] = eval_on_domain(model, tokenizer, domain, args.max_samples)
                results["base_model"] = base_results
                save_results(results, args.output)
            del model
            torch.cuda.empty_cache()
            gc.collect()

    # =========================================================================
    # PHASE: Individual LoRA evaluation
    # =========================================================================
    if "individual" in phases:
        logger.info("=" * 60)
        logger.info("  PHASE: Individual LoRA Evaluation")
        logger.info("=" * 60)
        set_peft_mode(True)  # PEFT LoRA: use training-compatible format
        individual = results.get("individual_loras", {})
        for domain in CORE_DOMAINS:
            if domain in individual and individual[domain]:
                logger.info("  Individual → %s (cached)", domain)
                continue
            adapter_path = os.path.join(args.lora_dir, domain)
            if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                logger.warning("  No adapter for %s", domain)
                continue
            logger.info("  Individual → %s", domain)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
            individual[domain] = eval_on_domain(model, tokenizer, domain, args.max_samples)
            results["individual_loras"] = individual
            save_results(results, args.output)
            del model
            torch.cuda.empty_cache()
            gc.collect()

    # =========================================================================
    # PHASE: GrassMerge pairwise evaluation (from pre-computed PEFT dirs)
    # =========================================================================
    if "grassmerge" in phases:
        logger.info("=" * 60)
        logger.info("  PHASE: GrassMerge Pairwise Evaluation")
        logger.info("=" * 60)
        gm_results = results.get("grassmerge", {})
        # GrassMerge: set per-pair depending on load method (PEFT vs delta)
        gm_dir = os.path.join(args.algebra_dir, "grassmerge")
        pairs = list(itertools.combinations(CORE_DOMAINS, 2))

        for idx, (d1, d2) in enumerate(pairs):
            pair_name = f"{d1}+{d2}"
            # Check if both domain evals for this pair exist
            key1 = f"{pair_name}_on_{d1}"
            key2 = f"{pair_name}_on_{d2}"
            if key1 in gm_results and key2 in gm_results:
                logger.info("  [%d/%d] GrassMerge %s (cached)", idx+1, len(pairs), pair_name)
                continue

            # Try both orderings since directory may use alphabetical order
            peft_dir = os.path.join(gm_dir, pair_name)
            pt_path = os.path.join(gm_dir, f"{pair_name}.pt")
            alt_name = f"{d2}+{d1}"
            alt_peft_dir = os.path.join(gm_dir, alt_name)
            alt_pt_path = os.path.join(gm_dir, f"{alt_name}.pt")

            # Find the correct path
            found_peft, found_pt = None, None
            for pd in [peft_dir, alt_peft_dir]:
                if os.path.isdir(pd) and os.path.exists(os.path.join(pd, "adapter_config.json")):
                    found_peft = pd
                    break
            for pp in [pt_path, alt_pt_path]:
                if os.path.exists(pp):
                    found_pt = pp
                    break

            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")

            if found_peft:
                logger.info("  [%d/%d] GrassMerge %s (PEFT)", idx+1, len(pairs), pair_name)
                model = PeftModel.from_pretrained(model, found_peft)
                set_peft_mode(True)
            elif found_pt:
                logger.info("  [%d/%d] GrassMerge %s (delta)", idx+1, len(pairs), pair_name)
                delta = torch.load(found_pt, map_location="cpu")
                apply_delta_to_model(model, delta)
                set_peft_mode(False)
                del delta
            else:
                logger.warning("  No GrassMerge output for %s (tried %s and %s)", pair_name, pair_name, alt_name)
                del model
                torch.cuda.empty_cache()
                continue

            model.eval()
            for d in [d1, d2]:
                key = f"{pair_name}_on_{d}"
                if key not in gm_results:
                    logger.info("    eval %s", key)
                    gm_results[key] = eval_on_domain(model, tokenizer, d, args.max_samples)
            results["grassmerge"] = gm_results
            save_results(results, args.output)
            del model
            torch.cuda.empty_cache()
            gc.collect()

    # =========================================================================
    # PHASE: All baseline methods (on-the-fly merge + eval)
    # =========================================================================
    if "baselines" in phases:
        logger.info("=" * 60)
        logger.info("  PHASE: Baseline Methods (merge + eval on-the-fly)")
        logger.info("=" * 60)
        set_peft_mode(False)  # Baselines apply delta weights to base model

        # Load all LoRAs into memory
        loras = {}
        for d in CORE_DOMAINS:
            path = os.path.join(args.lora_dir, d)
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                loras[d] = LoRAWeights.from_peft_dir(d, path)

        baseline_methods = [
            ("task_arithmetic", {"scaling": 1.0}),
            ("ties", {"density": 0.5}),
            ("dare", {"drop_rate": 0.5}),
            ("knots", {}),
            ("tspa", {}),
            ("svd_procrustes", {}),
            ("col_grassmann", {}),
        ]

        pairs = list(itertools.combinations(CORE_DOMAINS, 2))

        for method_name, method_kwargs in baseline_methods:
            result_key = f"baseline_{method_name}"
            method_results = results.get(result_key, {})

            for idx, (d1, d2) in enumerate(pairs):
                pair_name = f"{d1}+{d2}"
                key1 = f"{pair_name}_on_{d1}"
                key2 = f"{pair_name}_on_{d2}"
                if key1 in method_results and key2 in method_results:
                    logger.info("  [%s] %s (cached)", method_name, pair_name)
                    continue

                if d1 not in loras or d2 not in loras:
                    continue

                logger.info("  [%s] [%d/%d] merging %s", method_name, idx+1, len(pairs), pair_name)
                t0 = time.time()
                try:
                    delta = merge_on_the_fly([loras[d1], loras[d2]], method_name, method_kwargs)
                except Exception as e:
                    logger.error("  FAILED %s/%s: %s", method_name, pair_name, e)
                    continue
                merge_time = time.time() - t0
                logger.info("    merged in %.1fs", merge_time)

                # Load fresh base model and apply delta
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
                apply_delta_to_model(model, delta)
                model.eval()
                del delta
                torch.cuda.empty_cache()

                for d in [d1, d2]:
                    key = f"{pair_name}_on_{d}"
                    if key not in method_results:
                        logger.info("    eval %s → %s", method_name, key)
                        dr = eval_on_domain(model, tokenizer, d, args.max_samples)
                        dr["_merge_time_s"] = round(merge_time, 2)
                        method_results[key] = dr

                results[result_key] = method_results
                save_results(results, args.output)
                del model
                torch.cuda.empty_cache()
                gc.collect()

        del loras
        gc.collect()

    # =========================================================================
    # PHASE: N-way merges (3,4,6-way)
    # =========================================================================
    if "nway" in phases:
        logger.info("=" * 60)
        logger.info("  PHASE: N-way Merge Evaluation")
        logger.info("=" * 60)
        set_peft_mode(False)  # N-way uses delta weights

        loras = {}
        for d in CORE_DOMAINS:
            path = os.path.join(args.lora_dir, d)
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                loras[d] = LoRAWeights.from_peft_dir(d, path)

        nway_results = results.get("nway", {})

        for n in [3, 4, 6]:
            combos = list(itertools.combinations(CORE_DOMAINS, n))
            for combo in combos:
                name = "+".join(combo)

                for method_name in ["grassmerge", "task_arithmetic"]:
                    result_key = f"{method_name}_{n}way_{name}"
                    if result_key in nway_results and nway_results[result_key]:
                        logger.info("  [%s] %d-way %s (cached)", method_name, n, name)
                        continue

                    logger.info("  [%s] %d-way %s", method_name, n, name)
                    combo_loras = [loras[d] for d in combo if d in loras]
                    if len(combo_loras) != n:
                        continue

                    t0 = time.time()
                    delta = merge_on_the_fly(combo_loras, method_name)
                    merge_time = time.time() - t0

                    model = AutoModelForCausalLM.from_pretrained(
                        base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
                    apply_delta_to_model(model, delta)
                    model.eval()
                    del delta
                    torch.cuda.empty_cache()

                    combo_results = {"merge_time_s": round(merge_time, 2), "n": n, "method": method_name}
                    for domain in combo:
                        logger.info("    eval %s on %s", name, domain)
                        combo_results[domain] = eval_on_domain(model, tokenizer, domain, args.max_samples)

                    nway_results[result_key] = combo_results
                    results["nway"] = nway_results
                    save_results(results, args.output)
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()

        del loras
        gc.collect()

    # =========================================================================
    # PHASE: Profiling
    # =========================================================================
    if "profiling" in phases:
        logger.info("=" * 60)
        logger.info("  PHASE: Method Profiling")
        logger.info("=" * 60)

        loras = {}
        for d in CORE_DOMAINS:
            path = os.path.join(args.lora_dir, d)
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                loras[d] = LoRAWeights.from_peft_dir(d, path)

        test_pair = (CORE_DOMAINS[0], CORE_DOMAINS[1])  # math+code
        profile = []
        methods = {
            "GrassMerge": ("grassmerge", {}),
            "Task Arithmetic": ("task_arithmetic", {"scaling": 1.0}),
            "TIES (d=0.5)": ("ties", {"density": 0.5}),
            "DARE (p=0.5)": ("dare", {"drop_rate": 0.5}),
            "KnOTS": ("knots", {}),
            "TSPA": ("tspa", {}),
            "SVD-Procrustes": ("svd_procrustes", {}),
        }

        d1, d2 = test_pair
        for method_label, (method_name, kwargs) in methods.items():
            times = []
            for trial in range(3):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                delta = merge_on_the_fly([loras[d1], loras[d2]], method_name, kwargs)
                elapsed = time.time() - t0
                times.append(elapsed)
                peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
                del delta
                torch.cuda.empty_cache()

            profile.append({
                "method": method_label,
                "pair": f"{d1}+{d2}",
                "avg_time_s": round(np.mean(times), 3),
                "std_time_s": round(np.std(times), 3),
                "peak_gpu_gb": round(peak_mem, 2),
            })
            logger.info("  %s: %.3fs ± %.3fs, %.2f GB",
                        method_label, np.mean(times), np.std(times), peak_mem)

        results["profiling"] = profile
        save_results(results, args.output)
        del loras

    # =========================================================================
    # PHASE: BGD interference analysis
    # =========================================================================
    if "bgd" in phases:
        logger.info("=" * 60)
        logger.info("  PHASE: BGD Interference Analysis")
        logger.info("=" * 60)

        loras = {}
        for d in CORE_DOMAINS:
            path = os.path.join(args.lora_dir, d)
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                loras[d] = LoRAWeights.from_peft_dir(d, path)

        names = sorted(loras.keys())
        N = len(names)
        all_deltas = {k: loras[k].to_delta_weight() for k in names}
        rank = max(l.rank for l in loras.values())

        common_keys = set(all_deltas[names[0]].keys())
        for n in names[1:]:
            common_keys &= set(all_deltas[n].keys())
        sample_keys = sorted(common_keys)

        bgd_matrix = np.zeros((N, N))
        sbgd_matrix = np.zeros((N, N))
        cos_matrix = np.zeros((N, N))
        frob_matrix = np.zeros((N, N))

        for i in range(N):
            for j in range(i + 1, N):
                bgd_v, sbgd_v, cos_v, frob_v = [], [], [], []
                for key in sample_keys:
                    di, dj = all_deltas[names[i]][key], all_deltas[names[j]][key]
                    bgd_v.append(bilateral_grassmann_distance(di, dj, rank))
                    sbgd_v.append(spectral_weighted_bgd(di, dj, rank))
                    cos_v.append(cosine_interference(di, dj))
                    frob_v.append(frobenius_interference(di, dj))
                bgd_matrix[i][j] = bgd_matrix[j][i] = np.mean(bgd_v)
                sbgd_matrix[i][j] = sbgd_matrix[j][i] = np.mean(sbgd_v)
                cos_matrix[i][j] = cos_matrix[j][i] = np.mean(cos_v)
                frob_matrix[i][j] = frob_matrix[j][i] = np.mean(frob_v)

        results["interference"] = {
            "domain_names": names,
            "bgd_matrix": bgd_matrix.round(4).tolist(),
            "spectral_bgd_matrix": sbgd_matrix.round(4).tolist(),
            "cosine_interference_matrix": cos_matrix.round(4).tolist(),
            "frobenius_interference_matrix": frob_matrix.round(4).tolist(),
        }
        save_results(results, args.output)
        del loras, all_deltas
        gc.collect()

    save_results(results, args.output)
    logger.info("=" * 60)
    logger.info("  ALL PHASES COMPLETE")
    logger.info("  Results: %s", args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
