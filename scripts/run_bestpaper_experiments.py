#!/usr/bin/env python3
"""Master experiment script for NeurIPS best-paper-level evidence.

Phases:
  1. Generate ALL missing baseline merges (TIES, DARE, KnOTS, TSPA, SVD-Procrustes)
  2. Generate N-way GrassMerge compositions (3,4,6-way)
  3. Run comprehensive evaluation: base, individual, all pairwise, N-way
  4. Multi-seed evaluation for bootstrap CIs
  5. BGD-degradation correlation analysis
  6. Runtime/memory profiling
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lora_algebra import (
    ColumnOnlyGrassmannMerge,
    GrassMerge,
    KnOTSMerge,
    LoRAWeights,
    MergingBaselines,
    SVDProcrustesMerge,
    TSPAMerge,
    bilateral_grassmann_distance,
    spectral_weighted_bgd,
    cosine_interference,
    frobenius_interference,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORE_DOMAINS = ["math", "code", "medical", "science", "history", "philosophy"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_all_loras(lora_dir, domains):
    loras = {}
    for d in domains:
        path = os.path.join(lora_dir, d)
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            logger.info("Loading LoRA: %s", d)
            loras[d] = LoRAWeights.from_peft_dir(d, path)
        else:
            logger.warning("No adapter for %s, skipping", d)
    return loras


def save_as_peft(composed, peft_dir, config):
    import safetensors.torch
    os.makedirs(peft_dir, exist_ok=True)
    sd = composed.to_state_dict()
    safetensors.torch.save_file(sd, os.path.join(peft_dir, "adapter_model.safetensors"))
    lora_cfg = config.get("lora", {})
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": config.get("base_model", ""),
        "r": composed.rank,
        "lora_alpha": composed.rank,
        "target_modules": lora_cfg.get("target_modules", []),
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(peft_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)


def save_delta_as_peft(delta_dict, peft_dir, config, rank):
    """Re-factorize delta weights into LoRA A/B and save as PEFT."""
    import safetensors.torch
    os.makedirs(peft_dir, exist_ok=True)
    state_dict = {}
    for key, d in delta_dict.items():
        U, S, Vh = torch.linalg.svd(d.float(), full_matrices=False)
        r = min(rank, len(S))
        B = (U[:, :r] @ torch.diag(S[:r].sqrt())).to(torch.bfloat16)
        A = (torch.diag(S[:r].sqrt()) @ Vh[:r, :]).to(torch.bfloat16)
        # Convert key to PEFT format
        peft_key = key
        if not peft_key.startswith("base_model.model."):
            peft_key = "base_model.model." + peft_key
        peft_key_a = peft_key.replace(".weight", "").rstrip(".") + ".lora_A.weight"
        peft_key_b = peft_key.replace(".weight", "").rstrip(".") + ".lora_B.weight"
        state_dict[peft_key_a] = A
        state_dict[peft_key_b] = B
    safetensors.torch.save_file(state_dict, os.path.join(peft_dir, "adapter_model.safetensors"))
    lora_cfg = config.get("lora", {})
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": config.get("base_model", ""),
        "r": rank,
        "lora_alpha": rank,
        "target_modules": lora_cfg.get("target_modules", []),
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(peft_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)


# ==============================================================================
# Phase 1: Generate all baseline merges
# ==============================================================================

def phase1_generate_baselines(loras, output_dir, config):
    """Generate all missing pairwise baseline merges."""
    logger.info("=" * 60)
    logger.info("  PHASE 1: Generate ALL Pairwise Baselines")
    logger.info("=" * 60)

    baseline_dir = os.path.join(output_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)
    domains = sorted(loras.keys())
    pairs = list(itertools.combinations(domains, 2))
    rank = max(l.rank for l in loras.values())

    baseline_configs = [
        ("task_arithmetic", lambda ls, **kw: MergingBaselines.task_arithmetic_avg(ls, scaling=1.0), False),
        ("ties_d0.5", lambda ls, **kw: MergingBaselines.ties_merging(ls, density=0.5, scaling=1.0), False),
        ("dare_p0.3", lambda ls, **kw: MergingBaselines.dare_merging(ls, drop_rate=0.3, scaling=1.0), False),
        ("dare_p0.5", lambda ls, **kw: MergingBaselines.dare_merging(ls, drop_rate=0.5, scaling=1.0), False),
        ("knots", lambda ls, **kw: KnOTSMerge.merge(ls), True),
        ("tspa", lambda ls, **kw: TSPAMerge.merge(ls), True),
        ("svd_procrustes", lambda ls, **kw: SVDProcrustesMerge.merge(ls), True),
        ("col_grassmann", lambda ls, **kw: ColumnOnlyGrassmannMerge().merge(ls), True),
    ]

    stats = {}
    for method_name, method_fn, returns_lora in baseline_configs:
        method_dir = os.path.join(baseline_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        generated, cached = 0, 0

        for d1, d2 in pairs:
            name = f"{d1}+{d2}"
            pt_path = os.path.join(method_dir, f"{name}.pt")
            if os.path.exists(pt_path):
                cached += 1
                continue

            logger.info("  %s: %s", method_name, name)
            t0 = time.time()
            try:
                merged = method_fn([loras[d1], loras[d2]])
                elapsed = time.time() - t0

                if returns_lora:
                    delta = merged.to_delta_weight()
                    save_as_peft(merged, os.path.join(method_dir, name), config)
                else:
                    delta = merged
                    # Also save as PEFT for eval compatibility
                    save_delta_as_peft(delta, os.path.join(method_dir, name), config, rank)

                torch.save(delta, pt_path)
                generated += 1
                del delta, merged
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error("  FAILED %s/%s: %s", method_name, name, e)

        stats[method_name] = {"generated": generated, "cached": cached, "total": len(pairs)}
        logger.info("  %s: %d generated, %d cached / %d total",
                     method_name, generated, cached, len(pairs))

    return stats


# ==============================================================================
# Phase 2: Generate N-way GrassMerge compositions
# ==============================================================================

def phase2_nway_merges(loras, output_dir, config):
    """Generate 3-way, 4-way, and 6-way GrassMerge compositions."""
    logger.info("=" * 60)
    logger.info("  PHASE 2: N-way GrassMerge Compositions")
    logger.info("=" * 60)

    nway_dir = os.path.join(output_dir, "nway")
    os.makedirs(nway_dir, exist_ok=True)
    domains = sorted(loras.keys())
    merger = GrassMerge()
    results = {}

    for n in [3, 4, 6]:
        n_dir = os.path.join(nway_dir, f"{n}way")
        os.makedirs(n_dir, exist_ok=True)
        combos = list(itertools.combinations(domains, n))
        logger.info("  %d-way: %d combinations", n, len(combos))

        for combo in combos:
            name = "+".join(combo)
            pt_path = os.path.join(n_dir, f"{name}.pt")
            peft_path = os.path.join(n_dir, name)

            if os.path.exists(pt_path) and os.path.isdir(peft_path):
                logger.info("    %s (cached)", name)
                results[name] = {"n": n, "status": "cached"}
                continue

            logger.info("    %d-way GrassMerge: %s", n, name)
            t0 = time.time()
            try:
                combo_loras = [loras[d] for d in combo]
                composed = merger.merge(combo_loras, name=name)
                elapsed = time.time() - t0
                delta = composed.to_delta_weight()
                torch.save(delta, pt_path)
                save_as_peft(composed, peft_path, config)
                results[name] = {
                    "n": n, "domains": list(combo),
                    "time_seconds": round(elapsed, 3),
                    "rank": composed.rank,
                }
                del composed, delta
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error("    FAILED %s: %s", name, e)
                results[name] = {"n": n, "error": str(e)}

    # Also generate N-way Task Arithmetic for comparison
    ta_nway_dir = os.path.join(nway_dir, "task_arithmetic")
    os.makedirs(ta_nway_dir, exist_ok=True)
    rank = max(l.rank for l in loras.values())

    for n in [3, 4, 6]:
        combos = list(itertools.combinations(domains, n))
        for combo in combos:
            name = "+".join(combo)
            pt_path = os.path.join(ta_nway_dir, f"{name}.pt")
            peft_path = os.path.join(ta_nway_dir, name)
            if os.path.exists(pt_path):
                continue
            logger.info("    %d-way Task Arithmetic: %s", n, name)
            combo_loras = [loras[d] for d in combo]
            delta = MergingBaselines.task_arithmetic_avg(combo_loras, scaling=1.0)
            torch.save(delta, pt_path)
            save_delta_as_peft(delta, peft_path, config, rank)
            del delta

    with open(os.path.join(nway_dir, "nway_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ==============================================================================
# Phase 3: Comprehensive evaluation
# ==============================================================================

def phase3_comprehensive_eval(config, lora_dir, algebra_dir, output_dir, max_samples=200, seed=42):
    """Run eval_comprehensive.py with full coverage."""
    logger.info("=" * 60)
    logger.info("  PHASE 3: Comprehensive Evaluation (seed=%d, n=%d)", seed, max_samples)
    logger.info("=" * 60)

    eval_dir = os.path.join(output_dir, f"eval_seed{seed}")
    cmd = [
        sys.executable, "scripts/eval_comprehensive.py",
        "--config", "configs/domains.yaml",
        "--lora_dir", lora_dir,
        "--algebra_dir", algebra_dir,
        "--output_dir", eval_dir,
        "--max_samples", str(max_samples),
        "--seed", str(seed),
        "--domains", *CORE_DOMAINS,
    ]
    logger.info("  CMD: %s", " ".join(cmd))
    import subprocess
    result = subprocess.run(cmd, capture_output=False, cwd=str(Path(__file__).resolve().parent.parent))
    return result.returncode


# ==============================================================================
# Phase 4: N-way downstream evaluation
# ==============================================================================

def phase4_nway_eval(config, nway_dir, output_dir, max_samples=200, seed=42):
    """Evaluate N-way compositions on ALL constituent domains."""
    logger.info("=" * 60)
    logger.info("  PHASE 4: N-way Downstream Evaluation")
    logger.info("=" * 60)

    # Import eval helpers
    from scripts.eval_domain_accuracy import (
        DOMAIN_BENCHMARKS,
        evaluate_on_benchmark,
        evaluate_code_execution,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_model_name = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    nway_eval_path = os.path.join(output_dir, "nway_eval_results.json")

    for method in ["grassmerge", "task_arithmetic"]:
        method_label = "GrassMerge" if method == "grassmerge" else "TaskArith"
        method_results = {}

        for n in [3, 4, 6]:
            if method == "grassmerge":
                n_dir = os.path.join(nway_dir, f"{n}way")
            else:
                n_dir = os.path.join(nway_dir, "task_arithmetic")

            if not os.path.isdir(n_dir):
                continue

            combos = list(itertools.combinations(CORE_DOMAINS, n))
            for combo in combos:
                name = "+".join(combo)
                peft_dir = os.path.join(n_dir, name)
                pt_path = os.path.join(n_dir, f"{name}.pt")

                if os.path.isdir(peft_dir) and os.path.exists(os.path.join(peft_dir, "adapter_config.json")):
                    logger.info("  [%s] %d-way %s (PEFT)", method_label, n, name)
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
                    model = PeftModel.from_pretrained(model, peft_dir)
                    model.eval()
                elif os.path.exists(pt_path):
                    logger.info("  [%s] %d-way %s (delta)", method_label, n, name)
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
                    delta = torch.load(pt_path, map_location="cpu")
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
                    del delta
                else:
                    logger.warning("  No output for %s/%s", method, name)
                    continue

                combo_results = {}
                for domain in combo:
                    if domain not in DOMAIN_BENCHMARKS:
                        continue
                    for bench_name, bench_cfg in DOMAIN_BENCHMARKS[domain].items():
                        if bench_cfg.get("synthetic"):
                            continue
                        cfg = {**bench_cfg, "max_samples": max_samples}
                        key = f"{name}_on_{domain}_{bench_name}"
                        logger.info("    eval %s", key)
                        if domain == "code" and "mbpp" in bench_cfg.get("dataset_id", ""):
                            combo_results[key] = evaluate_code_execution(model, tokenizer, cfg)
                        else:
                            combo_results[key] = evaluate_on_benchmark(model, tokenizer, cfg, domain)

                method_results[f"{n}way_{name}"] = combo_results
                del model
                torch.cuda.empty_cache()

        results[method] = method_results

    with open(nway_eval_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("  N-way eval saved to %s", nway_eval_path)
    return results


# ==============================================================================
# Phase 5: BGD-degradation correlation
# ==============================================================================

def phase5_bgd_correlation(eval_results_path, interference_path, output_dir):
    """Compute correlation between BGD and actual downstream degradation."""
    logger.info("=" * 60)
    logger.info("  PHASE 5: BGD-Degradation Correlation")
    logger.info("=" * 60)

    with open(eval_results_path) as f:
        eval_results = json.load(f)
    with open(interference_path) as f:
        interference = json.load(f)

    names = interference["domain_names"]
    bgd_matrix = np.array(interference["bgd_matrix"])
    sbgd_matrix = np.array(interference.get("spectral_bgd_matrix", bgd_matrix))

    individual = eval_results.get("individual_loras", {})
    grassmerge = eval_results.get("grassmerge", {})

    # Compute degradation for each pair
    data_points = []
    for i, d1 in enumerate(names):
        for j, d2 in enumerate(names):
            if i >= j:
                continue
            pair_key = f"{d1}+{d2}"
            bgd_val = bgd_matrix[i][j]
            sbgd_val = sbgd_matrix[i][j]

            # Get individual accuracy
            for domain in [d1, d2]:
                indiv_acc = None
                gm_acc = None

                if domain in individual:
                    for bench_name, metrics in individual[domain].items():
                        if isinstance(metrics, dict) and "accuracy" in metrics:
                            indiv_acc = metrics["accuracy"]
                            break

                gm_key = f"{pair_key}_on_{domain}"
                if gm_key in grassmerge:
                    for bench_name, metrics in grassmerge[gm_key].items():
                        if isinstance(metrics, dict) and "accuracy" in metrics:
                            gm_acc = metrics["accuracy"]
                            break

                if indiv_acc is not None and gm_acc is not None and indiv_acc > 0:
                    retention = gm_acc / indiv_acc
                    degradation = 1.0 - retention
                    data_points.append({
                        "pair": pair_key,
                        "domain": domain,
                        "individual_acc": indiv_acc,
                        "grassmerge_acc": gm_acc,
                        "retention": round(retention, 4),
                        "degradation": round(degradation, 4),
                        "bgd": round(bgd_val, 4),
                        "spectral_bgd": round(sbgd_val, 4),
                    })

    if len(data_points) < 3:
        logger.warning("  Only %d data points, insufficient for correlation", len(data_points))
        return {"error": "insufficient data points", "n": len(data_points)}

    bgd_vals = [p["bgd"] for p in data_points]
    deg_vals = [p["degradation"] for p in data_points]
    ret_vals = [p["retention"] for p in data_points]

    from scipy import stats as scipy_stats
    pearson_r, pearson_p = scipy_stats.pearsonr(bgd_vals, deg_vals)
    spearman_r, spearman_p = scipy_stats.spearmanr(bgd_vals, deg_vals)

    correlation = {
        "n_data_points": len(data_points),
        "bgd_vs_degradation": {
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 6),
            "spearman_rho": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 6),
        },
        "data_points": data_points,
    }

    corr_path = os.path.join(output_dir, "bgd_correlation_full.json")
    with open(corr_path, "w") as f:
        json.dump(correlation, f, indent=2)
    logger.info("  Correlation (n=%d): Pearson r=%.4f (p=%.6f), Spearman ρ=%.4f (p=%.6f)",
                len(data_points), pearson_r, pearson_p, spearman_r, spearman_p)
    return correlation


# ==============================================================================
# Phase 6: Runtime + Memory profiling
# ==============================================================================

def phase6_profiling(loras, output_dir):
    """Profile all merge methods: wall-clock time and peak GPU memory."""
    logger.info("=" * 60)
    logger.info("  PHASE 6: Runtime & Memory Profiling")
    logger.info("=" * 60)

    domains = sorted(loras.keys())
    # Pick representative pairs
    test_pairs = [
        (domains[0], domains[1]),  # math+code
        (domains[2], domains[3]),  # medical+science
    ]

    methods = {
        "GrassMerge": lambda ls: GrassMerge().merge(ls),
        "Task Arithmetic": lambda ls: MergingBaselines.task_arithmetic_avg(ls, scaling=1.0),
        "TIES (d=0.5)": lambda ls: MergingBaselines.ties_merging(ls, density=0.5, scaling=1.0),
        "DARE (p=0.5)": lambda ls: MergingBaselines.dare_merging(ls, drop_rate=0.5, scaling=1.0),
        "KnOTS": lambda ls: KnOTSMerge.merge(ls),
        "TSPA": lambda ls: TSPAMerge.merge(ls),
        "SVD-Procrustes": lambda ls: SVDProcrustesMerge.merge(ls),
    }

    profile_results = []
    for method_name, merge_fn in methods.items():
        for d1, d2 in test_pairs:
            pair_name = f"{d1}+{d2}"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            times = []
            for trial in range(3):
                t0 = time.time()
                result = merge_fn([loras[d1], loras[d2]])
                elapsed = time.time() - t0
                times.append(elapsed)
                del result
                torch.cuda.empty_cache()

            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            avg_time = np.mean(times)
            std_time = np.std(times)

            profile_results.append({
                "method": method_name,
                "pair": pair_name,
                "avg_time_s": round(avg_time, 3),
                "std_time_s": round(std_time, 3),
                "peak_gpu_gb": round(peak_mem, 2),
                "n_trials": 3,
            })
            logger.info("  %s on %s: %.3fs ± %.3fs, %.2f GB peak",
                        method_name, pair_name, avg_time, std_time, peak_mem)

    profile_path = os.path.join(output_dir, "profile_results.json")
    with open(profile_path, "w") as f:
        json.dump(profile_results, f, indent=2)
    return profile_results


def main():
    parser = argparse.ArgumentParser(description="Best-paper experiment suite")
    parser.add_argument("--config", default="configs/domains.yaml")
    parser.add_argument("--lora_dir", default="results/domain_loras")
    parser.add_argument("--output_dir", default="results/algebra")
    parser.add_argument("--eval_output", default="results/eval_comprehensive")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--phase", type=int, default=0, help="Run specific phase (0=all)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Phase 1+2: Generate merges (need LoRAs in memory)
    if args.phase in [0, 1, 2, 6]:
        loras = load_all_loras(args.lora_dir, CORE_DOMAINS)
        if len(loras) < 2:
            logger.error("Need at least 2 LoRAs, found %d", len(loras))
            sys.exit(1)
        logger.info("Loaded %d LoRAs: %s", len(loras), sorted(loras.keys()))

    if args.phase in [0, 1]:
        stats = phase1_generate_baselines(loras, args.output_dir, config)
        logger.info("Phase 1 complete: %s", json.dumps(stats, indent=2))

    if args.phase in [0, 2]:
        nway = phase2_nway_merges(loras, args.output_dir, config)
        logger.info("Phase 2 complete: %d N-way merges", len(nway))

    # Phase 3: Comprehensive evaluation (multi-seed)
    if args.phase in [0, 3]:
        for seed in args.seeds:
            rc = phase3_comprehensive_eval(
                config, args.lora_dir, args.output_dir,
                os.path.join(args.eval_output, f"seed{seed}"),
                max_samples=args.max_samples, seed=seed,
            )
            logger.info("Phase 3 seed=%d: exit code %d", seed, rc)

    # Phase 4: N-way downstream eval
    if args.phase in [0, 4]:
        nway_dir = os.path.join(args.output_dir, "nway")
        phase4_nway_eval(
            config, nway_dir, args.eval_output,
            max_samples=args.max_samples, seed=args.seeds[0],
        )

    # Phase 5: BGD correlation
    if args.phase in [0, 5]:
        eval_path = os.path.join(args.eval_output, f"seed{args.seeds[0]}", "eval_results.json")
        interference_path = os.path.join(args.output_dir, "interference_metrics.json")
        if os.path.exists(eval_path) and os.path.exists(interference_path):
            phase5_bgd_correlation(eval_path, interference_path, args.eval_output)

    # Phase 6: Profiling
    if args.phase in [0, 6]:
        phase6_profiling(loras, args.eval_output)

    logger.info("=" * 60)
    logger.info("  ALL PHASES COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
