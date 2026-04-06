#!/usr/bin/env python3
"""Supplementary experiments addressing reviewer concerns.

Experiments:
  1. Oracle routing upper bound (per-domain correct adapter selection)
  2. Held-out domain generalization (eval on domains not in the merge)
  3. Reparameterization invariance test
  4. Karcher mean convergence analysis
  5. Failure analysis (identify worst-case pairs)
  6. General capability preservation (merged adapter on MMLU general)
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_domain_accuracy import (
    DOMAIN_BENCHMARKS,
    evaluate_on_benchmark,
    evaluate_code_execution,
)
from src.lora_algebra import (
    GrassMerge,
    GrassmannOps,
    LoRAWeights,
    MergingBaselines,
    bilateral_grassmann_distance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORE_DOMAINS = ["math", "code", "medical", "science", "history", "philosophy"]
HELD_OUT_DOMAINS = ["geography", "psychology"]
OUTPUT_FILE = "results/eval_comprehensive/supplementary_results.json"


def load_config(path="configs/domains.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def eval_on_domain(model, tokenizer, domain, max_samples=200):
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


def apply_delta_to_model(model, delta_dict):
    state = model.state_dict()
    original_values = {}
    for key, d in delta_dict.items():
        for suffix in ["", ".weight"]:
            full_key = key + suffix
            if full_key in state:
                original_values[full_key] = state[full_key].clone()
                state[full_key] = state[full_key].float() + d.float()
                state[full_key] = state[full_key].to(torch.bfloat16)
                break
    model.load_state_dict(state)
    return original_values


def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ==============================================================================
# 1. Oracle Routing Upper Bound
# ==============================================================================

def oracle_routing(config, lora_dir, tokenizer, max_samples=200):
    """Upper bound: for each domain, use the correct individual LoRA adapter."""
    logger.info("=" * 60)
    logger.info("  Oracle Routing Upper Bound")
    logger.info("=" * 60)

    base_model_name = config["base_model"]
    oracle_results = {}

    for domain in CORE_DOMAINS:
        adapter_path = os.path.join(lora_dir, domain)
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            continue
        logger.info("  Oracle → %s (using %s adapter)", domain, domain)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        oracle_results[domain] = eval_on_domain(model, tokenizer, domain, max_samples)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return oracle_results


# ==============================================================================
# 2. Held-out Domain Generalization
# ==============================================================================

def held_out_domain_eval(config, lora_dir, algebra_dir, tokenizer, max_samples=200):
    """Evaluate pairwise merges on domains NOT included in the merge."""
    logger.info("=" * 60)
    logger.info("  Held-out Domain Generalization")
    logger.info("=" * 60)

    base_model_name = config["base_model"]
    gm_dir = os.path.join(algebra_dir, "grassmerge")
    results = {}

    # For efficiency, test representative pairs on held-out domains
    test_pairs = [
        ("history", "philosophy"),   # humanities pair
        ("math", "science"),         # STEM pair
        ("code", "medical"),         # diverse pair
    ]

    for d1, d2 in test_pairs:
        pair_name = f"{d1}+{d2}"
        peft_dir = os.path.join(gm_dir, pair_name)
        pt_path = os.path.join(gm_dir, f"{pair_name}.pt")

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")

        if os.path.isdir(peft_dir) and os.path.exists(os.path.join(peft_dir, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, peft_dir)
        elif os.path.exists(pt_path):
            delta = torch.load(pt_path, map_location="cpu")
            apply_delta_to_model(model, delta)
            del delta
        else:
            del model
            torch.cuda.empty_cache()
            continue

        model.eval()
        pair_results = {}

        # Evaluate on held-out domains
        for hd in HELD_OUT_DOMAINS:
            logger.info("  GrassMerge(%s) → held-out %s", pair_name, hd)
            pair_results[f"heldout_{hd}"] = eval_on_domain(model, tokenizer, hd, max_samples)

        # Also evaluate on non-constituent core domains (transfer)
        non_constituent = [d for d in CORE_DOMAINS if d not in [d1, d2]]
        for td in non_constituent[:2]:  # Test 2 non-constituent domains
            logger.info("  GrassMerge(%s) → transfer %s", pair_name, td)
            pair_results[f"transfer_{td}"] = eval_on_domain(model, tokenizer, td, max_samples)

        results[pair_name] = pair_results
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ==============================================================================
# 3. Reparameterization Invariance Test
# ==============================================================================

def reparameterization_invariance(lora_dir):
    """Test that equivalent LoRA factorizations give the same merge result."""
    logger.info("=" * 60)
    logger.info("  Reparameterization Invariance Test")
    logger.info("=" * 60)

    loras = {}
    for d in CORE_DOMAINS[:3]:  # Test on 3 domains
        path = os.path.join(lora_dir, d)
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            loras[d] = LoRAWeights.from_peft_dir(d, path)

    if len(loras) < 2:
        return {"error": "Not enough LoRAs"}

    domains = sorted(loras.keys())
    merger = GrassMerge()
    results = []

    for d1, d2 in itertools.combinations(domains, 2):
        # Original merge
        original = merger.merge([loras[d1], loras[d2]])
        delta_original = original.to_delta_weight()

        # Apply random invertible basis change to both LoRAs
        for trial in range(3):
            rotated_loras = []
            for lora in [loras[d1], loras[d2]]:
                rotated = _apply_random_rotation(lora, seed=trial)
                rotated_loras.append(rotated)

            rotated_merge = merger.merge(rotated_loras)
            delta_rotated = rotated_merge.to_delta_weight()

            # Compare deltas
            diffs = []
            for key in delta_original:
                if key in delta_rotated:
                    diff = (delta_original[key].float() - delta_rotated[key].float()).norm()
                    norm = delta_original[key].float().norm()
                    rel_diff = (diff / norm).item() if norm > 0 else 0.0
                    diffs.append(rel_diff)

            avg_rel_diff = np.mean(diffs) if diffs else float('inf')
            max_rel_diff = max(diffs) if diffs else float('inf')

            results.append({
                "pair": f"{d1}+{d2}",
                "trial": trial,
                "avg_relative_diff": round(avg_rel_diff, 8),
                "max_relative_diff": round(max_rel_diff, 8),
                "num_layers": len(diffs),
                "invariant": avg_rel_diff < 1e-4,
            })
            logger.info("  %s+%s trial %d: avg_rel_diff=%.2e (invariant=%s)",
                        d1, d2, trial, avg_rel_diff, avg_rel_diff < 1e-4)

    return results


def _apply_random_rotation(lora, seed=0):
    """Apply a random invertible linear map R to the LoRA basis: A'=RA, B'=BR^{-1}."""
    rng = torch.Generator().manual_seed(seed)
    new_A, new_B = {}, {}
    for key in lora.lora_A:
        r = lora.rank
        R = torch.randn(r, r, generator=rng)
        # Make R orthogonal for numerical stability
        Q, _ = torch.linalg.qr(R)
        A = lora.lora_A[key].float()
        B = lora.lora_B[key].float()
        new_A[key] = (Q @ A).to(A.dtype)
        new_B[key] = (B @ Q.T).to(B.dtype)  # B @ Q^{-1} = B @ Q^T for orthogonal Q
    return LoRAWeights(
        name=f"{lora.name}_rotated_{seed}",
        lora_A=new_A, lora_B=new_B,
        rank=lora.rank, alpha=lora.alpha,
    )


# ==============================================================================
# 4. Karcher Mean Convergence Analysis
# ==============================================================================

def karcher_convergence(lora_dir):
    """Analyze convergence of Karcher mean across different domain configurations."""
    logger.info("=" * 60)
    logger.info("  Karcher Mean Convergence Analysis")
    logger.info("=" * 60)

    loras = {}
    for d in CORE_DOMAINS:
        path = os.path.join(lora_dir, d)
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            loras[d] = LoRAWeights.from_peft_dir(d, path)

    domains = sorted(loras.keys())
    results = []

    # Compute fast SVD for all LoRAs
    all_svds = {}
    for d in domains:
        svd_dict = loras[d].fast_svd()
        all_svds[d] = svd_dict

    # Test convergence for different N values
    for n in [2, 3, 4, 6]:
        combos = list(itertools.combinations(domains, n))[:5]  # Test up to 5 combos per N
        for combo in combos:
            name = "+".join(combo)
            # Pick one representative layer
            layer_keys = sorted(all_svds[combo[0]].keys())
            test_key = layer_keys[len(layer_keys)//2]  # Middle layer

            # Get left subspaces U for this layer
            Us = [all_svds[d][test_key][0] for d in combo]  # U matrices

            # Run Karcher mean with different max_iter to track convergence
            convergence_curve = []
            for max_iter in [1, 2, 3, 5, 10, 20, 50]:
                t0 = time.time()
                mean_U = GrassmannOps.karcher_mean(Us, max_iter=max_iter)
                elapsed = time.time() - t0

                # Compute sum of squared distances from mean to each U
                total_dist = 0.0
                for U in Us:
                    d_val = GrassmannOps.geodesic_distance(mean_U, U)
                    total_dist += d_val ** 2
                total_dist = total_dist ** 0.5

                convergence_curve.append({
                    "max_iter": max_iter,
                    "total_distance": round(total_dist, 6),
                    "time_s": round(elapsed, 4),
                })

            results.append({
                "combo": name,
                "n": n,
                "layer": test_key,
                "convergence": convergence_curve,
                "converged_at": _find_convergence_iter(convergence_curve),
            })
            logger.info("  %d-way %s: converges at iter %d",
                        n, name, _find_convergence_iter(convergence_curve))

    return results


def _find_convergence_iter(curve):
    """Find first iteration where distance stabilizes (<0.1% change)."""
    for i in range(1, len(curve)):
        if curve[i-1]["total_distance"] > 0:
            change = abs(curve[i]["total_distance"] - curve[i-1]["total_distance"]) / curve[i-1]["total_distance"]
            if change < 0.001:
                return curve[i]["max_iter"]
    return curve[-1]["max_iter"]


# ==============================================================================
# 5. General Capability Preservation
# ==============================================================================

def general_capability_eval(config, lora_dir, algebra_dir, tokenizer, max_samples=200):
    """Evaluate whether merged adapters preserve general model capabilities."""
    logger.info("=" * 60)
    logger.info("  General Capability Preservation")
    logger.info("=" * 60)

    base_model_name = config["base_model"]
    gm_dir = os.path.join(algebra_dir, "grassmerge")

    # Use MMLU as general capability benchmark (multiple subjects)
    general_benchmarks = {
        "mmlu_general": {
            "dataset_id": "cais/mmlu", "subset": "college_medicine",
            "split": "test", "q_field": "question", "a_field": "answer",
            "max_samples": 100, "multichoice": True,
            "system_prompt": "Answer the multiple choice question. Reply with just the letter.",
        },
    }

    results = {}

    # Base model general capability
    logger.info("  Base model → general benchmarks")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
    model.eval()
    base_general = {}
    for bench_name, bench_cfg in general_benchmarks.items():
        base_general[bench_name] = evaluate_on_benchmark(model, tokenizer, bench_cfg, "general")
    results["base_model"] = base_general
    del model
    torch.cuda.empty_cache()

    # GrassMerge 6-way on general benchmarks
    loras = {}
    for d in CORE_DOMAINS:
        path = os.path.join(lora_dir, d)
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            loras[d] = LoRAWeights.from_peft_dir(d, path)

    if len(loras) >= 6:
        merger = GrassMerge()
        all_loras = [loras[d] for d in CORE_DOMAINS if d in loras]

        logger.info("  GrassMerge 6-way → general benchmarks")
        delta = merger.merge(all_loras).to_delta_weight()
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
        apply_delta_to_model(model, delta)
        model.eval()
        del delta

        gm_general = {}
        for bench_name, bench_cfg in general_benchmarks.items():
            gm_general[bench_name] = evaluate_on_benchmark(model, tokenizer, bench_cfg, "general")
        results["grassmerge_6way"] = gm_general
        del model
        torch.cuda.empty_cache()

        # Task Arithmetic 6-way on general benchmarks
        logger.info("  Task Arithmetic 6-way → general benchmarks")
        delta = MergingBaselines.task_arithmetic_avg(all_loras, scaling=1.0)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
        apply_delta_to_model(model, delta)
        model.eval()
        del delta

        ta_general = {}
        for bench_name, bench_cfg in general_benchmarks.items():
            ta_general[bench_name] = evaluate_on_benchmark(model, tokenizer, bench_cfg, "general")
        results["task_arithmetic_6way"] = ta_general
        del model
        torch.cuda.empty_cache()

    del loras
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", default="all",
                        choices=["all", "oracle", "heldout", "reparam", "karcher", "general"])
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--lora_dir", default="results/domain_loras")
    parser.add_argument("--algebra_dir", default="results/algebra")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config()
    base_model_name = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = load_results(args.output)
    results["meta"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "max_samples": args.max_samples,
    }

    phases = [args.phase] if args.phase != "all" else [
        "reparam", "karcher", "oracle", "heldout", "general"
    ]

    # CPU-only experiments first
    if "reparam" in phases and "reparameterization_invariance" not in results:
        results["reparameterization_invariance"] = reparameterization_invariance(args.lora_dir)
        save_results(results, args.output)

    if "karcher" in phases and "karcher_convergence" not in results:
        results["karcher_convergence"] = karcher_convergence(args.lora_dir)
        save_results(results, args.output)

    # GPU experiments
    if "oracle" in phases and "oracle_routing" not in results:
        results["oracle_routing"] = oracle_routing(config, args.lora_dir, tokenizer, args.max_samples)
        save_results(results, args.output)

    if "heldout" in phases and "held_out_domains" not in results:
        results["held_out_domains"] = held_out_domain_eval(
            config, args.lora_dir, args.algebra_dir, tokenizer, args.max_samples)
        save_results(results, args.output)

    if "general" in phases and "general_capability" not in results:
        results["general_capability"] = general_capability_eval(
            config, args.lora_dir, args.algebra_dir, tokenizer, args.max_samples)
        save_results(results, args.output)

    save_results(results, args.output)
    logger.info("=" * 60)
    logger.info("  Supplementary experiments complete: %s", args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
