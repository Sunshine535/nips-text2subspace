#!/usr/bin/env python3
"""Ablation studies with DOWNSTREAM ACCURACY evaluation.

Unlike run_ablations.py (proxy metrics only), this script evaluates merged
adapters on actual domain benchmarks (MMLU, etc.) to measure real accuracy.

Ablations:
1. Rank ablation with downstream accuracy (history+philosophy pair)
2. N-way composition with downstream accuracy (N=2,3,4,6)
3. Geometry isolation ablation (GrassMerge vs baselines on history+philosophy)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lora_algebra import (
    ColumnOnlyGrassmannMerge,
    GrassMerge,
    LoRAAlgebra,
    LoRAWeights,
    MergingBaselines,
)
from scripts.eval_domain_accuracy import (
    DOMAIN_BENCHMARKS,
    evaluate_on_benchmark,
    load_model_with_adapter,
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


def save_as_peft(composed: LoRAWeights, peft_dir: str, config: dict) -> None:
    """Save a composed LoRAWeights as a PEFT-loadable directory."""
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
        "lora_dropout": lora_cfg.get("lora_dropout", 0.0),
        "bias": "none",
        "task_type": lora_cfg.get("task_type", "CAUSAL_LM"),
    }
    with open(os.path.join(peft_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)


def delta_to_lora_weights(delta: dict, rank: int, name: str) -> LoRAWeights:
    """Convert a delta-weight dict (from MergingBaselines) to LoRAWeights via SVD."""
    new_A, new_B = {}, {}
    for key, dW in delta.items():
        U, S, Vh = torch.linalg.svd(dW.float(), full_matrices=False)
        r = min(rank, U.shape[1])
        sqrt_S = torch.sqrt(S[:r].clamp(min=0))
        new_B[key] = U[:, :r] @ torch.diag(sqrt_S)
        new_A[key] = torch.diag(sqrt_S) @ Vh[:r, :]
    return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


def evaluate_adapter_on_domains(
    adapter_path: str,
    domains: list,
    base_model_name: str,
    tokenizer,
    max_samples: int,
) -> dict:
    """Load a PEFT adapter and evaluate on the given domains' benchmarks."""
    model = load_model_with_adapter(base_model_name, adapter_path)
    results = {}
    for domain in domains:
        if domain not in DOMAIN_BENCHMARKS:
            logger.warning("No benchmark defined for domain '%s', skipping", domain)
            continue
        domain_results = {}
        for bench_name, bench_cfg in DOMAIN_BENCHMARKS[domain].items():
            cfg = {**bench_cfg, "max_samples": max_samples}
            logger.info("    Evaluating %s/%s", domain, bench_name)
            domain_results[bench_name] = evaluate_on_benchmark(model, tokenizer, cfg, domain)
        results[domain] = domain_results
    del model
    torch.cuda.empty_cache()
    return results


# ===== Ablation 1: Rank with Downstream Accuracy =====

def ablation_rank_downstream(
    loras: dict,
    config: dict,
    tokenizer,
    output_dir: str,
    max_samples: int,
) -> dict:
    """Rank ablation on history+philosophy pair with downstream MMLU accuracy."""
    logger.info("=== Ablation 1: Rank with Downstream Accuracy ===")
    results = {}
    merger = GrassMerge()
    base_model_name = config["base_model"]

    pair = ["history", "philosophy"]
    for d in pair:
        if d not in loras:
            logger.error("Domain '%s' not found in loaded LoRAs, cannot run rank ablation", d)
            return {}

    ranks = [4, 8, 16, 32]
    for rank in ranks:
        logger.info("  Rank=%d", rank)
        lora1_r = refactorize_at_rank(loras[pair[0]], rank)
        lora2_r = refactorize_at_rank(loras[pair[1]], rank)

        t0 = time.time()
        composed = merger.merge([lora1_r, lora2_r], name=f"r{rank}_{pair[0]}+{pair[1]}")
        merge_time = time.time() - t0

        # Save as PEFT adapter
        peft_dir = os.path.join(output_dir, "rank_ablation", f"rank_{rank}")
        # Override config rank for this ablation
        cfg_override = {**config, "lora": {**config.get("lora", {}), "r": rank}}
        save_as_peft(composed, peft_dir, cfg_override)

        # Evaluate on both constituent domains
        eval_results = evaluate_adapter_on_domains(
            peft_dir, pair, base_model_name, tokenizer, max_samples,
        )

        results[f"rank_{rank}"] = {
            "rank": rank,
            "pair": pair,
            "merge_time": round(merge_time, 4),
            "downstream_accuracy": eval_results,
        }
        logger.info("  Rank=%d done: %s", rank, {
            d: {b: m.get("accuracy", -1) for b, m in benches.items()}
            for d, benches in eval_results.items()
        })

    return results


# ===== Ablation 2: N-way Composition with Downstream Accuracy =====

def ablation_nway_downstream(
    loras: dict,
    config: dict,
    tokenizer,
    output_dir: str,
    max_samples: int,
) -> dict:
    """N-way composition ablation with downstream accuracy on all N constituent domains."""
    logger.info("=== Ablation 2: N-way Composition with Downstream Accuracy ===")
    results = {}
    merger = GrassMerge()
    base_model_name = config["base_model"]
    domains = sorted(loras.keys())

    counts = [2, 3, 4, 6]
    for n in counts:
        if n > len(domains):
            logger.warning("  Requested %d domains but only %d available, skipping", n, len(domains))
            continue

        selected = domains[:n]
        logger.info("  N=%d, domains=%s", n, selected)

        t0 = time.time()
        selected_loras = [loras[d] for d in selected]
        composed = merger.merge(selected_loras, name=f"grassmerge_{n}d")
        merge_time = time.time() - t0

        # Save as PEFT adapter
        peft_dir = os.path.join(output_dir, "nway_ablation", f"n_{n}")
        save_as_peft(composed, peft_dir, config)

        # Evaluate on ALL N constituent domains
        eval_results = evaluate_adapter_on_domains(
            peft_dir, selected, base_model_name, tokenizer, max_samples,
        )

        # Compute average accuracy across domains
        all_accs = []
        per_domain_acc = {}
        for d, benches in eval_results.items():
            for bench_name, metrics in benches.items():
                acc = metrics.get("accuracy", -1)
                if acc >= 0:
                    all_accs.append(acc)
                    per_domain_acc[d] = acc

        avg_accuracy = round(float(np.mean(all_accs)), 4) if all_accs else 0.0

        results[f"n_{n}"] = {
            "num_domains": n,
            "domains": selected,
            "merge_time": round(merge_time, 4),
            "per_domain_accuracy": per_domain_acc,
            "average_accuracy": avg_accuracy,
            "downstream_accuracy": eval_results,
        }
        logger.info("  N=%d done: per-domain=%s, avg=%.4f", n, per_domain_acc, avg_accuracy)

    return results


# ===== Ablation 3: Geometry Isolation with Downstream Accuracy =====

def ablation_geometry_downstream(
    loras: dict,
    config: dict,
    tokenizer,
    output_dir: str,
    max_samples: int,
) -> dict:
    """Geometry isolation ablation: compare merge methods on history+philosophy."""
    logger.info("=== Ablation 3: Geometry Isolation with Downstream Accuracy ===")
    results = {}
    base_model_name = config["base_model"]

    pair = ["history", "philosophy"]
    for d in pair:
        if d not in loras:
            logger.error("Domain '%s' not found in loaded LoRAs, cannot run geometry ablation", d)
            return {}

    lora_list = [loras[pair[0]], loras[pair[1]]]
    rank = max(lora.rank for lora in lora_list)

    methods = {}

    # 1. GrassMerge (our method)
    logger.info("  Method: GrassMerge")
    merger = GrassMerge()
    methods["grassmerge"] = merger.merge(lora_list, name="grassmerge")

    # 2. Euclidean midpoint + orthogonalize (SVD re-project)
    logger.info("  Method: Euclidean midpoint + SVD re-project")
    delta_a = lora_list[0].to_delta_weight()
    delta_b = lora_list[1].to_delta_weight()
    euclid_delta = {}
    new_A, new_B = {}, {}
    common_keys = sorted(set(delta_a.keys()) & set(delta_b.keys()))
    for key in common_keys:
        midpoint = (delta_a[key].float() + delta_b[key].float()) / 2.0
        U, S, Vh = torch.linalg.svd(midpoint, full_matrices=False)
        r = min(rank, U.shape[1])
        sqrt_S = torch.sqrt(S[:r].clamp(min=0))
        new_B[key] = U[:, :r] @ torch.diag(sqrt_S)
        new_A[key] = torch.diag(sqrt_S) @ Vh[:r, :]
    methods["euclidean_svd_reproject"] = LoRAWeights(
        name="euclidean_svd_reproject", lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0,
    )

    # 3. Parameter averaging (simple (A+B)/2 on lora_A and lora_B directly)
    logger.info("  Method: Parameter averaging")
    avg_A, avg_B = {}, {}
    common_keys_ab = sorted(
        set(lora_list[0].lora_A.keys()) & set(lora_list[1].lora_A.keys())
        & set(lora_list[0].lora_B.keys()) & set(lora_list[1].lora_B.keys())
    )
    for key in common_keys_ab:
        avg_A[key] = (lora_list[0].lora_A[key].float() + lora_list[1].lora_A[key].float()) / 2.0
        avg_B[key] = (lora_list[0].lora_B[key].float() + lora_list[1].lora_B[key].float()) / 2.0
    methods["parameter_averaging"] = LoRAWeights(
        name="parameter_averaging", lora_A=avg_A, lora_B=avg_B, rank=rank, alpha=1.0,
    )

    # 4. Column-only Grassmann (right factor only)
    logger.info("  Method: Column-only Grassmann")
    col_merger = ColumnOnlyGrassmannMerge()
    methods["column_only_grassmann"] = col_merger.merge(lora_list, name="col_grassmann")

    # Save each method and evaluate
    for method_name, composed in methods.items():
        logger.info("  Evaluating method: %s", method_name)
        peft_dir = os.path.join(output_dir, "geometry_ablation", method_name)
        save_as_peft(composed, peft_dir, config)

        eval_results = evaluate_adapter_on_domains(
            peft_dir, pair, base_model_name, tokenizer, max_samples,
        )

        all_accs = []
        per_domain_acc = {}
        for d, benches in eval_results.items():
            for bench_name, metrics in benches.items():
                acc = metrics.get("accuracy", -1)
                if acc >= 0:
                    all_accs.append(acc)
                    per_domain_acc[d] = acc

        avg_accuracy = round(float(np.mean(all_accs)), 4) if all_accs else 0.0

        results[method_name] = {
            "method": method_name,
            "pair": pair,
            "per_domain_accuracy": per_domain_acc,
            "average_accuracy": avg_accuracy,
            "downstream_accuracy": eval_results,
        }
        logger.info("  %s: per-domain=%s, avg=%.4f", method_name, per_domain_acc, avg_accuracy)

    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation studies with downstream accuracy evaluation")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "domains.yaml"))
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory with trained domain LoRAs")
    parser.add_argument("--algebra_dir", type=str, default=None, help="Directory with algebra experiment outputs")
    parser.add_argument("--output_dir", type=str, default="results/ablations_downstream")
    parser.add_argument("--max_samples", type=int, default=200, help="Max evaluation samples per benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    base_model_name = config["base_model"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    domains = list(config["domains"].keys())
    logger.info("Loading LoRAs from %s", args.lora_dir)
    loras = load_all_loras(args.lora_dir, domains)

    if len(loras) < 2:
        logger.error("Need at least 2 trained LoRAs, found %d", len(loras))
        sys.exit(1)
    logger.info("Loaded %d LoRAs: %s", len(loras), sorted(loras.keys()))

    all_results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "domains_loaded": sorted(loras.keys()),
            "max_samples": args.max_samples,
            "seed": args.seed,
        }
    }

    # Ablation 1: Rank with downstream accuracy
    all_results["rank_ablation"] = ablation_rank_downstream(
        loras, config, tokenizer, args.output_dir, args.max_samples,
    )

    # Ablation 2: N-way composition with downstream accuracy
    all_results["nway_ablation"] = ablation_nway_downstream(
        loras, config, tokenizer, args.output_dir, args.max_samples,
    )

    # Ablation 3: Geometry isolation with downstream accuracy
    all_results["geometry_ablation"] = ablation_geometry_downstream(
        loras, config, tokenizer, args.output_dir, args.max_samples,
    )

    results_path = os.path.join(args.output_dir, "ablation_downstream_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("  Downstream ablation study complete")
    logger.info("  Rank ablation: %d configs", len(all_results["rank_ablation"]))
    logger.info("  N-way ablation: %d configs", len(all_results["nway_ablation"]))
    logger.info("  Geometry ablation: %d methods", len(all_results["geometry_ablation"]))
    logger.info("  Results: %s", results_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
