#!/usr/bin/env python3
"""Grassmannian Composition experiments: GrassMerge vs baselines on all domain pairs.

Experiments:
- GrassMerge (equal weights) on all C(12,2)=66 domain pairs
- Ablation A0: column-only Grassmann, SVD-Procrustes, parameter averaging
- Baselines: TIES, DARE, Task Arithmetic
- BGD matrix and interference analysis
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
    GrassmannOps,
    KnOTSMerge,
    LoRAAlgebra,
    LoRAWeights,
    MergingBaselines,
    SVDProcrustesMerge,
    TSPAMerge,
    bilateral_grassmann_distance,
    compute_bgd_matrix,
    cosine_interference,
    frobenius_interference,
    spectral_weighted_bgd,
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
            logger.warning("LoRA for '%s' not found at %s, skipping", domain, path)
            continue
        logger.info("Loading LoRA: %s", domain)
        loras[domain] = LoRAWeights.from_peft_dir(domain, path)
    return loras


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


def run_grassmerge_composition(merger: GrassMerge, loras: dict, output_dir: str, config: dict) -> dict:
    results = {}
    compose_dir = os.path.join(output_dir, "grassmerge")
    os.makedirs(compose_dir, exist_ok=True)
    domains = sorted(loras.keys())
    total_pairs = len(domains) * (len(domains) - 1) // 2
    logger.info("GrassMerge: composing all %d domain pairs...", total_pairs)

    for idx, (d1, d2) in enumerate(itertools.combinations(domains, 2)):
        name = f"{d1}+{d2}"
        logger.info("  [%d/%d] GrassMerge: %s", idx + 1, total_pairs, name)
        t0 = time.time()
        composed = merger.merge([loras[d1], loras[d2]], name=name)
        elapsed = time.time() - t0

        delta_c = composed.to_delta_weight()
        torch.save(delta_c, os.path.join(compose_dir, f"{name}.pt"))
        save_as_peft(composed, os.path.join(compose_dir, name), config)

        delta_a = loras[d1].to_delta_weight()
        delta_b = loras[d2].to_delta_weight()
        keys = sorted(set(delta_a.keys()) & set(delta_b.keys()) & set(delta_c.keys()))
        if keys:
            k = keys[0]
            cosine_a = torch.nn.functional.cosine_similarity(
                delta_c[k].flatten().unsqueeze(0), delta_a[k].flatten().unsqueeze(0)
            ).item()
            cosine_b = torch.nn.functional.cosine_similarity(
                delta_c[k].flatten().unsqueeze(0), delta_b[k].flatten().unsqueeze(0)
            ).item()
            bgd = bilateral_grassmann_distance(delta_a[k], delta_b[k], composed.rank)
        else:
            cosine_a, cosine_b, bgd = 0.0, 0.0, 0.0

        results[name] = {
            "method": "grassmerge",
            "domains": [d1, d2],
            "rank": composed.rank,
            "time_seconds": round(elapsed, 3),
            "cosine_to_d1": round(cosine_a, 4),
            "cosine_to_d2": round(cosine_b, 4),
            "bgd": round(bgd, 4),
        }

    return results


def run_ablation_a0(loras: dict, output_dir: str) -> dict:
    """A0: column-only vs bi-Grassmann vs Procrustes vs parameter-avg."""
    logger.info("=" * 50)
    logger.info("  Ablation A0: Representation Choice")
    logger.info("=" * 50)
    results = {}
    a0_dir = os.path.join(output_dir, "ablation_a0")
    os.makedirs(a0_dir, exist_ok=True)

    domains = sorted(loras.keys())
    pairs = list(itertools.combinations(domains, 2))

    methods = {
        "param_avg": lambda pair: _param_avg_merge(loras[pair[0]], loras[pair[1]]),
        "col_grassmann": lambda pair: ColumnOnlyGrassmannMerge().merge([loras[pair[0]], loras[pair[1]]]),
        "procrustes": lambda pair: SVDProcrustesMerge.merge([loras[pair[0]], loras[pair[1]]]),
        "knots": lambda pair: KnOTSMerge.merge([loras[pair[0]], loras[pair[1]]]),
        "tspa": lambda pair: TSPAMerge.merge([loras[pair[0]], loras[pair[1]]]),
        "grassmerge": lambda pair: GrassMerge().merge([loras[pair[0]], loras[pair[1]]]),
    }

    for method_name, merge_fn in methods.items():
        method_results = {}
        logger.info("  A0 method: %s (%d pairs)", method_name, len(pairs))
        for d1, d2 in pairs:
            name = f"{d1}+{d2}"
            t0 = time.time()
            composed = merge_fn((d1, d2))
            elapsed = time.time() - t0

            delta_c = composed.to_delta_weight()
            delta_a = loras[d1].to_delta_weight()
            delta_b = loras[d2].to_delta_weight()
            keys = sorted(set(delta_c.keys()) & set(delta_a.keys()) & set(delta_b.keys()))
            if keys:
                k = keys[0]
                cos_a = torch.nn.functional.cosine_similarity(
                    delta_c[k].flatten().unsqueeze(0), delta_a[k].flatten().unsqueeze(0)
                ).item()
                cos_b = torch.nn.functional.cosine_similarity(
                    delta_c[k].flatten().unsqueeze(0), delta_b[k].flatten().unsqueeze(0)
                ).item()
            else:
                cos_a, cos_b = 0.0, 0.0

            method_results[name] = {
                "cosine_to_d1": round(cos_a, 4),
                "cosine_to_d2": round(cos_b, 4),
                "time_seconds": round(elapsed, 3),
                "rank": composed.rank,
            }
        results[method_name] = method_results

    return results


def _param_avg_merge(lora_a: LoRAWeights, lora_b: LoRAWeights) -> LoRAWeights:
    delta_a = lora_a.to_delta_weight()
    delta_b = lora_b.to_delta_weight()
    rank = max(lora_a.rank, lora_b.rank)
    new_A, new_B = {}, {}
    for key in set(delta_a.keys()) & set(delta_b.keys()):
        avg = (delta_a[key] + delta_b[key]) / 2.0
        U, S, Vh = torch.linalg.svd(avg.float(), full_matrices=False)
        r = min(rank, U.shape[1])
        new_B[key] = U[:, :r] @ torch.diag(torch.sqrt(S[:r]))
        new_A[key] = torch.diag(torch.sqrt(S[:r])) @ Vh[:r, :]
    return LoRAWeights(name="param_avg", lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


def run_bgd_analysis(loras: dict, output_dir: str) -> dict:
    """Compute BGD and alternative interference metrics for all pairs."""
    logger.info("Computing interference metrics (BGD, spectral-BGD, cosine, Frobenius)...")
    names = sorted(loras.keys())
    N = len(names)
    all_deltas = {k: loras[k].to_delta_weight() for k in names}
    rank = max(lora.rank for lora in loras.values())

    bgd_matrix = np.zeros((N, N))
    sbgd_matrix = np.zeros((N, N))
    cosine_matrix = np.zeros((N, N))
    frob_matrix = np.zeros((N, N))

    common_keys = set(all_deltas[names[0]].keys())
    for n in names[1:]:
        common_keys &= set(all_deltas[n].keys())
    sample_keys = sorted(common_keys)[:5]

    for i in range(N):
        for j in range(i + 1, N):
            bgd_vals, sbgd_vals, cos_vals, frob_vals = [], [], [], []
            for key in sample_keys:
                di = all_deltas[names[i]][key]
                dj = all_deltas[names[j]][key]
                bgd_vals.append(bilateral_grassmann_distance(di, dj, rank))
                sbgd_vals.append(spectral_weighted_bgd(di, dj, rank))
                cos_vals.append(cosine_interference(di, dj))
                frob_vals.append(frobenius_interference(di, dj))
            bgd_matrix[i][j] = bgd_matrix[j][i] = float(np.mean(bgd_vals))
            sbgd_matrix[i][j] = sbgd_matrix[j][i] = float(np.mean(sbgd_vals))
            cosine_matrix[i][j] = cosine_matrix[j][i] = float(np.mean(cos_vals))
            frob_matrix[i][j] = frob_matrix[j][i] = float(np.mean(frob_vals))

    result = {
        "domain_names": names,
        "bgd_matrix": bgd_matrix.tolist(),
        "spectral_bgd_matrix": sbgd_matrix.tolist(),
        "cosine_interference_matrix": cosine_matrix.tolist(),
        "frobenius_interference_matrix": frob_matrix.tolist(),
    }
    with open(os.path.join(output_dir, "interference_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    logger.info("BGD matrix (first 5x5):\n%s",
                np.array2string(bgd_matrix[:5, :5], precision=3, suppress_small=True))
    return result


def _delta_dict_to_lora(delta: dict, name: str, rank: int) -> LoRAWeights:
    """Factorize a delta-weight dict into LoRA A/B factors via SVD."""
    new_A, new_B = {}, {}
    for key, d in delta.items():
        U, S, Vh = torch.linalg.svd(d.float(), full_matrices=False)
        r = min(rank, len(S))
        new_B[key] = U[:, :r] @ torch.diag(torch.sqrt(S[:r]))
        new_A[key] = torch.diag(torch.sqrt(S[:r])) @ Vh[:r, :]
    return LoRAWeights(name=name, lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


def run_pairwise_baselines(loras: dict, output_dir: str, config: dict) -> dict:
    """Run baselines pairwise to match GrassMerge evaluation protocol."""
    results = {}
    baseline_dir = os.path.join(output_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)
    domains = sorted(loras.keys())
    pairs = list(itertools.combinations(domains, 2))

    baseline_configs = [
        ("task_arithmetic", MergingBaselines.task_arithmetic_avg, {"scaling": 1.0}, False),
        ("ties_d0.5", MergingBaselines.ties_merging, {"density": 0.5, "scaling": 1.0}, False),
        ("dare_p0.5", MergingBaselines.dare_merging, {"drop_rate": 0.5, "scaling": 1.0}, False),
        ("knots", lambda pair_loras, **kw: KnOTSMerge.merge(pair_loras), {}, True),
        ("tspa", lambda pair_loras, **kw: TSPAMerge.merge(pair_loras), {}, True),
    ]

    for method_name, method_fn, kwargs, returns_lora_weights in baseline_configs:
        method_dir = os.path.join(baseline_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        method_results = {}
        logger.info("  Baseline: %s (%d pairs)", method_name, len(pairs))

        for d1, d2 in pairs:
            name = f"{d1}+{d2}"
            t0 = time.time()
            merged = method_fn([loras[d1], loras[d2]], **kwargs)
            elapsed = time.time() - t0

            if returns_lora_weights:
                delta = merged.to_delta_weight()
                save_as_peft(merged, os.path.join(method_dir, name), config)
            else:
                delta = merged
                rank = max(loras[d1].rank, loras[d2].rank)
                lora_w = _delta_dict_to_lora(delta, name, rank)
                save_as_peft(lora_w, os.path.join(method_dir, name), config)
            torch.save(delta, os.path.join(method_dir, f"{name}.pt"))
            method_results[name] = {
                "pair": [d1, d2],
                "num_layers": len(delta),
                "total_params": sum(v.numel() for v in delta.values()),
                "time_seconds": round(elapsed, 3),
            }
        results[method_name] = method_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Grassmannian Composition — Full Experiment Suite")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "domains.yaml"))
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/algebra")
    parser.add_argument("--skip_grassmerge", action="store_true")
    parser.add_argument("--skip_ablation_a0", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_bgd", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    domains = list(config["domains"].keys())
    logger.info("Loading %d domain LoRAs from %s", len(domains), args.lora_dir)
    loras = load_all_loras(args.lora_dir, domains)

    if len(loras) < 2:
        logger.error("Need at least 2 trained LoRAs, found %d", len(loras))
        sys.exit(1)
    logger.info("Loaded %d LoRAs: %s", len(loras), sorted(loras.keys()))

    merger = GrassMerge()
    all_results = {"meta": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domains_loaded": sorted(loras.keys()),
        "method": "grassmannian_composition",
    }}

    if not args.skip_grassmerge:
        logger.info("=" * 50)
        logger.info("  PHASE 1: GrassMerge Composition")
        logger.info("=" * 50)
        all_results["grassmerge"] = run_grassmerge_composition(merger, loras, args.output_dir, config)

    if not args.skip_ablation_a0:
        logger.info("=" * 50)
        logger.info("  PHASE 2: Ablation A0 — Representation Choice")
        logger.info("=" * 50)
        all_results["ablation_a0"] = run_ablation_a0(loras, args.output_dir)

    if not args.skip_bgd:
        logger.info("=" * 50)
        logger.info("  PHASE 3: BGD Analysis")
        logger.info("=" * 50)
        all_results["bgd"] = run_bgd_analysis(loras, args.output_dir)

    if not args.skip_baselines:
        logger.info("=" * 50)
        logger.info("  PHASE 4: Pairwise Baseline Comparisons")
        logger.info("=" * 50)
        all_results["baselines"] = run_pairwise_baselines(loras, args.output_dir, config)

    results_path = os.path.join(args.output_dir, "all_algebra_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("  All experiments complete")
    logger.info("  Results: %s", results_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
