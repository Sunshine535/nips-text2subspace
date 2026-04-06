#!/usr/bin/env python3
"""Interpolation experiment: geodesic vs linear interpolation smoothness (H2).

For each domain pair, sweep t ∈ [0.0, 0.1, ..., 1.0] along:
  (a) Grassmann geodesic interpolation
  (b) Linear interpolation in weight space

Evaluate at each t on both source tasks → smoothness curves.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lora_algebra import GrassMerge, GrassmannOps, LoRAWeights
from scripts.eval_domain_accuracy import (
    DOMAIN_BENCHMARKS,
    evaluate_on_benchmark,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def grassmann_interpolate(lora_a: LoRAWeights, lora_b: LoRAWeights, t: float, rank: int) -> LoRAWeights:
    """Interpolate along the Grassmann geodesic at parameter t."""
    svd_a = lora_a.fast_svd()
    svd_b = lora_b.fast_svd()
    common_keys = sorted(set(svd_a.keys()) & set(svd_b.keys()))

    new_A, new_B = {}, {}
    for key in common_keys:
        U_a, S_a, Vh_a = svd_a[key]
        U_b, S_b, Vh_b = svd_b[key]
        r = min(rank, U_a.shape[1], U_b.shape[1])

        # Geodesic on left Grassmannian G(r, d_out)
        U_t = GrassmannOps.geodesic_midpoint(U_a[:, :r], U_b[:, :r], t=t)
        # Geodesic on right Grassmannian G(r, d_in)
        V_t = GrassmannOps.geodesic_midpoint(Vh_a[:r, :].T, Vh_b[:r, :].T, t=t)

        # Interpolate core: S_t = (1-t)*S_a + t*S_b projected into the new basis
        # Use projected-core: project both LoRAs into the interpolated basis
        B_a, A_a = lora_a.lora_B.get(key), lora_a.lora_A.get(key)
        B_b, A_b = lora_b.lora_B.get(key), lora_b.lora_A.get(key)
        if B_a is None or A_a is None or B_b is None or A_b is None:
            continue

        S_a_proj = (U_t.T @ B_a.float()) @ (A_a.float() @ V_t) * lora_a.alpha
        S_b_proj = (U_t.T @ B_b.float()) @ (A_b.float() @ V_t) * lora_b.alpha
        S_t = (1 - t) * S_a_proj + t * S_b_proj

        U_core, sigma, Vh_core = torch.linalg.svd(S_t, full_matrices=False)
        U_final = U_t @ U_core
        V_final = V_t @ Vh_core.T
        sqrt_s = torch.sqrt(sigma.clamp(min=0))
        new_B[key] = U_final @ torch.diag(sqrt_s)
        new_A[key] = torch.diag(sqrt_s) @ V_final.T

    return LoRAWeights(name=f"geodesic_t{t:.2f}", lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


def linear_interpolate(lora_a: LoRAWeights, lora_b: LoRAWeights, t: float, rank: int) -> LoRAWeights:
    """Linear interpolation in weight space: (1-t)*ΔW_a + t*ΔW_b, then refactorize."""
    delta_a = lora_a.to_delta_weight()
    delta_b = lora_b.to_delta_weight()
    common_keys = sorted(set(delta_a.keys()) & set(delta_b.keys()))

    new_A, new_B = {}, {}
    for key in common_keys:
        interp = (1 - t) * delta_a[key].float() + t * delta_b[key].float()
        U, S, Vh = torch.linalg.svd(interp, full_matrices=False)
        r = min(rank, len(S))
        new_B[key] = U[:, :r] @ torch.diag(torch.sqrt(S[:r]))
        new_A[key] = torch.diag(torch.sqrt(S[:r])) @ Vh[:r, :]
    return LoRAWeights(name=f"linear_t{t:.2f}", lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


def apply_lora_weights_to_model(base_model, lora_w: LoRAWeights):
    """Apply LoRA delta weights directly to a base model's state dict."""
    delta = lora_w.to_delta_weight()
    state = base_model.state_dict()
    for key, d in delta.items():
        for suffix in ["", ".weight"]:
            full_key = key + suffix
            if full_key in state:
                state[full_key] = state[full_key].float() + d.float()
                state[full_key] = state[full_key].to(torch.bfloat16)
                break
    base_model.load_state_dict(state)
    return base_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/domains.yaml")
    parser.add_argument("--lora_dir", default="results/domain_loras")
    parser.add_argument("--output_dir", default="results/interpolation")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--t_steps", type=int, default=11, help="Number of interpolation steps")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Domain pairs like 'history+science'. Default: 5 representative pairs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    base_model_name = config["base_model"]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load LoRA adapters
    loras = {}
    for domain in ["math", "code", "medical", "science", "history", "philosophy"]:
        path = os.path.join(args.lora_dir, domain)
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            loras[domain] = LoRAWeights.from_peft_dir(domain, path)

    # Select pairs
    if args.pairs:
        pairs = [tuple(p.split("+")) for p in args.pairs]
    else:
        # Representative pairs: close (hist+phil), medium (hist+sci), far (code+phil)
        pairs = [
            ("history", "philosophy"),
            ("history", "science"),
            ("philosophy", "science"),
            ("history", "medical"),
            ("math", "science"),
        ]
    pairs = [(a, b) for a, b in pairs if a in loras and b in loras]

    t_values = np.linspace(0.0, 1.0, args.t_steps).tolist()
    rank = max(l.rank for l in loras.values())

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_model": base_model_name,
            "max_samples": args.max_samples,
            "t_values": t_values,
            "pairs": [f"{a}+{b}" for a, b in pairs],
        },
        "curves": {},
    }

    for pair_idx, (d1, d2) in enumerate(pairs):
        pair_name = f"{d1}+{d2}"
        logger.info("=" * 60)
        logger.info("  Pair [%d/%d]: %s", pair_idx + 1, len(pairs), pair_name)
        logger.info("=" * 60)

        geodesic_curve = {"d1_acc": [], "d2_acc": [], "t": []}
        linear_curve = {"d1_acc": [], "d2_acc": [], "t": []}

        for t in t_values:
            logger.info("  t=%.2f", t)

            # Geodesic interpolation
            geo_lora = grassmann_interpolate(loras[d1], loras[d2], t, rank)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
            )
            model.eval()
            model = apply_lora_weights_to_model(model, geo_lora)

            r1 = {}
            for bench_name, bench_cfg in DOMAIN_BENCHMARKS.get(d1, {}).items():
                if not bench_cfg.get("synthetic"):
                    r1[bench_name] = evaluate_on_benchmark(
                        model, tokenizer, {**bench_cfg, "max_samples": args.max_samples}, d1
                    )
            r2 = {}
            for bench_name, bench_cfg in DOMAIN_BENCHMARKS.get(d2, {}).items():
                if not bench_cfg.get("synthetic"):
                    r2[bench_name] = evaluate_on_benchmark(
                        model, tokenizer, {**bench_cfg, "max_samples": args.max_samples}, d2
                    )

            acc1 = np.mean([v["accuracy"] for v in r1.values()]) if r1 else 0.0
            acc2 = np.mean([v["accuracy"] for v in r2.values()]) if r2 else 0.0
            geodesic_curve["d1_acc"].append(float(acc1))
            geodesic_curve["d2_acc"].append(float(acc2))
            geodesic_curve["t"].append(t)
            logger.info("    geodesic: %s=%.3f, %s=%.3f", d1, acc1, d2, acc2)
            del model
            torch.cuda.empty_cache()

            # Linear interpolation
            lin_lora = linear_interpolate(loras[d1], loras[d2], t, rank)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
            )
            model.eval()
            model = apply_lora_weights_to_model(model, lin_lora)

            r1 = {}
            for bench_name, bench_cfg in DOMAIN_BENCHMARKS.get(d1, {}).items():
                if not bench_cfg.get("synthetic"):
                    r1[bench_name] = evaluate_on_benchmark(
                        model, tokenizer, {**bench_cfg, "max_samples": args.max_samples}, d1
                    )
            r2 = {}
            for bench_name, bench_cfg in DOMAIN_BENCHMARKS.get(d2, {}).items():
                if not bench_cfg.get("synthetic"):
                    r2[bench_name] = evaluate_on_benchmark(
                        model, tokenizer, {**bench_cfg, "max_samples": args.max_samples}, d2
                    )

            acc1 = np.mean([v["accuracy"] for v in r1.values()]) if r1 else 0.0
            acc2 = np.mean([v["accuracy"] for v in r2.values()]) if r2 else 0.0
            linear_curve["d1_acc"].append(float(acc1))
            linear_curve["d2_acc"].append(float(acc2))
            linear_curve["t"].append(t)
            logger.info("    linear:   %s=%.3f, %s=%.3f", d1, acc1, d2, acc2)
            del model
            torch.cuda.empty_cache()

        all_results["curves"][pair_name] = {
            "domains": [d1, d2],
            "geodesic": geodesic_curve,
            "linear": linear_curve,
        }

        # Save after each pair
        results_path = os.path.join(args.output_dir, "interpolation_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    logger.info("=" * 60)
    logger.info("  Interpolation experiments complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
