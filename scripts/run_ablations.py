#!/usr/bin/env python3
"""Ablation study for LoRA algebra experiments.

Ablations:
1. LoRA rank: r in [4, 8, 16, 32, 64]
2. Normalization: with/without Grassmann normalization
3. Interpolation type: linear vs geodesic vs chord
4. Number of domains composed: 2, 3, 4, 6, 12
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
    GrassMerge,
    GrassmannOps,
    GrassmannProjector,
    LoRAAlgebra,
    LoRAWeights,
    MergingBaselines,
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
    """Re-factorize a LoRA at a different rank using fast SVD (no full d×d matrix)."""
    svd_data = lora.fast_svd()  # Uses QR+compact SVD, O(d*r²) not O(d²*r)
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


def measure_composition_quality(lora_a: LoRAWeights, lora_b: LoRAWeights, composed: LoRAWeights) -> dict:
    """Measure how well composition preserves both domains."""
    delta_a = lora_a.to_delta_weight()
    delta_b = lora_b.to_delta_weight()
    delta_c = composed.to_delta_weight()
    keys = sorted(set(delta_a.keys()) & set(delta_b.keys()) & set(delta_c.keys()))
    if not keys:
        return {"cosine_a": 0.0, "cosine_b": 0.0, "reconstruction_error": 1.0}

    cos_a_vals, cos_b_vals, err_vals = [], [], []
    for k in keys[:10]:  # Sample 10 layers for speed
        cos_a_vals.append(torch.nn.functional.cosine_similarity(
            delta_c[k].flatten().unsqueeze(0), delta_a[k].flatten().unsqueeze(0),
        ).item())
        cos_b_vals.append(torch.nn.functional.cosine_similarity(
            delta_c[k].flatten().unsqueeze(0), delta_b[k].flatten().unsqueeze(0),
        ).item())
        ideal = delta_a[k] + delta_b[k]
        err_vals.append(torch.norm(delta_c[k] - ideal, "fro").item() / max(torch.norm(ideal, "fro").item(), 1e-8))
    import numpy as _np
    return {"cosine_a": round(float(_np.mean(cos_a_vals)), 4), "cosine_b": round(float(_np.mean(cos_b_vals)), 4), "reconstruction_error": round(float(_np.mean(err_vals)), 4)}


# ===== Ablation 1: LoRA Rank =====

def ablation_rank(loras: dict, ranks: list, output_dir: str) -> dict:
    """Ablation over LoRA ranks: GrassMerge composition quality at different ranks."""
    logger.info("=== Ablation: LoRA Rank ===")
    results = {}
    merger = GrassMerge()
    pair = list(sorted(loras.keys()))[:2]
    if len(pair) < 2:
        return {}

    d1, d2 = pair
    for rank in ranks:
        logger.info("  Rank=%d", rank)
        lora1_r = refactorize_at_rank(loras[d1], rank)
        lora2_r = refactorize_at_rank(loras[d2], rank)

        t0 = time.time()
        composed = merger.merge([lora1_r, lora2_r], name=f"r{rank}_{d1}+{d2}")
        elapsed = time.time() - t0

        quality = measure_composition_quality(lora1_r, lora2_r, composed)
        delta = composed.to_delta_weight()
        fro_norm = sum(torch.norm(v, "fro").item() for v in delta.values())

        results[f"rank_{rank}"] = {
            "rank": rank,
            "pair": [d1, d2],
            "compose_time": round(elapsed, 4),
            "frobenius_norm": round(fro_norm, 4),
            **quality,
        }

    return results


# ===== Ablation 2: Grassmann Normalization =====

def ablation_normalization(loras: dict, output_dir: str) -> dict:
    """Ablation: Grassmann-normalized vs raw composition."""
    logger.info("=== Ablation: Normalization ===")
    results = {}
    domains = sorted(loras.keys())
    pairs = list(itertools.combinations(domains, 2))[:10]

    for d1, d2 in pairs:
        name = f"{d1}+{d2}"

        # Raw composition (no Grassmann)
        delta_a = loras[d1].to_delta_weight()
        delta_b = loras[d2].to_delta_weight()
        keys = sorted(set(delta_a.keys()) & set(delta_b.keys()))
        raw_norms = []
        for k in keys[:5]:
            raw = delta_a[k] + delta_b[k]
            raw_norms.append(torch.norm(raw, "fro").item())

        # Grassmann-normalized composition
        projector = GrassmannProjector(svd_rank=16)
        grassmann_dists = []
        for k in keys[:5]:
            U_a = projector.to_grassmann(delta_a[k])
            U_b = projector.to_grassmann(delta_b[k])
            dist = projector.grassmann_distance(U_a, U_b)
            grassmann_dists.append(dist)

        # SVD-renormalized composition
        svd_norms = []
        for k in keys[:5]:
            combined = delta_a[k] + delta_b[k]
            U, S, Vh = torch.linalg.svd(combined.float(), full_matrices=False)
            svd_norms.append(S[:16].sum().item())

        results[name] = {
            "pair": [d1, d2],
            "raw_avg_norm": round(np.mean(raw_norms), 4) if raw_norms else 0.0,
            "grassmann_avg_distance": round(np.mean(grassmann_dists), 4) if grassmann_dists else 0.0,
            "svd_avg_energy": round(np.mean(svd_norms), 4) if svd_norms else 0.0,
        }

    return results


# ===== Ablation 3: Interpolation Type =====

def ablation_interpolation_type(loras: dict, output_dir: str) -> dict:
    """Ablation: linear vs geodesic vs chord interpolation."""
    logger.info("=== Ablation: Interpolation Type ===")
    results = {}
    algebra = LoRAAlgebra(grassmann_rank=32)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    pair = list(sorted(loras.keys()))[:2]
    if len(pair) < 2:
        return {}
    d1, d2 = pair

    for alpha in alphas:
        linear = LoRAAlgebra.interpolate(loras[d1], loras[d2], alpha=alpha, name="linear")
        delta_lin = linear.to_delta_weight()

        geodesic = algebra.grassmann_interpolate(loras[d1], loras[d2], t=alpha, name="geodesic")
        delta_geo = geodesic.to_delta_weight()

        # Chord (midpoint in ambient space, then project back)
        delta_a = loras[d1].to_delta_weight()
        delta_b = loras[d2].to_delta_weight()
        keys = sorted(set(delta_a.keys()) & set(delta_b.keys()) & set(delta_lin.keys()) & set(delta_geo.keys()))

        chord_norms, lin_norms, geo_norms = [], [], []
        lin_geo_diffs = []
        for k in keys[:5]:
            chord = alpha * delta_a[k] + (1 - alpha) * delta_b[k]
            U, S, Vh = torch.linalg.svd(chord.float(), full_matrices=False)
            r = min(16, len(S))
            chord_proj = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]

            chord_norms.append(torch.norm(chord_proj, "fro").item())
            lin_norms.append(torch.norm(delta_lin[k], "fro").item())
            geo_norms.append(torch.norm(delta_geo[k], "fro").item())
            lin_geo_diffs.append(torch.norm(delta_lin[k] - delta_geo[k], "fro").item())

        results[f"alpha_{alpha:.2f}"] = {
            "alpha": alpha,
            "linear_avg_norm": round(np.mean(lin_norms), 4) if lin_norms else 0.0,
            "geodesic_avg_norm": round(np.mean(geo_norms), 4) if geo_norms else 0.0,
            "chord_avg_norm": round(np.mean(chord_norms), 4) if chord_norms else 0.0,
            "linear_geodesic_avg_diff": round(np.mean(lin_geo_diffs), 6) if lin_geo_diffs else 0.0,
            "pair": [d1, d2],
        }

    return results


# ===== Ablation 4: Number of Domains Composed =====

def ablation_compose_count(loras: dict, counts: list, output_dir: str) -> dict:
    """Ablation: GrassMerge composition quality as number of domains increases."""
    logger.info("=== Ablation: Number of Domains Composed ===")
    results = {}
    merger = GrassMerge()
    domains = sorted(loras.keys())

    for n in counts:
        if n > len(domains):
            logger.warning("  Requested %d domains but only %d available, skipping", n, len(domains))
            continue

        selected = domains[:n]
        logger.info("  Composing %d domains: %s", n, selected)

        t0 = time.time()
        selected_loras = [loras[d] for d in selected]
        current = merger.merge(selected_loras, name=f"grassmerge_{n}d")
        elapsed = time.time() - t0

        delta = current.to_delta_weight()
        total_fro = sum(torch.norm(v, "fro").item() for v in delta.values())
        max_sv = 0.0
        for k in list(delta.keys())[:3]:
            _, S, _ = torch.linalg.svd(delta[k].float(), full_matrices=False)
            max_sv = max(max_sv, S[0].item())

        per_domain_cosines = {}
        for d in selected:
            d_delta = loras[d].to_delta_weight()
            keys = sorted(set(delta.keys()) & set(d_delta.keys()))
            if keys:
                cos = torch.nn.functional.cosine_similarity(
                    delta[keys[0]].flatten().unsqueeze(0),
                    d_delta[keys[0]].flatten().unsqueeze(0),
                ).item()
                per_domain_cosines[d] = round(cos, 4)

        results[f"n_{n}"] = {
            "num_domains": n,
            "domains": selected,
            "compose_time": round(elapsed, 4),
            "total_frobenius_norm": round(total_fro, 4),
            "max_singular_value": round(max_sv, 4),
            "rank": current.rank,
            "per_domain_cosine": per_domain_cosines,
            "avg_cosine": round(np.mean(list(per_domain_cosines.values())), 4) if per_domain_cosines else 0.0,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="LoRA Algebra Ablation Study")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "domains.yaml"))
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory with trained domain LoRAs")
    parser.add_argument("--output_dir", type=str, default="results/ablations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    domains = list(config["domains"].keys())
    logger.info("Loading LoRAs from %s", args.lora_dir)
    loras = load_all_loras(args.lora_dir, domains)

    if len(loras) < 2:
        logger.error("Need at least 2 trained LoRAs, found %d", len(loras))
        sys.exit(1)
    logger.info("Loaded %d LoRAs: %s", len(loras), sorted(loras.keys()))

    ablation_cfg = config.get("ablation", {})
    all_results = {"meta": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domains_loaded": sorted(loras.keys()),
    }}

    # Ablation 1: LoRA Rank
    ranks = ablation_cfg.get("ranks", [4, 8, 16, 32, 64])
    all_results["rank_ablation"] = ablation_rank(loras, ranks, args.output_dir)

    # Ablation 2: Normalization
    all_results["normalization_ablation"] = ablation_normalization(loras, args.output_dir)

    # Ablation 3: Interpolation Type
    all_results["interpolation_type_ablation"] = ablation_interpolation_type(loras, args.output_dir)

    # Ablation 4: Number of Domains
    counts = ablation_cfg.get("compose_counts", [2, 3, 4, 6, 12])
    all_results["compose_count_ablation"] = ablation_compose_count(loras, counts, args.output_dir)

    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("  Ablation study complete")
    logger.info("  Rank ablation: %d configs", len(all_results["rank_ablation"]))
    logger.info("  Normalization ablation: %d pairs", len(all_results["normalization_ablation"]))
    logger.info("  Interpolation type ablation: %d points", len(all_results["interpolation_type_ablation"]))
    logger.info("  Compose count ablation: %d configs", len(all_results["compose_count_ablation"]))
    logger.info("  Results: %s", results_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
