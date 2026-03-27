#!/usr/bin/env python3
"""Run complete LoRA algebra experiments.

Tests all algebraic operations on trained domain LoRAs:
- Composition: all 66 domain pairs → evaluate on both domains
- Interpolation: sweep alpha in [0, 0.25, 0.5, 0.75, 1.0] for selected pairs
- Subtraction: remove domain_i from composed(i,j) → evaluate on domain_j
- Projection: project domain_i onto subspace of domain_j
- Grassmann geodesic vs linear interpolation comparison
- Baseline comparison: TIES, DARE, Task Arithmetic
"""

import argparse
import itertools
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lora_algebra import (
    GrassmannProjector,
    LoRAAlgebra,
    LoRAWeights,
    MergingBaselines,
    compute_similarity_matrix,
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


def run_composition(algebra: LoRAAlgebra, loras: dict, output_dir: str) -> dict:
    """Compose all C(n,2) domain pairs."""
    results = {}
    compose_dir = os.path.join(output_dir, "composed")
    os.makedirs(compose_dir, exist_ok=True)
    domains = sorted(loras.keys())

    total_pairs = len(domains) * (len(domains) - 1) // 2
    logger.info("Composing all %d domain pairs...", total_pairs)

    for idx, (d1, d2) in enumerate(itertools.combinations(domains, 2)):
        name = f"{d1}+{d2}"
        logger.info("  [%d/%d] Composing: %s", idx + 1, total_pairs, name)
        t0 = time.time()
        composed = algebra.compose(loras[d1], loras[d2], name=name)
        elapsed = time.time() - t0

        sd = composed.to_state_dict()
        save_path = os.path.join(compose_dir, f"{name}.pt")
        torch.save(sd, save_path)

        delta_a = loras[d1].to_delta_weight()
        delta_b = loras[d2].to_delta_weight()
        delta_c = composed.to_delta_weight()
        first_key = sorted(set(delta_a.keys()) & set(delta_b.keys()) & set(delta_c.keys()))[0]
        cosine_a = torch.nn.functional.cosine_similarity(
            delta_c[first_key].flatten().unsqueeze(0),
            delta_a[first_key].flatten().unsqueeze(0),
        ).item()
        cosine_b = torch.nn.functional.cosine_similarity(
            delta_c[first_key].flatten().unsqueeze(0),
            delta_b[first_key].flatten().unsqueeze(0),
        ).item()

        results[name] = {
            "operation": "compose",
            "domains": [d1, d2],
            "rank": composed.rank,
            "num_params": sum(v.numel() for v in sd.values()),
            "time_seconds": round(elapsed, 3),
            "cosine_similarity_to_d1": round(cosine_a, 4),
            "cosine_similarity_to_d2": round(cosine_b, 4),
        }

    return results


def run_interpolation(algebra: LoRAAlgebra, loras: dict, alphas: list, output_dir: str) -> dict:
    """Linear interpolation sweep for selected pairs."""
    results = {}
    interp_dir = os.path.join(output_dir, "interpolated")
    os.makedirs(interp_dir, exist_ok=True)

    pairs = [
        ("math", "code"), ("medical", "science"), ("creative_writing", "philosophy"),
        ("legal", "finance"), ("history", "geography"), ("psychology", "philosophy"),
    ]

    for d1, d2 in pairs:
        if d1 not in loras or d2 not in loras:
            logger.warning("Skipping pair (%s, %s): missing LoRA", d1, d2)
            continue
        pair_results = {}
        for alpha in alphas:
            name = f"{d1}_{d2}_a{alpha:.2f}"
            logger.info("  Interpolating: %s (alpha=%.2f)", name, alpha)
            interp = algebra.interpolate(loras[d1], loras[d2], alpha=alpha, name=name)
            sd = interp.to_state_dict()
            torch.save(sd, os.path.join(interp_dir, f"{name}.pt"))

            delta_interp = interp.to_delta_weight()
            delta_a = loras[d1].to_delta_weight()
            delta_b = loras[d2].to_delta_weight()
            keys = sorted(set(delta_interp.keys()) & set(delta_a.keys()) & set(delta_b.keys()))
            if keys:
                fro_a = sum(torch.norm(delta_interp[k] - delta_a[k], "fro").item() for k in keys)
                fro_b = sum(torch.norm(delta_interp[k] - delta_b[k], "fro").item() for k in keys)
            else:
                fro_a, fro_b = 0.0, 0.0

            pair_results[str(alpha)] = {
                "alpha": alpha,
                "rank": interp.rank,
                "frobenius_dist_to_d1": round(fro_a, 4),
                "frobenius_dist_to_d2": round(fro_b, 4),
            }
        results[f"{d1}+{d2}"] = pair_results

    return results


def run_subtraction(algebra: LoRAAlgebra, loras: dict, output_dir: str) -> dict:
    """Subtract domain_i from composed(i,j), evaluate residual for domain_j."""
    results = {}
    sub_dir = os.path.join(output_dir, "subtracted")
    os.makedirs(sub_dir, exist_ok=True)

    test_pairs = [
        ("math", "code"), ("medical", "legal"), ("creative_writing", "science"),
        ("finance", "philosophy"), ("history", "geography"), ("psychology", "science"),
    ]

    for d1, d2 in test_pairs:
        if d1 not in loras or d2 not in loras:
            continue

        composed = algebra.compose(loras[d1], loras[d2], name=f"temp_{d1}+{d2}")
        residual = algebra.subtract(composed, loras[d1], name=f"residual_{d2}_from_{d1}+{d2}")

        delta_res = residual.to_delta_weight()
        delta_d2 = loras[d2].to_delta_weight()
        keys = sorted(set(delta_res.keys()) & set(delta_d2.keys()))

        if keys:
            cosine = torch.nn.functional.cosine_similarity(
                delta_res[keys[0]].flatten().unsqueeze(0),
                delta_d2[keys[0]].flatten().unsqueeze(0),
            ).item()
            fro_diff = torch.norm(delta_res[keys[0]] - delta_d2[keys[0]], "fro").item()
            fro_d2 = torch.norm(delta_d2[keys[0]], "fro").item()
            reconstruction_error = fro_diff / max(fro_d2, 1e-8)
        else:
            cosine, reconstruction_error = 0.0, 1.0

        sd = residual.to_state_dict()
        torch.save(sd, os.path.join(sub_dir, f"residual_{d2}_from_{d1}+{d2}.pt"))

        name = f"({d1}+{d2})-{d1}"
        results[name] = {
            "composed": [d1, d2],
            "removed": d1,
            "expected_residual": d2,
            "cosine_to_d2": round(cosine, 4),
            "relative_reconstruction_error": round(reconstruction_error, 4),
            "rank": residual.rank,
        }

    return results


def run_projection(algebra: LoRAAlgebra, loras: dict, output_dir: str) -> dict:
    """Project domain_i onto subspace of domain_j."""
    results = {}
    proj_dir = os.path.join(output_dir, "projected")
    os.makedirs(proj_dir, exist_ok=True)

    test_pairs = [
        ("math", ["code", "science"]),
        ("medical", ["science", "psychology"]),
        ("creative_writing", ["philosophy", "history"]),
        ("legal", ["finance", "philosophy"]),
    ]

    for target_domain, basis_domains in test_pairs:
        if target_domain not in loras:
            continue
        basis = [loras[d] for d in basis_domains if d in loras]
        if len(basis) < 1:
            continue

        name = f"proj_{target_domain}_onto_{'_'.join(basis_domains)}"
        logger.info("  Projecting: %s → span(%s)", target_domain, basis_domains)
        projected = algebra.project_onto_subspace(loras[target_domain], basis, name=name)

        delta_orig = loras[target_domain].to_delta_weight()
        delta_proj = projected.to_delta_weight()
        keys = sorted(set(delta_orig.keys()) & set(delta_proj.keys()))

        if keys:
            retained_energy = sum(
                torch.norm(delta_proj[k], "fro").item() ** 2 for k in keys
            ) / max(sum(torch.norm(delta_orig[k], "fro").item() ** 2 for k in keys), 1e-8)
        else:
            retained_energy = 0.0

        sd = projected.to_state_dict()
        torch.save(sd, os.path.join(proj_dir, f"{name}.pt"))

        results[name] = {
            "target": target_domain,
            "basis": basis_domains,
            "retained_energy_ratio": round(retained_energy, 4),
            "rank": projected.rank,
        }

    return results


def run_grassmann_comparison(algebra: LoRAAlgebra, loras: dict, alphas: list, output_dir: str) -> dict:
    """Compare Grassmann geodesic vs linear interpolation."""
    results = {}
    geo_dir = os.path.join(output_dir, "grassmann_comparison")
    os.makedirs(geo_dir, exist_ok=True)

    pairs = [("math", "code"), ("medical", "science"), ("legal", "finance")]

    for d1, d2 in pairs:
        if d1 not in loras or d2 not in loras:
            continue
        pair_results = []
        for alpha in alphas:
            linear = algebra.interpolate(loras[d1], loras[d2], alpha=alpha, name="linear")
            geodesic = algebra.grassmann_interpolate(loras[d1], loras[d2], t=alpha, name="geodesic")

            delta_lin = linear.to_delta_weight()
            delta_geo = geodesic.to_delta_weight()
            keys = sorted(set(delta_lin.keys()) & set(delta_geo.keys()))

            if keys:
                fro_diff = sum(torch.norm(delta_lin[k] - delta_geo[k], "fro").item() for k in keys)
                fro_lin = sum(torch.norm(delta_lin[k], "fro").item() for k in keys)
                relative_diff = fro_diff / max(fro_lin, 1e-8)
            else:
                relative_diff = 0.0

            pair_results.append({
                "alpha": alpha,
                "relative_difference": round(relative_diff, 6),
                "linear_fro_norm": round(fro_lin, 4) if keys else 0.0,
            })

        results[f"{d1}+{d2}"] = pair_results

    return results


def run_grassmann_distances(loras: dict, config: dict, output_dir: str) -> dict:
    """Compute pairwise Grassmann distance matrix."""
    svd_rank = config.get("algebra", {}).get("svd_rank", 32)
    projector = GrassmannProjector(svd_rank=svd_rank)
    names = sorted(loras.keys())
    lora_list = [loras[n] for n in names]

    logger.info("Computing Grassmann distance matrix for %d domains...", len(names))
    dist_matrix = compute_similarity_matrix(lora_list, projector)

    result = {"domain_names": names, "distance_matrix": dist_matrix.tolist()}
    with open(os.path.join(output_dir, "grassmann_distances.json"), "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Distance matrix (first 5x5):\n%s",
                np.array2string(dist_matrix[:5, :5], precision=3, suppress_small=True))
    return result


def run_baselines(loras: dict, output_dir: str) -> dict:
    """Compare with TIES, DARE, Task Arithmetic baselines."""
    results = {}
    baseline_dir = os.path.join(output_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)
    lora_list = list(loras.values())

    configs = [
        ("task_arithmetic_s0.5", MergingBaselines.task_arithmetic, {"scaling": 0.5}),
        ("task_arithmetic_s1.0", MergingBaselines.task_arithmetic, {"scaling": 1.0}),
        ("ties_d0.3_s0.5", MergingBaselines.ties_merging, {"density": 0.3, "scaling": 0.5}),
        ("ties_d0.5_s0.5", MergingBaselines.ties_merging, {"density": 0.5, "scaling": 0.5}),
        ("ties_d0.7_s0.5", MergingBaselines.ties_merging, {"density": 0.7, "scaling": 0.5}),
        ("dare_p0.3_s0.5", MergingBaselines.dare_merging, {"drop_rate": 0.3, "scaling": 0.5}),
        ("dare_p0.5_s0.5", MergingBaselines.dare_merging, {"drop_rate": 0.5, "scaling": 0.5}),
        ("dare_p0.7_s0.5", MergingBaselines.dare_merging, {"drop_rate": 0.7, "scaling": 0.5}),
    ]

    for method_name, method_fn, kwargs in configs:
        logger.info("  Baseline: %s", method_name)
        t0 = time.time()
        merged = method_fn(lora_list, **kwargs)
        elapsed = time.time() - t0
        torch.save(merged, os.path.join(baseline_dir, f"{method_name}.pt"))
        results[method_name] = {
            "method": method_name,
            "num_layers": len(merged),
            "total_params": sum(v.numel() for v in merged.values()),
            "time_seconds": round(elapsed, 3),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="LoRA Algebra — Full Experiment Suite")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "domains.yaml"))
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory with trained domain LoRAs")
    parser.add_argument("--output_dir", type=str, default="results/algebra")
    parser.add_argument("--skip_compose", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    domains = list(config["domains"].keys())
    logger.info("Loading %d domain LoRAs from %s", len(domains), args.lora_dir)
    loras = load_all_loras(args.lora_dir, domains)

    if len(loras) < 2:
        logger.error("Need at least 2 trained LoRAs, found %d", len(loras))
        sys.exit(1)
    logger.info("Loaded %d LoRAs: %s", len(loras), sorted(loras.keys()))

    algebra = LoRAAlgebra(grassmann_rank=config.get("algebra", {}).get("svd_rank", 32))
    alphas = config.get("algebra", {}).get("interpolation_alphas", [0.0, 0.25, 0.5, 0.75, 1.0])
    all_results = {"meta": {"timestamp": datetime.now(timezone.utc).isoformat(), "domains_loaded": sorted(loras.keys())}}

    if not args.skip_compose:
        logger.info("=" * 50)
        logger.info("  PHASE 1: Composition (all %d pairs)", len(loras) * (len(loras) - 1) // 2)
        logger.info("=" * 50)
        all_results["composition"] = run_composition(algebra, loras, args.output_dir)

    logger.info("=" * 50)
    logger.info("  PHASE 2: Interpolation")
    logger.info("=" * 50)
    all_results["interpolation"] = run_interpolation(algebra, loras, alphas, args.output_dir)

    logger.info("=" * 50)
    logger.info("  PHASE 3: Subtraction")
    logger.info("=" * 50)
    all_results["subtraction"] = run_subtraction(algebra, loras, args.output_dir)

    logger.info("=" * 50)
    logger.info("  PHASE 4: Projection")
    logger.info("=" * 50)
    all_results["projection"] = run_projection(algebra, loras, args.output_dir)

    logger.info("=" * 50)
    logger.info("  PHASE 5: Grassmann Geodesic vs Linear Comparison")
    logger.info("=" * 50)
    all_results["grassmann_comparison"] = run_grassmann_comparison(algebra, loras, alphas, args.output_dir)
    all_results["grassmann_distances"] = run_grassmann_distances(loras, config, args.output_dir)

    if not args.skip_baselines:
        logger.info("=" * 50)
        logger.info("  PHASE 6: Baseline Comparisons")
        logger.info("=" * 50)
        all_results["baselines"] = run_baselines(loras, args.output_dir)

    results_path = os.path.join(args.output_dir, "all_algebra_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("  All algebra experiments complete")
    logger.info("  Results: %s", results_path)
    logger.info("  Composition pairs: %d", len(all_results.get("composition", {})))
    logger.info("  Interpolation configs: %d", sum(len(v) for v in all_results.get("interpolation", {}).values()))
    logger.info("  Subtraction tests: %d", len(all_results.get("subtraction", {})))
    logger.info("  Projection tests: %d", len(all_results.get("projection", {})))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
