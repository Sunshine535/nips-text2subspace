#!/usr/bin/env python3
"""Fast merging experiments using GPU and compact low-rank operations.

Avoids materializing full (d_out x d_in) delta matrices by working
entirely in the rank-r factor space. Runs GrassMerge + baselines + BGD
on all C(N,2) pairs in minutes instead of hours.
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
from src.lora_algebra import GrassMerge, GrassmannOps, LoRAWeights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_loras(lora_dir, domains):
    loras = {}
    for d in domains:
        p = os.path.join(lora_dir, d)
        if os.path.exists(os.path.join(p, "adapter_model.safetensors")):
            logger.info("Loading LoRA: %s", d)
            loras[d] = LoRAWeights.from_peft_dir(d, p)
    return loras


def move_lora_to_device(lora, device):
    """Move LoRA factors to GPU for fast computation."""
    for key in lora.lora_A:
        lora.lora_A[key] = lora.lora_A[key].to(device)
    for key in lora.lora_B:
        lora.lora_B[key] = lora.lora_B[key].to(device)


def fast_bgd_from_svds(svd_i, svd_j, rank):
    """Compute BGD using pre-computed SVDs (no full delta matrix needed)."""
    common_keys = sorted(set(svd_i.keys()) & set(svd_j.keys()))
    if not common_keys:
        return 0.0
    bgd_sum = 0.0
    for key in common_keys:
        U_i, _, Vh_i = svd_i[key]
        U_j, _, Vh_j = svd_j[key]
        r = min(rank, U_i.shape[1], U_j.shape[1])
        d_left = GrassmannOps.geodesic_distance(U_i[:, :r], U_j[:, :r])
        d_right = GrassmannOps.geodesic_distance(Vh_i[:r, :].T, Vh_j[:r, :].T)
        bgd_sum += float(np.sqrt(d_left ** 2 + d_right ** 2))
    return bgd_sum / len(common_keys)


def fast_task_arithmetic(lora_a, lora_b, rank):
    """Task Arithmetic in compact space: average B and A factors, then refactorize."""
    new_A, new_B = {}, {}
    for key in sorted(set(lora_a.lora_A.keys()) & set(lora_b.lora_A.keys())):
        B_a, A_a = lora_a.lora_B[key].float(), lora_a.lora_A[key].float()
        B_b, A_b = lora_b.lora_B[key].float(), lora_b.lora_A[key].float()
        alpha_a, alpha_b = lora_a.alpha, lora_b.alpha
        # Compute (B_a @ A_a * alpha_a + B_b @ A_b * alpha_b) / 2
        # Using compact factored form: stack [B_a*sqrt(alpha_a), B_b*sqrt(alpha_b)] etc.
        B_cat = torch.cat([B_a * np.sqrt(abs(alpha_a)), B_b * np.sqrt(abs(alpha_b))], dim=1)
        A_cat = torch.cat([A_a * np.sqrt(abs(alpha_a)), A_b * np.sqrt(abs(alpha_b))], dim=0)
        # QR + compact SVD to get rank-r approximation
        Q_B, R_B = torch.linalg.qr(B_cat)
        Q_A, R_A = torch.linalg.qr(A_cat.T)
        M = (R_B @ R_A.T) / 2.0  # average
        U_s, S, Vh_s = torch.linalg.svd(M, full_matrices=False)
        r = min(rank, len(S))
        sqrt_S = torch.sqrt(S[:r].clamp(min=0))
        new_B[key] = (Q_B @ U_s[:, :r] @ torch.diag(sqrt_S)).cpu()
        new_A[key] = (torch.diag(sqrt_S) @ Vh_s[:r, :] @ Q_A.T).cpu()
    return LoRAWeights(name="task_arithmetic", lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


def fast_ties(lora_a, lora_b, rank, density=0.5):
    """TIES-Merging using per-element trim+elect on the compact factors."""
    new_A, new_B = {}, {}
    for key in sorted(set(lora_a.lora_A.keys()) & set(lora_b.lora_A.keys())):
        B_a, A_a = lora_a.lora_B[key].float(), lora_a.lora_A[key].float()
        B_b, A_b = lora_b.lora_B[key].float(), lora_b.lora_A[key].float()
        # For TIES we need element-wise operations, so compute compact delta
        # Use factored representation: delta = B @ A
        delta_a = (B_a @ A_a) * lora_a.alpha
        delta_b = (B_b @ A_b) * lora_b.alpha
        stack = torch.stack([delta_a, delta_b])
        # Trim
        k = int(density * stack[0].numel())
        trimmed = stack.clone()
        for i in range(2):
            flat = trimmed[i].abs().flatten()
            if k < flat.numel():
                threshold = flat.topk(k).values[-1]
                trimmed[i][trimmed[i].abs() < threshold] = 0.0
        # Elect sign
        sign_votes = torch.sign(trimmed).sum(dim=0)
        elected = torch.sign(sign_votes)
        elected[elected == 0] = 1.0
        mask = torch.sign(trimmed) == elected.unsqueeze(0)
        trimmed[~mask] = 0.0
        counts = mask.float().sum(dim=0).clamp(min=1.0)
        merged = trimmed.sum(dim=0) / counts
        # Refactorize to rank-r
        U, S, Vh = torch.linalg.svd(merged, full_matrices=False)
        r = min(rank, len(S))
        sqrt_S = torch.sqrt(S[:r].clamp(min=0))
        new_B[key] = (U[:, :r] @ torch.diag(sqrt_S)).cpu()
        new_A[key] = (torch.diag(sqrt_S) @ Vh[:r, :]).cpu()
        del delta_a, delta_b, stack, trimmed, merged  # free GPU memory
    return LoRAWeights(name="ties", lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


def fast_dare(lora_a, lora_b, rank, drop_rate=0.5):
    """DARE-Merging: random drop + rescale."""
    new_A, new_B = {}, {}
    for key in sorted(set(lora_a.lora_A.keys()) & set(lora_b.lora_A.keys())):
        B_a, A_a = lora_a.lora_B[key].float(), lora_a.lora_A[key].float()
        B_b, A_b = lora_b.lora_B[key].float(), lora_b.lora_A[key].float()
        delta_a = (B_a @ A_a) * lora_a.alpha
        delta_b = (B_b @ A_b) * lora_b.alpha
        stack = torch.stack([delta_a, delta_b])
        mask = torch.bernoulli(torch.full_like(stack, 1.0 - drop_rate))
        rescaled = stack * mask / (1.0 - drop_rate)
        merged = rescaled.mean(dim=0)
        U, S, Vh = torch.linalg.svd(merged, full_matrices=False)
        r = min(rank, len(S))
        sqrt_S = torch.sqrt(S[:r].clamp(min=0))
        new_B[key] = (U[:, :r] @ torch.diag(sqrt_S)).cpu()
        new_A[key] = (torch.diag(sqrt_S) @ Vh[:r, :]).cpu()
        del delta_a, delta_b, stack, merged
    return LoRAWeights(name="dare", lora_A=new_A, lora_B=new_B, rank=rank, alpha=1.0)


def save_peft(lora, peft_dir, config):
    os.makedirs(peft_dir, exist_ok=True)
    import safetensors.torch
    sd = lora.to_state_dict()
    safetensors.torch.save_file(sd, os.path.join(peft_dir, "adapter_model.safetensors"))
    lora_cfg = config.get("lora", {})
    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": config.get("base_model", ""),
        "r": lora.rank, "lora_alpha": lora.rank,
        "target_modules": lora_cfg.get("target_modules", []),
        "lora_dropout": 0.0, "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(peft_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/domains.yaml")
    parser.add_argument("--lora_dir", required=True)
    parser.add_argument("--output_dir", default="results/algebra")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load only trained domains
    all_domains = list(config["domains"].keys())
    loras = load_loras(args.lora_dir, all_domains)
    if len(loras) < 2:
        logger.error("Need >= 2 LoRAs, got %d", len(loras))
        sys.exit(1)
    logger.info("Loaded %d LoRAs: %s", len(loras), sorted(loras.keys()))

    names = sorted(loras.keys())
    rank = max(l.rank for l in loras.values())
    pairs = list(itertools.combinations(names, 2))

    # Move all LoRAs to GPU
    logger.info("Moving LoRAs to %s...", DEVICE)
    for l in loras.values():
        move_lora_to_device(l, DEVICE)

    # Pre-compute fast SVDs (GPU)
    logger.info("Pre-computing fast SVDs...")
    t0 = time.time()
    all_svds = {name: loras[name].fast_svd() for name in names}
    logger.info("  SVDs: %.1fs", time.time() - t0)

    # ===== Phase 1: GrassMerge =====
    logger.info("=" * 50)
    logger.info("  Phase 1: GrassMerge (%d pairs)", len(pairs))
    logger.info("=" * 50)
    gm_dir = os.path.join(args.output_dir, "grassmerge")
    os.makedirs(gm_dir, exist_ok=True)
    merger = GrassMerge(karcher_max_iter=10)
    gm_results = {}

    for idx, (d1, d2) in enumerate(pairs):
        name = f"{d1}+{d2}"
        logger.info("  [%d/%d] %s", idx + 1, len(pairs), name)
        t0 = time.time()
        merged = merger.merge([loras[d1], loras[d2]], name=name)
        elapsed = time.time() - t0

        # Compute BGD from pre-computed SVDs
        bgd = fast_bgd_from_svds(all_svds[d1], all_svds[d2], rank)

        # Move merged to CPU before saving (avoid GPU OOM on to_delta_weight)
        for k in merged.lora_A:
            merged.lora_A[k] = merged.lora_A[k].cpu()
        for k in merged.lora_B:
            merged.lora_B[k] = merged.lora_B[k].cpu()
        save_peft(merged, os.path.join(gm_dir, name), config)
        delta = merged.to_delta_weight()
        torch.save(delta, os.path.join(gm_dir, f"{name}.pt"))

        gm_results[name] = {
            "method": "grassmerge", "domains": [d1, d2],
            "rank": merged.rank, "time_seconds": round(elapsed, 2),
            "bgd": round(bgd, 4),
        }
        logger.info("    %.1fs, BGD=%.4f", elapsed, bgd)
        del merged, delta
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # ===== Phase 2: Baselines =====
    logger.info("=" * 50)
    logger.info("  Phase 2: Baselines (%d pairs)", len(pairs))
    logger.info("=" * 50)
    baseline_dir = os.path.join(args.output_dir, "baselines")
    baseline_results = {}

    methods = [
        ("task_arithmetic", lambda a, b, r: fast_task_arithmetic(a, b, r)),
        ("ties_d0.5", lambda a, b, r: fast_ties(a, b, r, density=0.5)),
        ("dare_p0.5", lambda a, b, r: fast_dare(a, b, r, drop_rate=0.5)),
    ]

    for method_name, method_fn in methods:
        method_dir = os.path.join(baseline_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        method_data = {}
        logger.info("  Baseline: %s", method_name)

        for d1, d2 in pairs:
            name = f"{d1}+{d2}"
            t0 = time.time()
            merged = method_fn(loras[d1], loras[d2], rank)
            elapsed = time.time() - t0

            save_peft(merged, os.path.join(method_dir, name), config)
            delta = merged.to_delta_weight()
            torch.save(delta, os.path.join(method_dir, f"{name}.pt"))

            method_data[name] = {
                "pair": [d1, d2], "time_seconds": round(elapsed, 2),
                "num_layers": len(delta),
            }
            del merged, delta
            torch.cuda.empty_cache() if DEVICE == "cuda" else None

        baseline_results[method_name] = method_data
        logger.info("    %s: %d pairs done", method_name, len(pairs))

    # ===== Phase 3: BGD Matrix =====
    logger.info("=" * 50)
    logger.info("  Phase 3: BGD Matrix")
    logger.info("=" * 50)
    N = len(names)
    bgd_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            bgd = fast_bgd_from_svds(all_svds[names[i]], all_svds[names[j]], rank)
            bgd_matrix[i][j] = bgd_matrix[j][i] = bgd

    logger.info("BGD matrix:\n%s", np.array2string(bgd_matrix, precision=3))
    interference = {
        "domain_names": names,
        "bgd_matrix": bgd_matrix.tolist(),
    }
    with open(os.path.join(args.output_dir, "interference_metrics.json"), "w") as f:
        json.dump(interference, f, indent=2)

    # ===== Save All Results =====
    all_results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "domains_loaded": names,
            "device": DEVICE,
        },
        "grassmerge": gm_results,
        "baselines": baseline_results,
        "bgd": interference,
    }
    with open(os.path.join(args.output_dir, "all_algebra_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("  ALL EXPERIMENTS COMPLETE")
    logger.info("  GrassMerge: %d pairs", len(gm_results))
    logger.info("  Baselines: %s", list(baseline_results.keys()))
    logger.info("  Results: %s", os.path.join(args.output_dir, "all_algebra_results.json"))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
