#!/usr/bin/env python3
"""Memory-efficient reparameterization invariance test.

Tests that GrassMerge produces identical merged LoRA when inputs
are rotated by arbitrary invertible basis changes: A'=QA, B'=BQ^T.
Compares per-layer to avoid materializing full 7B-param deltas.
"""

import json
import logging
import os
import sys
import itertools
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lora_algebra import GrassMerge, LoRAWeights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORE_DOMAINS = ["math", "code", "medical", "science", "history", "philosophy"]


def apply_random_rotation(lora, seed=0):
    """Apply orthogonal rotation to LoRA basis: A'=QA, B'=BQ^T."""
    rng = torch.Generator().manual_seed(seed)
    new_A, new_B = {}, {}
    for key in lora.lora_A:
        r = lora.rank
        R = torch.randn(r, r, generator=rng)
        Q, _ = torch.linalg.qr(R)
        A = lora.lora_A[key].float()
        B = lora.lora_B[key].float()
        new_A[key] = (Q @ A).to(A.dtype)
        new_B[key] = (B @ Q.T).to(B.dtype)
    return LoRAWeights(
        name=f"{lora.name}_rot{seed}",
        lora_A=new_A, lora_B=new_B,
        rank=lora.rank, alpha=lora.alpha,
    )


def compare_merges_per_layer(original_merge, rotated_merge):
    """Compare two LoRAWeights per-layer without materializing full deltas."""
    results = []
    keys = sorted(set(original_merge.lora_A.keys()) & set(rotated_merge.lora_A.keys()))
    for key in keys:
        # Compute delta per-layer: delta = B @ A (small: d_out x d_in but only one layer)
        delta_orig = (original_merge.lora_B[key].float() @ original_merge.lora_A[key].float())
        delta_rot = (rotated_merge.lora_B[key].float() @ rotated_merge.lora_A[key].float())

        diff_norm = (delta_orig - delta_rot).norm().item()
        orig_norm = delta_orig.norm().item()
        rel_diff = diff_norm / orig_norm if orig_norm > 0 else 0.0
        results.append({"layer": key, "rel_diff": rel_diff, "abs_diff": diff_norm})
        del delta_orig, delta_rot
    return results


def main():
    lora_dir = "results/domain_loras"
    output_path = "results/eval_comprehensive/reparam_invariance.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load LoRAs
    loras = {}
    for d in CORE_DOMAINS[:4]:  # Use 4 domains for testing
        path = os.path.join(lora_dir, d)
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            logger.info("Loading LoRA: %s", d)
            loras[d] = LoRAWeights.from_peft_dir(d, path)

    domains = sorted(loras.keys())
    merger = GrassMerge()
    all_results = []

    for d1, d2 in itertools.combinations(domains, 2):
        # Original merge
        logger.info("Merging %s + %s (original)", d1, d2)
        original = merger.merge([loras[d1], loras[d2]])

        for trial in range(3):
            # Rotate both LoRAs
            rot1 = apply_random_rotation(loras[d1], seed=trial * 100)
            rot2 = apply_random_rotation(loras[d2], seed=trial * 100 + 1)

            # Verify rotation preserves delta: B'A' = BQ^T QA = BA
            sample_key = list(loras[d1].lora_A.keys())[0]
            orig_delta = loras[d1].lora_B[sample_key].float() @ loras[d1].lora_A[sample_key].float()
            rot_delta = rot1.lora_B[sample_key].float() @ rot1.lora_A[sample_key].float()
            rotation_check = (orig_delta - rot_delta).norm().item() / orig_delta.norm().item()
            logger.info("  Rotation preserves delta: rel_diff = %.2e", rotation_check)
            del orig_delta, rot_delta

            # Merge rotated
            logger.info("  Merging rotated (trial %d)", trial)
            rotated = merger.merge([rot1, rot2])

            # Compare per-layer
            layer_diffs = compare_merges_per_layer(original, rotated)
            avg_diff = np.mean([r["rel_diff"] for r in layer_diffs])
            max_diff = max(r["rel_diff"] for r in layer_diffs)

            result = {
                "pair": f"{d1}+{d2}",
                "trial": trial,
                "rotation_preserves_delta": rotation_check < 1e-6,
                "avg_relative_diff": round(avg_diff, 8),
                "max_relative_diff": round(max_diff, 8),
                "num_layers": len(layer_diffs),
                "invariant": avg_diff < 1e-4,
            }
            all_results.append(result)
            logger.info("  Result: avg_rel_diff=%.2e, max_rel_diff=%.2e, invariant=%s",
                        avg_diff, max_diff, result["invariant"])

            del rot1, rot2, rotated
            torch.cuda.empty_cache()

        del original
        torch.cuda.empty_cache()

    # Summary
    invariant_count = sum(1 for r in all_results if r["invariant"])
    total = len(all_results)
    logger.info("=" * 60)
    logger.info("  SUMMARY: %d/%d tests passed (avg_rel_diff < 1e-4)", invariant_count, total)
    logger.info("=" * 60)

    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "passed": invariant_count,
                "total": total,
                "pass_rate": round(invariant_count / max(total, 1), 4),
            },
            "results": all_results,
        }, f, indent=2)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
