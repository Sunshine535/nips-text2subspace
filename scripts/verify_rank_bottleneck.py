#!/usr/bin/env python3
"""Verify Rank Bottleneck Theorem on synthetic and real adapters (Block 1)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import json
from datetime import datetime
from src.rank_bottleneck import (
    compute_composition_rank,
    identify_bottleneck_directions,
    verify_theorem_synthetic,
    BottleneckAwareComposition,
    BACConfig,
)
from src.lora_algebra import LoRAWeights


def run_synthetic_verification():
    """Block 1A: Verify theorem on synthetic adapters with controlled overlap."""
    print("=" * 70)
    print("Block 1A: Synthetic Theorem Verification")
    print("=" * 70)

    results = verify_theorem_synthetic(
        d_out=256, d_in=256, rank=16,
        overlap_dims=[0, 2, 4, 8, 12, 16],
        seed=42,
    )

    print(f"\n{'Overlap':>8} {'Exp r_c':>8} {'Meas r_c':>9} {'Predicted':>10} {'Actual':>10} {'Tightness':>10} {'CRS':>6}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['overlap_dim']:>8} {r['expected_rc']:>8} {r['measured_rc']:>9} "
            f"{r['predicted_lower_bound']:>10.4f} {r['actual_loss']:>10.4f} "
            f"{r['bound_tight']:>10.4f} {r['crs']:>6.3f}"
        )

    # Verify: predicted bound <= actual loss (theorem correctness)
    all_valid = all(r["predicted_lower_bound"] <= r["actual_loss"] * 1.01 for r in results)
    print(f"\nTheorem 1 valid (bound <= actual for all cases): {all_valid}")

    # Verify: r_c matches expected
    rc_matches = all(r["measured_rc"] == r["expected_rc"] for r in results)
    print(f"Composition rank matches expected: {rc_matches}")

    return results


def run_real_adapter_analysis():
    """Block 1B: Analyze composition rank of real trained LoRAs."""
    print("\n" + "=" * 70)
    print("Block 1B: Real Adapter Composition Rank Analysis")
    print("=" * 70)

    lora_dir = os.path.join(os.path.dirname(__file__), "..", "results", "domain_loras")
    if not os.path.isdir(lora_dir):
        print(f"No trained LoRAs found at {lora_dir}, skipping real adapter analysis.")
        return None

    # Find available trained adapters
    domains = []
    for d in sorted(os.listdir(lora_dir)):
        peft_path = os.path.join(lora_dir, d)
        if os.path.isfile(os.path.join(peft_path, "adapter_config.json")):
            domains.append(d)

    if len(domains) < 2:
        print(f"Need >= 2 trained adapters, found {len(domains)}: {domains}")
        return None

    print(f"Found {len(domains)} trained adapters: {domains}")

    # Load all adapters
    loras = []
    for d in domains:
        peft_path = os.path.join(lora_dir, d)
        try:
            lw = LoRAWeights.from_peft_dir(d, peft_path)
            loras.append(lw)
            print(f"  Loaded {d}: rank={lw.rank}, layers={len(lw.lora_A)}")
        except Exception as e:
            print(f"  Failed to load {d}: {e}")

    if len(loras) < 2:
        return None

    # Pairwise CRS analysis
    print(f"\n{'Pair':<35} {'Global CRS':>10} {'Tail Energy':>12} {'Max k_bot':>10} {'Bot Layers':>10}")
    print("-" * 80)

    pair_results = []
    for i in range(len(loras)):
        for j in range(i + 1, len(loras)):
            pair = [loras[i], loras[j]]
            analysis = compute_composition_rank(pair)
            max_k = max(v.bottleneck_dim for v in analysis.layers.values())
            n_bot = sum(1 for v in analysis.layers.values() if v.bottleneck_dim > 0)
            pair_name = f"{loras[i].name}+{loras[j].name}"
            print(
                f"{pair_name:<35} {analysis.global_crs:>10.4f} "
                f"{analysis.total_tail_energy:>12.4f} {max_k:>10} {n_bot:>10}"
            )
            pair_results.append({
                "pair": pair_name,
                "crs": analysis.global_crs,
                "tail_energy": analysis.total_tail_energy,
                "max_bottleneck_dim": max_k,
                "bottleneck_layers": n_bot,
            })

    # Full BAC analysis on one pair (lowest CRS = hardest to merge)
    if pair_results:
        hardest = min(pair_results, key=lambda x: x["crs"])
        print(f"\nDetailed analysis of hardest pair: {hardest['pair']}")
        names = hardest["pair"].split("+")
        pair_loras = [l for l in loras if l.name in names]
        if len(pair_loras) == 2:
            bac = BottleneckAwareComposition(BACConfig(static_rank=16))
            bac.analyze(pair_loras)
            print(bac.summary())

    return pair_results


def main():
    results = {}

    # Synthetic verification
    synth = run_synthetic_verification()
    results["synthetic"] = synth

    # Real adapter analysis
    real = run_real_adapter_analysis()
    if real is not None:
        results["real_pairs"] = real

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rank_bottleneck_verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # Convert for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        return obj

    import numpy as np
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=to_serializable)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
