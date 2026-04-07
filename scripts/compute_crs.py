#!/usr/bin/env python3
"""Compute Composition Rank Score (CRS) for all adapter pairs.

Outputs per-pair CRS, per-layer breakdown, and phase diagram data.
"""

import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lora_algebra import LoRAWeights
from src.rank_bottleneck import compute_composition_rank, compare_diagnostics

CORE_DOMAINS = ["math", "code", "medical", "science", "history", "philosophy"]


def load_adapters(lora_dir: str, domains: list) -> dict:
    adapters = {}
    for d in domains:
        peft_path = os.path.join(lora_dir, d)
        cfg = os.path.join(peft_path, "adapter_config.json")
        if not os.path.isfile(cfg):
            continue
        try:
            adapters[d] = LoRAWeights.from_peft_dir(d, peft_path)
        except Exception as e:
            print(f"  WARN: Failed to load {d}: {e}")
    return adapters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir", type=str, default="results/domain_loras")
    parser.add_argument("--output", type=str, default="results/crs")
    parser.add_argument("--domains", nargs="+", default=CORE_DOMAINS)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    adapters = load_adapters(args.lora_dir, args.domains)
    if len(adapters) < 2:
        print(f"Need >= 2 adapters, found {len(adapters)}: {list(adapters.keys())}")
        sys.exit(1)

    print(f"Loaded {len(adapters)} adapters: {list(adapters.keys())}")

    # Pairwise CRS
    pair_results = []
    names = sorted(adapters.keys())
    for a, b in itertools.combinations(names, 2):
        pair = [adapters[a], adapters[b]]
        analysis = compute_composition_rank(pair)
        diag = compare_diagnostics(pair)

        entry = {
            "pair": f"{a}+{b}",
            "crs": analysis.global_crs,
            "tail_energy": analysis.total_tail_energy,
            "total_energy": analysis.total_energy,
            **{k: v for k, v in diag.items() if k not in ("crs", "tail_energy")},
        }

        # Per-layer CRS for layers with bottleneck
        bot_layers = {k: {"rc": v.composition_rank, "k_bot": v.bottleneck_dim, "crs": v.crs}
                      for k, v in analysis.layers.items() if v.bottleneck_dim > 0}
        entry["bottleneck_layers"] = len(bot_layers)
        entry["per_layer"] = bot_layers

        pair_results.append(entry)
        print(f"  {entry['pair']:<30} CRS={entry['crs']:.4f}  tail={entry['tail_energy']:.4f}  bot_layers={entry['bottleneck_layers']}")

    # N-way CRS (all adapters)
    if len(adapters) >= 3:
        all_loras = [adapters[n] for n in names]
        nway = compute_composition_rank(all_loras)
        nway_entry = {
            "pair": "+".join(names),
            "crs": nway.global_crs,
            "tail_energy": nway.total_tail_energy,
            "n_adapters": len(names),
        }
        pair_results.append(nway_entry)
        print(f"  {nway_entry['pair']:<30} CRS={nway_entry['crs']:.4f} (N-way)")

    # Save
    out_path = os.path.join(args.output, "crs_results.json")
    with open(out_path, "w") as f:
        json.dump(pair_results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
