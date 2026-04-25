#!/usr/bin/env python3
"""Aggregate eval_carr.py multi-seed runs with bootstrap CIs.

GPT-5.5 Round 3 Task 7 expected verification command:
    python scripts/aggregate_results.py \
        --input_dir results-synced/carr_minimal_verified \
        --ci bootstrap

Reads `index.json` written by eval_carr.py (or scans for results_*.json) and
produces a method × domain accuracy table with mean / std / 95% bootstrap CI
across (seed, sample_seed) combinations.
"""
import argparse
import json
import math
import os
import random
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def load_runs(input_dir: str) -> List[dict]:
    runs = []
    idx = Path(input_dir) / "index.json"
    if idx.is_file():
        with idx.open() as f:
            data = json.load(f)
        for r in data.get("runs", []):
            if "error" in r:
                continue
            runs.append(r)
    if runs:
        return runs

    # Fallback: scan for results_seed*_sample*.json
    for jf in glob(os.path.join(input_dir, "**/results_seed*_sample*.json"), recursive=True):
        with open(jf) as f:
            runs.append(json.load(f))
    return runs


def collect(runs: Sequence[dict], method: str, domain: str) -> List[float]:
    accs = []
    for r in runs:
        if method not in r:
            continue
        scores = r[method]
        if domain not in scores:
            continue
        a = scores[domain].get("accuracy", -1)
        if a >= 0:
            accs.append(float(a))
    return accs


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * (alpha / 2))]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return float(lo), float(hi)


def percentile_pairwise_win(runs: Sequence[dict], a: str, b: str,
                            domains: Sequence[str]) -> Tuple[int, int]:
    """Counts (seed,sample_seed) tuples where method a's mean-over-domains > b's."""
    a_wins = 0
    total = 0
    for r in runs:
        if a not in r or b not in r:
            continue
        a_acc = sum(r[a][d]["accuracy"] for d in domains) / len(domains)
        b_acc = sum(r[b][d]["accuracy"] for d in domains) / len(domains)
        if a_acc > b_acc:
            a_wins += 1
        total += 1
    return a_wins, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--ci", default="bootstrap", choices=["bootstrap", "none"])
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--methods", default="base,single,static_only,carr_no_mechanism,carr_full,"
                                          "no_reliability,no_conflict,no_base_fallback")
    p.add_argument("--domains", default=None,
                   help="comma-separated; default: infer from index.json")
    p.add_argument("--output", default=None,
                   help="json output path; default: <input_dir>/aggregate.json")
    args = p.parse_args()

    runs = load_runs(args.input_dir)
    if not runs:
        print(f"No runs found in {args.input_dir}")
        sys.exit(1)

    if args.domains:
        domains = args.domains.split(",")
    else:
        idx_file = Path(args.input_dir) / "index.json"
        if idx_file.is_file():
            with idx_file.open() as f:
                domains = json.load(f).get("domains", ["science", "medical"])
        else:
            domains = list(runs[0].get("domains", ["science", "medical"]))

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    table = {}
    for method in methods:
        per_domain = {}
        for d in domains:
            accs = collect(runs, method, d)
            if not accs:
                continue
            mean = sum(accs) / len(accs)
            std = math.sqrt(sum((a - mean) ** 2 for a in accs) / max(len(accs) - 1, 1))
            ci = bootstrap_ci(accs, n_boot=args.n_boot) if args.ci == "bootstrap" else (None, None)
            per_domain[d] = {
                "n": len(accs), "mean": mean, "std": std,
                "ci_lo": ci[0], "ci_hi": ci[1], "values": accs,
            }
        if per_domain:
            # Mean-of-means across reported domains
            domain_means = [v["mean"] for v in per_domain.values()]
            table[method] = {
                "per_domain": per_domain,
                "mean_over_domains": sum(domain_means) / len(domain_means),
            }

    # Pairwise win rates: full CARR vs each baseline
    pairwise = {}
    if "carr_full" in table:
        for opp in ["static_only", "carr_no_mechanism", "no_reliability",
                    "no_conflict", "no_base_fallback"]:
            if opp not in table:
                continue
            wins, total = percentile_pairwise_win(runs, "carr_full", opp, domains)
            pairwise[f"carr_full_vs_{opp}"] = {
                "wins": wins, "total": total,
                "rate": wins / max(total, 1),
            }

    # Print Markdown-style table
    print("\nMethod" + " " * 22 + " | ".join(f"{d:>14}" for d in domains) + " |   Mean")
    print("-" * (28 + len(domains) * 17 + 12))
    for method, row in table.items():
        cols = []
        for d in domains:
            if d not in row["per_domain"]:
                cols.append(f"{'—':>14}")
                continue
            r = row["per_domain"][d]
            if r["ci_lo"] is not None:
                cols.append(f"{r['mean']:.3f}±{(r['ci_hi'] - r['ci_lo'])/2:.3f}")
            else:
                cols.append(f"{r['mean']:.3f}±{r['std']:.3f}")
        print(f"{method:<28}" + " | ".join(f"{c:>14}" for c in cols)
              + f" | {row['mean_over_domains']:.4f}")

    if pairwise:
        print("\nPairwise win rates (carr_full mean over domains > opponent's):")
        for k, v in pairwise.items():
            print(f"  {k}: {v['wins']}/{v['total']}  ({v['rate']:.0%})")

    out_path = args.output or os.path.join(args.input_dir, "aggregate.json")
    with open(out_path, "w") as f:
        json.dump({
            "input_dir": args.input_dir, "n_runs": len(runs),
            "domains": domains, "ci": args.ci, "n_boot": args.n_boot,
            "table": table, "pairwise": pairwise,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
