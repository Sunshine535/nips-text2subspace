#!/usr/bin/env python3
"""Collect and summarize all experiment results into a single report.

Reads from results/{algebra,eval,ablations,analysis}/ and generates
a comprehensive summary suitable for paper writing.
"""

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        logger.warning("Not found: %s", path)
        return {}
    with open(path) as f:
        return json.load(f)


def summarize_grassmerge_vs_baselines(eval_results: dict) -> str:
    """Generate comparison table text."""
    lines = ["## Main Results: GrassMerge vs Baselines\n"]
    grassmerge = eval_results.get("grassmerge", {})
    baselines = eval_results.get("baselines", {})
    individual = eval_results.get("individual_loras", {})
    base = eval_results.get("base_model", {})

    if not grassmerge:
        lines.append("No GrassMerge results found.\n")
        return "\n".join(lines)

    lines.append("| Pair | GrassMerge | Task Arith. | TIES | DARE | KnOTS | TSPA |")
    lines.append("|------|-----------|-------------|------|------|-------|------|")

    pair_gm = {}
    for key, metrics in grassmerge.items():
        parts = key.split("_on_")
        if len(parts) >= 2:
            pair_name = parts[0]
            acc = metrics.get("accuracy", -1)
            if pair_name not in pair_gm:
                pair_gm[pair_name] = []
            if acc >= 0:
                pair_gm[pair_name].append(acc)

    for pair_name, accs in sorted(pair_gm.items()):
        gm_avg = sum(accs) / len(accs) if accs else -1

        bl_accs = {}
        for method, method_data in baselines.items():
            m_accs = []
            for key, metrics in method_data.items():
                if key.startswith(pair_name + "_"):
                    acc = metrics.get("accuracy", -1)
                    if acc >= 0:
                        m_accs.append(acc)
            bl_accs[method] = sum(m_accs) / len(m_accs) if m_accs else -1

        ta = bl_accs.get("task_arithmetic", -1)
        ties = bl_accs.get("ties_d0.5", -1)
        dare = bl_accs.get("dare_p0.5", -1)
        knots = bl_accs.get("knots", -1)
        tspa = bl_accs.get("tspa", -1)

        def fmt(v):
            return f"{v:.3f}" if v >= 0 else "-"

        lines.append(f"| {pair_name} | {fmt(gm_avg)} | {fmt(ta)} | {fmt(ties)} | {fmt(dare)} | {fmt(knots)} | {fmt(tspa)} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect and summarize results")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/RESULTS_SUMMARY.md")
    args = parser.parse_args()

    rdir = args.results_dir
    eval_results = load_json(os.path.join(rdir, "eval", "eval_results.json"))
    algebra_results = load_json(os.path.join(rdir, "algebra", "all_algebra_results.json"))
    ablation_results = load_json(os.path.join(rdir, "ablations", "ablation_results.json"))
    correlation = load_json(os.path.join(rdir, "analysis", "bgd_correlation.json"))
    interference = load_json(os.path.join(rdir, "algebra", "interference_metrics.json"))

    lines = [
        "# GrassMerge Experiment Results Summary\n",
        f"Generated automatically from `{rdir}/`\n",
    ]

    lines.append(summarize_grassmerge_vs_baselines(eval_results))

    if correlation:
        lines.append("\n## Interference Metric Correlations\n")
        lines.append("| Metric | Spearman ρ | p-value | Pearson r | p-value | n |")
        lines.append("|--------|-----------|---------|-----------|---------|---|")
        for metric, stats in correlation.items():
            if isinstance(stats, dict) and "spearman_rho" in stats:
                lines.append(
                    f"| {metric} | {stats['spearman_rho']:.4f} | {stats['spearman_p']:.4f} | "
                    f"{stats['pearson_r']:.4f} | {stats['pearson_p']:.4f} | {stats['n']} |"
                )

    if ablation_results:
        lines.append("\n## Ablation Studies\n")
        rank_abl = ablation_results.get("rank_ablation", {})
        if rank_abl:
            lines.append("### Rank Ablation\n")
            lines.append("| Rank | Cosine to D1 | Cosine to D2 | Recon. Error | Time (s) |")
            lines.append("|------|-------------|-------------|-------------|----------|")
            for key, data in sorted(rank_abl.items()):
                lines.append(
                    f"| {data.get('rank', '?')} | {data.get('cosine_a', '?')} | "
                    f"{data.get('cosine_b', '?')} | {data.get('reconstruction_error', '?')} | "
                    f"{data.get('compose_time', '?')} |"
                )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    logger.info("Summary written to %s", args.output)


if __name__ == "__main__":
    main()
