#!/usr/bin/env python3
"""Generate comprehensive paper-quality results tables and analysis.

Reads all experiment outputs and generates:
1. Main results table: GrassMerge vs all baselines on all domain pairs
2. Ablation summary tables
3. BGD correlation analysis
4. Per-domain performance breakdown
5. LaTeX-formatted tables ready for paper
"""

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        logger.warning("File not found: %s", path)
        return {}
    with open(path) as f:
        return json.load(f)


def generate_main_results_table(eval_results: dict) -> str:
    """Generate the main comparison table: individual vs merged accuracy."""
    individual = eval_results.get("individual_loras", {})
    grassmerge = eval_results.get("grassmerge", {})
    baselines = eval_results.get("baselines", {})
    base_model = eval_results.get("base_model", {})

    lines = ["# Main Results: GrassMerge vs Baselines\n"]

    # Individual LoRA performance
    lines.append("## Individual Domain LoRA Performance\n")
    lines.append("| Domain | Benchmark | Base Model | + LoRA | Improvement |")
    lines.append("|--------|-----------|------------|--------|-------------|")
    for domain, benches in sorted(individual.items()):
        for bench_name, metrics in sorted(benches.items()):
            acc = metrics.get("accuracy", -1)
            if acc < 0:
                continue
            base_acc = 0.0
            if domain in base_model:
                for bn, bm in base_model[domain].items():
                    if isinstance(bm, dict) and "accuracy" in bm:
                        base_acc = bm["accuracy"]
            improvement = acc - base_acc
            lines.append(f"| {domain} | {bench_name} | {base_acc:.4f} | {acc:.4f} | +{improvement:.4f} |")

    # Pairwise merge comparison
    lines.append("\n## Pairwise Merge Comparison\n")

    # Collect all methods and their results
    all_methods = {"GrassMerge": {}}
    for key, metrics in grassmerge.items():
        parts = key.split("_on_")
        if len(parts) >= 2:
            pair = parts[0]
            target = parts[1].split("_")[0]
            bench = "_".join(parts[1].split("_")[1:])
            all_methods["GrassMerge"][(pair, target, bench)] = metrics.get("accuracy", -1)

    for method_name, method_data in baselines.items():
        all_methods[method_name] = {}
        for key, metrics in method_data.items():
            parts = key.split("_on_")
            if len(parts) >= 2:
                pair = parts[0]
                target = parts[1].split("_")[0]
                bench = "_".join(parts[1].split("_")[1:])
                all_methods[method_name][(pair, target, bench)] = metrics.get("accuracy", -1)

    # Build comparison
    method_names = sorted(all_methods.keys())
    if method_names:
        header = "| Pair | Domain | " + " | ".join(method_names) + " | Individual |"
        separator = "|------|--------|" + "|".join(["--------"] * len(method_names)) + "|------------|"
        lines.append(header)
        lines.append(separator)

        all_pairs = set()
        for m in all_methods.values():
            for (pair, target, bench) in m:
                all_pairs.add((pair, target, bench))

        for pair, target, bench in sorted(all_pairs):
            # Get individual accuracy
            ind_acc = 0.0
            if target in individual:
                for bn, bm in individual[target].items():
                    if isinstance(bm, dict) and "accuracy" in bm:
                        ind_acc = bm["accuracy"]

            cells = []
            for m_name in method_names:
                acc = all_methods[m_name].get((pair, target, bench), -1)
                if acc >= 0:
                    cells.append(f"{acc:.4f}")
                else:
                    cells.append("-")
            lines.append(f"| {pair} | {target} | " + " | ".join(cells) + f" | {ind_acc:.4f} |")

    # Summary statistics
    lines.append("\n## Summary: Average Accuracy Retention\n")
    lines.append("| Method | Avg Accuracy | Avg Retention vs Individual |")
    lines.append("|--------|-------------|---------------------------|")

    for m_name in method_names:
        accs = [v for v in all_methods[m_name].values() if v >= 0]
        if accs:
            avg_acc = np.mean(accs)
            lines.append(f"| {m_name} | {avg_acc:.4f} | - |")

    return "\n".join(lines)


def generate_bgd_analysis(algebra_results: dict, eval_results: dict) -> str:
    """Generate BGD analysis section."""
    bgd_data = algebra_results.get("bgd", {})
    if not bgd_data:
        return "# BGD Analysis\n\nNo BGD data available."

    lines = ["# BGD Analysis\n"]
    names = bgd_data.get("domain_names", [])
    bgd_matrix = np.array(bgd_data.get("bgd_matrix", []))

    if len(names) > 0 and bgd_matrix.size > 0:
        lines.append("## BGD Heatmap (lower = more aligned subspaces)\n")
        lines.append("| " + " | ".join([""] + names) + " |")
        lines.append("|" + "|".join(["---"] * (len(names) + 1)) + "|")
        for i, name in enumerate(names):
            row = [name]
            for j in range(len(names)):
                if i == j:
                    row.append("0")
                else:
                    row.append(f"{bgd_matrix[i][j]:.3f}")
            lines.append("| " + " | ".join(row) + " |")

    # GrassMerge results: BGD vs composition quality
    grassmerge_data = algebra_results.get("grassmerge", {})
    if grassmerge_data:
        lines.append("\n## BGD vs Composition Quality\n")
        lines.append("| Pair | BGD | Cosine to D1 | Cosine to D2 | Time (s) |")
        lines.append("|------|-----|-------------|-------------|----------|")
        for pair, data in sorted(grassmerge_data.items()):
            bgd_val = data.get("bgd", 0)
            cos1 = data.get("cosine_to_d1", 0)
            cos2 = data.get("cosine_to_d2", 0)
            time_s = data.get("time_seconds", 0)
            lines.append(f"| {pair} | {bgd_val:.3f} | {cos1:.4f} | {cos2:.4f} | {time_s:.2f} |")

    return "\n".join(lines)


def generate_ablation_summary(ablation_results: dict) -> str:
    """Generate ablation study summary."""
    lines = ["# Ablation Studies\n"]

    # Rank ablation
    rank_data = ablation_results.get("rank_ablation", {})
    if rank_data:
        lines.append("## Rank Sensitivity\n")
        lines.append("| Rank | Cosine to D1 | Cosine to D2 | Reconstruction Error | Time (s) |")
        lines.append("|------|-------------|-------------|---------------------|----------|")
        for key, data in sorted(rank_data.items()):
            r = data.get("rank", 0)
            cos_a = data.get("cosine_a", 0)
            cos_b = data.get("cosine_b", 0)
            err = data.get("reconstruction_error", 0)
            time_s = data.get("compose_time", 0)
            lines.append(f"| {r} | {cos_a:.4f} | {cos_b:.4f} | {err:.4f} | {time_s:.3f} |")

    # N-way composition
    compose_data = ablation_results.get("compose_count_ablation", {})
    if compose_data:
        lines.append("\n## N-way Composition Scalability\n")
        lines.append("| N Domains | Avg Cosine | Max SV | Frobenius Norm | Time (s) |")
        lines.append("|-----------|-----------|--------|----------------|----------|")
        for key, data in sorted(compose_data.items()):
            n = data.get("num_domains", 0)
            avg_cos = data.get("avg_cosine", 0)
            max_sv = data.get("max_singular_value", 0)
            fro = data.get("total_frobenius_norm", 0)
            time_s = data.get("compose_time", 0)
            lines.append(f"| {n} | {avg_cos:.4f} | {max_sv:.4f} | {fro:.2f} | {time_s:.3f} |")

    # Interpolation
    interp_data = ablation_results.get("interpolation_type_ablation", {})
    if interp_data:
        lines.append("\n## Interpolation: Linear vs Geodesic\n")
        lines.append("| Alpha | Linear Norm | Geodesic Norm | Difference |")
        lines.append("|-------|-----------|-------------|-----------|")
        for key, data in sorted(interp_data.items()):
            alpha = data.get("alpha", 0)
            lin = data.get("linear_avg_norm", 0)
            geo = data.get("geodesic_avg_norm", 0)
            diff = data.get("linear_geodesic_avg_diff", 0)
            lines.append(f"| {alpha:.2f} | {lin:.4f} | {geo:.4f} | {diff:.6f} |")

    return "\n".join(lines)


def generate_latex_table(eval_results: dict) -> str:
    """Generate LaTeX-formatted main results table."""
    lines = [
        "% Auto-generated LaTeX table",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{GrassMerge vs baselines: pairwise merge accuracy on 6 core domains.}",
        "\\label{tab:main_results}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & Math & Code & Medical & Science & History & Philosophy \\\\",
        "\\midrule",
    ]

    individual = eval_results.get("individual_loras", {})
    # Get individual accuracies
    ind_accs = {}
    for domain in ["math", "code", "medical", "science", "history", "philosophy"]:
        if domain in individual:
            for bn, bm in individual[domain].items():
                if isinstance(bm, dict) and "accuracy" in bm:
                    ind_accs[domain] = bm["accuracy"]

    domains_order = ["math", "code", "medical", "science", "history", "philosophy"]
    ind_cells = " & ".join([f"{ind_accs.get(d, 0):.2f}" for d in domains_order])
    lines.append(f"Individual LoRA & {ind_cells} \\\\")
    lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    results_dir = Path("/home/claude/nips-text2subspace/results")

    eval_results = load_json(str(results_dir / "eval" / "eval_results.json"))
    algebra_results = load_json(str(results_dir / "algebra" / "all_algebra_results.json"))
    ablation_results = load_json(str(results_dir / "ablations" / "ablation_results.json"))
    correlation_results = load_json(str(results_dir / "analysis" / "bgd_correlation.json"))

    output_dir = results_dir / "paper_results"
    os.makedirs(output_dir, exist_ok=True)

    # Generate all reports
    if eval_results:
        main_table = generate_main_results_table(eval_results)
        with open(output_dir / "main_results.md", "w") as f:
            f.write(main_table)
        logger.info("Main results: %s", output_dir / "main_results.md")

        latex = generate_latex_table(eval_results)
        with open(output_dir / "main_table.tex", "w") as f:
            f.write(latex)
        logger.info("LaTeX table: %s", output_dir / "main_table.tex")

    if algebra_results:
        bgd = generate_bgd_analysis(algebra_results, eval_results)
        with open(output_dir / "bgd_analysis.md", "w") as f:
            f.write(bgd)
        logger.info("BGD analysis: %s", output_dir / "bgd_analysis.md")

    if ablation_results:
        ablation = generate_ablation_summary(ablation_results)
        with open(output_dir / "ablation_summary.md", "w") as f:
            f.write(ablation)
        logger.info("Ablation summary: %s", output_dir / "ablation_summary.md")

    if correlation_results:
        lines = ["# BGD-Performance Correlation\n"]
        lines.append("| Metric | Spearman rho | p-value | Pearson r | p-value | N |")
        lines.append("|--------|-------------|---------|-----------|---------|---|")
        for metric, data in sorted(correlation_results.items()):
            if "error" in data:
                lines.append(f"| {metric} | - | - | - | - | {data.get('n', 0)} |")
            else:
                lines.append(
                    f"| {metric} | {data.get('spearman_rho', 0):.4f} | "
                    f"{data.get('spearman_p', 1):.4f} | "
                    f"{data.get('pearson_r', 0):.4f} | "
                    f"{data.get('pearson_p', 1):.4f} | "
                    f"{data.get('n', 0)} |"
                )
        with open(output_dir / "correlation.md", "w") as f:
            f.write("\n".join(lines))
        logger.info("Correlation: %s", output_dir / "correlation.md")

    # Overall summary
    summary = ["# GrassMerge Experiment Summary\n"]
    summary.append(f"- Eval results: {'available' if eval_results else 'MISSING'}")
    summary.append(f"- Algebra results: {'available' if algebra_results else 'MISSING'}")
    summary.append(f"- Ablation results: {'available' if ablation_results else 'MISSING'}")
    summary.append(f"- Correlation results: {'available' if correlation_results else 'MISSING'}")

    with open(output_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(summary))

    logger.info("Summary: %s", output_dir / "SUMMARY.md")
    logger.info("All paper results generated in %s", output_dir)


if __name__ == "__main__":
    main()
