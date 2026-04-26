#!/usr/bin/env python3
"""UCAR P0 Gate: Oracle utility ceiling over candidate pool.

Runs base + single adapters + static TA/TIES/DARE on eval items and computes
the per-item oracle-best candidate. If oracle accuracy <= base, no routing
method can help with this candidate pool.

Usage:
    python scripts/eval_utility_oracle.py \
      --config configs/ucar_minimal.yaml \
      --domains science,medical \
      --seed 1 --sample_seed 1 \
      --max_samples 50 \
      --output_dir results-synced/ucar_oracle_seed1_sample1
"""
import argparse
import gc
import json
import logging
import os
import sys
import tempfile
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ucar_oracle")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ucar_minimal.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--domains", default="science,medical")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sample_seed", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--metric_mode", default="logprob_mcq")
    p.add_argument("--output_dir", default="results-synced/ucar_oracle_seed1_sample1")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import yaml
        cfg = yaml.safe_load(open(args.config))
    except Exception:
        cfg = {}

    base_model = args.model or cfg.get("base_model", "/root/models/Qwen3.5-9B")
    adapter_dir = args.adapter_dir or cfg.get("adapter_dir", "/root/nips-text2subspace/results/sfc_loras_test")
    dataset_dir = args.dataset_dir or cfg.get("dataset_dir", "/root/datasets")
    domains = args.domains.split(",")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "effective_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)

    from scripts.eval_sfc_downstream import (
        load_eval_dataset, format_mcq_prompt, DOMAIN_EVAL,
        load_adapter_weights, merge_task_arithmetic,
        merge_ties, merge_dare, save_merged_adapter,
    )
    from src.utility_oracle import (
        evaluate_candidate_on_items, build_utility_table,
        summarize_oracle, save_utility_table,
    )

    # Build evaluation items
    log.info("Building eval items (sample_seed=%d, max_samples=%d)...",
             args.sample_seed, args.max_samples)
    items = []
    for d in domains:
        cfg_d = DOMAIN_EVAL[d]
        ds = load_eval_dataset(d, dataset_dir, args.max_samples,
                               sample_seed=args.sample_seed)
        for i, ex in enumerate(ds):
            prompt, gold = format_mcq_prompt(ex, cfg_d["type"])
            if not gold:
                continue
            gold_idx = ["A", "B", "C", "D"].index(gold.upper()) if gold.upper() in "ABCD" else 0
            items.append({
                "item_idx": len(items),
                "domain": d,
                "prompt": prompt,
                "gold_label": gold,
                "gold_idx": gold_idx,
            })
    log.info("  %d eval items across %s", len(items), domains)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("Loading model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def fresh_model():
        return AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa", trust_remote_code=True)

    model = fresh_model()
    all_candidate_results = {}

    # 1. Base
    log.info("=== Candidate: base ===")
    all_candidate_results["base"] = evaluate_candidate_on_items(
        model, tokenizer, items, "base")
    base_acc = sum(1 for r in all_candidate_results["base"] if r.correct) / len(items)
    log.info("  base accuracy: %.4f", base_acc)

    # 2. Single adapters
    from peft import PeftModel
    for d in domains:
        cname = f"single_{d}"
        log.info("=== Candidate: %s ===", cname)
        pm = PeftModel.from_pretrained(model, os.path.join(adapter_dir, d))
        all_candidate_results[cname] = evaluate_candidate_on_items(
            pm, tokenizer, items, cname)
        acc = sum(1 for r in all_candidate_results[cname] if r.correct) / len(items)
        log.info("  %s accuracy: %.4f", cname, acc)
        model = pm.merge_and_unload()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = fresh_model()

    # 3. Static merges (TA, TIES, DARE)
    d1, d2 = domains[0], domains[1]
    wa = load_adapter_weights(os.path.join(adapter_dir, d1))
    wb = load_adapter_weights(os.path.join(adapter_dir, d2))

    for mname, mfn, mkwargs in [
        ("static_TA", merge_task_arithmetic, {}),
        ("static_TIES", merge_ties, {"density": 0.5}),
        ("static_DARE", merge_dare, {"drop_rate": 0.5}),
    ]:
        log.info("=== Candidate: %s ===", mname)
        merged = mfn(wa, wb, **mkwargs)
        with tempfile.TemporaryDirectory() as tmp:
            save_merged_adapter(merged, os.path.join(adapter_dir, d1), tmp)
            del merged
            gc.collect()
            pm = PeftModel.from_pretrained(model, tmp)
            all_candidate_results[mname] = evaluate_candidate_on_items(
                pm, tokenizer, items, mname)
            acc = sum(1 for r in all_candidate_results[mname] if r.correct) / len(items)
            log.info("  %s accuracy: %.4f", mname, acc)
            model = pm.merge_and_unload()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = fresh_model()
    del wa, wb
    gc.collect()

    # Build oracle table
    log.info("=== Oracle Analysis ===")
    utility_rows = build_utility_table(all_candidate_results)
    candidate_names = list(all_candidate_results.keys())
    summary = summarize_oracle(utility_rows, candidate_names)

    log.info("  base accuracy:       %.4f", summary["base_accuracy"])
    log.info("  oracle accuracy:     %.4f", summary["oracle_accuracy"])
    log.info("  oracle lift:         %.4f", summary["oracle_accuracy_lift"])
    log.info("  complementarity:     %d / %d (%.1f%%)",
             summary["complementarity_items"], summary["n_items"],
             100 * summary["complementarity_rate"])
    log.info("  oracle selection:    %s", summary["oracle_selection_counts"])
    log.info("  oracle beats base:   %s", summary["oracle_beats_base"])

    # Save
    save_utility_table(utility_rows,
                       os.path.join(args.output_dir, "utility_table.jsonl"))
    with open(os.path.join(args.output_dir, "oracle_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Write report
    report_path = os.path.join(args.output_dir, "UCAR_ORACLE_SUMMARY.md")
    with open(report_path, "w") as f:
        f.write("# UCAR Oracle Utility Ceiling\n\n")
        f.write(f"Seed: {args.seed}, Sample seed: {args.sample_seed}\n")
        f.write(f"Domains: {domains}, Items: {summary['n_items']}\n")
        f.write(f"Metric: {args.metric_mode}\n\n")
        f.write("## Per-candidate accuracy\n\n")
        f.write("| Candidate | Accuracy |\n|---|---|\n")
        for c, acc in sorted(summary["per_candidate_accuracy"].items(),
                             key=lambda x: -x[1]):
            f.write(f"| {c} | {acc:.4f} |\n")
        f.write(f"\n## Oracle\n\n")
        f.write(f"- Oracle accuracy: **{summary['oracle_accuracy']:.4f}**\n")
        f.write(f"- Base accuracy: {summary['base_accuracy']:.4f}\n")
        f.write(f"- Oracle lift: **{summary['oracle_accuracy_lift']:.4f}**\n")
        f.write(f"- Complementarity items: {summary['complementarity_items']} / {summary['n_items']}\n")
        f.write(f"- Oracle selection counts: {summary['oracle_selection_counts']}\n")
        f.write(f"\n## Gate Decision\n\n")
        if summary["oracle_beats_base"]:
            f.write("**PASS**: Oracle > Base. Routing has potential with this candidate pool.\n")
        else:
            f.write("**FAIL**: Oracle <= Base. No routing method can help. "
                    "Candidate pool must be improved before method work.\n")

    log.info("Report -> %s", report_path)
    log.info("Gate result: %s", "PASS" if summary["oracle_beats_base"] else "FAIL")

    if not summary["oracle_beats_base"]:
        fail_path = os.path.join("reports", "CANDIDATE_POOL_NO_ORACLE_GAIN.md")
        os.makedirs("reports", exist_ok=True)
        with open(fail_path, "w") as f:
            f.write("# CANDIDATE_POOL_NO_ORACLE_GAIN\n\n")
            f.write(f"Oracle accuracy ({summary['oracle_accuracy']:.4f}) <= "
                    f"Base accuracy ({summary['base_accuracy']:.4f}).\n")
            f.write("No routing method can help with the current candidate pool.\n")
            f.write("Action: retrain/replace adapters or change evaluation domains.\n")
        log.info("STOP: %s", fail_path)


if __name__ == "__main__":
    main()
