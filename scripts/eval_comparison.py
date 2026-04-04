#!/usr/bin/env python3
"""Quick head-to-head eval: GrassMerge vs Task Arithmetic on key pairs."""
import json, os, sys, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import eval functions
from scripts.eval_domain_accuracy import (
    evaluate_on_benchmark, DOMAIN_BENCHMARKS, generate_response
)

PAIRS = [
    ("history", "philosophy"),
    ("history", "science"),
    ("philosophy", "science"),
]
MAX_SAMPLES = 50
BASE_MODEL = "Qwen/Qwen3-8B"


def load_and_eval(base_model_name, adapter_path, domain, tokenizer, max_samples=50):
    """Load model with adapter and evaluate."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    results = {}
    for bench_name, bench_cfg in DOMAIN_BENCHMARKS.get(domain, {}).items():
        cfg = {**bench_cfg, "max_samples": max_samples}
        results[bench_name] = evaluate_on_benchmark(model, tokenizer, cfg, domain)

    del model
    torch.cuda.empty_cache()
    return results


def main():
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    for d1, d2 in PAIRS:
        pair = f"{d1}+{d2}"
        print(f"\n{'='*60}", flush=True)
        print(f"  Evaluating pair: {pair}", flush=True)
        print(f"{'='*60}", flush=True)

        gm_path = f"results/algebra/grassmerge/{pair}"
        ta_path = f"results/algebra/baselines/task_arithmetic/{pair}"

        pair_results = {}

        for domain in [d1, d2]:
            print(f"\n  --- {domain} benchmark ---", flush=True)

            # GrassMerge
            print(f"  GrassMerge on {domain}...", flush=True)
            t0 = time.time()
            gm_res = load_and_eval(BASE_MODEL, gm_path, domain, tokenizer, MAX_SAMPLES)
            t1 = time.time()
            for bn, bm in gm_res.items():
                gm_acc = bm.get("accuracy", -1)
                print(f"    GrassMerge: {gm_acc:.4f} ({t1-t0:.0f}s)", flush=True)

            # Task Arithmetic
            print(f"  Task Arithmetic on {domain}...", flush=True)
            t0 = time.time()
            ta_res = load_and_eval(BASE_MODEL, ta_path, domain, tokenizer, MAX_SAMPLES)
            t1 = time.time()
            for bn, bm in ta_res.items():
                ta_acc = bm.get("accuracy", -1)
                print(f"    Task Arithmetic: {ta_acc:.4f} ({t1-t0:.0f}s)", flush=True)

            pair_results[domain] = {"grassmerge": gm_res, "task_arithmetic": ta_res}

        all_results[pair] = pair_results

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("  COMPARISON SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Pair':<25} {'Domain':<12} {'GrassMerge':>12} {'TaskArith':>12} {'Winner':>10}", flush=True)

    gm_wins, ta_wins, ties = 0, 0, 0
    for pair, domains in all_results.items():
        for domain, methods in domains.items():
            gm_acc = 0
            for bn, bm in methods.get("grassmerge", {}).items():
                gm_acc = bm.get("accuracy", 0)
            ta_acc = 0
            for bn, bm in methods.get("task_arithmetic", {}).items():
                ta_acc = bm.get("accuracy", 0)

            if gm_acc > ta_acc:
                winner = "GrassMerge"
                gm_wins += 1
            elif ta_acc > gm_acc:
                winner = "TaskArith"
                ta_wins += 1
            else:
                winner = "Tie"
                ties += 1

            print(f"{pair:<25} {domain:<12} {gm_acc:>12.4f} {ta_acc:>12.4f} {winner:>10}", flush=True)

    print(f"\nGrassMerge wins: {gm_wins}, Task Arithmetic wins: {ta_wins}, Ties: {ties}", flush=True)

    with open("results/eval/comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("Saved to results/eval/comparison_results.json", flush=True)


if __name__ == "__main__":
    main()
