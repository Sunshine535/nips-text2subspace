#!/usr/bin/env python3
"""Evaluate BCFF vs TA/TIES baselines."""
import gc, json, os, sys, time, torch, tempfile
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("bcff")

MODEL = os.environ.get("MODEL", "/root/models/Qwen3.5-9B")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "/root/nips-text2subspace/results/sfc_loras_test")
DATASET_DIR = os.environ.get("DATASET_DIR", "/root/datasets")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "50"))
OUTPUT = os.environ.get("OUTPUT", "/root/nips-text2subspace/results/bcff_results.json")

PAIRS = [("math","medical"), ("math","science"), ("science","philosophy"),
         ("medical","science"), ("medical","philosophy")]

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk

from src.cross_factor_fusion import load_lora_factors_v2, collect_module_inputs_for_bcff, bcff_merge, save_bcff_adapter
from scripts.eval_sfc_downstream import evaluate_model_mcq, load_adapter_weights, merge_task_arithmetic, merge_ties, save_merged_adapter

CALIB_MAP = {
    "math": ("gsm8k", "train", "question"),
    "medical": ("medmcqa", "train", "question"),
    "science": ("arc_challenge", "train", "question"),
    "philosophy": ("mmlu", "auxiliary_train", "question"),
}
PHIL_SUBJECTS = {"philosophy", "moral_disputes", "moral_scenarios"}


def load_calib_texts(domain, n=200):
    cfg = CALIB_MAP[domain]
    ds = load_from_disk(os.path.join(DATASET_DIR, cfg[0]))
    split = cfg[1] if cfg[1] in ds else list(ds.keys())[0]
    data = ds[split]
    if domain == "philosophy":
        data = data.filter(lambda x: x.get("subject","") in PHIL_SUBJECTS)
    data = data.shuffle(seed=42).select(range(min(n, len(data))))
    return [str(row[cfg[2]]) for row in data]


def main():
    log.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", trust_remote_code=True)

    # Base eval
    log.info("=== Base Model Eval ===")
    base_scores = {}
    for domain in sorted(set(d for p in PAIRS for d in p)):
        if domain == "code":
            continue
        s = evaluate_model_mcq(model, tokenizer, domain, DATASET_DIR, N_SAMPLES, "cuda")
        base_scores[domain] = s
        log.info("  Base | %s: %.4f", domain, s["accuracy"])

    # Single adapter eval
    log.info("=== Single Adapter Eval ===")
    single_scores = {}
    for domain in sorted(set(d for p in PAIRS for d in p)):
        if domain == "code":
            continue
        pm = PeftModel.from_pretrained(model, os.path.join(ADAPTER_DIR, domain))
        s = evaluate_model_mcq(pm, tokenizer, domain, DATASET_DIR, N_SAMPLES, "cuda")
        single_scores[domain] = s
        log.info("  Single(%s) | %s: %.4f", domain, domain, s["accuracy"])
        model = pm.merge_and_unload()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa", trust_remote_code=True)
        del pm; gc.collect(); torch.cuda.empty_cache()

    # Pairs
    pair_results = []
    for d1, d2 in PAIRS:
        log.info("\n" + "=" * 60)
        log.info("Pair: %s+%s", d1, d2)
        log.info("=" * 60)

        # Load LoRA factors
        f1 = load_lora_factors_v2(os.path.join(ADAPTER_DIR, d1))
        f2 = load_lora_factors_v2(os.path.join(ADAPTER_DIR, d2))
        log.info("  Loaded %d/%d modules for %s/%s", len(f1), len(f2), d1, d2)

        # Collect calibration inputs
        calib_1 = load_calib_texts(d1, 200)
        calib_2 = load_calib_texts(d2, 200)
        all_calib = calib_1 + calib_2

        target_mods = sorted(f1.keys())
        log.info("  Collecting calibration inputs (%d texts, %d modules)...", len(all_calib), len(target_mods))
        calib_inputs = collect_module_inputs_for_bcff(
            model, tokenizer, all_calib, target_mods, batch_size=4, max_length=128)
        log.info("  Got inputs for %d modules", len(calib_inputs))

        # BCFF merge
        log.info("  BCFF merging...")
        merged_bcff, diag = bcff_merge(f1, f2, calib_inputs, rank=16)
        del calib_inputs; gc.collect()

        # Evaluate BCFF
        bcff_scores = {}
        with tempfile.TemporaryDirectory() as tmp:
            save_bcff_adapter(merged_bcff, os.path.join(ADAPTER_DIR, d1), tmp)
            del merged_bcff; gc.collect()
            pm = PeftModel.from_pretrained(model, tmp)
            for d in [d1, d2]:
                if d == "code":
                    continue
                s = evaluate_model_mcq(pm, tokenizer, d, DATASET_DIR, N_SAMPLES, "cuda")
                bcff_scores[d] = s
                log.info("  BCFF | %s: %.4f", d, s["accuracy"])
            model = pm.merge_and_unload()
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="sdpa", trust_remote_code=True)
            del pm; gc.collect(); torch.cuda.empty_cache()

        # TA baseline
        log.info("  TA merging...")
        wa = load_adapter_weights(os.path.join(ADAPTER_DIR, d1))
        wb = load_adapter_weights(os.path.join(ADAPTER_DIR, d2))
        merged_ta = merge_task_arithmetic(wa, wb)
        ta_scores = {}
        with tempfile.TemporaryDirectory() as tmp:
            save_merged_adapter(merged_ta, os.path.join(ADAPTER_DIR, d1), tmp)
            del merged_ta; gc.collect()
            pm = PeftModel.from_pretrained(model, tmp)
            for d in [d1, d2]:
                if d == "code":
                    continue
                s = evaluate_model_mcq(pm, tokenizer, d, DATASET_DIR, N_SAMPLES, "cuda")
                ta_scores[d] = s
                log.info("  TA   | %s: %.4f", d, s["accuracy"])
            model = pm.merge_and_unload()
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="sdpa", trust_remote_code=True)
            del pm; gc.collect(); torch.cuda.empty_cache()

        # TIES baseline
        log.info("  TIES merging...")
        merged_ties = merge_ties(wa, wb)
        ties_scores = {}
        with tempfile.TemporaryDirectory() as tmp:
            save_merged_adapter(merged_ties, os.path.join(ADAPTER_DIR, d1), tmp)
            del merged_ties, wa, wb; gc.collect()
            pm = PeftModel.from_pretrained(model, tmp)
            for d in [d1, d2]:
                if d == "code":
                    continue
                s = evaluate_model_mcq(pm, tokenizer, d, DATASET_DIR, N_SAMPLES, "cuda")
                ties_scores[d] = s
                log.info("  TIES | %s: %.4f", d, s["accuracy"])
            model = pm.merge_and_unload()
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="sdpa", trust_remote_code=True)
            del pm; gc.collect(); torch.cuda.empty_cache()

        result = {
            "pair": "%s+%s" % (d1, d2),
            "bcff": bcff_scores, "ta": ta_scores, "ties": ties_scores,
            "bcff_diagnostics": {mod: d["coefficients"] for mod, d in diag.items()} if diag else {},
        }
        pair_results.append(result)

        # Summary table
        log.info("\n  %-12s %-12s %-12s %-8s", "Method", d1, d2, "Mean")
        log.info("  " + "-" * 44)
        for name, scores in [("Base", {d1: base_scores.get(d1,{}), d2: base_scores.get(d2,{})}),
                              ("BCFF", bcff_scores), ("TA", ta_scores), ("TIES", ties_scores)]:
            a = scores.get(d1,{}).get("accuracy",-1)
            b = scores.get(d2,{}).get("accuracy",-1)
            m = (a+b)/2 if a>=0 and b>=0 else -1
            log.info("  %-12s %-12.4f %-12.4f %-8.4f", name, a, b, m)

    final = {
        "method": "Bilinear Cross-Factor Fusion (BCFF)",
        "base_model": base_scores, "single_adapter": single_scores,
        "pairs": pair_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2,
                  default=lambda o: float(o) if hasattr(o,"item") else str(o))
    log.info("Results saved to %s", OUTPUT)


if __name__ == "__main__":
    main()
