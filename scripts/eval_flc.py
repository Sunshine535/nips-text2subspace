#!/usr/bin/env python3
"""Evaluate Functional LoRA Composition vs baselines."""
import gc, json, os, sys, time, torch, tempfile
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("flc_eval")

MODEL = os.environ.get("MODEL", "/root/models/Qwen3.5-9B")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "/root/nips-text2subspace/results/sfc_loras_test")
DATASET_DIR = os.environ.get("DATASET_DIR", "/root/datasets")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "50"))
CALIB_SAMPLES = int(os.environ.get("CALIB_SAMPLES", "200"))
OUTPUT = os.environ.get("OUTPUT", "/root/nips-text2subspace/results/flc_results.json")

PAIRS = [("math","medical"), ("math","science"), ("science","philosophy"),
         ("medical","science"), ("medical","philosophy")]

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk

from src.functional_composition import (
    load_lora_delta_w, collect_module_inputs,
    functional_merge, save_functional_adapter,
)
from scripts.eval_sfc_downstream import (
    evaluate_model_mcq, load_adapter_weights,
    merge_task_arithmetic, merge_ties, save_merged_adapter,
)

LOCAL_DATASETS = {
    "math": ("gsm8k", "train", "question"),
    "medical": ("medmcqa", "train", "question"),
    "science": ("arc_challenge", "train", "question"),
    "philosophy": ("mmlu", "auxiliary_train", "question"),
    "history": ("mmlu", "auxiliary_train", "question"),
    "code": ("mbpp", "train", "text"),
}

PHILOSOPHY_SUBJECTS = {"philosophy", "moral_disputes", "moral_scenarios"}
HISTORY_SUBJECTS = {"high_school_us_history", "high_school_world_history", "prehistory", "high_school_european_history"}


def load_calibration_texts(domain, n=200):
    cfg = LOCAL_DATASETS[domain]
    ds = load_from_disk(os.path.join(DATASET_DIR, cfg[0]))
    split = cfg[1] if cfg[1] in ds else list(ds.keys())[0]
    data = ds[split]
    if domain == "philosophy":
        data = data.filter(lambda x: x.get("subject","") in PHILOSOPHY_SUBJECTS)
    elif domain == "history":
        data = data.filter(lambda x: x.get("subject","") in HISTORY_SUBJECTS)
    data = data.shuffle(seed=42).select(range(min(n, len(data))))
    field = cfg[2]
    return [str(row[field]) for row in data]


def get_lora_target_modules(adapter_path):
    import json as j
    cfg = j.load(open(os.path.join(adapter_path, "adapter_config.json")))
    targets = cfg.get("target_modules", [])
    return targets


def main():
    log.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", trust_remote_code=True)

    # Get LoRA target module names from first adapter
    first_adapter = os.path.join(ADAPTER_DIR, "math")
    target_module_types = get_lora_target_modules(first_adapter)
    log.info("LoRA targets: %s", target_module_types)

    # Build full module name list by matching model modules
    full_target_names = []
    for name, _ in model.named_modules():
        for t in target_module_types:
            if name.endswith("." + t):
                full_target_names.append(name)
    log.info("Found %d LoRA target modules in model", len(full_target_names))

    # Base model eval
    log.info("=== Base Model Eval ===")
    base_scores = {}
    all_domains = set(d for pair in PAIRS for d in pair)
    for domain in sorted(all_domains):
        if domain == "code":
            continue
        s = evaluate_model_mcq(model, tokenizer, domain, DATASET_DIR, N_SAMPLES, "cuda")
        base_scores[domain] = s
        log.info("  Base | %s: %.4f", domain, s["accuracy"])

    # Single adapter eval
    log.info("=== Single Adapter Eval ===")
    single_scores = {}
    for domain in sorted(all_domains):
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

    # Evaluate each pair
    pair_results = []
    for d1, d2 in PAIRS:
        log.info("\n" + "=" * 60)
        log.info("Pair: %s+%s", d1, d2)
        log.info("=" * 60)

        # Load delta_W for both adapters
        dw1 = load_lora_delta_w(os.path.join(ADAPTER_DIR, d1))
        dw2 = load_lora_delta_w(os.path.join(ADAPTER_DIR, d2))

        # Collect calibration inputs
        log.info("  Collecting calibration data...")
        calib_texts_1 = load_calibration_texts(d1, CALIB_SAMPLES)
        calib_texts_2 = load_calibration_texts(d2, CALIB_SAMPLES)

        target_keys = sorted(dw1.keys())
        log.info("  Collecting module inputs for %s (%d texts)...", d1, len(calib_texts_1))
        inputs_1 = collect_module_inputs(model, tokenizer, calib_texts_1, target_keys, batch_size=4, max_length=128)
        log.info("  Collecting module inputs for %s (%d texts)...", d2, len(calib_texts_2))
        inputs_2 = collect_module_inputs(model, tokenizer, calib_texts_2, target_keys, batch_size=4, max_length=128)

        # FLC merge
        log.info("  FLC merging...")
        merged_flc, diag = functional_merge([dw1, dw2], [inputs_1, inputs_2], rank=16)
        del inputs_1, inputs_2; gc.collect()

        # Save and evaluate FLC
        flc_scores = {}
        with tempfile.TemporaryDirectory() as tmp:
            state = {}
            for mod, (B, A) in merged_flc.items():
                state[mod + ".lora_A.weight"] = A.to(torch.bfloat16)
                state[mod + ".lora_B.weight"] = B.to(torch.bfloat16)
            from safetensors.torch import save_file
            save_file(state, os.path.join(tmp, "adapter_model.safetensors"))
            import shutil
            shutil.copy2(os.path.join(ADAPTER_DIR, d1, "adapter_config.json"),
                        os.path.join(tmp, "adapter_config.json"))
            pm = PeftModel.from_pretrained(model, tmp)
            for d in [d1, d2]:
                if d == "code": continue
                s = evaluate_model_mcq(pm, tokenizer, d, DATASET_DIR, N_SAMPLES, "cuda")
                flc_scores[d] = s
                log.info("  FLC  | %s: %.4f", d, s["accuracy"])
            model = pm.merge_and_unload()
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="sdpa", trust_remote_code=True)
            del pm; gc.collect(); torch.cuda.empty_cache()
        del merged_flc

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
                if d == "code": continue
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
                if d == "code": continue
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
            "flc": flc_scores, "ta": ta_scores, "ties": ties_scores,
        }
        pair_results.append(result)

        # Summary
        log.info("\n  %-12s %-12s %-12s %-8s", "Method", d1, d2, "Mean")
        log.info("  " + "-" * 44)
        for name, scores in [("Base", {d1: base_scores.get(d1,{}), d2: base_scores.get(d2,{})}),
                              ("FLC", flc_scores), ("TA", ta_scores), ("TIES", ties_scores)]:
            a = scores.get(d1,{}).get("accuracy",-1)
            b = scores.get(d2,{}).get("accuracy",-1)
            m = (a+b)/2 if a>=0 and b>=0 else -1
            log.info("  %-12s %-12.4f %-12.4f %-8.4f", name, a, b, m)

    final = {
        "method": "Functional LoRA Composition (FLC)",
        "base_model": base_scores,
        "single_adapter": single_scores,
        "pairs": pair_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2,
                  default=lambda o: float(o) if hasattr(o,"item") else str(o))
    log.info("Results saved to %s", OUTPUT)


if __name__ == "__main__":
    main()
