#!/usr/bin/env python3
"""Memory-lean SFC evaluation — process one pair at a time."""
import gc, json, os, sys, time, torch
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, "/root/nips-text2subspace")

MODEL = "/root/models/Qwen3.5-9B"
SAE_DIR = "/root/saes/qwen3.5-9b"
ADAPTER_DIR = "/root/nips-text2subspace/results/sfc_loras_test"
DATASET_DIR = "/root/datasets"
LAYERS = [7, 15, 23]
N_SAMPLES = 50
PROBE_SIZE = 100
PAIRS = [("math","medical"), ("math","science"), ("science","philosophy")]
OUTPUT = "/root/nips-text2subspace/results/sfc_downstream.json"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("eval")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

log.info("Loading tokenizer + model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="sdpa", trust_remote_code=True)

log.info("Loading SAEs to GPU...")
from scripts.run_sfc_pilot import load_saes
saes = load_saes(SAE_DIR, LAYERS, device="cuda")
log.info("SAEs loaded: %s", list(saes.keys()))

import random
random.seed(42)
topics = ["gravity","evolution","democracy","calculus","photosynthesis",
          "economics","philosophy","medicine","programming","history"]
probes = []
for i in range(PROBE_SIZE):
    probes.append("Explain the concept of %s in detail." % topics[i % len(topics)])

from src.sae_decomposition import collect_activations, compute_feature_profile, AdapterFeatureMap
from src.sparse_feature_composition import sfc_compose, compute_fds
from scripts.eval_sfc_downstream import (
    evaluate_model_mcq, load_adapter_weights, merge_task_arithmetic,
    merge_ties, save_merged_adapter
)

log.info("Collecting base activations...")
base_acts = collect_activations(model, tokenizer, probes, list(saes.keys()),
    batch_size=2, max_length=128, device="cuda")
for k, v in base_acts.items():
    log.info("  %s: %s", k, v.shape)

def compute_profile(domain):
    path = os.path.join(ADAPTER_DIR, domain)
    log.info("Computing profile: %s", domain)
    pm = PeftModel.from_pretrained(model, path)
    acts = collect_activations(pm, tokenizer, probes, list(saes.keys()),
        batch_size=2, max_length=128, device="cuda")
    del pm
    gc.collect()
    torch.cuda.empty_cache()

    profiles = {}
    total_active = 0
    total_features = 0
    for ln, sae in saes.items():
        if ln in base_acts and ln in acts:
            p = compute_feature_profile(base_acts[ln], acts[ln], sae, domain, ln,
                threshold_multiplier=3.0, device="cuda")
            profiles[ln] = p
            total_active += p.support.numel()
            total_features += p.total_features
            log.info("  %s: %d/%d active (%.2f%%)", ln, p.support.numel(), p.total_features, p.sparsity*100)
    del acts
    gc.collect()

    gs = total_active / max(total_features, 1)
    return AdapterFeatureMap(adapter_name=domain, profiles=profiles,
        global_sparsity=gs, total_active_features=total_active, total_features=total_features)

# Base model eval
all_domains = set(d for pair in PAIRS for d in pair)
log.info("=== Base Model Eval ===")
base_scores = {}
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
    del pm
    gc.collect()
    torch.cuda.empty_cache()

# Pairs
pair_results = []
for d1, d2 in PAIRS:
    sep = "=" * 60
    log.info("\n%s", sep)
    log.info("Pair: %s+%s", d1, d2)
    log.info("%s", sep)

    fm1 = compute_profile(d1)
    fm2 = compute_profile(d2)

    fds = compute_fds(fm1, fm2)
    log.info("  FDS: %.4f (overlap=%d, union=%d)", fds["global_fds"], fds["total_overlap"], fds["total_union"])

    composed = sfc_compose([fm1, fm2])

    # SFC-Exact
    from scripts.eval_sfc_downstream import apply_sfc_hooks
    handles = apply_sfc_hooks(model, saes, composed)
    sfc_scores = {}
    for d in [d1, d2]:
        s = evaluate_model_mcq(model, tokenizer, d, DATASET_DIR, N_SAMPLES, "cuda")
        sfc_scores[d] = s
        log.info("  SFC  | %s: %.4f", d, s["accuracy"])
    for h in handles:
        h.remove()

    # Task Arithmetic
    import tempfile
    wa = load_adapter_weights(os.path.join(ADAPTER_DIR, d1))
    wb = load_adapter_weights(os.path.join(ADAPTER_DIR, d2))

    merged_ta = merge_task_arithmetic(wa, wb)
    ta_scores = {}
    with tempfile.TemporaryDirectory() as tmp:
        save_merged_adapter(merged_ta, os.path.join(ADAPTER_DIR, d1), tmp)
        pm = PeftModel.from_pretrained(model, tmp)
        for d in [d1, d2]:
            s = evaluate_model_mcq(pm, tokenizer, d, DATASET_DIR, N_SAMPLES, "cuda")
            ta_scores[d] = s
            log.info("  TA   | %s: %.4f", d, s["accuracy"])
        del pm
        gc.collect()
        torch.cuda.empty_cache()
    del merged_ta

    # TIES
    merged_ties = merge_ties(wa, wb)
    ties_scores = {}
    with tempfile.TemporaryDirectory() as tmp:
        save_merged_adapter(merged_ties, os.path.join(ADAPTER_DIR, d1), tmp)
        pm = PeftModel.from_pretrained(model, tmp)
        for d in [d1, d2]:
            s = evaluate_model_mcq(pm, tokenizer, d, DATASET_DIR, N_SAMPLES, "cuda")
            ties_scores[d] = s
            log.info("  TIES | %s: %.4f", d, s["accuracy"])
        del pm
        gc.collect()
        torch.cuda.empty_cache()
    del merged_ties, wa, wb, fm1, fm2, composed
    gc.collect()
    torch.cuda.empty_cache()

    pair_results.append({
        "pair": "%s+%s" % (d1, d2), "fds": fds["global_fds"],
        "sfc_exact": sfc_scores, "task_arithmetic": ta_scores,
        "ties": ties_scores,
    })

    # Summary
    log.info("\n  %-15s %-15s %-15s %-10s", "Method", d1, d2, "Mean")
    log.info("  " + "-" * 55)
    for name, scores in [("Base", {d1: base_scores.get(d1,{}), d2: base_scores.get(d2,{})}),
                          ("SFC", sfc_scores), ("TA", ta_scores), ("TIES", ties_scores)]:
        a = scores.get(d1,{}).get("accuracy",-1)
        b = scores.get(d2,{}).get("accuracy",-1)
        m = (a+b)/2 if a>=0 and b>=0 else -1
        log.info("  %-15s %-15.4f %-15.4f %-10.4f", name, a, b, m)

results = {"base_model": base_scores, "single_adapter": single_scores,
           "pairs": pair_results, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2,
              default=lambda o: float(o) if hasattr(o, "item") else str(o))
log.info("Results saved to %s", OUTPUT)
