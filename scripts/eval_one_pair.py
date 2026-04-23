#!/usr/bin/env python3
"""Evaluate ONE method on ONE pair. Called as subprocess to isolate memory."""
import argparse, gc, json, os, sys, time, torch
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("eval1")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["sfc","ta","ties","single_a","single_b","base"])
    p.add_argument("--d1", required=True)
    p.add_argument("--d2", required=True)
    p.add_argument("--model", default="/root/models/Qwen3.5-9B")
    p.add_argument("--sae-dir", default="/root/saes/qwen3.5-9b")
    p.add_argument("--adapter-dir", default="/root/nips-text2subspace/results/sfc_loras_test")
    p.add_argument("--dataset-dir", default="/root/datasets")
    p.add_argument("--layers", default="7,15,23")
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--output", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", trust_remote_code=True)

    from scripts.eval_sfc_downstream import evaluate_model_mcq

    result = {"method": args.method, "pair": "%s+%s" % (args.d1, args.d2)}

    if args.method == "base":
        scores = {}
        for d in [args.d1, args.d2]:
            if d == "code": continue
            s = evaluate_model_mcq(model, tokenizer, d, args.dataset_dir, args.n_samples, "cuda")
            scores[d] = s
            log.info("Base | %s: %.4f", d, s["accuracy"])
        result["scores"] = scores

    elif args.method in ("single_a", "single_b"):
        from peft import PeftModel
        domain = args.d1 if args.method == "single_a" else args.d2
        adapter_path = os.path.join(args.adapter_dir, domain)
        pm = PeftModel.from_pretrained(model, adapter_path)
        s = evaluate_model_mcq(pm, tokenizer, domain, args.dataset_dir, args.n_samples, "cuda")
        log.info("Single(%s) | %s: %.4f", domain, domain, s["accuracy"])
        result["scores"] = {domain: s}

    elif args.method == "sfc":
        log.info("Loading SAEs...")
        from scripts.run_sfc_pilot import load_saes
        saes = load_saes(args.sae_dir, layers, device="cuda")

        import random
        random.seed(42)
        topics = ["gravity","evolution","democracy","calculus","photosynthesis",
                  "economics","philosophy","medicine","programming","history"]
        probes = ["Explain the concept of %s in detail." % topics[i%10] for i in range(100)]

        from src.sae_decomposition import collect_activations, compute_feature_profile, AdapterFeatureMap
        from src.sparse_feature_composition import sfc_compose, compute_fds
        from peft import PeftModel

        log.info("Collecting base activations...")
        base_acts = collect_activations(model, tokenizer, probes, list(saes.keys()),
            batch_size=2, max_length=128, device="cuda")

        feature_maps = {}
        for domain in [args.d1, args.d2]:
            log.info("Feature profile: %s", domain)
            pm = PeftModel.from_pretrained(model, os.path.join(args.adapter_dir, domain))
            acts = collect_activations(pm, tokenizer, probes, list(saes.keys()),
                batch_size=2, max_length=128, device="cuda")
            # Unload adapter properly
            model = pm.merge_and_unload()
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="sdpa", trust_remote_code=True)
            del pm; gc.collect(); torch.cuda.empty_cache()

            profiles = {}
            ta, tf = 0, 0
            for ln, sae in saes.items():
                if ln in base_acts and ln in acts:
                    p = compute_feature_profile(base_acts[ln], acts[ln], sae, domain, ln,
                        threshold_multiplier=3.0, device="cuda")
                    profiles[ln] = p
                    ta += p.support.numel()
                    tf += p.total_features
                    log.info("  %s: %d/%d active (%.2f%%)", ln, p.support.numel(), p.total_features, p.sparsity*100)
            del acts; gc.collect()
            gs = ta / max(tf, 1)
            feature_maps[domain] = AdapterFeatureMap(adapter_name=domain, profiles=profiles,
                global_sparsity=gs, total_active_features=ta, total_features=tf)

        fds = compute_fds(feature_maps[args.d1], feature_maps[args.d2])
        log.info("FDS: %.4f", fds["global_fds"])
        result["fds"] = fds["global_fds"]
        result["sparsity"] = {d: fm.global_sparsity for d, fm in feature_maps.items()}

        composed = sfc_compose([feature_maps[args.d1], feature_maps[args.d2]])

        from scripts.eval_sfc_downstream import apply_sfc_hooks
        handles = apply_sfc_hooks(model, saes, composed)
        scores = {}
        for d in [args.d1, args.d2]:
            if d == "code": continue
            s = evaluate_model_mcq(model, tokenizer, d, args.dataset_dir, args.n_samples, "cuda")
            scores[d] = s
            log.info("SFC | %s: %.4f", d, s["accuracy"])
        for h in handles: h.remove()
        result["scores"] = scores

    elif args.method in ("ta", "ties"):
        from peft import PeftModel
        from scripts.eval_sfc_downstream import load_adapter_weights, merge_task_arithmetic, merge_ties, save_merged_adapter
        import tempfile

        wa = load_adapter_weights(os.path.join(args.adapter_dir, args.d1))
        wb = load_adapter_weights(os.path.join(args.adapter_dir, args.d2))

        if args.method == "ta":
            merged = merge_task_arithmetic(wa, wb)
        else:
            merged = merge_ties(wa, wb)
        del wa, wb; gc.collect()

        scores = {}
        with tempfile.TemporaryDirectory() as tmp:
            save_merged_adapter(merged, os.path.join(args.adapter_dir, args.d1), tmp)
            del merged; gc.collect()
            pm = PeftModel.from_pretrained(model, tmp)
            for d in [args.d1, args.d2]:
                if d == "code": continue
                s = evaluate_model_mcq(pm, tokenizer, d, args.dataset_dir, args.n_samples, "cuda")
                scores[d] = s
                log.info("%s | %s: %.4f", args.method.upper(), d, s["accuracy"])
        result["scores"] = scores

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=lambda o: float(o) if hasattr(o,"item") else str(o))
    log.info("Saved to %s", args.output)

if __name__ == "__main__":
    main()
