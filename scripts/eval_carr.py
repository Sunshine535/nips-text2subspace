#!/usr/bin/env python3
"""Evaluate CARR vs baselines with A/B/C comparison."""
import argparse, gc, json, logging, os, sys, time
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("carr_eval")

CALIB_MAP = {
    "science": ("arc_challenge", "train", "question"),
    "medical": ("medmcqa", "train", "question"),
    "math": ("gsm8k", "train", "question"),
}

def load_calib_texts(domain, dataset_dir, n=200, seed=42):
    from datasets import load_from_disk
    cfg = CALIB_MAP[domain]
    ds = load_from_disk(os.path.join(dataset_dir, cfg[0]))
    split = cfg[1] if cfg[1] in ds else list(ds.keys())[0]
    data = ds[split].shuffle(seed=seed).select(range(min(n, len(ds[split]))))
    return [str(row[cfg[2]]) for row in data]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen3.5-9B")
    p.add_argument("--adapter_dir", default="/root/nips-text2subspace/results/sfc_loras_test")
    p.add_argument("--dataset_dir", default="/root/datasets")
    p.add_argument("--domains", default="science,medical")
    p.add_argument("--mode", default="all", choices=["static_only","carr_full","carr_no_mechanism","all"])
    p.add_argument("--carr_checkpoint", default="/root/nips-text2subspace/results/carr_checkpoints/carr_router.pt")
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--calib_samples", type=int, default=200)
    p.add_argument("--train_steps", type=int, default=200)
    p.add_argument("--output", default="/root/nips-text2subspace/results/carr_abc.json")
    return p.parse_args()

def evaluate_model(model, tokenizer, domain, dataset_dir, n_samples, device="cuda"):
    """Simple MCQ evaluation."""
    from scripts.eval_sfc_downstream import evaluate_model_mcq
    return evaluate_model_mcq(model, tokenizer, domain, dataset_dir, n_samples, device)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    domains = args.domains.split(",")
    d1, d2 = domains[0], domains[1]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def fresh_model():
        return AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa", trust_remote_code=True)

    model = fresh_model()
    results = {"seed": args.seed, "domains": domains, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # Base model eval
    log.info("=== Base Model ===")
    base_scores = {}
    for d in domains:
        s = evaluate_model(model, tokenizer, d, args.dataset_dir, args.max_samples)
        base_scores[d] = s
        log.info("  Base | %s: %.4f", d, s["accuracy"])
    results["base"] = base_scores

    # Single adapter eval
    log.info("=== Single Adapters ===")
    from peft import PeftModel
    single_scores = {}
    for d in domains:
        pm = PeftModel.from_pretrained(model, os.path.join(args.adapter_dir, d))
        s = evaluate_model(pm, tokenizer, d, args.dataset_dir, args.max_samples)
        single_scores[d] = s
        log.info("  Single(%s) | %s: %.4f", d, d, s["accuracy"])
        model = pm.merge_and_unload()
        del model; gc.collect(); torch.cuda.empty_cache()
        model = fresh_model()
    results["single"] = single_scores

    # Load adapter factors
    from src.cross_factor_fusion import load_lora_factors_v2
    f1 = load_lora_factors_v2(os.path.join(args.adapter_dir, d1))
    f2 = load_lora_factors_v2(os.path.join(args.adapter_dir, d2))
    target_modules = sorted(f1.keys())

    static_delta_ws = {}
    adapter_delta_ws = [{}, {}]
    for mod in target_modules:
        B1, A1 = f1[mod]
        B2, A2 = f2[mod]
        dw1 = (B1 @ A1).cuda()
        dw2 = (B2 @ A2).cuda()
        static_delta_ws[mod] = (dw1 + dw2) / 2
        adapter_delta_ws[0][mod] = dw1
        adapter_delta_ws[1][mod] = dw2

    modes_to_run = ["static_only", "carr_full", "carr_no_mechanism"] if args.mode == "all" else [args.mode]

    for mode in modes_to_run:
        log.info("\n=== Mode: %s ===", mode)

        if mode == "static_only":
            # A: Just apply static TA merge via hooks
            import tempfile
            from scripts.eval_sfc_downstream import load_adapter_weights, merge_task_arithmetic, save_merged_adapter
            wa = load_adapter_weights(os.path.join(args.adapter_dir, d1))
            wb = load_adapter_weights(os.path.join(args.adapter_dir, d2))
            merged = merge_task_arithmetic(wa, wb)
            scores = {}
            with tempfile.TemporaryDirectory() as tmp:
                save_merged_adapter(merged, os.path.join(args.adapter_dir, d1), tmp)
                del merged, wa, wb; gc.collect()
                pm = PeftModel.from_pretrained(model, tmp)
                for d in domains:
                    s = evaluate_model(pm, tokenizer, d, args.dataset_dir, args.max_samples)
                    scores[d] = s
                    log.info("  %s | %s: %.4f", mode, d, s["accuracy"])
                model = pm.merge_and_unload()
                del model; gc.collect(); torch.cuda.empty_cache()
                model = fresh_model()
            results[mode] = scores

        else:
            # CARR modes: train a router first
            from src.conflict_aware_routing import CARRConfig, ConflictAwareResidualRouter, CARRHook
            from src.conflict_diagnostics import compute_activation_gram, compute_pair_conflict
            from src.cross_factor_fusion import collect_module_inputs_for_bcff

            d_model = model.config.hidden_size

            if mode == "carr_no_mechanism":
                config = CARRConfig(n_adapters=2, d_model=d_model, gate_hidden_dim=128,
                                   use_reliability=False, use_conflict=False, use_base_fallback=True)
            else:
                config = CARRConfig(n_adapters=2, d_model=d_model, gate_hidden_dim=128,
                                   use_reliability=True, use_conflict=True, use_base_fallback=True)

            router = ConflictAwareResidualRouter(config).cuda()

            # Compute conflict scores
            module_conflict_scores = {}
            if config.use_conflict:
                calib_texts = load_calib_texts(d1, args.dataset_dir, args.calib_samples//2) + \
                              load_calib_texts(d2, args.dataset_dir, args.calib_samples//2)
                module_inputs = collect_module_inputs_for_bcff(
                    model, tokenizer, calib_texts, target_modules,
                    batch_size=4, max_length=128)
                dw_list = [
                    {mod: adapter_delta_ws[0][mod].cpu() for mod in target_modules},
                    {mod: adapter_delta_ws[1][mod].cpu() for mod in target_modules},
                ]
                grams = compute_activation_gram(dw_list, module_inputs)
                for mod in target_modules:
                    if mod in grams:
                        m = compute_pair_conflict(grams[mod])
                        module_conflict_scores[mod] = torch.tensor(
                            [m["interference_ratio"], 1 - m["interference_ratio"]])
                del module_inputs, grams; gc.collect()

            # Quick train router
            log.info("  Training %s router (%d steps)...", mode, args.train_steps)
            router.train()
            hook = CARRHook(router, static_delta_ws, adapter_delta_ws,
                           module_conflict_scores=module_conflict_scores, training=True)
            hook.attach(model)

            calib = load_calib_texts(d1, args.dataset_dir, 100) + load_calib_texts(d2, args.dataset_dir, 100)
            calib_enc = [tokenizer(t, return_tensors="pt", truncation=True,
                                  max_length=128, padding="max_length") for t in calib]

            optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)
            for step in range(args.train_steps):
                idx = step % len(calib_enc)
                inp = {k: v.cuda() for k, v in calib_enc[idx].items() if k in ("input_ids","attention_mask")}
                out = model(**inp, labels=inp["input_ids"])
                loss = out.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                hook.gate_stats.clear()
                if step % 50 == 0:
                    log.info("    step=%d loss=%.4f", step, loss.item())

            hook.detach()

            # Evaluate
            router.eval()
            hook2 = CARRHook(router, static_delta_ws, adapter_delta_ws,
                            module_conflict_scores=module_conflict_scores, training=False)
            hook2.attach(model)
            scores = {}
            for d in domains:
                s = evaluate_model(model, tokenizer, d, args.dataset_dir, args.max_samples)
                scores[d] = s
                log.info("  %s | %s: %.4f", mode, d, s["accuracy"])

            gate_stats = hook2.get_aggregated_stats()
            hook2.detach()
            del router; gc.collect(); torch.cuda.empty_cache()

            results[mode] = scores
            results[mode + "_gate_stats"] = {
                mod: {k: v for k, v in s.items() if not isinstance(v, list)}
                for mod, s in gate_stats.items()
            }

    # Summary
    log.info("\n" + "=" * 60)
    log.info("A/B/C COMPARISON (seed=%d)", args.seed)
    log.info("=" * 60)
    log.info("%-25s %-12s %-12s %-8s", "Method", d1, d2, "Mean")
    log.info("-" * 57)
    for name in ["base", "single", "static_only", "carr_no_mechanism", "carr_full"]:
        if name not in results:
            continue
        s = results[name]
        a1 = s.get(d1, {}).get("accuracy", -1)
        a2 = s.get(d2, {}).get("accuracy", -1)
        mean = (a1 + a2) / 2 if a1 >= 0 and a2 >= 0 else -1
        label = {"base": "Base", "single": "Single best", "static_only": "A: Static TA",
                 "carr_no_mechanism": "B: CARR no mech", "carr_full": "C: Full CARR"}.get(name, name)
        log.info("%-25s %-12.4f %-12.4f %-8.4f", label, a1, a2, mean)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))
    log.info("Results saved to %s", args.output)

if __name__ == "__main__":
    main()
