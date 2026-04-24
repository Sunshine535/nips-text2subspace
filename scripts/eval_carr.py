#!/usr/bin/env python3
"""Evaluate CARR vs baselines with A/B/C comparison.

Config is the source of truth (configs/carr_minimal.yaml). CLI overrides when needed.
Saves effective_config.json + sample_manifest.jsonl + predictions.jsonl per run.
"""
import argparse
import gc
import json
import logging
import os
import sys
import time

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
    p.add_argument("--config", default="configs/carr_minimal.yaml",
                   help="YAML source of truth for CARR hyperparams")
    p.add_argument("--model", default=None, help="override config.base_model")
    p.add_argument("--adapter_dir", default=None, help="override config.adapter_dir")
    p.add_argument("--dataset_dir", default=None, help="override config.dataset_dir")
    p.add_argument("--domains", default="science,medical")
    p.add_argument("--mode", default="all",
                   choices=["static_only", "carr_full", "carr_no_mechanism",
                            "no_reliability", "no_conflict", "no_base_fallback",
                            "all", "ablations"])
    p.add_argument("--carr_checkpoint", default=None,
                   help="path to pretrained router.pt; if None, train fresh")
    p.add_argument("--max_samples", type=int, default=None, help="override config.eval.max_samples")
    p.add_argument("--metric_mode", default=None,
                   choices=[None, "generation", "logprob_mcq"],
                   help="override config.eval.metric_mode")
    p.add_argument("--seed", type=int, default=1, help="router/train seed")
    p.add_argument("--sample_seed", type=int, default=42, help="eval sample shuffle seed")
    p.add_argument("--calib_samples", type=int, default=None, help="override config.carr.calib_samples")
    p.add_argument("--train_steps", type=int, default=None, help="override config.carr.max_steps")
    p.add_argument("--output_dir", default="/root/nips-text2subspace/results/carr_abc_verified",
                   help="run directory for all output artefacts")
    p.add_argument("--dry_run_effective_config", action="store_true",
                   help="print effective config and exit (no GPU)")
    return p.parse_args()


def evaluate_model(model, tokenizer, domain, dataset_dir, n_samples, device="cuda",
                   sample_seed=42, metric_mode="generation",
                   manifest_out=None, predictions_out=None):
    from scripts.eval_sfc_downstream import evaluate_model_mcq
    return evaluate_model_mcq(
        model, tokenizer, domain, dataset_dir, n_samples,
        device=device, sample_seed=sample_seed, metric_mode=metric_mode,
        manifest_out=manifest_out, predictions_out=predictions_out,
    )


def main():
    args = parse_args()

    # ---- Phase 0: load config, resolve effective values ----
    from src.carr_config_loader import (
        load_run_config, save_effective_config, save_sample_manifest,
        assert_config_applied, MODE_OVERRIDES,
    )
    cfg = load_run_config(args.config)

    base_model = args.model or cfg.base_model
    adapter_dir = args.adapter_dir or cfg.adapter_dir
    dataset_dir = args.dataset_dir or cfg.dataset_dir
    max_samples = args.max_samples if args.max_samples is not None else cfg.eval_max_samples
    metric_mode = args.metric_mode or cfg.metric_mode
    calib_samples = args.calib_samples if args.calib_samples is not None else int(cfg.carr.get("calib_samples", 200))
    train_steps = args.train_steps if args.train_steps is not None else int(cfg.carr.get("max_steps", 200))

    os.makedirs(args.output_dir, exist_ok=True)
    eff_extra = {
        "cli_args": vars(args),
        "resolved": {
            "base_model": base_model, "adapter_dir": adapter_dir,
            "dataset_dir": dataset_dir, "max_samples": max_samples,
            "metric_mode": metric_mode, "calib_samples": calib_samples,
            "train_steps": train_steps,
        },
    }
    eff_path = save_effective_config(cfg, args.output_dir, extra=eff_extra)
    log.info("Effective config -> %s", eff_path)
    log.info("  base_model=%s  adapter_dir=%s", base_model, adapter_dir)
    log.info("  max_samples=%d  metric_mode=%s  sample_seed=%d",
             max_samples, metric_mode, args.sample_seed)
    log.info("  carr.top_k=%s  use_reliability=%s  use_conflict=%s  use_base_fallback=%s",
             cfg.carr.get("top_k"),
             cfg.carr.get("use_reliability"),
             cfg.carr.get("use_conflict"),
             cfg.carr.get("use_base_fallback"))

    if args.dry_run_effective_config:
        print(json.dumps({"effective_config": eff_path, "resolved": eff_extra["resolved"]},
                         indent=2))
        return

    torch.manual_seed(args.seed)
    domains = args.domains.split(",")
    d1, d2 = domains[0], domains[1]

    manifest_path = os.path.join(args.output_dir, "sample_manifest.jsonl")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def fresh_model():
        return AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa", trust_remote_code=True)

    model = fresh_model()
    results = {
        "seed": args.seed, "sample_seed": args.sample_seed,
        "domains": domains, "metric_mode": metric_mode,
        "max_samples": max_samples,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "effective_config_path": eff_path,
    }

    def eval_on(m, tag):
        predictions = os.path.join(args.output_dir, f"predictions_{tag}.jsonl")
        if os.path.exists(predictions):
            os.remove(predictions)
        scores = {}
        for d in domains:
            s = evaluate_model(
                m, tokenizer, d, dataset_dir, max_samples,
                sample_seed=args.sample_seed, metric_mode=metric_mode,
                manifest_out=manifest_path, predictions_out=predictions,
            )
            scores[d] = s
            log.info("  %s | %s: %.4f", tag, d, s["accuracy"])
        return scores

    # Base model
    log.info("=== Base Model ===")
    results["base"] = eval_on(model, "base")

    # Single adapters
    log.info("=== Single Adapters ===")
    from peft import PeftModel
    single_scores = {}
    for d in domains:
        pm = PeftModel.from_pretrained(model, os.path.join(adapter_dir, d))
        s_tag = f"single_{d}"
        predictions = os.path.join(args.output_dir, f"predictions_{s_tag}.jsonl")
        if os.path.exists(predictions):
            os.remove(predictions)
        for eval_d in domains:
            s = evaluate_model(
                pm, tokenizer, eval_d, dataset_dir, max_samples,
                sample_seed=args.sample_seed, metric_mode=metric_mode,
                manifest_out=manifest_path, predictions_out=predictions,
            )
            single_scores.setdefault(f"single_{d}", {})[eval_d] = s
            log.info("  Single(%s) | %s: %.4f", d, eval_d, s["accuracy"])
        model = pm.merge_and_unload()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = fresh_model()
    results["single_detail"] = single_scores
    # For backwards-compatible "single best" aggregate (diagonal)
    results["single"] = {d: single_scores[f"single_{d}"][d] for d in domains}

    # Load adapter factors
    from src.cross_factor_fusion import load_lora_factors_v2
    f1 = load_lora_factors_v2(os.path.join(adapter_dir, d1))
    f2 = load_lora_factors_v2(os.path.join(adapter_dir, d2))
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

    if args.mode == "all":
        modes_to_run = ["static_only", "carr_full", "carr_no_mechanism"]
    elif args.mode == "ablations":
        modes_to_run = ["carr_full", "no_reliability", "no_conflict", "no_base_fallback"]
    else:
        modes_to_run = [args.mode]

    for mode in modes_to_run:
        log.info("\n=== Mode: %s ===", mode)

        if mode == "static_only":
            import tempfile
            from scripts.eval_sfc_downstream import (
                load_adapter_weights, merge_task_arithmetic, save_merged_adapter,
            )
            wa = load_adapter_weights(os.path.join(adapter_dir, d1))
            wb = load_adapter_weights(os.path.join(adapter_dir, d2))
            merged = merge_task_arithmetic(wa, wb)
            with tempfile.TemporaryDirectory() as tmp:
                save_merged_adapter(merged, os.path.join(adapter_dir, d1), tmp)
                del merged, wa, wb
                gc.collect()
                pm = PeftModel.from_pretrained(model, tmp)
                scores = eval_on(pm, mode)
                model = pm.merge_and_unload()
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = fresh_model()
            results[mode] = scores

        else:
            from src.conflict_aware_routing import (
                CARRConfig, ConflictAwareResidualRouter, CARRHook,
            )
            from src.conflict_diagnostics import (
                compute_activation_gram, compute_pair_conflict,
            )
            from src.cross_factor_fusion import collect_module_inputs_for_bcff

            d_model = model.config.hidden_size

            # Build CARR config from yaml + mode overlay. d_model forced from model.
            carr_kwargs = cfg.carr_kwargs_for_mode(mode)
            carr_kwargs["d_model"] = d_model
            # Whitelist only what CARRConfig accepts
            carr_valid = {
                "n_adapters", "d_model", "gate_hidden_dim",
                "use_reliability", "use_conflict", "use_base_fallback",
                "top_k", "temperature",
                "base_kl_weight", "conflict_weight", "sparsity_weight",
            }
            carr_init_kwargs = {k: v for k, v in carr_kwargs.items() if k in carr_valid}
            carr_config = CARRConfig(**carr_init_kwargs)

            # Verify realization matches YAML request (except d_model, forced)
            yaml_request = {k: v for k, v in carr_kwargs.items() if k in carr_valid and k != "d_model"}
            actual = {k: getattr(carr_config, k) for k in yaml_request if hasattr(carr_config, k)}
            assert_config_applied(yaml_request, actual)

            log.info("  CARRConfig realized: top_k=%s use_rel=%s use_conf=%s use_base=%s",
                     carr_config.top_k, carr_config.use_reliability,
                     carr_config.use_conflict, carr_config.use_base_fallback)

            router = ConflictAwareResidualRouter(carr_config).cuda()

            # Load pretrained router checkpoint if given
            if args.carr_checkpoint and os.path.isfile(args.carr_checkpoint):
                ckpt = torch.load(args.carr_checkpoint, map_location="cuda")
                try:
                    router.load_state_dict(ckpt["router_state_dict"])
                    log.info("  Loaded router checkpoint: %s", args.carr_checkpoint)
                except Exception as e:
                    log.warning("  Failed to load router checkpoint: %s (training fresh)", e)

            module_conflict_scores = {}
            if carr_config.use_conflict:
                calib_texts = (
                    load_calib_texts(d1, dataset_dir, calib_samples // 2)
                    + load_calib_texts(d2, dataset_dir, calib_samples // 2)
                )
                module_inputs = collect_module_inputs_for_bcff(
                    model, tokenizer, calib_texts, target_modules,
                    batch_size=int(cfg.carr.get("batch_size", 4)),
                    max_length=int(cfg.carr.get("max_length", 128)))
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
                del module_inputs, grams
                gc.collect()

            # Train router (unless a checkpoint is loaded and user wants to skip)
            if not args.carr_checkpoint:
                log.info("  Training %s router (%d steps, lr=%.1e)...",
                         mode, train_steps, float(cfg.carr.get("lr", 1e-3)))
                router.train()
                hook = CARRHook(router, static_delta_ws, adapter_delta_ws,
                                module_conflict_scores=module_conflict_scores,
                                training=True)
                hook.attach(model)

                calib = (
                    load_calib_texts(d1, dataset_dir, 100)
                    + load_calib_texts(d2, dataset_dir, 100)
                )
                max_length = int(cfg.carr.get("max_length", 128))
                calib_enc = [
                    tokenizer(t, return_tensors="pt", truncation=True,
                              max_length=max_length, padding="max_length")
                    for t in calib
                ]

                optimizer = torch.optim.Adam(router.parameters(),
                                             lr=float(cfg.carr.get("lr", 1e-3)))
                for step in range(train_steps):
                    idx = step % len(calib_enc)
                    inp = {k: v.cuda() for k, v in calib_enc[idx].items()
                           if k in ("input_ids", "attention_mask")}
                    out = model(**inp, labels=inp["input_ids"])
                    loss = out.loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    hook.gate_stats.clear()
                    if step % 50 == 0:
                        log.info("    step=%d loss=%.4f", step, loss.item())
                hook.detach()

            # Evaluate with router
            router.eval()
            hook2 = CARRHook(router, static_delta_ws, adapter_delta_ws,
                             module_conflict_scores=module_conflict_scores,
                             training=False)
            hook2.attach(model)
            scores = eval_on(model, mode)

            gate_stats = hook2.get_aggregated_stats()
            hook2.detach()
            del router
            gc.collect()
            torch.cuda.empty_cache()

            results[mode] = scores
            results[mode + "_gate_stats"] = {
                mod: {k: v for k, v in s.items() if not isinstance(v, list)}
                for mod, s in gate_stats.items()
            }

    # Summary
    log.info("\n" + "=" * 60)
    log.info("A/B/C COMPARISON (seed=%d, sample_seed=%d, metric=%s)",
             args.seed, args.sample_seed, metric_mode)
    log.info("=" * 60)
    log.info("%-25s %-12s %-12s %-8s", "Method", d1, d2, "Mean")
    log.info("-" * 57)
    for name in ["base", "single", "static_only", "carr_no_mechanism", "carr_full",
                 "no_reliability", "no_conflict", "no_base_fallback"]:
        if name not in results:
            continue
        s = results[name]
        a1 = s.get(d1, {}).get("accuracy", -1)
        a2 = s.get(d2, {}).get("accuracy", -1)
        mean = (a1 + a2) / 2 if a1 >= 0 and a2 >= 0 else -1
        label = {
            "base": "Base", "single": "Single best",
            "static_only": "A: Static TA",
            "carr_no_mechanism": "B: CARR no mech",
            "carr_full": "C: Full CARR",
        }.get(name, name)
        log.info("%-25s %-12.4f %-12.4f %-8.4f", label, a1, a2, mean)

    out_json = os.path.join(args.output_dir, f"results_seed{args.seed}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))
    log.info("Results saved to %s", out_json)


if __name__ == "__main__":
    main()
