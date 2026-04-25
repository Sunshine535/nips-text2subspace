#!/usr/bin/env python3
"""Evaluate CARR vs baselines with A/B/C comparison.

GPT-5.5 Round 3 verified pipeline:
  - --config configs/carr_minimal.yaml is the source of truth.
  - --seeds and --sample_seeds accept comma lists; loops every combination.
  - CARR modes train router with the multi-term objective (Task 4) — L_task +
    L_base_KL + L_conf + L_sparse + L_cal. Reliability labels (Task 5) are
    precomputed once per (sample_seed) and reused across CARR modes.
  - Eval uses logprob MCQ metric (Task 2) by default.
  - Static baseline (Task 8): TA / TIES / DARE; strongest selected by mean
    accuracy across domains and reported as A.
  - Per-run effective_config.json + sample_manifest.jsonl + predictions_*.jsonl.
"""
import argparse
import gc
import json
import logging
import os
import sys
import time
from typing import List, Optional

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
    p.add_argument("--config", default="configs/carr_minimal.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--domains", default="science,medical")
    p.add_argument("--mode", default="all",
                   choices=["static_only", "static_baselines",
                            "carr_full", "carr_no_mechanism",
                            "no_reliability", "no_conflict", "no_base_fallback",
                            "all", "ablations", "full"])
    p.add_argument("--carr_checkpoint", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--metric_mode", default=None,
                   choices=[None, "generation", "logprob_mcq"])
    p.add_argument("--seeds", default="1", help="comma-separated router seeds")
    p.add_argument("--sample_seeds", default=None,
                   help="comma-separated eval sample shuffle seeds; "
                        "default: same as seeds")
    p.add_argument("--seed", type=int, default=None,
                   help="(deprecated) single seed; use --seeds 1,2,3 instead")
    p.add_argument("--sample_seed", type=int, default=None,
                   help="(deprecated) single sample_seed")
    p.add_argument("--calib_samples", type=int, default=None)
    p.add_argument("--train_steps", type=int, default=None)
    p.add_argument("--reliability_calib_samples", type=int, default=60,
                   help="MCQ items per domain for reliability label precompute")
    p.add_argument("--objective", default="carr_full",
                   choices=["lm_loss", "carr_full"],
                   help="lm_loss = legacy LM on calib texts; carr_full = multi-term MCQ")
    p.add_argument("--output_dir", default="/root/nips-text2subspace/results/carr_abc_verified")
    p.add_argument("--dry_run_effective_config", action="store_true")
    p.add_argument("--dry_run_samples", action="store_true",
                   help="emit sample manifest only and exit (verifies sample_seed plumbing)")
    return p.parse_args()


def evaluate_model(model, tokenizer, domain, dataset_dir, n_samples,
                   sample_seed=42, metric_mode="generation",
                   manifest_out=None, predictions_out=None):
    from scripts.eval_sfc_downstream import evaluate_model_mcq
    return evaluate_model_mcq(
        model, tokenizer, domain, dataset_dir, n_samples,
        device="cuda", sample_seed=sample_seed, metric_mode=metric_mode,
        manifest_out=manifest_out, predictions_out=predictions_out,
    )


def parse_seed_list(spec, fallback) -> List[int]:
    if spec is None or spec == "":
        return [int(fallback)] if fallback is not None else [1]
    return [int(s.strip()) for s in str(spec).split(",") if s.strip()]


def run_one_seed_combo(
    seed, sample_seed, args, cfg, base_model, adapter_dir, dataset_dir,
    max_samples, metric_mode, calib_samples, train_steps, modes_to_run,
):
    run_dir = os.path.join(args.output_dir, f"seed{seed}_sample{sample_seed}")
    os.makedirs(run_dir, exist_ok=True)

    from src.carr_config_loader import (
        save_effective_config, save_sample_manifest, assert_config_applied,
    )
    extra = {
        "cli_args": vars(args),
        "resolved": {
            "base_model": base_model, "adapter_dir": adapter_dir,
            "dataset_dir": dataset_dir, "max_samples": max_samples,
            "metric_mode": metric_mode, "calib_samples": calib_samples,
            "train_steps": train_steps, "seed": seed, "sample_seed": sample_seed,
            "objective": args.objective,
            "modes_to_run": modes_to_run,
        },
    }
    eff_path = save_effective_config(cfg, run_dir, extra=extra)
    log.info("[seed=%d sample_seed=%d] effective config -> %s",
             seed, sample_seed, eff_path)

    torch.manual_seed(seed)
    domains = args.domains.split(",")
    d1, d2 = domains[0], domains[1]

    manifest_path = os.path.join(run_dir, "sample_manifest.jsonl")
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

    if args.dry_run_samples:
        log.info("Dry-run sample manifest only.")
        for d in domains:
            evaluate_model(model, tokenizer, d, dataset_dir, max_samples,
                           sample_seed=sample_seed, metric_mode=metric_mode,
                           manifest_out=manifest_path,
                           predictions_out=os.path.join(run_dir, f"dry_predictions_{d}.jsonl"))
        return {"dry_run": True}

    results = {
        "seed": seed, "sample_seed": sample_seed,
        "domains": domains, "metric_mode": metric_mode,
        "max_samples": max_samples, "objective": args.objective,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "effective_config_path": eff_path,
    }

    def eval_on(m, tag):
        predictions = os.path.join(run_dir, f"predictions_{tag}.jsonl")
        if os.path.exists(predictions):
            os.remove(predictions)
        scores = {}
        for d in domains:
            s = evaluate_model(
                m, tokenizer, d, dataset_dir, max_samples,
                sample_seed=sample_seed, metric_mode=metric_mode,
                manifest_out=manifest_path, predictions_out=predictions,
            )
            scores[d] = s
            log.info("  %s | %s: %.4f", tag, d, s["accuracy"])
        return scores

    log.info("=== Base Model ===")
    results["base"] = eval_on(model, "base")

    log.info("=== Single Adapters ===")
    from peft import PeftModel
    single_scores = {}
    for d in domains:
        pm = PeftModel.from_pretrained(model, os.path.join(adapter_dir, d))
        for eval_d in domains:
            s = evaluate_model(
                pm, tokenizer, eval_d, dataset_dir, max_samples,
                sample_seed=sample_seed, metric_mode=metric_mode,
                manifest_out=manifest_path,
                predictions_out=os.path.join(run_dir, f"predictions_single_{d}.jsonl"),
            )
            single_scores.setdefault(f"single_{d}", {})[eval_d] = s
            log.info("  Single(%s) | %s: %.4f", d, eval_d, s["accuracy"])
        model = pm.merge_and_unload()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = fresh_model()
    results["single_detail"] = single_scores
    results["single"] = {d: single_scores[f"single_{d}"][d] for d in domains}

    # Load adapters with LoRA scaling
    from src.cross_factor_fusion import load_lora_factors_v2, get_lora_scaling
    f1 = load_lora_factors_v2(os.path.join(adapter_dir, d1))
    f2 = load_lora_factors_v2(os.path.join(adapter_dir, d2))
    scale1 = get_lora_scaling(os.path.join(adapter_dir, d1))
    scale2 = get_lora_scaling(os.path.join(adapter_dir, d2))
    log.info("LoRA scaling: %s=%.3f  %s=%.3f", d1, scale1, d2, scale2)
    target_modules = sorted(f1.keys())
    static_delta_ws = {}
    adapter_delta_ws = [{}, {}]
    for mod in target_modules:
        B1, A1 = f1[mod]
        B2, A2 = f2[mod]
        dw1 = (scale1 * (B1 @ A1)).cuda()
        dw2 = (scale2 * (B2 @ A2)).cuda()
        static_delta_ws[mod] = (dw1 + dw2) / 2
        adapter_delta_ws[0][mod] = dw1
        adapter_delta_ws[1][mod] = dw2

    # Static baselines (TA/TIES/DARE) — Task 8
    if any(m in ("static_only", "static_baselines") for m in modes_to_run):
        import tempfile
        from scripts.eval_sfc_downstream import (
            load_adapter_weights, merge_task_arithmetic,
            merge_ties, merge_dare, save_merged_adapter,
        )
        wa = load_adapter_weights(os.path.join(adapter_dir, d1))
        wb = load_adapter_weights(os.path.join(adapter_dir, d2))
        all_static = {}
        for mname, mfn, mkwargs in [
            ("TA", merge_task_arithmetic, {}),
            ("TIES", merge_ties, {"density": 0.5}),
            ("DARE", merge_dare, {"drop_rate": 0.5}),
        ]:
            log.info("=== Static: %s ===", mname)
            merged = mfn(wa, wb, **mkwargs)
            with tempfile.TemporaryDirectory() as tmp:
                save_merged_adapter(merged, os.path.join(adapter_dir, d1), tmp)
                del merged
                gc.collect()
                pm = PeftModel.from_pretrained(model, tmp)
                scores = eval_on(pm, f"static_{mname}")
                model = pm.merge_and_unload()
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = fresh_model()
            all_static[mname] = scores
        del wa, wb
        gc.collect()
        best_name = max(
            all_static.keys(),
            key=lambda k: sum(all_static[k][d]["accuracy"] for d in domains) / len(domains),
        )
        log.info("strongest static = %s", best_name)
        results["static_baselines"] = all_static
        results["static_best_method"] = best_name
        results["static_only"] = all_static[best_name]

    # CARR modes — Task 4 multi-term objective
    carr_modes = [m for m in modes_to_run if m not in ("static_only", "static_baselines")]
    if not carr_modes:
        return results

    from src.conflict_aware_routing import (
        CARRConfig, ConflictAwareResidualRouter, CARRHook,
    )
    from src.conflict_diagnostics import (
        compute_activation_gram, compute_pair_conflict,
    )
    from src.cross_factor_fusion import collect_module_inputs_for_bcff
    from src.carr_objective import (
        build_mcq_calib_items, precompute_base_option_logprobs,
        precompute_reliability_labels, ReliabilityLabels,
        train_router_multiterm, LossWeights,
        compute_ece, compute_brier,
    )

    d_model = model.config.hidden_size

    # Compute module-level conflict scores once (shared across CARR modes that use_conflict)
    log.info("=== Pre-compute: module-level conflict (Gram-based) ===")
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
    module_conflict_scores = {}
    for mod in target_modules:
        if mod in grams:
            m = compute_pair_conflict(grams[mod])
            module_conflict_scores[mod] = torch.tensor(
                [m["interference_ratio"], 1 - m["interference_ratio"]])
    del module_inputs, grams
    gc.collect()

    # Multi-term objective prerequisites: MCQ calib items + base logprobs + reliability labels
    if args.objective == "carr_full":
        log.info("=== Pre-compute: MCQ calib items + base logprobs + reliability labels ===")
        max_length = int(cfg.carr.get("max_length", 128))
        mcq_items = build_mcq_calib_items(
            domains, dataset_dir, args.reliability_calib_samples, tokenizer,
            sample_seed=sample_seed, max_length=max_length * 3)
        if not mcq_items:
            raise RuntimeError("No MCQ calib items; objective=carr_full requires arc/medmcqa")
        log.info("  MCQ items: %d", len(mcq_items))

        log.info("  base option logprobs ...")
        base_lps = precompute_base_option_logprobs(model, mcq_items, device="cuda")
        log.info("  base option logprobs shape=%s", tuple(base_lps.shape))

        rel_path = os.path.join(run_dir, "reliability_labels.pt")
        adapter_paths = [os.path.join(adapter_dir, d) for d in domains]
        rel_labels = precompute_reliability_labels(
            base_model, adapter_paths, tokenizer, mcq_items,
            device="cuda", save_path=rel_path)
        log.info("  reliability labels: base_acc=%.3f  adapter_acc=%s",
                 rel_labels.base_correct.float().mean().item(),
                 rel_labels.correctness.float().mean(0).tolist())
        # precompute_reliability_labels mutates `base` in place; we need a clean model now
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = fresh_model()
    else:
        mcq_items = None
        base_lps = None
        rel_labels = None

    weights = LossWeights(
        task=float(cfg.carr.get("task_weight", 1.0)),
        base_kl=float(cfg.carr.get("base_kl_weight", 0.1)),
        conflict=float(cfg.carr.get("conflict_weight", 0.05)),
        sparse=float(cfg.carr.get("sparsity_weight", 0.01)),
        calibration=float(cfg.carr.get("calibration_weight", 0.1)),
    )
    log.info("Loss weights: %s", weights)

    for mode in carr_modes:
        log.info("\n=== Mode: %s ===", mode)
        carr_kwargs = cfg.carr_kwargs_for_mode(mode)
        carr_kwargs["d_model"] = d_model
        carr_valid = {
            "n_adapters", "d_model", "gate_hidden_dim",
            "use_reliability", "use_conflict", "use_base_fallback",
            "top_k", "temperature", "conflict_mode",
            "base_kl_weight", "conflict_weight", "sparsity_weight",
        }
        carr_init = {k: v for k, v in carr_kwargs.items() if k in carr_valid}
        carr_config = CARRConfig(**carr_init)
        yaml_request = {k: v for k, v in carr_kwargs.items()
                        if k in carr_valid and k != "d_model"}
        actual = {k: getattr(carr_config, k) for k in yaml_request if hasattr(carr_config, k)}
        assert_config_applied(yaml_request, actual)
        log.info("  realized: top_k=%s use_rel=%s use_conf=%s use_base=%s conflict_mode=%s",
                 carr_config.top_k, carr_config.use_reliability,
                 carr_config.use_conflict, carr_config.use_base_fallback,
                 carr_config.conflict_mode)

        torch.manual_seed(seed)
        router = ConflictAwareResidualRouter(carr_config).cuda()
        if args.carr_checkpoint and os.path.isfile(args.carr_checkpoint):
            ckpt = torch.load(args.carr_checkpoint, map_location="cuda",
                              weights_only=False)
            try:
                router.load_state_dict(ckpt["router_state_dict"])
                log.info("  loaded router checkpoint: %s", args.carr_checkpoint)
            except Exception as e:
                log.warning("  failed to load checkpoint: %s — training fresh", e)

        hook = CARRHook(router, static_delta_ws, adapter_delta_ws,
                        module_conflict_scores=module_conflict_scores,
                        training=True)
        hook.attach(model)

        if not args.carr_checkpoint:
            log.info("  Training (%s objective, %d steps, lr=%.1e)...",
                     args.objective, train_steps, float(cfg.carr.get("lr", 1e-3)))
            if args.objective == "carr_full":
                history = train_router_multiterm(
                    model=model, hook=hook, router=router,
                    calib_items=mcq_items, base_option_lps=base_lps,
                    rel_labels=rel_labels,
                    module_conflict_scores=module_conflict_scores,
                    weights=weights, max_steps=train_steps,
                    lr=float(cfg.carr.get("lr", 1e-3)),
                    device="cuda", overfit_one_batch=False,
                    log_every=max(1, train_steps // 20),
                    log_fn=log.info,
                )
                # Save history
                with open(os.path.join(run_dir, f"loss_history_{mode}.json"), "w") as f:
                    json.dump(history, f, indent=2)
            else:
                # legacy LM-loss path (fallback)
                router.train()
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
                    inp = {k: v.cuda() for k, v in calib_enc[step % len(calib_enc)].items()
                           if k in ("input_ids", "attention_mask")}
                    out = model(**inp, labels=inp["input_ids"])
                    loss = out.loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    hook.clear_step_buffers()
                    if step % 50 == 0:
                        log.info("    step=%d loss=%.4f", step, loss.item())

        # Reliability calibration metrics (Task 5)
        rel_metrics = {}
        if (args.objective == "carr_full" and carr_config.use_reliability
                and rel_labels is not None and mcq_items is not None):
            router.eval()
            preds_all = []
            with torch.no_grad():
                for it in mcq_items[:min(64, len(mcq_items))]:
                    hook.clear_step_buffers()
                    prompt_ids = it.prompt_ids.unsqueeze(0).cuda()
                    model(prompt_ids)
                    pr = hook.mean_reliability_across_modules()
                    if pr is not None:
                        preds_all.append(pr.mean(0).cpu())
            if preds_all:
                preds_all = torch.stack(preds_all)
                labels_sub = rel_labels.correctness[:len(preds_all)].float()
                for k in range(rel_labels.n_adapters):
                    rel_metrics[f"adapter_{k}"] = {
                        "ECE": compute_ece(preds_all[:, k], labels_sub[:, k]),
                        "Brier": compute_brier(preds_all[:, k], labels_sub[:, k]),
                        "pred_mean": float(preds_all[:, k].mean().item()),
                        "label_mean": float(labels_sub[:, k].mean().item()),
                    }
                    log.info("  reliability adapter%d: ECE=%.4f Brier=%.4f",
                             k, rel_metrics[f"adapter_{k}"]["ECE"],
                             rel_metrics[f"adapter_{k}"]["Brier"])
            router.train()

        router.eval()
        hook2 = CARRHook(router, static_delta_ws, adapter_delta_ws,
                         module_conflict_scores=module_conflict_scores,
                         training=False)
        hook.detach()
        hook2.attach(model)
        scores = eval_on(model, mode)
        gate_stats = hook2.get_aggregated_stats()
        hook2.detach()

        # Save router ckpt
        ckpt_path = os.path.join(run_dir, f"router_{mode}.pt")
        torch.save({
            "router_state_dict": router.state_dict(),
            "config": carr_config.__dict__,
            "mode": mode, "seed": seed, "sample_seed": sample_seed,
        }, ckpt_path)

        del router
        gc.collect()
        torch.cuda.empty_cache()

        results[mode] = scores
        results[mode + "_gate_stats"] = {
            mod: {k: v for k, v in s.items() if not isinstance(v, list)}
            for mod, s in gate_stats.items()
        }
        if rel_metrics:
            results[mode + "_reliability"] = rel_metrics

    # Summary
    log.info("\n" + "=" * 60)
    log.info("[seed=%d sample_seed=%d metric=%s] A/B/C", seed, sample_seed, metric_mode)
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
            "static_only": "A: Static (best-of)",
            "carr_no_mechanism": "B: CARR no mech",
            "carr_full": "C: Full CARR",
        }.get(name, name)
        log.info("%-25s %-12.4f %-12.4f %-8.4f", label, a1, a2, mean)

    out_json = os.path.join(run_dir, f"results_seed{seed}_sample{sample_seed}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))
    log.info("Results saved to %s", out_json)
    return results


def main():
    args = parse_args()

    from src.carr_config_loader import load_run_config, save_effective_config
    cfg = load_run_config(args.config)

    base_model = args.model or cfg.base_model
    adapter_dir = args.adapter_dir or cfg.adapter_dir
    dataset_dir = args.dataset_dir or cfg.dataset_dir
    max_samples = args.max_samples if args.max_samples is not None else cfg.eval_max_samples
    metric_mode = args.metric_mode or cfg.metric_mode
    calib_samples = args.calib_samples if args.calib_samples is not None else int(cfg.carr.get("calib_samples", 200))
    train_steps = args.train_steps if args.train_steps is not None else int(cfg.carr.get("max_steps", 200))

    if args.dry_run_effective_config:
        os.makedirs(args.output_dir, exist_ok=True)
        eff_path = save_effective_config(cfg, args.output_dir, extra={
            "cli_args": vars(args),
            "resolved": {"base_model": base_model, "adapter_dir": adapter_dir,
                         "dataset_dir": dataset_dir, "max_samples": max_samples,
                         "metric_mode": metric_mode, "calib_samples": calib_samples,
                         "train_steps": train_steps},
        })
        print(json.dumps({"effective_config": eff_path,
                          "metric_mode": metric_mode,
                          "top_k": cfg.carr.get("top_k"),
                          "objective": args.objective}, indent=2))
        return

    seeds = parse_seed_list(args.seeds, args.seed)
    sample_seeds = parse_seed_list(args.sample_seeds, args.sample_seed) if (args.sample_seeds or args.sample_seed) else seeds

    if args.mode == "all":
        modes_to_run = ["static_baselines", "carr_full", "carr_no_mechanism"]
    elif args.mode == "ablations":
        modes_to_run = ["carr_full", "no_reliability", "no_conflict", "no_base_fallback"]
    elif args.mode == "full":
        modes_to_run = ["static_baselines", "carr_no_mechanism",
                        "carr_full", "no_reliability", "no_conflict", "no_base_fallback"]
    else:
        modes_to_run = [args.mode]

    log.info("Run grid: seeds=%s sample_seeds=%s modes=%s metric=%s objective=%s",
             seeds, sample_seeds, modes_to_run, metric_mode, args.objective)

    all_runs = []
    for seed in seeds:
        for sample_seed in sample_seeds:
            t0 = time.time()
            try:
                res = run_one_seed_combo(
                    seed=seed, sample_seed=sample_seed, args=args, cfg=cfg,
                    base_model=base_model, adapter_dir=adapter_dir,
                    dataset_dir=dataset_dir, max_samples=max_samples,
                    metric_mode=metric_mode, calib_samples=calib_samples,
                    train_steps=train_steps, modes_to_run=modes_to_run,
                )
                res["wall_seconds"] = time.time() - t0
                all_runs.append(res)
            except Exception as e:
                log.exception("FAILED seed=%d sample_seed=%d: %s", seed, sample_seed, e)
                all_runs.append({"seed": seed, "sample_seed": sample_seed,
                                 "error": str(e)})

    # Combined index file
    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump({
            "config": args.config, "modes": modes_to_run,
            "seeds": seeds, "sample_seeds": sample_seeds,
            "metric_mode": metric_mode, "objective": args.objective,
            "domains": args.domains.split(","),
            "max_samples": max_samples,
            "runs": all_runs,
        }, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    log.info("Index -> %s", index_path)


if __name__ == "__main__":
    main()
