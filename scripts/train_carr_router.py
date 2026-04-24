#!/usr/bin/env python3
"""Train CARR router on calibration data. Config is the source of truth."""
import argparse
import gc
import json
import logging
import os
import sys
import time

import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("carr_train")


CALIB_MAP = {
    "science": ("arc_challenge", "train", "question"),
    "medical": ("medmcqa", "train", "question"),
    "math": ("gsm8k", "train", "question"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/carr_minimal.yaml",
                   help="YAML source of truth")
    p.add_argument("--model", default=None)
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--domains", default="science,medical")
    p.add_argument("--calib_samples", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--overfit_one_batch", action="store_true")
    p.add_argument("--output_dir", default="/root/nips-text2subspace/results/carr_checkpoints")
    p.add_argument("--seed", type=int, default=1, help="router init/train seed")
    p.add_argument("--sample_seed", type=int, default=42, help="calib shuffle seed")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry_run_effective_config", action="store_true")
    p.add_argument("--objective", default="carr_full",
                   choices=["lm_loss", "carr_full"],
                   help="lm_loss = legacy LM on question text; carr_full = Round 3 multi-term MCQ loss")
    p.add_argument("--reliability_labels", default=None,
                   help="path to precomputed reliability labels .pt; if missing, computed on the fly")
    p.add_argument("--reliability_calib_samples", type=int, default=60,
                   help="calib MCQ items for reliability label precompute (per domain)")
    p.add_argument("--log_reliability_calibration", action="store_true")
    return p.parse_args()


def load_calib_texts(domain, dataset_dir, n=200, seed=42):
    from datasets import load_from_disk

    cfg = CALIB_MAP[domain]
    ds = load_from_disk(os.path.join(dataset_dir, cfg[0]))
    split = cfg[1] if cfg[1] in ds else list(ds.keys())[0]
    data = ds[split].shuffle(seed=seed).select(range(min(n, len(ds[split]))))
    return [str(row[cfg[2]]) for row in data]


def main():
    args = parse_args()

    from src.carr_config_loader import (
        load_run_config, save_effective_config, assert_config_applied,
    )
    cfg = load_run_config(args.config)

    base_model = args.model or cfg.base_model
    adapter_dir = args.adapter_dir or cfg.adapter_dir
    dataset_dir = args.dataset_dir or cfg.dataset_dir
    max_steps = args.max_steps if args.max_steps is not None else int(cfg.carr.get("max_steps", 500))
    lr = args.lr if args.lr is not None else float(cfg.carr.get("lr", 1e-3))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.carr.get("batch_size", 4))
    max_length = args.max_length if args.max_length is not None else int(cfg.carr.get("max_length", 128))
    calib_samples = args.calib_samples if args.calib_samples is not None else int(cfg.carr.get("calib_samples", 200))

    os.makedirs(args.output_dir, exist_ok=True)
    eff_extra = {
        "cli_args": vars(args),
        "resolved": {
            "base_model": base_model, "adapter_dir": adapter_dir,
            "dataset_dir": dataset_dir, "max_steps": max_steps, "lr": lr,
            "batch_size": batch_size, "max_length": max_length,
            "calib_samples": calib_samples,
        },
    }
    eff_path = save_effective_config(cfg, args.output_dir, extra=eff_extra)
    log.info("Effective config -> %s", eff_path)
    log.info("  top_k=%s use_rel=%s use_conf=%s use_base=%s",
             cfg.carr.get("top_k"), cfg.carr.get("use_reliability"),
             cfg.carr.get("use_conflict"), cfg.carr.get("use_base_fallback"))

    if args.dry_run_effective_config:
        print(json.dumps(eff_extra["resolved"], indent=2))
        return

    torch.manual_seed(args.seed)
    domains = args.domains.split(",")
    assert len(domains) == 2, "CARR v0 requires exactly 2 domains"
    d1, d2 = domains

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("Loading model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", trust_remote_code=True)
    model.eval()

    from src.cross_factor_fusion import load_lora_factors_v2, get_lora_scaling
    log.info("Loading adapter factors: %s, %s", d1, d2)
    f1 = load_lora_factors_v2(os.path.join(adapter_dir, d1))
    f2 = load_lora_factors_v2(os.path.join(adapter_dir, d2))
    scale1 = get_lora_scaling(os.path.join(adapter_dir, d1))
    scale2 = get_lora_scaling(os.path.join(adapter_dir, d2))
    log.info("LoRA scaling: %s=%.3f  %s=%.3f", d1, scale1, d2, scale2)
    target_modules = sorted(f1.keys())
    log.info("Target modules: %d", len(target_modules))

    static_delta_ws = {}
    for mod in target_modules:
        B1, A1 = f1[mod]
        B2, A2 = f2[mod]
        static_delta_ws[mod] = (
            (scale1 * (B1 @ A1) + scale2 * (B2 @ A2)) / 2
        ).to(args.device)

    adapter_delta_ws = []
    for factors, s in [(f1, scale1), (f2, scale2)]:
        dws = {}
        for mod in target_modules:
            B, A = factors[mod]
            dws[mod] = (s * (B @ A)).to(args.device)
        adapter_delta_ws.append(dws)

    log.info("Computing conflict diagnostics...")
    from src.conflict_diagnostics import compute_activation_gram, compute_pair_conflict
    from src.cross_factor_fusion import collect_module_inputs_for_bcff

    calib_texts = (
        load_calib_texts(d1, dataset_dir, calib_samples // 2, seed=args.sample_seed)
        + load_calib_texts(d2, dataset_dir, calib_samples // 2, seed=args.sample_seed)
    )
    log.info("Collecting module inputs (%d texts)...", len(calib_texts))
    module_inputs = collect_module_inputs_for_bcff(
        model, tokenizer, calib_texts, target_modules,
        batch_size=batch_size, max_length=max_length)

    delta_w_list = [
        {mod: (f1[mod][0] @ f1[mod][1]) for mod in target_modules},
        {mod: (f2[mod][0] @ f2[mod][1]) for mod in target_modules},
    ]
    grams = compute_activation_gram(delta_w_list, module_inputs)

    module_conflict_scores = {}
    for mod in target_modules:
        if mod in grams:
            metrics = compute_pair_conflict(grams[mod])
            cs = torch.tensor([metrics["interference_ratio"], 1 - metrics["interference_ratio"]])
            module_conflict_scores[mod] = cs

    # CARR config from YAML
    d_model = model.config.hidden_size
    from src.conflict_aware_routing import (
        CARRConfig, ConflictAwareResidualRouter, CARRHook,
    )
    carr_kwargs = cfg.carr_kwargs_for_mode("carr_full")
    carr_kwargs["d_model"] = d_model
    carr_valid = {
        "n_adapters", "d_model", "gate_hidden_dim",
        "use_reliability", "use_conflict", "use_base_fallback",
        "top_k", "temperature", "conflict_mode",
        "base_kl_weight", "conflict_weight", "sparsity_weight",
    }
    carr_init = {k: v for k, v in carr_kwargs.items() if k in carr_valid}
    config = CARRConfig(**carr_init)

    yaml_request = {k: v for k, v in carr_kwargs.items() if k in carr_valid and k != "d_model"}
    actual = {k: getattr(config, k) for k in yaml_request if hasattr(config, k)}
    assert_config_applied(yaml_request, actual)

    log.info("CARRConfig realized: top_k=%s use_rel=%s use_conf=%s use_base=%s",
             config.top_k, config.use_reliability, config.use_conflict,
             config.use_base_fallback)

    router = ConflictAwareResidualRouter(config).to(args.device)
    router.train()
    log.info("Router params: %d", sum(p.numel() for p in router.parameters()))

    optimizer = torch.optim.Adam(router.parameters(), lr=lr)

    log.info("Starting CARR router training (objective=%s, max_steps=%d, overfit=%s)...",
             args.objective, max_steps, args.overfit_one_batch)

    hook = CARRHook(router, static_delta_ws, adapter_delta_ws,
                    module_conflict_scores=module_conflict_scores, training=True)
    hook.attach(model)

    loss_history = []

    if args.objective == "lm_loss":
        # Legacy path: LM loss on question text only
        log.info("Tokenizing calibration data for LM loss...")
        all_inputs = []
        for text in calib_texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_length, padding="max_length")
            all_inputs.append(enc)
        train_indices = (list(range(min(batch_size, len(all_inputs))))
                         if args.overfit_one_batch else list(range(len(all_inputs))))
        for step in range(max_steps):
            idx = train_indices[step % len(train_indices)] if not args.overfit_one_batch else train_indices[0]
            inp = {k: v.to(args.device) for k, v in all_inputs[idx].items()
                   if k in ("input_ids", "attention_mask")}
            outputs = model(**inp, labels=inp["input_ids"])
            loss_task = outputs.loss
            total_entropy = 0.0
            n_hooks = 0
            for mod_stats in hook.gate_stats.values():
                if mod_stats:
                    total_entropy += mod_stats[-1]["gate_entropy"]
                    n_hooks += 1
            gate_entropy = total_entropy / max(n_hooks, 1)
            loss = loss_task + config.sparsity_weight * gate_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            hook.clear_step_buffers()
            loss_history.append({"step": step, "loss": float(loss.item()),
                                 "loss_task": float(loss_task.item()),
                                 "gate_entropy": float(gate_entropy)})
            if step % 10 == 0:
                log.info("step=%d loss=%.4f task=%.4f ent=%.4f",
                         step, loss.item(), loss_task.item(), gate_entropy)
                sys.stdout.flush()
            if args.overfit_one_batch and step >= 49:
                break
    else:
        # Round 3 Task 4: multi-term MCQ objective
        from src.carr_objective import (
            build_mcq_calib_items, compute_option_logprobs,
            precompute_reliability_labels, ReliabilityLabels,
            carr_multi_term_loss, LossWeights, compute_ece, compute_brier,
        )
        log.info("Building MCQ calib items (%d per domain)...", args.reliability_calib_samples)
        mcq_items = build_mcq_calib_items(
            domains, dataset_dir, args.reliability_calib_samples, tokenizer,
            sample_seed=args.sample_seed, max_length=max_length)
        if not mcq_items:
            raise RuntimeError("No MCQ calib items built; cannot train carr_full objective")

        # Precompute base option logprobs (for L_base_KL)
        log.info("Precomputing base option logprobs (%d items)...", len(mcq_items))
        hook.detach()  # base pass w/o hooks
        model.eval()
        with torch.no_grad():
            base_option_lps = []
            for it in mcq_items:
                lp = compute_option_logprobs(model, it, device=args.device, require_grad=False)
                base_option_lps.append(lp.cpu())
        base_option_lps = torch.stack(base_option_lps)  # (N, n_options)
        log.info("  base option logprobs: shape=%s", tuple(base_option_lps.shape))

        # Precompute or load reliability labels
        if args.reliability_labels and os.path.isfile(args.reliability_labels):
            rel_labels = ReliabilityLabels.load(args.reliability_labels)
            log.info("Loaded reliability labels: %s  (n_items=%d, n_adapters=%d)",
                     args.reliability_labels, rel_labels.n_items, rel_labels.n_adapters)
        else:
            log.info("Precomputing reliability labels (base + %d PEFT adapters)...", len(domains))
            adapter_paths = [os.path.join(adapter_dir, d) for d in domains]
            rel_path = args.reliability_labels or os.path.join(
                args.output_dir, "reliability_labels.pt")
            rel_labels = precompute_reliability_labels(
                base_model, adapter_paths, tokenizer, mcq_items,
                device=args.device, save_path=rel_path)

        # Re-attach hook for CARR training forward
        hook.attach(model)

        weights = LossWeights(
            task=float(cfg.carr.get("task_weight", 1.0)),
            base_kl=float(cfg.carr.get("base_kl_weight", 0.1)),
            conflict=float(cfg.carr.get("conflict_weight", 0.05)),
            sparse=float(cfg.carr.get("sparsity_weight", 0.01)),
            calibration=float(cfg.carr.get("calibration_weight", 0.1)),
        )
        log.info("Loss weights: %s", weights)

        n_items = len(mcq_items)
        train_order = list(range(n_items))
        if args.overfit_one_batch:
            train_order = train_order[:batch_size]

        router.train()
        for step in range(max_steps):
            item_idx = train_order[step % len(train_order)]
            item = mcq_items[item_idx]

            # Forward each option with CARR active to get option logprobs (with grad)
            option_lps = []
            for opt_ids in item.option_token_ids:
                hook.clear_step_buffers()
                prompt_ids = item.prompt_ids.unsqueeze(0).to(args.device)
                opt_tensor = torch.tensor([opt_ids], device=args.device,
                                          dtype=prompt_ids.dtype)
                full = torch.cat([prompt_ids, opt_tensor], dim=-1)
                logits = model(full).logits
                target = full[:, prompt_ids.shape[1]:]
                log_probs = F.log_softmax(
                    logits[:, prompt_ids.shape[1] - 1:-1, :].float(), dim=-1)
                lp = log_probs.gather(-1, target.unsqueeze(-1)).sum() / max(len(opt_ids), 1)
                option_lps.append(lp)
            carr_lps = torch.stack(option_lps)  # (n_options,)

            # Grab gates + reliability from the last option forward (prompt-dominated)
            last_gates = {mod: g for mod, g in hook.last_gates.items()}
            rel_pred = hook.mean_reliability_across_modules()

            losses = carr_multi_term_loss(
                carr_option_logprobs=carr_lps,
                base_option_logprobs=base_option_lps[item_idx],
                gold_idx=item.gold_idx,
                last_gates=last_gates,
                module_conflict_scores=module_conflict_scores,
                reliability_pred=rel_pred,
                reliability_label=rel_labels.correctness[item_idx],
                weights=weights,
                adapter_start_idx=router.adapter_start_idx,
                static_idx=router.static_idx,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            loss_history.append({
                "step": step,
                "total": float(losses["total"].item()),
                "L_task": float(losses["L_task"].item()),
                "L_base_KL": float(losses["L_base_KL"].item()),
                "L_conf": float(losses["L_conf"].item()),
                "L_sparse": float(losses["L_sparse"].item()),
                "L_cal": float(losses["L_cal"].item()),
                "item_idx": item_idx, "domain": item.domain,
            })
            if step % 10 == 0:
                log.info("step=%d total=%.4f task=%.4f kl=%.4f conf=%.4f sparse=%.4f cal=%.4f",
                         step, losses["total"].item(),
                         losses["L_task"].item(), losses["L_base_KL"].item(),
                         losses["L_conf"].item(), losses["L_sparse"].item(),
                         losses["L_cal"].item())
                sys.stdout.flush()
            if args.overfit_one_batch and step >= 49:
                break

        # Reliability calibration metrics
        if args.log_reliability_calibration and config.use_reliability:
            log.info("Computing reliability calibration metrics...")
            router.eval()
            preds_all = []
            with torch.no_grad():
                for it in mcq_items[:min(64, len(mcq_items))]:
                    hook.clear_step_buffers()
                    prompt_ids = it.prompt_ids.unsqueeze(0).to(args.device)
                    model(prompt_ids)
                    pr = hook.mean_reliability_across_modules()
                    if pr is not None:
                        preds_all.append(pr.mean(0).cpu())
            if preds_all:
                preds_all = torch.stack(preds_all)
                labels_sub = rel_labels.correctness[:len(preds_all)].float()
                for k in range(rel_labels.n_adapters):
                    ece = compute_ece(preds_all[:, k], labels_sub[:, k])
                    brier = compute_brier(preds_all[:, k], labels_sub[:, k])
                    log.info("  adapter %d: ECE=%.4f  Brier=%.4f  pred_mean=%.4f  label_mean=%.4f",
                             k, ece, brier,
                             preds_all[:, k].mean().item(), labels_sub[:, k].mean().item())
            router.train()

    hook.detach()

    ckpt_path = os.path.join(args.output_dir, "carr_router.pt")
    torch.save({
        "router_state_dict": router.state_dict(),
        "config": config.__dict__,
        "domains": domains,
        "step": step,
        "seed": args.seed,
        "sample_seed": args.sample_seed,
        "effective_config_path": eff_path,
    }, ckpt_path)
    log.info("Router saved to %s", ckpt_path)

    # Save loss history
    with open(os.path.join(args.output_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f, indent=2)

    # Final gate stats (diagnostic)
    hook2 = CARRHook(router, static_delta_ws, adapter_delta_ws,
                     module_conflict_scores=module_conflict_scores, training=False)
    hook2.attach(model)
    with torch.no_grad():
        inp = {k: v.to(args.device) for k, v in all_inputs[0].items()
               if k in ("input_ids", "attention_mask")}
        model(**inp)
    stats = hook2.get_aggregated_stats()
    hook2.detach()

    log.info("Final gate stats:")
    for mod, s in stats.items():
        log.info("  %s: base=%.4f static=%.4f entropy=%.4f",
                 mod, s.get("base_gate_mean", 0), s["static_gate_mean"], s["gate_entropy"])

    log.info("CARR training complete")


if __name__ == "__main__":
    main()
