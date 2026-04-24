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

    from src.cross_factor_fusion import load_lora_factors_v2
    log.info("Loading adapter factors: %s, %s", d1, d2)
    f1 = load_lora_factors_v2(os.path.join(adapter_dir, d1))
    f2 = load_lora_factors_v2(os.path.join(adapter_dir, d2))
    target_modules = sorted(f1.keys())
    log.info("Target modules: %d", len(target_modules))

    static_delta_ws = {}
    for mod in target_modules:
        B1, A1 = f1[mod]
        B2, A2 = f2[mod]
        static_delta_ws[mod] = ((B1 @ A1 + B2 @ A2) / 2).to(args.device)

    adapter_delta_ws = []
    for factors in [f1, f2]:
        dws = {}
        for mod in target_modules:
            B, A = factors[mod]
            dws[mod] = (B @ A).to(args.device)
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
        "top_k", "temperature",
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

    log.info("Tokenizing calibration data...")
    all_inputs = []
    for text in calib_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length, padding="max_length")
        all_inputs.append(enc)

    optimizer = torch.optim.Adam(router.parameters(), lr=lr)

    log.info("Starting CARR router training (max_steps=%d, overfit=%s)...",
             max_steps, args.overfit_one_batch)

    hook = CARRHook(router, static_delta_ws, adapter_delta_ws,
                    module_conflict_scores=module_conflict_scores, training=True)
    hook.attach(model)

    if args.overfit_one_batch:
        train_indices = list(range(min(batch_size, len(all_inputs))))
    else:
        train_indices = list(range(len(all_inputs)))

    loss_history = []
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
        hook.gate_stats.clear()

        loss_history.append({
            "step": step, "loss": float(loss.item()),
            "loss_task": float(loss_task.item()), "gate_entropy": float(gate_entropy),
        })
        if step % 10 == 0:
            log.info("step=%d loss=%.4f task_loss=%.4f gate_entropy=%.4f",
                     step, loss.item(), loss_task.item(), gate_entropy)
            sys.stdout.flush()

        if args.overfit_one_batch and step >= 49:
            break

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
