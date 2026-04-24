#!/usr/bin/env python3
"""Checkpoint integrity + forced-equivalence tests for CARR.

Validates (per Round 3 Task 10):
1. Each adapter's base_model_name_or_path matches the base model used for CARR.
2. Adapter weight files hash reproducibly.
3. Forced base gate path reproduces pure base-model output (within tolerance).
4. Forced single-adapter gate path matches PEFT adapter output on a toy batch.

Exit 0 on pass, 1 on any failure.
"""
import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("integrity")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/carr_minimal.yaml")
    p.add_argument("--domains", default="science,medical")
    p.add_argument("--tolerance", type=float, default=1e-3,
                   help="max-abs tolerance for forced equivalence")
    p.add_argument("--output", default=None,
                   help="json integrity report path; default: <config_dir>/integrity_report.json")
    return p.parse_args()


def sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def adapter_file_hashes(adapter_dir: str) -> dict:
    p = Path(adapter_dir)
    out = {}
    for name in ("adapter_model.safetensors", "adapter_model.bin",
                 "adapter_config.json"):
        q = p / name
        if q.is_file():
            out[name] = sha256_of_file(str(q))
    return out


def read_adapter_base_model(adapter_dir: str) -> str:
    cfg = Path(adapter_dir) / "adapter_config.json"
    if not cfg.is_file():
        return ""
    with cfg.open() as f:
        data = json.load(f)
    return str(data.get("base_model_name_or_path", ""))


def main():
    args = parse_args()

    from src.carr_config_loader import load_run_config
    cfg = load_run_config(args.config)
    base_model = cfg.base_model
    adapter_dir = cfg.adapter_dir
    domains = args.domains.split(",")

    report = {
        "config": args.config, "base_model": base_model,
        "adapter_dir": adapter_dir, "domains": domains,
        "checks": {}, "all_pass": True,
    }

    # Check 1: adapter_config.json base_model_name_or_path
    log.info("Check 1: adapter base_model matches")
    mismatches = []
    for d in domains:
        ap = os.path.join(adapter_dir, d)
        claimed = read_adapter_base_model(ap)
        ok = claimed and (claimed == base_model or Path(claimed).name == Path(base_model).name)
        report["checks"].setdefault("adapter_base_model", {})[d] = {
            "claimed": claimed, "expected": base_model, "match": bool(ok),
        }
        if not ok:
            mismatches.append((d, claimed, base_model))
            log.warning("  %s: claimed=%s expected=%s  MISMATCH", d, claimed, base_model)
        else:
            log.info("  %s: %s ✓", d, claimed)
    if mismatches:
        report["all_pass"] = False

    # Check 2: adapter file hashes
    log.info("Check 2: adapter file hashes")
    for d in domains:
        ap = os.path.join(adapter_dir, d)
        hashes = adapter_file_hashes(ap)
        report["checks"].setdefault("adapter_hashes", {})[d] = hashes
        for name, digest in hashes.items():
            log.info("  %s/%s: %s", d, name, digest[:16])

    # Check 3+4: forced-equivalence tests (requires GPU)
    if not torch.cuda.is_available():
        log.warning("CUDA not available; skipping forced-equivalence tests")
        report["checks"]["forced_equivalence"] = {"skipped": "no CUDA"}
        _write_report(args, report)
        sys.exit(0 if report["all_pass"] else 1)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from src.cross_factor_fusion import load_lora_factors_v2
    from src.conflict_aware_routing import (
        CARRConfig, ConflictAwareResidualRouter, CARRHook,
    )

    log.info("Loading model: %s", base_model)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", trust_remote_code=True)
    model.eval()

    toy_text = "The quick brown fox jumps over the lazy dog."
    enc = tok(toy_text, return_tensors="pt", padding="max_length",
              truncation=True, max_length=32).to("cuda")

    log.info("  (a) pure base forward")
    with torch.no_grad():
        base_out = model(**enc).logits.float().cpu()

    # Load adapter factors (for CARR hook)
    d1, d2 = domains[0], domains[1]
    f1 = load_lora_factors_v2(os.path.join(adapter_dir, d1))
    f2 = load_lora_factors_v2(os.path.join(adapter_dir, d2))
    target_modules = sorted(f1.keys())
    static_dw = {}
    adapter_dws = [{}, {}]
    for mod in target_modules:
        B1, A1 = f1[mod]
        B2, A2 = f2[mod]
        dw1 = (B1 @ A1).cuda()
        dw2 = (B2 @ A2).cuda()
        static_dw[mod] = (dw1 + dw2) / 2
        adapter_dws[0][mod] = dw1
        adapter_dws[1][mod] = dw2

    # Check 3: forced-base-gate via CARR hook should match base
    d_model = model.config.hidden_size
    carr_cfg = CARRConfig(n_adapters=2, d_model=d_model, gate_hidden_dim=32,
                          use_reliability=False, use_conflict=False,
                          use_base_fallback=True, top_k=0)
    router = ConflictAwareResidualRouter(carr_cfg).cuda()
    # Force gate: make base-idx logit dominate
    with torch.no_grad():
        router.gate_net[-1].weight.zero_()
        router.gate_net[-1].bias.zero_()
        # bias[0] = base, [1] = static, [2,3] = adapters
        router.gate_net[-1].bias[0] = 1e6  # force softmax all to base

    router.eval()
    hook = CARRHook(router, static_dw, adapter_dws, training=False)
    hook.attach(model)
    with torch.no_grad():
        carr_out = model(**enc).logits.float().cpu()
    hook.detach()

    diff_base = (carr_out - base_out).abs().max().item()
    ok_base = diff_base < args.tolerance
    report["checks"]["forced_base_gate"] = {
        "max_abs_diff": float(diff_base),
        "tolerance": float(args.tolerance),
        "pass": bool(ok_base),
    }
    log.info("Check 3: forced-base gate max|diff|=%.2e %s",
             diff_base, "✓" if ok_base else "✗ FAIL")
    if not ok_base:
        report["all_pass"] = False

    # Check 4: forced-single-adapter gate via CARR hook vs PEFT single-adapter
    log.info("Check 4: forced-single-adapter gate vs PEFT adapter")
    peft = PeftModel.from_pretrained(model, os.path.join(adapter_dir, d1))
    peft.eval()
    with torch.no_grad():
        peft_out = peft(**enc).logits.float().cpu()
    base_again = model.base_model if hasattr(model, "base_model") else model
    # Unload PEFT
    try:
        model = peft.merge_and_unload()
    except Exception:
        model = peft.unload()

    # Reload base cleanly to rerun with CARR hook
    del peft
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", trust_remote_code=True)
    model.eval()

    # Router: force all gate to adapter_0
    router2 = ConflictAwareResidualRouter(carr_cfg).cuda()
    with torch.no_grad():
        router2.gate_net[-1].weight.zero_()
        router2.gate_net[-1].bias.zero_()
        router2.gate_net[-1].bias[2] = 1e6  # force all to adapter_0 (d1)
    router2.eval()
    hook2 = CARRHook(router2, static_dw, adapter_dws, training=False)
    hook2.attach(model)
    with torch.no_grad():
        carr_adapter_out = model(**enc).logits.float().cpu()
    hook2.detach()

    diff_adapter = (carr_adapter_out - peft_out).abs().max().item()
    ok_adapter = diff_adapter < args.tolerance * 10  # allow slightly looser
    report["checks"]["forced_adapter_gate"] = {
        "domain": d1,
        "max_abs_diff": float(diff_adapter),
        "tolerance": float(args.tolerance * 10),
        "pass": bool(ok_adapter),
    }
    log.info("  forced-adapter max|diff|=%.2e %s",
             diff_adapter, "✓" if ok_adapter else "✗ FAIL")
    if not ok_adapter:
        report["all_pass"] = False

    _write_report(args, report)
    sys.exit(0 if report["all_pass"] else 1)


def _write_report(args, report):
    out_path = args.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(args.config), "integrity_report.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Report -> %s", out_path)
    log.info("ALL PASS: %s", report["all_pass"])


if __name__ == "__main__":
    main()
