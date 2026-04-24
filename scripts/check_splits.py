#!/usr/bin/env python3
"""Check train/calibration/test split separation.

Modes:
  --mode audit_all:    List all domains, flag leaky ones, exit 0 (informational)
  --mode require_safe: Exit 1 if ANY checked domain has leakage risk
"""
import argparse
import sys
import yaml
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Verify no train/test overlap in domain splits")
    p.add_argument("--splits", default="configs/splits.yaml")
    p.add_argument("--mode", choices=["audit_all", "require_safe"], default="require_safe")
    p.add_argument("--domains", default=None, help="Comma-separated domains to check (default: all)")
    args = p.parse_args()

    splits_path = Path(args.splits)
    if not splits_path.exists():
        print(f"ERROR: splits file not found: {splits_path}")
        sys.exit(1)

    with open(splits_path) as f:
        config = yaml.safe_load(f)

    domains = config.get("domains", {})
    if args.domains:
        selected = set(args.domains.split(","))
        domains = {k: v for k, v in domains.items() if k in selected}

    leaky = []
    safe = []

    for domain, cfg in sorted(domains.items()):
        train_ds = cfg.get("train_dataset", "unknown")
        eval_ds = cfg.get("eval_dataset", "unknown")
        eval_split = cfg.get("eval_split", "unknown")
        is_leaky = cfg.get("leakage_risk", False)
        reason = cfg.get("leakage_reason", "")

        hist_split = cfg.get("historical_train_split", cfg.get("train_split", "unknown"))
        curr_split = cfg.get("current_train_split", cfg.get("train_split", "unknown"))

        if is_leaky:
            leaky.append(domain)
            status = "LEAKY"
        else:
            safe.append(domain)
            status = "SAFE"

        print(f"  {domain:<15} {status:<8} train={train_ds}:{curr_split} (historical:{hist_split})  eval={eval_ds}:{eval_split}")
        if reason:
            print(f"  {'':15} reason: {reason}")

    total = len(safe) + len(leaky)
    print(f"\nSummary: {len(safe)}/{total} safe, {len(leaky)}/{total} leaky")
    print(f"  Safe: {', '.join(safe) if safe else 'none'}")
    print(f"  Leaky: {', '.join(leaky) if leaky else 'none'}")

    if args.mode == "require_safe":
        if leaky:
            print(f"\nFAIL (require_safe): {len(leaky)} domains have leakage risk")
            sys.exit(1)
        else:
            print("\nPASS: All checked domains are safe")
            sys.exit(0)
    else:
        if leaky:
            print(f"\nAUDIT: {len(leaky)} domains flagged as leaky (informational, exit 0)")
        else:
            print("\nAUDIT: No leakage detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
