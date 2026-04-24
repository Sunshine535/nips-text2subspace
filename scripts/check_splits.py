#!/usr/bin/env python3
"""Check train/calibration/test split separation. Fails on any leakage."""
import argparse
import sys
import yaml
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Verify no train/test overlap in domain splits")
    p.add_argument("--splits", default="configs/splits.yaml")
    p.add_argument("--fail_on_leakage", action="store_true", default=True)
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
    total = 0

    for domain, cfg in sorted(domains.items()):
        total += 1
        train_ds = cfg.get("train_dataset", "unknown")
        train_split = cfg.get("train_split", "unknown")
        eval_ds = cfg.get("eval_dataset", "unknown")
        eval_split = cfg.get("eval_split", "unknown")
        is_leaky = cfg.get("leakage_risk", False)
        reason = cfg.get("leakage_reason", "")

        same_dataset = (train_ds == eval_ds)
        same_split = (train_split == eval_split)

        if is_leaky:
            leaky.append(domain)
            status = "LEAKY"
        elif same_dataset and same_split:
            leaky.append(domain)
            status = "OVERLAP DETECTED"
        else:
            safe.append(domain)
            status = "SAFE"

        print(f"  {domain:<15} {status:<20} train={train_ds}:{train_split}  eval={eval_ds}:{eval_split}")
        if reason:
            print(f"  {'':15} reason: {reason}")

    print(f"\nSummary: {len(safe)}/{total} safe, {len(leaky)}/{total} leaky")
    print(f"  Safe domains: {', '.join(safe)}")
    print(f"  Leaky domains: {', '.join(leaky)}")

    if leaky and args.fail_on_leakage:
        print(f"\nFAIL: {len(leaky)} domains have train/test leakage risk")
        print("These domains' adapter results CANNOT be used as strong evidence")
        print("CARR evaluation must use only safe domains: " + ", ".join(safe))
        sys.exit(1)

    print("\nPASS: No leakage detected in checked domains")
    return 0


if __name__ == "__main__":
    main()
