#!/usr/bin/env python3
"""Train all 12 domain-specific LoRA adapters on Qwen/Qwen3.5-9B.

Orchestrates sequential training across domains using torchrun + train_domain_lora.py.
Supports resuming, selective domain training, and progress tracking.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

ALL_DOMAINS = [
    "math", "code", "medical", "legal", "finance", "science",
    "history", "geography", "philosophy", "psychology",
    "creative_writing", "translation",
]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def is_domain_trained(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, "adapter_config.json"))


def train_single_domain(
    domain: str,
    config_path: str,
    output_dir: str,
    num_gpus: int,
    master_port: int,
    resume_from: str | None = None,
) -> dict:
    """Launch torchrun for a single domain training."""
    worker_script = str(SCRIPT_DIR / "train_domain_lora.py")

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        f"--master_port={master_port}",
        worker_script,
        "--config", config_path,
        "--domain", domain,
        "--output_dir", output_dir,
        "--num_gpus", str(num_gpus),
    ]
    if resume_from:
        cmd.extend(["--resume_from_checkpoint", resume_from])

    log_path = os.path.join(output_dir, "train.log")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting training for domain '%s'", domain)
    logger.info("  Output: %s", output_dir)
    logger.info("  Log: %s", log_path)
    logger.info("  Command: %s", " ".join(cmd))

    t0 = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_DIR),
        )
    elapsed = time.time() - t0

    result = {
        "domain": domain,
        "return_code": proc.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": output_dir,
        "trained": is_domain_trained(output_dir),
    }
    if proc.returncode != 0:
        logger.error("Training FAILED for domain '%s' (rc=%d). Check %s", domain, proc.returncode, log_path)
    else:
        logger.info("Training complete for '%s' in %.1f seconds", domain, elapsed)
    return result


def main():
    parser = argparse.ArgumentParser(description="Train all 12 domain LoRA adapters")
    parser.add_argument("--config", type=str, default=str(PROJECT_DIR / "configs" / "domains.yaml"))
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root dir for LoRA outputs (default: from config)")
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Specific domains to train (default: all 12)")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--master_port_start", type=int, default=29500)
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if adapter_config.json exists")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print training plan without executing")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = args.output_root or config.get("output_root", "./results/domain_loras")
    if not os.path.isabs(output_root):
        output_root = str(PROJECT_DIR / output_root)

    domains = args.domains or ALL_DOMAINS
    invalid = [d for d in domains if d not in config["domains"]]
    if invalid:
        logger.error("Unknown domains: %s. Available: %s", invalid, list(config["domains"].keys()))
        sys.exit(1)

    lora_cfg = config["lora"]
    train_cfg = config["training"]

    logger.info("=" * 60)
    logger.info("  LoRA Algebra — Train %d Domain LoRAs", len(domains))
    logger.info("  Base model: %s", config["base_model"])
    logger.info("  LoRA r=%d, alpha=%d, epochs=%d",
                lora_cfg["r"], lora_cfg["lora_alpha"], train_cfg["num_train_epochs"])
    logger.info("  GPUs: %d", args.num_gpus)
    logger.info("  Output root: %s", output_root)
    logger.info("=" * 60)

    plan = []
    for domain in domains:
        domain_dir = os.path.join(output_root, domain)
        trained = is_domain_trained(domain_dir)
        skip = trained and not args.force
        plan.append({"domain": domain, "output": domain_dir, "skip": skip, "already_trained": trained})

    to_train = [p for p in plan if not p["skip"]]
    to_skip = [p for p in plan if p["skip"]]

    if to_skip:
        logger.info("Skipping %d already-trained domains: %s",
                     len(to_skip), [p["domain"] for p in to_skip])
    logger.info("Will train %d domains: %s", len(to_train), [p["domain"] for p in to_train])

    if args.dry_run:
        logger.info("[DRY RUN] Would train domains: %s", [p["domain"] for p in to_train])
        return

    results = []
    for idx, p in enumerate(to_train):
        logger.info("")
        logger.info("=" * 50)
        logger.info("  [%d/%d] Domain: %s", idx + 1, len(to_train), p["domain"])
        logger.info("=" * 50)

        result = train_single_domain(
            domain=p["domain"],
            config_path=args.config,
            output_dir=p["output"],
            num_gpus=args.num_gpus,
            master_port=args.master_port_start + idx,
        )
        results.append(result)

    summary_path = os.path.join(output_root, "training_summary.json")
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": config["base_model"],
        "lora_config": lora_cfg,
        "num_domains": len(domains),
        "num_trained": sum(1 for r in results if r["trained"]),
        "num_failed": sum(1 for r in results if not r["trained"]),
        "total_time_seconds": sum(r["elapsed_seconds"] for r in results),
        "results": results,
    }
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Training Summary")
    logger.info("  Trained: %d/%d  |  Failed: %d  |  Skipped: %d",
                summary["num_trained"], len(domains), summary["num_failed"], len(to_skip))
    logger.info("  Total time: %.1f seconds", summary["total_time_seconds"])
    logger.info("  Summary: %s", summary_path)
    logger.info("=" * 60)

    if summary["num_failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
