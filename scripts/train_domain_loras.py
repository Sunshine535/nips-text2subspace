#!/usr/bin/env python3
"""Train all 12 domain-specific LoRA adapters on Qwen/Qwen3.5-9B.

Orchestrates sequential training across domains using torchrun + train_domain_lora.py.
Supports resuming, selective domain training, and progress tracking.
"""

import argparse
import json
import logging
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
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


def _adapter_weight_exists(output_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(output_dir, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(output_dir, "adapter_model.bin"))
    )


def is_domain_trained(output_dir: str) -> bool:
    if not os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        return False
    return _adapter_weight_exists(output_dir)


def verify_adapter_integrity(output_dir: str) -> bool:
    """Verify an adapter directory is loadable, not just that files exist."""
    if not is_domain_trained(output_dir):
        return False
    try:
        import safetensors.torch
        safetensors_path = os.path.join(output_dir, "adapter_model.safetensors")
        bin_path = os.path.join(output_dir, "adapter_model.bin")
        if os.path.exists(safetensors_path):
            sd = safetensors.torch.load_file(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):
            sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            return False
        has_a = any("lora_A" in k for k in sd.keys())
        has_b = any("lora_B" in k for k in sd.keys())
        if not (has_a and has_b):
            logger.warning("Adapter at %s lacks lora_A or lora_B keys", output_dir)
            return False
        return True
    except Exception as e:
        logger.warning("Adapter integrity check failed for %s: %s", output_dir, e)
        return False


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the path of the latest checkpoint-* dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = sorted(
        (d for d in os.listdir(output_dir) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[1]) if d.split("-")[1].isdigit() else 0,
    )
    if ckpts:
        return os.path.join(output_dir, ckpts[-1])
    return None


def _torchrun_exe() -> str:
    raw = os.environ.get("TORCHRUN", "").strip()
    if not raw:
        return "torchrun"
    return shlex.split(raw)[0]


def train_single_domain(
    domain: str,
    config_path: str,
    output_dir: str,
    num_gpus: int,
    master_port: int,
    resume_from: str | None = None,
    seed: int = 42,
) -> dict:
    """Launch torchrun for a single domain training."""
    worker_script = str(SCRIPT_DIR / "train_domain_lora.py")

    cmd = [
        _torchrun_exe(),
        f"--nproc_per_node={num_gpus}",
        f"--master_port={master_port}",
        worker_script,
        "--config", config_path,
        "--domain", domain,
        "--output_dir", output_dir,
        "--num_gpus", str(num_gpus),
        "--seed", str(seed),
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
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Number of GPUs (0 = auto-detect via torch.cuda.device_count())")
    parser.add_argument("--master_port_start", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if adapter_config.json exists")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print training plan without executing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.master_port_start is None:
        args.master_port_start = random.randint(20000, 28000)

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

    # Auto-detect GPU count if not specified
    if args.num_gpus <= 0:
        args.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.num_gpus == 0:
        logger.error("No GPUs detected. Training requires at least 1 GPU.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  LoRA Algebra — Train %d Domain LoRAs", len(domains))
    logger.info("  Base model: %s", config["base_model"])
    logger.info("  LoRA r=%d, alpha=%d, epochs=%d",
                lora_cfg["r"], lora_cfg["lora_alpha"], train_cfg["num_train_epochs"])
    logger.info("  GPUs: %d (auto-detected)" if args.num_gpus == torch.cuda.device_count() else "  GPUs: %d", args.num_gpus)
    logger.info("  Output root: %s", output_root)
    logger.info("=" * 60)

    plan = []
    for domain in domains:
        domain_dir = os.path.join(output_root, domain)
        intact = verify_adapter_integrity(domain_dir)
        skip = intact and not args.force
        if is_domain_trained(domain_dir) and not intact and not args.force:
            logger.warning("Domain '%s' has incomplete adapter at %s — will retrain", domain, domain_dir)
        plan.append({"domain": domain, "output": domain_dir, "skip": skip, "already_trained": intact})

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

        resume_ckpt = find_latest_checkpoint(p["output"])
        if resume_ckpt:
            logger.info("  Resuming from checkpoint: %s", resume_ckpt)

        result = train_single_domain(
            domain=p["domain"],
            config_path=args.config,
            output_dir=p["output"],
            num_gpus=args.num_gpus,
            master_port=args.master_port_start + idx,
            resume_from=resume_ckpt,
            seed=args.seed,
        )
        if result["trained"] and not verify_adapter_integrity(p["output"]):
            logger.error("Post-train integrity check FAILED for '%s'", p["domain"])
            result["trained"] = False
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
