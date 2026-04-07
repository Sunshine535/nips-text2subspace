#!/usr/bin/env python3
"""Parallel evaluation across multiple GPUs.

Distributes domain-pair evaluations across available GPUs for maximum throughput.
Each GPU runs one evaluation at a time; pairs are assigned round-robin.
"""

import argparse
import itertools
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

CORE_DOMAINS = ["math", "code", "medical", "science", "history", "philosophy"]


def detect_gpus() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def run_eval_on_gpu(gpu_id: int, pair: tuple, eval_script: str,
                    output_dir: str, model_name: str, lora_dir: str,
                    max_samples: int) -> dict:
    """Run a single evaluation on a specific GPU."""
    domain_a, domain_b = pair
    pair_name = f"{domain_a}+{domain_b}"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, eval_script,
        "--model", model_name,
        "--lora_dir", lora_dir,
        "--domains", domain_a, domain_b,
        "--output_dir", output_dir,
        "--max_samples", str(max_samples),
    ]

    logger.info(f"[GPU {gpu_id}] Starting eval: {pair_name}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=3600
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            logger.info(f"[GPU {gpu_id}] Finished {pair_name} in {elapsed:.0f}s")
            return {"pair": pair_name, "gpu": gpu_id, "status": "ok", "time": elapsed}
        else:
            logger.error(f"[GPU {gpu_id}] Failed {pair_name}: {result.stderr[-500:]}")
            return {"pair": pair_name, "gpu": gpu_id, "status": "error",
                    "stderr": result.stderr[-500:]}
    except subprocess.TimeoutExpired:
        logger.error(f"[GPU {gpu_id}] Timeout on {pair_name}")
        return {"pair": pair_name, "gpu": gpu_id, "status": "timeout"}


def main():
    parser = argparse.ArgumentParser(description="Parallel multi-GPU evaluation")
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Number of GPUs (0 = auto-detect)")
    parser.add_argument("--eval_script", type=str,
                        default=str(SCRIPT_DIR / "eval_domain_accuracy.py"))
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--lora_dir", type=str, default="results/domain_loras")
    parser.add_argument("--output_dir", type=str, default="results/eval_parallel")
    parser.add_argument("--domains", nargs="+", default=CORE_DOMAINS)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--composition", choices=["pairwise", "nway"], default="pairwise")
    args = parser.parse_args()

    num_gpus = args.num_gpus if args.num_gpus > 0 else detect_gpus()
    if num_gpus == 0:
        logger.error("No GPUs detected.")
        sys.exit(1)

    logger.info(f"Using {num_gpus} GPUs for parallel evaluation")

    # Generate all pairs
    if args.composition == "pairwise":
        pairs = list(itertools.combinations(args.domains, 2))
    else:
        pairs = [tuple(args.domains)]  # single N-way

    logger.info(f"Total evaluation jobs: {len(pairs)}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Distribute pairs across GPUs using ProcessPoolExecutor
    # max_workers = num_gpus ensures one job per GPU at a time
    results = []
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {}
        for i, pair in enumerate(pairs):
            gpu_id = i % num_gpus
            future = executor.submit(
                run_eval_on_gpu, gpu_id, pair, args.eval_script,
                args.output_dir, args.model, args.lora_dir, args.max_samples,
            )
            futures[future] = pair

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            ok = sum(1 for r in results if r["status"] == "ok")
            logger.info(f"Progress: {len(results)}/{len(pairs)} ({ok} ok)")

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] != "ok")
    total_time = sum(r.get("time", 0) for r in results)
    logger.info(f"\nDone: {ok} ok, {failed} failed, total GPU-time: {total_time:.0f}s")

    if failed > 0:
        for r in results:
            if r["status"] != "ok":
                logger.warning(f"  FAILED: {r['pair']} on GPU {r['gpu']}: {r['status']}")


if __name__ == "__main__":
    main()
