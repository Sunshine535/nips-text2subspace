#!/usr/bin/env python3
"""Train multitask LoRA baselines on the UNION of two domain datasets (oracle upper bound).

For each specified domain pair (e.g., history+philosophy), loads both domain datasets,
concatenates them, and trains a single LoRA adapter on the combined data.
This serves as an oracle upper-bound baseline: a model that has seen training data
from both domains simultaneously.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import concatenate_datasets
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_domain_lora import (
    _has_tensorboard,
    _seed_everything,
    load_config,
    load_domain_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_pairs(pairs_str: str) -> list[list[str]]:
    """Parse comma-separated pair specs like 'history+philosophy,history+science'."""
    pairs = []
    for pair_spec in pairs_str.split(","):
        pair_spec = pair_spec.strip()
        if not pair_spec:
            continue
        domains = pair_spec.split("+")
        if len(domains) < 2:
            logger.warning("Skipping invalid pair spec '%s' (need at least 2 domains)", pair_spec)
            continue
        pairs.append(domains)
    return pairs


def train_multitask_pair(
    domains: list[str],
    config: dict,
    output_dir: str,
    seed: int,
):
    """Train a single LoRA on the concatenation of multiple domain datasets."""
    pair_name = "+".join(domains)
    pair_output = os.path.join(output_dir, pair_name)
    os.makedirs(pair_output, exist_ok=True)

    base_model = config["base_model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    logger.info("=" * 60)
    logger.info("  Training multitask LoRA: %s", pair_name)
    logger.info("  Base model: %s", base_model)
    logger.info("  Output: %s", pair_output)
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and concatenate datasets
    datasets_list = []
    for domain in domains:
        if domain not in config["domains"]:
            logger.error("Domain '%s' not found. Available: %s", domain, list(config["domains"].keys()))
            sys.exit(1)
        domain_cfg = config["domains"][domain]
        ds = load_domain_dataset(domain_cfg, tokenizer)
        logger.info("  %s: %d examples", domain, len(ds))
        datasets_list.append(ds)

    combined_dataset = concatenate_datasets(datasets_list)
    combined_dataset = combined_dataset.shuffle(seed=seed)
    logger.info("Combined dataset: %d examples", len(combined_dataset))

    # Load model
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    logger.info("Using attention implementation: %s", attn_impl)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        device_map={"": 0},
    )
    model.config.use_cache = False

    # LoRA config
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # SFT config (same compatibility handling as train_domain_lora.py)
    sft_kwargs = dict(
        output_dir=pair_output,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=2,
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to="tensorboard" if _has_tensorboard() else "none",
        ddp_find_unused_parameters=False,
        dataset_text_field="text",
    )
    # trl < 0.16: max_seq_length; trl >= 0.16: max_length
    try:
        SFTConfig(output_dir="/tmp/_probe", max_seq_length=512)
        sft_kwargs["max_seq_length"] = train_cfg["max_seq_length"]
    except TypeError:
        sft_kwargs["max_length"] = train_cfg["max_seq_length"]
    # dataset_text_field might be removed in future trl
    try:
        SFTConfig(output_dir="/tmp/_probe", dataset_text_field="text")
    except TypeError:
        sft_kwargs.pop("dataset_text_field", None)

    sft_config = SFTConfig(**sft_kwargs)

    # Trainer (processing_class vs tokenizer compatibility)
    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=combined_dataset,
        peft_config=peft_config,
    )
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)

    logger.info("Starting training for %s...", pair_name)
    trainer.train()

    logger.info("Saving model to %s", pair_output)
    trainer.save_model(pair_output)
    tokenizer.save_pretrained(pair_output)

    with open(os.path.join(pair_output, "training_args.json"), "w") as f:
        json.dump({
            "pair": pair_name,
            "domains": domains,
            "base_model": base_model,
            "lora_config": lora_cfg,
            "total_examples": len(combined_dataset),
        }, f, indent=2)

    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    logger.info("=== Training complete for multitask pair: %s ===", pair_name)


def main():
    parser = argparse.ArgumentParser(description="Train multitask LoRA baselines (oracle upper bound)")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "domains.yaml"))
    parser.add_argument("--output_dir", type=str, default="results/multitask_loras")
    parser.add_argument("--pairs", type=str, required=True,
                        help="Comma-separated domain pairs, e.g. 'history+philosophy,history+science'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _seed_everything(args.seed)

    config = load_config(args.config)
    pairs = parse_pairs(args.pairs)

    if not pairs:
        logger.error("No valid pairs specified. Use format: 'domain1+domain2,domain3+domain4'")
        sys.exit(1)

    logger.info("Will train %d multitask pair(s): %s", len(pairs), ["+".join(p) for p in pairs])
    os.makedirs(args.output_dir, exist_ok=True)

    for domains in pairs:
        train_multitask_pair(domains, config, args.output_dir, args.seed)

    logger.info("All multitask baselines complete.")


if __name__ == "__main__":
    main()
