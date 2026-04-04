#!/usr/bin/env python3
"""Train a single domain-specific LoRA on Qwen/Qwen3.5-9B using TRL SFTTrainer + PEFT."""

import argparse
import json
import logging
import os
import sys

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _has_tensorboard() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import tensorboardX  # noqa: F401
        return True
    except ImportError:
        return False


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


MMLU_LETTERS = ["A", "B", "C", "D"]


def format_example(example: dict, system_prompt: str, domain: str) -> dict:
    """Format dataset examples into chat-style text for SFT.
    
    Handles diverse HuggingFace dataset schemas: instruction/output,
    question/answer, MMLU-style with choices, camel-ai message pairs,
    writingprompts prompt/story, sciq support/question/answer, etc.
    """
    choices = example.get("choices", [])
    answer_raw = example.get("answer", example.get("output", example.get("response", "")))

    if choices and isinstance(choices, list) and len(choices) >= 2:
        question = str(example.get("question", example.get("input", ""))).strip()
        if not question:
            return {"text": ""}
        for i, c in enumerate(choices):
            if i < len(MMLU_LETTERS):
                question += f"\n{MMLU_LETTERS[i]}. {c}"
        question += "\n\nAnswer with the letter of the correct choice."
        if isinstance(answer_raw, int) and 0 <= answer_raw < len(MMLU_LETTERS):
            answer = MMLU_LETTERS[answer_raw]
        else:
            answer = str(answer_raw).strip()
        if not answer:
            return {"text": ""}
        text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\nThe answer is {answer}.<|im_end|>"
        )
        return {"text": text}

    instruction = (
        example.get("instruction")
        or example.get("question")
        or example.get("input")
        or example.get("prompt")
        or example.get("message_1")
        or example.get("support", "") + " " + str(example.get("question", ""))
        or example.get("text")
        or ""
    )
    output = (
        example.get("output")
        or example.get("answer")
        or example.get("correct_answer")
        or example.get("response")
        or example.get("message_2")
        or example.get("story")
        or example.get("label")
        or ""
    )
    if isinstance(instruction, list):
        instruction = " ".join(str(x) for x in instruction)
    if isinstance(output, list):
        output = " ".join(str(x) for x in output)
    instruction = str(instruction).strip()
    output = str(output).strip()
    if not instruction or not output:
        return {"text": ""}
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    return {"text": text}


SPLIT_OVERRIDES = {
    "cais/mmlu": "test",  # cais/mmlu: test has 237-311 samples; dev/validation too small
}


def load_domain_dataset(domain_cfg: dict, tokenizer):
    """Load and format a domain-specific dataset."""
    ds_name = domain_cfg["datasets"][0]
    subset = domain_cfg.get("subset", None)
    max_samples = domain_cfg.get("max_samples", 20000)
    system_prompt = domain_cfg.get("system_prompt", "You are a helpful assistant.")
    split = SPLIT_OVERRIDES.get(ds_name, "train")

    logger.info("Loading dataset %s (subset=%s, split=%s, max=%d)", ds_name, subset, split, max_samples)
    try:
        if subset:
            ds = load_dataset(ds_name, subset, split=split)
        else:
            ds = load_dataset(ds_name, split=split)
    except Exception as e:
        logger.error("FAILED to load %s (subset=%s, split=%s): %s", ds_name, subset, split, e)
        logger.error("Will NOT silently generate synthetic data. Fix the dataset config.")
        raise RuntimeError(f"Dataset loading failed for {ds_name}: {e}") from e

    if len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    ds = ds.map(
        lambda ex: format_example(ex, system_prompt, domain_cfg["name"]),
        remove_columns=ds.column_names,
    )
    ds = ds.filter(lambda ex: len(ex["text"]) > 0)
    if len(ds) == 0:
        raise RuntimeError(
            f"Dataset {ds_name} produced 0 valid examples after formatting. "
            f"Check field mapping in format_example()."
        )
    logger.info("Loaded %d examples for domain %s", len(ds), domain_cfg["name"])
    return ds



def _seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train a single domain LoRA")
    parser.add_argument("--config", type=str, default="configs/domains.yaml")
    parser.add_argument("--domain", type=str, required=True, help="Domain key from config")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None, help="Override max_samples from config")
    args = parser.parse_args()

    _seed_everything(args.seed)

    config = load_config(args.config)
    if args.domain not in config["domains"]:
        logger.error("Domain '%s' not found. Available: %s", args.domain, list(config["domains"].keys()))
        sys.exit(1)

    domain_cfg = config["domains"][args.domain]
    if args.max_samples is not None:
        domain_cfg["max_samples"] = args.max_samples
    output_dir = args.output_dir or os.path.join(config["output_root"], args.domain)
    os.makedirs(output_dir, exist_ok=True)

    base_model = config["base_model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    logger.info("=== Training LoRA for domain: %s ===", args.domain)
    logger.info("Base model: %s", base_model)
    logger.info("Output: %s", output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    dataset = load_domain_dataset(domain_cfg, tokenizer)

    import trl
    logger.info("trl version: %s", trl.__version__)

    sft_kwargs = dict(
        output_dir=output_dir,
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

    # processing_class (trl >= 0.12) vs tokenizer (older trl)
    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Saving model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        json.dump({"domain": args.domain, "base_model": base_model, "lora_config": lora_cfg}, f, indent=2)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Destroyed distributed process group")

    logger.info("=== Training complete for domain: %s ===", args.domain)


if __name__ == "__main__":
    main()
