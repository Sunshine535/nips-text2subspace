#!/usr/bin/env python3
"""Domain-specific evaluation for LoRA algebra experiments.

Evaluates:
- Individual domain LoRAs on their native benchmarks
- Composed LoRAs on both constituent domains
- Baseline merging methods (TIES, DARE, Task Arithmetic)

Benchmarks per domain:
  math → GSM8K, MATH  |  code → HumanEval, MBPP
  medical → MedQA      |  legal → LegalBench  |  etc.

Metrics: accuracy, perplexity, domain-specific F1
Outputs: comparison tables in JSON + markdown
"""

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lora_algebra import LoRAWeights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)



DOMAIN_BENCHMARKS = {
    "math": {
        "gsm8k": {"dataset_id": "openai/gsm8k", "subset": "main", "split": "test",
                   "q_field": "question", "a_field": "answer", "max_samples": 500,
                   "system_prompt": "Solve this math problem step by step. End your solution with #### followed by the final numeric answer (e.g., #### 42)."},
    },
    "code": {
        "mbpp": {"dataset_id": "google-research-datasets/mbpp", "subset": "sanitized",
                 "split": "test", "q_field": "prompt", "a_field": "code", "max_samples": 500,
                 "system_prompt": "Write clean Python code. Provide only the code."},
    },
    "medical": {
        "medqa": {"dataset_id": "GBaker/MedQA-USMLE-4-options", "split": "test",
                  "q_field": "question", "a_field": "answer_idx", "options_field": "options",
                  "max_samples": 500, "multichoice": True,
                  "system_prompt": "You are a medical expert. Answer with the letter of the correct choice."},
    },
    "legal": {
        "legalbench_subset": {"dataset_id": "nguha/legalbench", "subset": "abercrombie",
                              "split": "test", "q_field": "text", "a_field": "label", "max_samples": 200},
    },
    "finance": {
        "finance_qa": {"dataset_id": "AdaptLLM/finance-tasks", "split": "test",
                       "q_field": "input", "a_field": "output", "max_samples": 500},
    },
    "science": {
        "arc_challenge": {"dataset_id": "allenai/ai2_arc", "subset": "ARC-Challenge",
                          "split": "test", "q_field": "question", "a_field": "answerKey", "max_samples": 500, "multichoice": True,
                          "system_prompt": "Answer the multiple choice question. Reply with just the letter of the correct answer (A, B, C, or D)."},
    },
    "history": {
        "mmlu_history": {"dataset_id": "cais/mmlu", "subset": "high_school_world_history",
                         "split": "test", "q_field": "question", "a_field": "answer", "max_samples": 500, "multichoice": True,
                         "system_prompt": "Answer the multiple choice question. Reply with just the letter of the correct answer (A, B, C, or D)."},
    },
    "geography": {
        "mmlu_geography": {"dataset_id": "cais/mmlu", "subset": "high_school_geography",
                           "split": "test", "q_field": "question", "a_field": "answer", "max_samples": 500, "multichoice": True,
                           "system_prompt": "Answer the multiple choice question. Reply with just the letter of the correct answer (A, B, C, or D)."},
    },
    "philosophy": {
        "mmlu_philosophy": {"dataset_id": "cais/mmlu", "subset": "philosophy",
                            "split": "test", "q_field": "question", "a_field": "answer", "max_samples": 500, "multichoice": True,
                            "system_prompt": "Answer the multiple choice question. Reply with just the letter of the correct answer (A, B, C, or D)."},
    },
    "psychology": {
        "mmlu_psychology": {"dataset_id": "cais/mmlu", "subset": "high_school_psychology",
                            "split": "test", "q_field": "question", "a_field": "answer", "max_samples": 500, "multichoice": True,
                            "system_prompt": "Answer the multiple choice question. Reply with just the letter of the correct answer (A, B, C, or D)."},
    },
    "creative_writing": {
        "creative_writing_synthetic": {"synthetic": True, "max_samples": 5},
    },
    "translation": {
        "translation_synthetic": {"synthetic": True, "max_samples": 5},
    },
}

ALL_EVAL_DOMAINS = list(DOMAIN_BENCHMARKS.keys())

MMLU_CHOICES = ["A", "B", "C", "D"]


def extract_answer(text: str) -> str:
    patterns = [
        r"####\s*([\-\d,]+\.?\d*)",
        r"(?:the answer is|answer:)\s*\$?\\?boxed\{([^}]+)\}",
        r"(?:the answer is|answer:)\s*\(?([A-D])\)?",
        # MCQ: "correct answer is A", "answer: B", "**A.**", etc.
        r"(?:correct answer|the answer)\s*(?:is)?[:\s]*\**\(?([A-D])\)?",
        r"(?:correct answer|the answer)\s*(?:is)?[:\s]*\**([A-D])[\.\s\*]",
        r"(?:the answer is|answer:)\s*\$?\s*([\-\d,]+\.?\d*)",
        # Bold letter at start: "**A. description**"
        r"^\**([A-D])[\.\)]",
        # Standalone letter on a line
        r"^([A-D])\s*$",
        r"\b([A-D])\b\s*$",
        r"([\-\d,]+\.?\d*)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "").strip()
    return text.strip().split("\n")[-1].strip()


def _extract_code_block(text: str) -> str:
    """Extract python code from markdown fences or return raw text."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _run_code_with_tests(code: str, test_list: list, timeout: int = 10) -> bool:
    """Execute generated code against test assertions, return True if all pass."""
    combined = code + "\n" + "\n".join(test_list)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(combined)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, timeout=timeout,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        os.unlink(tmp_path)


def format_mmlu_question(example: dict) -> str:
    q = example.get("question", "")
    choices = example.get("choices", example.get("options", {}))
    if isinstance(choices, dict) and "text" in choices:
        choice_texts = choices["text"]
        labels = choices.get("label", MMLU_CHOICES[:len(choice_texts)])
        for label, text in zip(labels, choice_texts):
            q += f"\n{label}. {text}"
    elif isinstance(choices, list):
        for i, c in enumerate(choices):
            if i < len(MMLU_CHOICES):
                q += f"\n{MMLU_CHOICES[i]}. {c}"
    elif isinstance(choices, dict):
        for key in sorted(choices.keys()):
            q += f"\n{key}. {choices[key]}"
    q += "\n\nAnswer with the letter of the correct choice."
    return q


def decode_answer(example: dict, bench_cfg: dict) -> str:
    """Decode the gold answer, handling ClassLabel integers, GSM8K format, and lettered answers."""
    raw = example.get(bench_cfg.get("a_field", "answer"), "")
    if isinstance(raw, int) and 0 <= raw < len(MMLU_CHOICES):
        return MMLU_CHOICES[raw]
    raw_str = str(raw).strip()
    # GSM8K: answer field contains reasoning ending with "#### NUMBER"
    gsm_match = re.search(r"####\s*([\-\d,]+\.?\d*)", raw_str)
    if gsm_match:
        return gsm_match.group(1).replace(",", "").strip()
    return raw_str


def _strip_think_tags(text: str) -> str:
    """Strip Qwen3 <think>...</think> reasoning blocks, keeping only the final answer."""
    # Remove complete <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Handle truncated thinking: if <think> exists without closing </think>,
    # the model ran out of tokens mid-thought. Remove from <think> to end.
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
    # If stripping removed everything, fall back to original
    return cleaned if cleaned else text


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 2048, system_prompt: str = "") -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    # Always use enable_thinking=False for direct answers.
    # This works for both base model and PEFT/merged models:
    # - Base model: answers directly (designed API)
    # - PEFT LoRA: answers directly (tested to work correctly)
    # - Delta-weight merged: answers directly
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return _strip_think_tags(response)


def evaluate_code_execution(model, tokenizer, bench_cfg: dict) -> dict:
    """Execution-based evaluation for MBPP: run generated code against test assertions."""
    ds_id = bench_cfg["dataset_id"]
    subset = bench_cfg.get("subset")
    split = bench_cfg.get("split", "test")
    max_samples = bench_cfg.get("max_samples", 500)

    try:
        ds = load_dataset(ds_id, subset, split=split) if subset else load_dataset(ds_id, split=split)
    except Exception as e:
        logger.warning("Failed to load %s/%s: %s", ds_id, subset, e)
        return {"accuracy": 0.0, "total": 0, "error": str(e)}

    if len(ds) > max_samples:
        ds = ds.shuffle(seed=sample_seed).select(range(max_samples))

    correct, total = 0, 0
    t0 = time.time()
    for ex in ds:
        task_desc = ex.get("prompt", ex.get("text", ""))
        prompt = f"Write a Python function.\n\n{task_desc}\n\nProvide only the code, no explanations."
        response = generate_response(model, tokenizer, prompt)
        code = _extract_code_block(response)
        test_list = ex.get("test_list", [])
        if test_list and _run_code_with_tests(code, test_list):
            correct += 1
        total += 1

    elapsed = time.time() - t0
    return {
        "accuracy": round(correct / max(total, 1), 4),
        "correct": correct,
        "total": total,
        "eval_mode": "execution",
        "time_seconds": round(elapsed, 1),
    }


def evaluate_on_benchmark(model, tokenizer, bench_cfg: dict, domain: str, sample_seed: int = 42) -> dict:
    """Evaluate model on a single benchmark."""
    if bench_cfg.get("synthetic"):
        return evaluate_synthetic(model, tokenizer, domain, bench_cfg.get("max_samples", 100))

    if domain == "code" and "mbpp" in bench_cfg.get("dataset_id", ""):
        return evaluate_code_execution(model, tokenizer, bench_cfg)

    ds_id = bench_cfg["dataset_id"]
    subset = bench_cfg.get("subset")
    split = bench_cfg.get("split", "test")
    q_field = bench_cfg.get("q_field", "question")
    a_field = bench_cfg.get("a_field", "answer")
    max_samples = bench_cfg.get("max_samples", 500)
    is_mc = bench_cfg.get("multichoice", False)

    try:
        if subset:
            ds = load_dataset(ds_id, subset, split=split)
        else:
            ds = load_dataset(ds_id, split=split)
    except Exception as e:
        logger.warning("Failed to load %s/%s: %s", ds_id, subset, e)
        return {"accuracy": 0.0, "total": 0, "error": str(e)}

    if len(ds) > max_samples:
        ds = ds.shuffle(seed=sample_seed).select(range(max_samples))

    correct, total, total_tokens = 0, 0, 0
    t0 = time.time()

    options_field = bench_cfg.get("options_field", None)
    for ex in ds:
        if is_mc:
            if options_field and options_field in ex:
                ex_copy = dict(ex)
                ex_copy["options"] = ex[options_field]
            else:
                ex_copy = ex
            question = format_mmlu_question(ex_copy)
        else:
            question = str(ex.get(q_field, ""))

        gold_extracted = decode_answer(ex, bench_cfg)
        if not gold_extracted:
            gold = str(ex.get(a_field, ""))
            gold_extracted = extract_answer(gold)

        sys_prompt = bench_cfg.get("system_prompt", "")
        response = generate_response(model, tokenizer, question, system_prompt=sys_prompt)
        pred = extract_answer(response)
        total_tokens += len(tokenizer.encode(response))

        try:
            is_correct = abs(float(pred) - float(gold_extracted)) < 1e-3
        except (ValueError, TypeError):
            is_correct = pred.strip().upper() == gold_extracted.strip().upper()

        if is_correct:
            correct += 1
        total += 1

    elapsed = time.time() - t0
    return {
        "accuracy": round(correct / max(total, 1), 4),
        "correct": correct,
        "total": total,
        "avg_tokens": round(total_tokens / max(total, 1), 1),
        "time_seconds": round(elapsed, 1),
    }


def evaluate_synthetic(model, tokenizer, domain: str, n: int) -> dict:
    """Synthetic quality evaluation for creative_writing / translation."""
    prompts = {
        "creative_writing": [
            "Write a short poem about the ocean at sunset.",
            "Create a compelling opening paragraph for a mystery novel.",
            "Write a haiku about artificial intelligence.",
        ],
        "translation": [
            "Translate 'Knowledge is power' into Chinese, French, and German.",
            "Translate 'The early bird catches the worm' into Japanese and Spanish.",
        ],
    }
    domain_prompts = prompts.get(domain, prompts["creative_writing"])
    results = []
    for prompt in domain_prompts[:n]:
        response = generate_response(model, tokenizer, prompt)
        results.append({"prompt": prompt, "response_length": len(response), "tokens": len(tokenizer.encode(response))})

    return {
        "accuracy": -1.0,
        "total": len(results),
        "avg_response_length": sum(r["response_length"] for r in results) / max(len(results), 1),
        "avg_tokens": sum(r["tokens"] for r in results) / max(len(results), 1),
        "note": "synthetic benchmark — qualitative evaluation only",
    }


def load_model_with_adapter(base_model_name: str, adapter_path: str | None, device_map: str = "auto"):
    """Load base model optionally with a PEFT adapter."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device_map,
    )
    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def apply_delta_weights(base_model_name: str, delta_path: str, device_map: str = "auto"):
    """Load base model and apply pre-computed delta weights."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device_map,
    )
    delta = torch.load(delta_path, map_location="cpu")
    state = model.state_dict()
    for key, d in delta.items():
        for suffix in ["", ".weight"]:
            full_key = key + suffix
            if full_key in state:
                state[full_key] = state[full_key].float() + d.float()
                state[full_key] = state[full_key].to(torch.bfloat16)
                break
    model.load_state_dict(state)
    model.eval()
    return model


def generate_comparison_table(all_results: dict, output_dir: str):
    """Generate markdown comparison table."""
    lines = ["# LoRA Algebra — Domain Accuracy Comparison", ""]
    lines.append("| Method | Domain | Benchmark | Accuracy | Avg Tokens |")
    lines.append("|--------|--------|-----------|----------|------------|")

    for section, section_data in all_results.items():
        if section == "meta":
            continue
        if isinstance(section_data, dict):
            for key, val in section_data.items():
                if isinstance(val, dict):
                    for bench, metrics in val.items():
                        if isinstance(metrics, dict) and "accuracy" in metrics:
                            acc = metrics["accuracy"]
                            tok = metrics.get("avg_tokens", "-")
                            lines.append(f"| {section} | {key} | {bench} | {acc:.4f} | {tok} |")

    md_path = os.path.join(output_dir, "comparison_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Comparison table: %s", md_path)


def main():
    parser = argparse.ArgumentParser(description="Domain-specific LoRA evaluation")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "domains.yaml"))
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory with trained domain LoRAs")
    parser.add_argument("--algebra_dir", type=str, default=None, help="Directory with algebra experiment outputs")
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--domains", nargs="+", default=None, help="Specific domains to evaluate")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_base", action="store_true", help="Skip base model evaluation")
    parser.add_argument("--skip_composed", action="store_true", help="Skip composed LoRA evaluation")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip baseline merging evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    base_model_name = config["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_domains = args.domains or ALL_EVAL_DOMAINS
    all_results = {"meta": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model_name,
        "domains": eval_domains,
    }}

    # --- Phase 1: Base model (no LoRA) ---
    if not args.skip_base:
        logger.info("=" * 50)
        logger.info("  PHASE 1: Base Model Evaluation")
        logger.info("=" * 50)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
        )
        model.eval()
        base_results = {}
        for domain in eval_domains:
            if domain not in DOMAIN_BENCHMARKS:
                continue
            domain_results = {}
            for bench_name, bench_cfg in DOMAIN_BENCHMARKS[domain].items():
                if args.max_samples:
                    bench_cfg = {**bench_cfg, "max_samples": args.max_samples}
                logger.info("  Base model → %s/%s", domain, bench_name)
                domain_results[bench_name] = evaluate_on_benchmark(model, tokenizer, bench_cfg, domain)
            base_results[domain] = domain_results
        all_results["base_model"] = base_results
        del model
        torch.cuda.empty_cache()

    # --- Phase 2: Individual domain LoRAs ---
    logger.info("=" * 50)
    logger.info("  PHASE 2: Individual Domain LoRAs")
    logger.info("=" * 50)
    individual_results = {}
    for domain in eval_domains:
        adapter_path = os.path.join(args.lora_dir, domain)
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            logger.warning("No adapter for '%s', skipping", domain)
            continue
        if domain not in DOMAIN_BENCHMARKS:
            continue

        logger.info("  Evaluating LoRA: %s", domain)
        model = load_model_with_adapter(base_model_name, adapter_path)
        domain_results = {}
        for bench_name, bench_cfg in DOMAIN_BENCHMARKS[domain].items():
            if args.max_samples:
                bench_cfg = {**bench_cfg, "max_samples": args.max_samples}
            logger.info("    %s/%s", domain, bench_name)
            domain_results[bench_name] = evaluate_on_benchmark(model, tokenizer, bench_cfg, domain)
        individual_results[domain] = domain_results
        del model
        torch.cuda.empty_cache()
    all_results["individual_loras"] = individual_results

    # --- Phase 3: Composed LoRAs (GrassMerge pairwise) ---
    if not args.skip_composed and args.algebra_dir:
        logger.info("=" * 50)
        logger.info("  PHASE 3: Composed LoRAs (GrassMerge)")
        logger.info("=" * 50)
        composed_results = {}
        compose_dir = os.path.join(args.algebra_dir, "grassmerge")
        if not os.path.isdir(compose_dir):
            compose_dir = os.path.join(args.algebra_dir, "composed")
        if os.path.isdir(compose_dir):
            pt_files = sorted(f for f in os.listdir(compose_dir) if f.endswith(".pt"))
            for fname in pt_files:
                name = fname.replace(".pt", "")
                domains_in = name.split("+")
                logger.info("  Evaluating GrassMerge: %s", name)
                peft_subdir = os.path.join(compose_dir, name)
                if os.path.isdir(peft_subdir) and os.path.exists(os.path.join(peft_subdir, "adapter_config.json")):
                    model = load_model_with_adapter(base_model_name, peft_subdir)
                else:
                    model = apply_delta_weights(base_model_name, os.path.join(compose_dir, fname))
                for d in domains_in:
                    if d in DOMAIN_BENCHMARKS:
                        for bench_name, bench_cfg in DOMAIN_BENCHMARKS[d].items():
                            if args.max_samples:
                                bench_cfg = {**bench_cfg, "max_samples": args.max_samples}
                            key = f"{name}_on_{d}_{bench_name}"
                            logger.info("    %s", key)
                            composed_results[key] = evaluate_on_benchmark(model, tokenizer, bench_cfg, d)
                del model
                torch.cuda.empty_cache()
        all_results["grassmerge"] = composed_results

    # --- Phase 4: Pairwise baseline merging methods ---
    if not args.skip_baselines and args.algebra_dir:
        logger.info("=" * 50)
        logger.info("  PHASE 4: Pairwise Baseline Merging Methods")
        logger.info("=" * 50)
        baseline_results = {}
        baseline_dir = os.path.join(args.algebra_dir, "baselines")
        if os.path.isdir(baseline_dir):
            for method_name in sorted(os.listdir(baseline_dir)):
                method_dir = os.path.join(baseline_dir, method_name)
                if not os.path.isdir(method_dir):
                    continue
                logger.info("  Baseline method: %s", method_name)
                method_results = {}
                pt_files = sorted(f for f in os.listdir(method_dir) if f.endswith(".pt"))
                for fname in pt_files:
                    name = fname.replace(".pt", "")
                    domains_in = name.split("+")
                    logger.info("    Evaluating %s: %s", method_name, name)
                    peft_subdir = os.path.join(method_dir, name)
                    if os.path.isdir(peft_subdir) and os.path.exists(os.path.join(peft_subdir, "adapter_config.json")):
                        model = load_model_with_adapter(base_model_name, peft_subdir)
                    else:
                        model = apply_delta_weights(
                            base_model_name, os.path.join(method_dir, fname)
                        )
                    for d in domains_in:
                        if d in DOMAIN_BENCHMARKS:
                            for bench_name, bench_cfg in DOMAIN_BENCHMARKS[d].items():
                                if args.max_samples:
                                    bench_cfg = {**bench_cfg, "max_samples": args.max_samples}
                                key = f"{name}_on_{d}_{bench_name}"
                                method_results[key] = evaluate_on_benchmark(
                                    model, tokenizer, bench_cfg, d
                                )
                    del model
                    torch.cuda.empty_cache()
                baseline_results[method_name] = method_results
        all_results["baselines"] = baseline_results

    # --- Save results ---
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    generate_comparison_table(all_results, args.output_dir)

    logger.info("=" * 60)
    logger.info("  Evaluation complete")
    logger.info("  Results: %s", results_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
