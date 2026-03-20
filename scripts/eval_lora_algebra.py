#!/usr/bin/env python3
"""Evaluate composed LoRAs on domain-specific benchmarks."""

import argparse
import json
import logging
import os
import sys
import time

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.lora_algebra import LoRAWeights


DOMAIN_EVAL_PROMPTS = {
    "math": [
        "Solve: What is 15% of 240?",
        "If f(x) = 3x^2 - 2x + 1, find f(4).",
        "A rectangle has perimeter 40cm and area 96cm². Find its dimensions.",
    ],
    "code": [
        "Write a Python function to find the longest common subsequence of two strings.",
        "Implement binary search in Python.",
        "Write a function to detect cycles in a linked list.",
    ],
    "medical": [
        "What are the symptoms and treatment options for Type 2 Diabetes?",
        "Explain the mechanism of action of ACE inhibitors.",
        "What is the differential diagnosis for acute chest pain?",
    ],
    "legal": [
        "Explain the doctrine of res judicata.",
        "What are the elements required to prove negligence?",
        "Describe the difference between civil and criminal liability.",
    ],
    "creative": [
        "Write a short poem about the ocean at sunset.",
        "Create a compelling opening paragraph for a mystery novel.",
        "Write a haiku about artificial intelligence.",
    ],
    "science": [
        "Explain the process of nuclear fusion in stars.",
        "What is the difference between mitosis and meiosis?",
        "Describe the principle of conservation of energy.",
    ],
    "history": [
        "What were the main causes of World War I?",
        "Describe the significance of the Magna Carta.",
        "What led to the fall of the Roman Empire?",
    ],
    "finance": [
        "Explain the concept of compound interest with an example.",
        "What is the Capital Asset Pricing Model (CAPM)?",
        "Describe the difference between systematic and unsystematic risk.",
    ],
    "philosophy": [
        "Explain Kant's categorical imperative.",
        "What is the problem of other minds?",
        "Describe the trolley problem and its ethical implications.",
    ],
    "multilingual": [
        "Translate 'Knowledge is power' into Chinese, French, and German.",
        "用中文解释什么是人工智能。",
        "Expliquez le concept de liberté en philosophie.",
    ],
}


def load_base_model_and_tokenizer(model_name: str):
    logger.info("Loading base model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    return model, tokenizer


def apply_delta_weights(base_model, delta_weights: dict):
    """Apply pre-computed delta weights directly to the base model."""
    state = base_model.state_dict()
    for key, delta in delta_weights.items():
        full_key = key + ".weight" if key + ".weight" in state else key
        if full_key in state:
            state[full_key] = state[full_key].float() + delta.float()
            state[full_key] = state[full_key].to(torch.bfloat16)
    base_model.load_state_dict(state)
    return base_model


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    with torch.no_grad():
        output = model.generate(**inputs, generation_config=gen_config)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate_domain(model, tokenizer, domain: str, prompts: list) -> dict:
    """Evaluate model on domain-specific prompts. Returns qualitative results + timing."""
    results = []
    total_tokens = 0
    t0 = time.time()
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt)
        tokens = len(tokenizer.encode(response))
        total_tokens += tokens
        results.append({"prompt": prompt, "response": response, "tokens": tokens})
    elapsed = time.time() - t0
    return {
        "domain": domain,
        "num_prompts": len(prompts),
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / len(prompts),
        "time_seconds": round(elapsed, 2),
        "results": results,
    }


def evaluate_peft_adapter(base_model_name: str, adapter_path: str, domain: str, tokenizer):
    """Evaluate a PEFT adapter by loading it onto the base model."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    prompts = DOMAIN_EVAL_PROMPTS.get(domain, DOMAIN_EVAL_PROMPTS["math"])
    return evaluate_domain(model, tokenizer, domain, prompts)


def main():
    parser = argparse.ArgumentParser(description="Evaluate composed LoRAs")
    parser.add_argument("--config", type=str, default="configs/domains.yaml")
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--algebra_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    base_model_name = config["base_model"]
    model, tokenizer = load_base_model_and_tokenizer(base_model_name)
    model.eval()

    all_eval_results = {}

    # 1. Evaluate base model (no LoRA)
    logger.info("=== Evaluating base model (no LoRA) ===")
    base_results = {}
    for domain, prompts in DOMAIN_EVAL_PROMPTS.items():
        logger.info("  Base model on domain: %s", domain)
        base_results[domain] = evaluate_domain(model, tokenizer, domain, prompts)
    all_eval_results["base_model"] = base_results

    # 2. Evaluate individual domain LoRAs
    logger.info("=== Evaluating individual domain LoRAs ===")
    domain_results = {}
    for domain in config["domains"]:
        adapter_path = os.path.join(args.lora_dir, domain)
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            logger.warning("No adapter found for domain %s, skipping", domain)
            continue
        logger.info("  Evaluating domain LoRA: %s", domain)
        result = evaluate_peft_adapter(base_model_name, adapter_path, domain, tokenizer)
        domain_results[domain] = result
    all_eval_results["domain_loras"] = domain_results

    # 3. Evaluate composed LoRAs
    logger.info("=== Evaluating composed LoRAs ===")
    composed_results = {}
    compose_dir = os.path.join(args.algebra_dir, "composed")
    if os.path.isdir(compose_dir):
        for fname in sorted(os.listdir(compose_dir)):
            if not fname.endswith(".pt"):
                continue
            name = fname.replace(".pt", "")
            logger.info("  Evaluating composed LoRA: %s", name)
            delta = torch.load(os.path.join(compose_dir, fname), map_location="cpu")
            model_copy = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
            )
            model_copy = apply_delta_weights(model_copy, delta)
            model_copy.eval()
            domains_in_name = name.split("+")
            for d in domains_in_name:
                if d in DOMAIN_EVAL_PROMPTS:
                    composed_results[f"{name}_on_{d}"] = evaluate_domain(
                        model_copy, tokenizer, d, DOMAIN_EVAL_PROMPTS[d]
                    )
            del model_copy
            torch.cuda.empty_cache()
    all_eval_results["composed"] = composed_results

    # 4. Evaluate baseline merging methods
    logger.info("=== Evaluating baseline merging methods ===")
    baseline_results = {}
    baseline_dir = os.path.join(args.algebra_dir, "baselines")
    if os.path.isdir(baseline_dir):
        for fname in sorted(os.listdir(baseline_dir)):
            if not fname.endswith(".pt"):
                continue
            method = fname.replace(".pt", "")
            logger.info("  Evaluating baseline: %s", method)
            delta = torch.load(os.path.join(baseline_dir, fname), map_location="cpu")
            model_copy = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
            )
            model_copy = apply_delta_weights(model_copy, delta)
            model_copy.eval()
            method_results = {}
            for domain in ["math", "code", "medical"]:
                method_results[domain] = evaluate_domain(
                    model_copy, tokenizer, domain, DOMAIN_EVAL_PROMPTS[domain]
                )
            baseline_results[method] = method_results
            del model_copy
            torch.cuda.empty_cache()
    all_eval_results["baselines"] = baseline_results

    # Save all results
    output_path = os.path.join(args.output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(all_eval_results, f, indent=2, ensure_ascii=False)

    logger.info("=== Evaluation complete. Results: %s ===", output_path)

    # Print summary
    logger.info("\n=== EVALUATION SUMMARY ===")
    for section, data in all_eval_results.items():
        if isinstance(data, dict):
            logger.info("Section: %s (%d entries)", section, len(data))


if __name__ == "__main__":
    main()
