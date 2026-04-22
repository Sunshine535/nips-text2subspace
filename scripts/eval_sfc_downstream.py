#!/usr/bin/env python3
"""
SFC Downstream Evaluation: Compare SFC composition vs weight-space baselines.

For each adapter pair:
  1. Base model (lower bound)
  2. Single adapter A on domain A (upper bound)
  3. Single adapter B on domain B (upper bound)
  4. Task Arithmetic merge → eval on both domains
  5. TIES merge → eval on both domains
  6. SFC-Exact composition → eval on both domains

Outputs a JSON comparison table.

Usage:
    python scripts/eval_sfc_downstream.py \
        --model /root/models/Qwen3.5-9B \
        --sae-dir /root/saes/qwen3.5-9b \
        --adapter-dir /root/nips-text2subspace/results/sfc_loras_test \
        --dataset-dir /root/datasets \
        --layers 7,15,23 \
        --n-samples 100 \
        --output /root/nips-text2subspace/results/sfc_downstream.json
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_sfc")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen3.5-9B")
    p.add_argument("--sae-dir", default="/root/saes/qwen3.5-9b")
    p.add_argument("--adapter-dir", default="/root/nips-text2subspace/results/sfc_loras_test")
    p.add_argument("--dataset-dir", default="/root/datasets")
    p.add_argument("--layers", default="7,15,23")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--probe-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--threshold-multiplier", type=float, default=3.0)
    p.add_argument("--output", default="/root/nips-text2subspace/results/sfc_downstream.json")
    p.add_argument("--pairs", default=None, help="Comma-separated pairs like 'math+medical,science+philosophy'")
    return p.parse_args()


# ---------------------------------------------------------------------------
# MCQ Evaluation
# ---------------------------------------------------------------------------

DOMAIN_EVAL = {
    "math": {"local": "gsm8k", "split": "test", "type": "gsm8k"},
    "code": {"local": "mbpp", "split": "test", "type": "mbpp"},
    "medical": {"local": "medmcqa", "split": "validation", "type": "medmcqa"},
    "science": {"local": "arc_challenge", "split": "test", "type": "arc"},
    "history": {"local": "mmlu", "split": "test", "type": "mmlu", "subject_filter": ["high_school_us_history", "high_school_world_history", "prehistory"]},
    "philosophy": {"local": "mmlu", "split": "test", "type": "mmlu", "subject_filter": ["philosophy", "moral_scenarios", "moral_disputes"]},
}

CHOICE_LABELS = ["A", "B", "C", "D"]


def load_eval_dataset(domain, dataset_dir, n_samples, split=None):
    """Load evaluation dataset from local cache."""
    from datasets import load_from_disk

    cfg = DOMAIN_EVAL[domain]
    local_path = Path(dataset_dir) / cfg["local"]
    ds = load_from_disk(str(local_path))

    use_split = split or cfg["split"]
    if hasattr(ds, "keys") and use_split in ds:
        ds = ds[use_split]
    elif hasattr(ds, "keys"):
        ds = ds[list(ds.keys())[0]]

    if "subject_filter" in cfg:
        subjects = cfg["subject_filter"]
        ds = ds.filter(lambda x: x.get("subject", "") in subjects)

    ds = ds.shuffle(seed=42)
    return ds.select(range(min(n_samples, len(ds))))


def format_mcq_prompt(example, eval_type):
    """Format a single MCQ example into prompt + gold answer."""
    if eval_type == "arc":
        q = example["question"]
        choices = example["choices"]
        labels = choices["label"]
        texts = choices["text"]
        options = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        gold = example["answerKey"]
        prompt = f"Question: {q}\n{options}\nAnswer:"
        return prompt, gold

    elif eval_type == "medmcqa":
        q = example["question"]
        opts = [example.get(f"op{c}", "") for c in "abcd"]
        options = "\n".join(f"{CHOICE_LABELS[i]}) {o}" for i, o in enumerate(opts))
        gold_idx = int(example.get("cop", 0))
        gold = CHOICE_LABELS[min(gold_idx, 3)]
        prompt = f"Question: {q}\n{options}\nAnswer:"
        return prompt, gold

    elif eval_type == "mmlu":
        q = example["question"]
        choices = example["choices"]
        options = "\n".join(f"{CHOICE_LABELS[i]}) {c}" for i, c in enumerate(choices))
        gold_idx = int(example.get("answer", 0))
        gold = CHOICE_LABELS[min(gold_idx, 3)]
        prompt = f"Question: {q}\n{options}\nAnswer:"
        return prompt, gold

    elif eval_type == "gsm8k":
        q = example["question"]
        answer_str = example.get("answer", "")
        if "####" in answer_str:
            gold = answer_str.split("####")[-1].strip()
        else:
            gold = answer_str.strip()
        prompt = f"Question: {q}\nAnswer: Let's solve step by step."
        return prompt, gold

    elif eval_type == "mbpp":
        q = example.get("text", "")
        gold = example.get("test_list", [""])[0] if "test_list" in example else ""
        prompt = f"Write a Python function: {q}\n```python\n"
        return prompt, gold

    return str(example), ""


def evaluate_model_mcq(model, tokenizer, domain, dataset_dir, n_samples, device="cuda"):
    """Evaluate model on domain MCQ benchmark. Returns accuracy and details."""
    cfg = DOMAIN_EVAL[domain]
    eval_type = cfg["type"]

    if eval_type in ("mbpp",):
        logger.info(f"  Skipping {domain} (code eval not supported in quick mode)")
        return {"accuracy": -1, "n_samples": 0, "skipped": True}

    ds = load_eval_dataset(domain, dataset_dir, n_samples)
    logger.info(f"  Evaluating {domain}: {len(ds)} samples, type={eval_type}")

    correct = 0
    total = 0
    model.eval()

    input_device = next(model.parameters()).device

    for example in ds:
        prompt, gold = format_mcq_prompt(example, eval_type)
        if not gold:
            continue

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=512).to(input_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=32, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        if eval_type == "gsm8k":
            import re
            nums = re.findall(r'-?\d+\.?\d*', response)
            pred = nums[-1] if nums else ""
            gold_clean = gold.replace(",", "").strip()
            is_correct = pred == gold_clean
        else:
            pred = response.strip()[:1].upper()
            is_correct = pred == gold.upper()

        if is_correct:
            correct += 1
        total += 1

    acc = correct / max(total, 1)
    return {"accuracy": float(acc), "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Weight-space baselines
# ---------------------------------------------------------------------------

def load_adapter_weights(adapter_path):
    """Load LoRA adapter weights from safetensors."""
    from safetensors.torch import load_file
    p = Path(adapter_path)
    if (p / "adapter_model.safetensors").exists():
        return load_file(str(p / "adapter_model.safetensors"))
    elif (p / "adapter_model.bin").exists():
        return torch.load(str(p / "adapter_model.bin"), map_location="cpu")
    raise FileNotFoundError(f"No adapter weights in {adapter_path}")


def _materialize_delta_w(weights, module_prefix):
    """Compute delta_W = B @ A for a single LoRA module, return as flat tensor."""
    a_key = f"{module_prefix}.lora_A.weight"
    b_key = f"{module_prefix}.lora_B.weight"
    if a_key in weights and b_key in weights:
        return weights[b_key].float() @ weights[a_key].float()
    return None


def _get_lora_modules(weights):
    """Extract unique LoRA module prefixes from weight keys."""
    modules = set()
    for key in weights:
        if ".lora_A.weight" in key:
            modules.add(key.replace(".lora_A.weight", ""))
    return sorted(modules)


def merge_task_arithmetic(weights_a, weights_b, weight=0.5):
    """Task Arithmetic: average in delta_W = B@A space, then refactor to LoRA."""
    modules = _get_lora_modules(weights_a)
    merged = {}
    for mod in modules:
        dw_a = _materialize_delta_w(weights_a, mod)
        dw_b = _materialize_delta_w(weights_b, mod)
        if dw_a is not None and dw_b is not None:
            dw_merged = weight * dw_a + (1 - weight) * dw_b
            U, S, Vh = torch.linalg.svd(dw_merged, full_matrices=False)
            rank = weights_a[f"{mod}.lora_A.weight"].shape[0]
            r = min(rank, U.shape[1])
            sqrt_S = torch.sqrt(S[:r].clamp(min=0))
            merged[f"{mod}.lora_B.weight"] = (U[:, :r] @ torch.diag(sqrt_S)).to(weights_a[f"{mod}.lora_B.weight"].dtype)
            merged[f"{mod}.lora_A.weight"] = (torch.diag(sqrt_S) @ Vh[:r, :]).to(weights_a[f"{mod}.lora_A.weight"].dtype)
    return merged


def merge_ties(weights_a, weights_b, density=0.5):
    """TIES: Trim, Elect sign, Merge — in delta_W space."""
    modules = _get_lora_modules(weights_a)
    merged = {}
    for mod in modules:
        dw_a = _materialize_delta_w(weights_a, mod)
        dw_b = _materialize_delta_w(weights_b, mod)
        if dw_a is None or dw_b is None:
            continue

        # Trim: zero out small-magnitude elements
        for dw in [dw_a, dw_b]:
            flat = dw.abs().flatten()
            threshold = torch.quantile(flat, 1 - density)
            dw[dw.abs() < threshold] = 0

        # Elect sign: majority vote
        signs = torch.sign(dw_a) + torch.sign(dw_b)
        elected_sign = torch.sign(signs)
        elected_sign[elected_sign == 0] = 1

        # Merge: average magnitudes, apply elected sign
        avg_mag = (dw_a.abs() + dw_b.abs()) / 2
        dw_merged = elected_sign * avg_mag

        # Refactor to LoRA
        rank = weights_a[f"{mod}.lora_A.weight"].shape[0]
        U, S, Vh = torch.linalg.svd(dw_merged, full_matrices=False)
        r = min(rank, U.shape[1])
        sqrt_S = torch.sqrt(S[:r].clamp(min=0))
        merged[f"{mod}.lora_B.weight"] = (U[:, :r] @ torch.diag(sqrt_S)).to(weights_a[f"{mod}.lora_B.weight"].dtype)
        merged[f"{mod}.lora_A.weight"] = (torch.diag(sqrt_S) @ Vh[:r, :]).to(weights_a[f"{mod}.lora_A.weight"].dtype)
    return merged


def merge_dare(weights_a, weights_b, drop_rate=0.5):
    """DARE: Drop And REscale — in delta_W space."""
    modules = _get_lora_modules(weights_a)
    merged = {}
    for mod in modules:
        dw_a = _materialize_delta_w(weights_a, mod)
        dw_b = _materialize_delta_w(weights_b, mod)
        if dw_a is None or dw_b is None:
            continue

        # Random binary masks + rescale
        mask_a = (torch.rand_like(dw_a) > drop_rate).float()
        mask_b = (torch.rand_like(dw_b) > drop_rate).float()
        scale = 1.0 / (1.0 - drop_rate + 1e-8)
        dw_merged = (dw_a * mask_a * scale + dw_b * mask_b * scale) / 2

        # Refactor to LoRA
        rank = weights_a[f"{mod}.lora_A.weight"].shape[0]
        U, S, Vh = torch.linalg.svd(dw_merged, full_matrices=False)
        r = min(rank, U.shape[1])
        sqrt_S = torch.sqrt(S[:r].clamp(min=0))
        merged[f"{mod}.lora_B.weight"] = (U[:, :r] @ torch.diag(sqrt_S)).to(weights_a[f"{mod}.lora_B.weight"].dtype)
        merged[f"{mod}.lora_A.weight"] = (torch.diag(sqrt_S) @ Vh[:r, :]).to(weights_a[f"{mod}.lora_A.weight"].dtype)
    return merged


def save_merged_adapter(merged_weights, template_adapter_path, output_dir):
    """Save merged weights as a PEFT adapter (copy config from template)."""
    from safetensors.torch import save_file
    import shutil

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_file(merged_weights, str(out / "adapter_model.safetensors"))

    template = Path(template_adapter_path)
    for cfg_file in ["adapter_config.json"]:
        src = template / cfg_file
        if src.exists():
            shutil.copy2(str(src), str(out / cfg_file))


# ---------------------------------------------------------------------------
# SFC Composition
# ---------------------------------------------------------------------------

def compute_all_feature_profiles(
    model, tokenizer, saes, adapter_paths, probe_texts,
    batch_size, max_length, threshold_multiplier, device,
):
    """Compute feature profiles for all adapters."""
    from src.sae_decomposition import collect_activations, compute_feature_profile, AdapterFeatureMap
    from peft import PeftModel

    logger.info("Collecting base model activations...")
    base_acts = collect_activations(
        model, tokenizer, probe_texts, list(saes.keys()),
        batch_size=batch_size, max_length=max_length, device=device,
    )

    feature_maps = {}
    for domain, adapter_path in adapter_paths.items():
        logger.info(f"Computing feature profile: {domain}")
        peft_model = PeftModel.from_pretrained(model, adapter_path)

        lora_acts = collect_activations(
            peft_model, tokenizer, probe_texts, list(saes.keys()),
            batch_size=batch_size, max_length=max_length, device=device,
        )
        del peft_model
        torch.cuda.empty_cache()

        profiles = {}
        total_active = 0
        total_features = 0

        for layer_name, sae in saes.items():
            if layer_name not in base_acts or layer_name not in lora_acts:
                continue
            profile = compute_feature_profile(
                base_acts[layer_name], lora_acts[layer_name],
                sae, domain, layer_name,
                threshold_multiplier=threshold_multiplier, device=device,
            )
            profiles[layer_name] = profile
            total_active += profile.support.numel()
            total_features += profile.total_features
            logger.info(f"  {layer_name}: {profile.support.numel()}/{profile.total_features} active")

        global_sparsity = total_active / max(total_features, 1)
        feature_maps[domain] = AdapterFeatureMap(
            adapter_name=domain, profiles=profiles,
            global_sparsity=global_sparsity,
            total_active_features=total_active, total_features=total_features,
        )

    return feature_maps, base_acts


def apply_sfc_hooks(model, saes, composed_profiles):
    """Apply SFC-Exact hooks to model. Returns handles for cleanup."""
    handles = []
    for name, module in model.named_modules():
        for layer_name, profile in composed_profiles.items():
            if name == layer_name or name.endswith(layer_name):
                sae = saes[layer_name]

                def make_hook(sae, composed):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            act = output[0]
                            rest = output[1:]
                        else:
                            act = output
                            rest = None

                        device = act.device
                        dtype = act.dtype
                        original_shape = act.shape
                        flat = act.reshape(-1, act.shape[-1]).float()

                        sae_dev = sae.W_enc.device
                        flat_on_sae = flat.to(sae_dev)
                        f_base = sae.encode(flat_on_sae)

                        f_mod = f_base.clone()
                        support = composed.support.to(sae_dev)
                        coeffs = composed.coefficients.to(sae_dev)
                        f_mod[:, support] += coeffs.unsqueeze(0)

                        delta = sae.decode(f_mod) - sae.decode(f_base)
                        delta = delta.to(device).to(dtype)
                        modified = act + delta.reshape(original_shape)

                        if rest is not None:
                            return (modified,) + rest
                        return modified
                    return hook_fn

                h = module.register_forward_hook(make_hook(sae, profile))
                handles.append(h)
                logger.info(f"  SFC hook attached: {name} → {layer_name}")
                break
    return handles


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    start_time = time.time()
    results = {"config": vars(args), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=True,
    )
    device = "auto"

    # Load SAEs
    from scripts.run_sfc_pilot import load_saes
    logger.info("Loading SAEs...")
    saes = load_saes(args.sae_dir, layers, device="cpu")
    if not saes:
        logger.error("No SAEs found. Train SAEs first.")
        sys.exit(1)
    logger.info(f"Loaded SAEs for layers: {list(saes.keys())}")

    # Find adapters
    adapter_dir = Path(args.adapter_dir)
    adapter_paths = {}
    for d in adapter_dir.iterdir():
        if d.is_dir() and (d / "adapter_config.json").exists():
            adapter_paths[d.name] = str(d)
    logger.info(f"Found adapters: {list(adapter_paths.keys())}")

    # Determine pairs
    if args.pairs:
        pairs = [tuple(p.split("+")) for p in args.pairs.split(",")]
    else:
        all_domains = sorted(adapter_paths.keys())
        pairs = list(combinations(all_domains, 2))

    # Generate probes and compute feature profiles
    from scripts.run_sfc_pilot import generate_probe_texts
    probe_texts = generate_probe_texts(args.probe_size)

    feature_maps, base_acts = compute_all_feature_profiles(
        model, tokenizer, saes, adapter_paths, probe_texts,
        args.batch_size, args.max_length, args.threshold_multiplier, device,
    )

    # Sparsity summary
    sparsity_summary = {}
    for domain, fm in feature_maps.items():
        sparsity_summary[domain] = {
            "global_sparsity": fm.global_sparsity,
            "active": fm.total_active_features,
            "total": fm.total_features,
        }
    results["sparsity"] = sparsity_summary

    # FDS matrix
    from src.sparse_feature_composition import compute_fds
    fds_results = {}
    for d1, d2 in pairs:
        if d1 in feature_maps and d2 in feature_maps:
            fds = compute_fds(feature_maps[d1], feature_maps[d2])
            fds_results[f"{d1}+{d2}"] = fds["global_fds"]
    results["fds"] = fds_results

    # Evaluate base model on all domains (once)
    logger.info("\n=== Base Model Evaluation ===")
    base_scores = {}
    for domain in set(d for pair in pairs for d in pair):
        if domain == "code":
            continue
        score = evaluate_model_mcq(model, tokenizer, domain, args.dataset_dir, args.n_samples, device)
        base_scores[domain] = score
        logger.info(f"  Base | {domain}: {score['accuracy']:.4f}")
    results["base_model"] = base_scores

    # Evaluate single adapters
    logger.info("\n=== Single Adapter Evaluation ===")
    from peft import PeftModel
    single_scores = {}
    for domain in set(d for pair in pairs for d in pair):
        if domain == "code" or domain not in adapter_paths:
            continue
        peft_model = PeftModel.from_pretrained(model, adapter_paths[domain])
        score = evaluate_model_mcq(peft_model, tokenizer, domain, args.dataset_dir, args.n_samples, device)
        single_scores[domain] = score
        logger.info(f"  Single({domain}) | {domain}: {score['accuracy']:.4f}")
        del peft_model
        torch.cuda.empty_cache()
    results["single_adapter"] = single_scores

    # Evaluate each pair
    pair_results = []
    for d1, d2 in pairs:
        if d1 not in adapter_paths or d2 not in adapter_paths:
            continue
        if d1 == "code" or d2 == "code":
            continue

        pair_name = f"{d1}+{d2}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Pair: {pair_name} (FDS={fds_results.get(pair_name, '?')})")
        logger.info(f"{'='*60}")

        pair_result = {"pair": pair_name, "fds": fds_results.get(pair_name, None)}

        # --- SFC-Exact ---
        logger.info("  [SFC-Exact]")
        from src.sparse_feature_composition import sfc_compose
        composed = sfc_compose([feature_maps[d1], feature_maps[d2]])
        handles = apply_sfc_hooks(model, saes, composed)
        sfc_scores = {}
        for domain in [d1, d2]:
            score = evaluate_model_mcq(model, tokenizer, domain, args.dataset_dir, args.n_samples, device)
            sfc_scores[domain] = score
            logger.info(f"    SFC | {domain}: {score['accuracy']:.4f}")
        for h in handles:
            h.remove()
        pair_result["sfc_exact"] = sfc_scores

        # --- Task Arithmetic ---
        logger.info("  [Task Arithmetic]")
        try:
            wa = load_adapter_weights(adapter_paths[d1])
            wb = load_adapter_weights(adapter_paths[d2])
            merged_ta = merge_task_arithmetic(wa, wb)
            with tempfile.TemporaryDirectory() as tmpdir:
                save_merged_adapter(merged_ta, adapter_paths[d1], tmpdir)
                peft_ta = PeftModel.from_pretrained(model, tmpdir)
                ta_scores = {}
                for domain in [d1, d2]:
                    score = evaluate_model_mcq(peft_ta, tokenizer, domain, args.dataset_dir, args.n_samples, device)
                    ta_scores[domain] = score
                    logger.info(f"    TA  | {domain}: {score['accuracy']:.4f}")
                del peft_ta
                torch.cuda.empty_cache()
            pair_result["task_arithmetic"] = ta_scores
        except Exception as e:
            logger.error(f"    TA failed: {e}")
            pair_result["task_arithmetic"] = {"error": str(e)}

        # --- TIES ---
        logger.info("  [TIES]")
        try:
            merged_ties = merge_ties(wa, wb, density=0.5)
            with tempfile.TemporaryDirectory() as tmpdir:
                save_merged_adapter(merged_ties, adapter_paths[d1], tmpdir)
                peft_ties = PeftModel.from_pretrained(model, tmpdir)
                ties_scores = {}
                for domain in [d1, d2]:
                    score = evaluate_model_mcq(peft_ties, tokenizer, domain, args.dataset_dir, args.n_samples, device)
                    ties_scores[domain] = score
                    logger.info(f"    TIES| {domain}: {score['accuracy']:.4f}")
                del peft_ties
                torch.cuda.empty_cache()
            pair_result["ties"] = ties_scores
        except Exception as e:
            logger.error(f"    TIES failed: {e}")
            pair_result["ties"] = {"error": str(e)}

        # --- DARE ---
        logger.info("  [DARE]")
        try:
            torch.manual_seed(42)
            merged_dare = merge_dare(wa, wb, drop_rate=0.5)
            with tempfile.TemporaryDirectory() as tmpdir:
                save_merged_adapter(merged_dare, adapter_paths[d1], tmpdir)
                peft_dare = PeftModel.from_pretrained(model, tmpdir)
                dare_scores = {}
                for domain in [d1, d2]:
                    score = evaluate_model_mcq(peft_dare, tokenizer, domain, args.dataset_dir, args.n_samples, device)
                    dare_scores[domain] = score
                    logger.info(f"    DARE| {domain}: {score['accuracy']:.4f}")
                del peft_dare
                torch.cuda.empty_cache()
            pair_result["dare"] = dare_scores
        except Exception as e:
            logger.error(f"    DARE failed: {e}")
            pair_result["dare"] = {"error": str(e)}

        pair_results.append(pair_result)

        # Log comparison table
        logger.info(f"\n  === Comparison: {pair_name} ===")
        logger.info(f"  {'Method':<20} {'Domain A (' + d1 + ')':<20} {'Domain B (' + d2 + ')':<20} {'Mean':<10}")
        logger.info(f"  {'-'*70}")
        for method_name, method_scores in [
            ("Base", {d1: base_scores.get(d1, {}), d2: base_scores.get(d2, {})}),
            ("SFC-Exact", sfc_scores),
            ("Task Arithmetic", pair_result.get("task_arithmetic", {})),
            ("TIES", pair_result.get("ties", {})),
            ("DARE", pair_result.get("dare", {})),
        ]:
            if isinstance(method_scores, dict) and "error" not in method_scores:
                a_acc = method_scores.get(d1, {}).get("accuracy", -1)
                b_acc = method_scores.get(d2, {}).get("accuracy", -1)
                valid = [x for x in [a_acc, b_acc] if x >= 0]
                mean = np.mean(valid) if valid else -1
                logger.info(f"  {method_name:<20} {a_acc:<20.4f} {b_acc:<20.4f} {mean:<10.4f}")

    results["pairs"] = pair_results
    results["elapsed_seconds"] = time.time() - start_time

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.bool_)) else str(o))
    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
