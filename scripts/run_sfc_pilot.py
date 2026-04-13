#!/usr/bin/env python3
"""
SFC Pilot Experiment (E0): Verify the Sparse Feature Decomposition hypothesis.

Kill criterion: if adapter effects are NOT sparse in SAE feature space
(>20% features modified), the entire SFC framework is invalid.

This script:
1. Loads Gemma-2-9B base model
2. Trains (or loads) 6 domain-specific LoRA adapters
3. Loads Gemma Scope SAEs for selected layers
4. Decomposes each adapter's effects through the SAEs
5. Reports sparsity stats and pairwise FDS

Usage:
    python scripts/run_sfc_pilot.py --model Qwen/Qwen3.5-9B \
        --sae-repo saes/qwen3.5-9b \
        --layers 8,12,16,24,32 \
        --output results/sfc_pilot.json

Requirements:
    - GPU with >= 24GB VRAM (for 9B model in bfloat16)
    - Pre-trained LoRA adapters OR training data access
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sfc_pilot")


def parse_args():
    p = argparse.ArgumentParser(description="SFC Pilot: Sparsity Verification")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B",
                    help="Base model name or path")
    p.add_argument("--sae-repo", default="saes/qwen3.5-9b",
                    help="HuggingFace repo for SAEs")
    p.add_argument("--sae-width", default="16k",
                    help="SAE width (16k, 32k, 65k, 131k)")
    p.add_argument("--layers", default="8,12,16,24,32",
                    help="Comma-separated layer indices for SAEs")
    p.add_argument("--adapter-dir", default="results/domain_loras",
                    help="Directory containing LoRA adapter subdirs")
    p.add_argument("--probe-size", type=int, default=256,
                    help="Number of probe samples for feature decomposition")
    p.add_argument("--threshold-multiplier", type=float, default=3.0,
                    help="Features with mean|Δf| > mean + k*std are active (default: 3σ)")
    p.add_argument("--output", default="results/sfc_pilot.json",
                    help="Output JSON path")
    p.add_argument("--device", default="auto",
                    help="Device (auto, cuda, cpu)")
    p.add_argument("--batch-size", type=int, default=2,
                    help="Batch size for activation collection")
    p.add_argument("--max-length", type=int, default=256,
                    help="Max sequence length for probing")
    # Training args (if adapters don't exist yet)
    p.add_argument("--train-adapters", action="store_true",
                    help="Train LoRA adapters if they don't exist")
    p.add_argument("--train-samples", type=int, default=5000,
                    help="Training samples per domain")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--train-epochs", type=int, default=2)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Probe data generation
# ---------------------------------------------------------------------------

PROBE_TEMPLATES = [
    # Math
    "Solve the following math problem step by step: What is {a} times {b} plus {c}?",
    # Code
    "Write a Python function that {task}.",
    # Medical
    "A patient presents with {symptom}. What is the most likely diagnosis?",
    # Science
    "Explain the scientific principle behind {topic}.",
    # History
    "Describe the historical significance of {event}.",
    # Philosophy
    "What is {philosopher}'s argument about {concept}?",
    # General
    "Summarize the key points of {topic} in three sentences.",
    "What are the advantages and disadvantages of {topic}?",
]


def generate_probe_texts(n: int) -> list:
    """Generate diverse probe texts for feature decomposition."""
    from datasets import load_dataset

    texts = []

    # Try to load diverse text from C4
    try:
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        for i, example in enumerate(ds):
            if i >= n:
                break
            text = example["text"][:512]  # truncate long texts
            if len(text) > 50:  # skip very short
                texts.append(text)
    except Exception as e:
        logger.warning(f"Could not load C4: {e}. Using synthetic probes.")

    # Fill remaining with synthetic probes
    while len(texts) < n:
        import random
        template = random.choice(PROBE_TEMPLATES)
        # Simple placeholder filling
        text = template.format(
            a=random.randint(10, 999),
            b=random.randint(2, 99),
            c=random.randint(1, 999),
            task=random.choice(["sorts a list", "finds primes", "computes factorial"]),
            symptom=random.choice(["chest pain", "headache", "fever"]),
            topic=random.choice(["gravity", "evolution", "quantum mechanics"]),
            event=random.choice(["the Renaissance", "World War II", "the Moon landing"]),
            philosopher=random.choice(["Kant", "Aristotle", "Nietzsche"]),
            concept=random.choice(["free will", "justice", "knowledge"]),
        )
        texts.append(text)

    return texts[:n]


# ---------------------------------------------------------------------------
# LoRA training (if needed)
# ---------------------------------------------------------------------------

DOMAIN_DATASETS = {
    "math": ("gsm8k", "main", "question", "answer"),
    "science": ("allenai/ai2_arc", "ARC-Challenge", "question", "answerKey"),
    "medical": ("bigbio/med_qa", "med_qa_en_4options_source", "question", "answer_idx"),
    "history": ("cais/mmlu", "all", "question", "answer"),  # filter for history
    "philosophy": ("cais/mmlu", "all", "question", "answer"),  # filter for philosophy
    "code": ("mbpp", "full", "text", "code"),
}


def train_single_adapter(
    model_name: str,
    domain: str,
    output_dir: str,
    n_samples: int = 5000,
    rank: int = 16,
    alpha: int = 32,
    epochs: int = 2,
    device: str = "cuda",
):
    """Train a single domain-specific LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset

    logger.info(f"Training {domain} adapter: {n_samples} samples, rank={rank}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if n_gpus > 1 else device,
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Load domain data
    ds_name, ds_config, q_field, a_field = DOMAIN_DATASETS.get(
        domain, ("gsm8k", "main", "question", "answer")
    )

    try:
        ds = load_dataset(ds_name, ds_config, split="train")
    except Exception:
        ds = load_dataset(ds_name, split="train")

    # Format for SFT
    def format_example(example):
        q = str(example.get(q_field, ""))
        a = str(example.get(a_field, ""))
        return {"text": f"Question: {q}\nAnswer: {a}"}

    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    ds = ds.map(format_example, remove_columns=ds.column_names)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=512,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model, trainer
    torch.cuda.empty_cache()
    logger.info(f"Adapter saved to {output_dir}")


# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------

def load_saes(
    sae_path: str,
    layers: list,
    width: str = "16k",
    device: str = "cpu",
) -> dict:
    """Load SAEs for specified layers. Tries local path first, then HuggingFace."""
    from src.sae_decomposition import SparseAutoencoder

    saes = {}
    for layer_idx in layers:
        layer_name = f"model.layers.{layer_idx}"
        logger.info(f"Loading SAE for layer {layer_idx}...")

        # Try local directory first (from train_sae.py output)
        local_dir = Path(sae_path) / f"layer_{layer_idx}"
        if local_dir.exists() and any(local_dir.glob("*.safetensors")):
            sae = SparseAutoencoder.from_pretrained(str(local_dir), device=device)
            saes[layer_name] = sae
            logger.info(f"  → Local: {sae.n_features} features, d_model={sae.d_model}")
            continue

        # Fallback: try as HuggingFace repo (e.g., google/gemma-scope-9b-pt-res)
        try:
            sae = SparseAutoencoder.from_huggingface(
                sae_path, layer=layer_idx, width=width, device=device,
            )
            saes[layer_name] = sae
            logger.info(f"  → HuggingFace: {sae.n_features} features, d_model={sae.d_model}")
        except Exception as e:
            logger.error(f"  → FAILED: {e}")
            logger.error(f"  → Run: python scripts/train_sae.py --model <MODEL> --layers {layer_idx} --output {sae_path}")

    return saes


# ---------------------------------------------------------------------------
# Main pilot experiment
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Device setup — auto-detect multi-GPU
    if args.device == "auto":
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
            logger.info(f"Found {n_gpus} GPU(s): {gpu_names}")
            # Use "auto" device_map for multi-GPU model sharding
            device = "auto" if n_gpus > 1 else "cuda"
        else:
            device = "cpu"
            n_gpus = 0
    else:
        device = args.device
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    layers = [int(x) for x in args.layers.split(",")]

    start_time = time.time()
    results = {
        "config": vars(args),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": device,
    }

    # Step 1: Check/train LoRA adapters
    adapter_dir = Path(args.adapter_dir)
    domains = ["math", "code", "medical", "science", "history", "philosophy"]
    adapter_paths = {}

    for domain in domains:
        domain_dir = adapter_dir / domain
        if (domain_dir / "adapter_config.json").exists():
            logger.info(f"Found existing adapter: {domain}")
            adapter_paths[domain] = str(domain_dir)
        elif args.train_adapters:
            logger.info(f"Training adapter: {domain}")
            train_single_adapter(
                model_name=args.model,
                domain=domain,
                output_dir=str(domain_dir),
                n_samples=args.train_samples,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                epochs=args.train_epochs,
                device=device,
            )
            adapter_paths[domain] = str(domain_dir)
        else:
            logger.warning(f"No adapter for {domain}. Use --train-adapters to create.")

    if len(adapter_paths) < 2:
        logger.error("Need at least 2 adapters. Run with --train-adapters.")
        sys.exit(1)

    results["adapters_found"] = list(adapter_paths.keys())

    # Step 2: Load SAEs
    logger.info("Loading SAEs...")
    saes = load_saes(args.sae_repo, layers, args.sae_width, device="cpu")
    results["sae_layers"] = list(saes.keys())
    results["sae_features_per_layer"] = {
        k: v.n_features for k, v in saes.items()
    }

    if not saes:
        logger.error("No SAEs loaded. Check repo ID and layer indices.")
        sys.exit(1)

    # Step 3: Load base model
    logger.info(f"Loading base model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto" if n_gpus > 1 else device,
        attn_implementation="sdpa",
    )

    # Step 4: Generate probe data
    logger.info(f"Generating {args.probe_size} probe samples...")
    probe_texts = generate_probe_texts(args.probe_size)
    results["probe_size"] = len(probe_texts)

    # Step 5: Collect base activations (once)
    from src.sae_decomposition import collect_activations, compute_feature_profile

    logger.info("Collecting base model activations...")
    base_acts = collect_activations(
        model, tokenizer, probe_texts, list(saes.keys()),
        batch_size=args.batch_size, max_length=args.max_length, device=device,
    )
    results["base_activation_shapes"] = {k: list(v.shape) for k, v in base_acts.items()}

    # Step 6: For each adapter, collect activations and compute feature profiles
    from peft import PeftModel
    from src.sae_decomposition import FeatureProfile, AdapterFeatureMap

    feature_maps = {}
    sparsity_results = {}

    for domain, adapter_path in adapter_paths.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Decomposing adapter: {domain}")
        logger.info(f"{'='*60}")

        # Load adapter (skip .to() when using device_map="auto" on multi-GPU)
        peft_model = PeftModel.from_pretrained(model, adapter_path)
        if n_gpus <= 1:
            peft_model.to(device)

        # Collect adapter activations
        lora_acts = collect_activations(
            peft_model, tokenizer, probe_texts, list(saes.keys()),
            batch_size=args.batch_size, max_length=args.max_length, device=device,
        )

        # Unload adapter
        del peft_model
        torch.cuda.empty_cache()

        # Compute feature profiles per layer
        profiles = {}
        total_active = 0
        total_features = 0

        for layer_name, sae in saes.items():
            if layer_name not in base_acts or layer_name not in lora_acts:
                continue

            profile = compute_feature_profile(
                base_activations=base_acts[layer_name],
                lora_activations=lora_acts[layer_name],
                sae=sae,
                adapter_name=domain,
                layer_name=layer_name,
                threshold_multiplier=args.threshold_multiplier,
                device=device,
            )
            profiles[layer_name] = profile
            total_active += profile.support.numel()
            total_features += profile.total_features

            logger.info(
                f"  {layer_name}: {profile.support.numel()}/{profile.total_features} "
                f"active features ({profile.sparsity*100:.2f}%)"
            )

        global_sparsity = total_active / max(total_features, 1)
        feature_maps[domain] = AdapterFeatureMap(
            adapter_name=domain,
            profiles=profiles,
            global_sparsity=global_sparsity,
            total_active_features=total_active,
            total_features=total_features,
        )

        sparsity_results[domain] = {
            "global_sparsity": global_sparsity,
            "total_active": total_active,
            "total_features": total_features,
            "per_layer": {
                ln: {
                    "active": p.support.numel(),
                    "total": p.total_features,
                    "sparsity": p.sparsity,
                }
                for ln, p in profiles.items()
            },
        }

        logger.info(
            f"  GLOBAL: {total_active}/{total_features} = "
            f"{global_sparsity*100:.2f}% features active"
        )

    results["sparsity"] = sparsity_results

    # Step 7: Compute pairwise FDS
    from src.sparse_feature_composition import compute_fds, compute_fds_matrix

    domain_list = list(feature_maps.keys())
    maps_list = [feature_maps[d] for d in domain_list]

    fds_matrix, names = compute_fds_matrix(maps_list)
    results["fds_matrix"] = {
        "names": names,
        "matrix": fds_matrix.tolist(),
    }

    # Pairwise FDS details
    fds_pairs = {}
    for i in range(len(domain_list)):
        for j in range(i + 1, len(domain_list)):
            pair_name = f"{domain_list[i]}+{domain_list[j]}"
            fds_result = compute_fds(maps_list[i], maps_list[j])
            fds_pairs[pair_name] = {
                "global_fds": fds_result["global_fds"],
                "total_overlap": fds_result["total_overlap"],
                "total_union": fds_result["total_union"],
            }

    results["fds_pairs"] = fds_pairs

    # Step 8: Kill criterion check
    mean_sparsity = np.mean([s["global_sparsity"] for s in sparsity_results.values()])
    max_sparsity = np.max([s["global_sparsity"] for s in sparsity_results.values()])
    mean_fds = np.mean([p["global_fds"] for p in fds_pairs.values()])

    results["summary"] = {
        "mean_sparsity": float(mean_sparsity),
        "max_sparsity": float(max_sparsity),
        "mean_pairwise_fds": float(mean_fds),
        "kill_criterion_sparsity": max_sparsity < 0.20,
        "kill_criterion_verdict": "PASS" if max_sparsity < 0.20 else "FAIL",
        "elapsed_seconds": time.time() - start_time,
    }

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("PILOT EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Mean adapter sparsity: {mean_sparsity*100:.2f}%")
    logger.info(f"Max adapter sparsity:  {max_sparsity*100:.2f}%")
    logger.info(f"Mean pairwise FDS:     {mean_fds:.4f}")
    logger.info(f"Kill criterion (sparsity < 20%): "
                f"{'PASS ✓' if max_sparsity < 0.20 else 'FAIL ✗'}")
    logger.info(f"Elapsed: {time.time() - start_time:.1f}s")

    if max_sparsity < 0.05:
        logger.info("🎯 EXCELLENT: Adapter effects are very sparse (<5%). SFC framework is well-supported.")
    elif max_sparsity < 0.10:
        logger.info("✓ GOOD: Adapter effects are sparse (<10%). SFC framework should work well.")
    elif max_sparsity < 0.20:
        logger.info("⚠ MARGINAL: Adapter effects are moderately sparse (<20%). SFC may work but needs careful tuning.")
    else:
        logger.info("✗ FAIL: Adapter effects are NOT sparse (>20%). SFC framework may not be viable.")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
