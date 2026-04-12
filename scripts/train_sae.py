#!/usr/bin/env python3
"""
Train Sparse Autoencoders (SAEs) for models without pre-trained SAEs.

Uses sae-lens to train JumpReLU SAEs on specified layers of any HuggingFace model.
Supports multi-GPU: one SAE per GPU in parallel.

Usage:
    # Train SAEs for 5 layers, 8 GPUs available → 5 parallel jobs
    python scripts/train_sae.py \
        --model Qwen/Qwen3.5-9B \
        --layers 8,12,16,24,32 \
        --output saes/qwen3.5-9b \
        --n-features 16384 \
        --training-tokens 100_000_000

    # Single layer on specific GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/train_sae.py \
        --model Qwen/Qwen3.5-9B \
        --layers 16 \
        --output saes/qwen3.5-9b
"""

import argparse
import json
import logging
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_sae")


def parse_args():
    p = argparse.ArgumentParser(description="Train SAEs for SFC experiments")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B",
                    help="Base model to train SAEs on")
    p.add_argument("--layers", default="8,12,16,24,32",
                    help="Comma-separated layer indices")
    p.add_argument("--output", default="saes/qwen3.5-9b",
                    help="Output directory for trained SAEs")
    p.add_argument("--n-features", type=int, default=16384,
                    help="SAE dictionary size (default: 16k)")
    p.add_argument("--training-tokens", type=int, default=100_000_000,
                    help="Number of training tokens (default: 100M)")
    p.add_argument("--batch-size", type=int, default=4096,
                    help="SAE training batch size (tokens)")
    p.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate")
    p.add_argument("--l1-coefficient", type=float, default=5.0,
                    help="L1 sparsity coefficient")
    p.add_argument("--dataset", default="allenai/c4",
                    help="Training dataset")
    p.add_argument("--dataset-config", default="en",
                    help="Dataset config name")
    p.add_argument("--context-size", type=int, default=256,
                    help="Context window for SAE training")
    p.add_argument("--parallel", action="store_true", default=True,
                    help="Train layers in parallel across GPUs")
    p.add_argument("--no-parallel", action="store_true",
                    help="Train layers sequentially on GPU 0")
    return p.parse_args()


def train_sae_single_layer(
    model_name: str,
    layer: int,
    output_dir: str,
    n_features: int = 16384,
    training_tokens: int = 100_000_000,
    batch_size: int = 4096,
    lr: float = 3e-4,
    l1_coefficient: float = 5.0,
    dataset: str = "allenai/c4",
    dataset_config: str = "en",
    context_size: int = 256,
    gpu_id: int = 0,
):
    """Train a single SAE for one layer on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch

    layer_dir = Path(output_dir) / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    # Check if already trained
    if (layer_dir / "sae_weights.safetensors").exists():
        logger.info(f"[GPU {gpu_id}] Layer {layer}: already trained, skipping")
        return

    logger.info(f"[GPU {gpu_id}] Training SAE for layer {layer}: "
                f"{n_features} features, {training_tokens/1e6:.0f}M tokens")

    train_sae_manual(
        model_name, layer, str(layer_dir), n_features,
        training_tokens, batch_size, lr, l1_coefficient,
        dataset, dataset_config, context_size,
    )


def train_sae_manual(
    model_name: str,
    layer: int,
    output_dir: str,
    n_features: int = 16384,
    training_tokens: int = 100_000_000,
    batch_size: int = 4096,
    lr: float = 3e-4,
    l1_coefficient: float = 5.0,
    dataset: str = "allenai/c4",
    dataset_config: str = "en",
    context_size: int = 256,
):
    """Fallback: train SAE without sae-lens using plain PyTorch."""
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from safetensors.torch import save_file

    logger.info(f"Manual SAE training for layer {layer}")

    # Set offline mode via env var (more reliable than local_files_only kwarg)
    if not os.path.isdir(model_name):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", trust_remote_code=True,
    )
    model.eval()

    # Get d_model from the model
    d_model = model.config.hidden_size

    # Simple TopK SAE
    W_enc = nn.Parameter(torch.randn(n_features, d_model, device="cuda") * 0.01)
    W_dec = nn.Parameter(torch.randn(n_features, d_model, device="cuda") * 0.01)
    b_enc = nn.Parameter(torch.zeros(n_features, device="cuda"))
    b_dec = nn.Parameter(torch.zeros(d_model, device="cuda"))

    # Normalize decoder columns
    with torch.no_grad():
        W_dec.data = W_dec.data / W_dec.data.norm(dim=1, keepdim=True)

    optimizer = Adam([W_enc, W_dec, b_enc, b_dec], lr=lr)

    # Activation hook
    activations = []
    hook_name = f"model.layers.{layer}"

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations.append(output[0].detach().float())
        else:
            activations.append(output.detach().float())

    for name, module in model.named_modules():
        if name == hook_name:
            module.register_forward_hook(hook_fn)
            break

    # Stream training data — try real dataset, fall back to random tokens
    try:
        ds = load_dataset(dataset, dataset_config, split="train", streaming=True)
        use_random = False
        logger.info(f"Using dataset: {dataset}/{dataset_config}")
    except Exception as e:
        logger.warning(f"Cannot load {dataset}: {e}. Using random token inputs.")
        use_random = True
        ds = None

    tokens_seen = 0
    step = 0
    k = max(1, n_features // 100)  # TopK sparsity

    logger.info(f"Training: d_model={d_model}, n_features={n_features}, k={k}")

    def data_iter():
        """Yield tokenized inputs from dataset or random tokens."""
        if use_random:
            vocab_size = tokenizer.vocab_size
            while True:
                random_ids = torch.randint(100, vocab_size - 100, (1, context_size)).to("cuda")
                yield {"input_ids": random_ids, "attention_mask": torch.ones_like(random_ids)}
        else:
            for example in ds:
                text = example.get("text", "")
                if len(text) < 50:
                    continue
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=context_size, padding=False).to("cuda")
                yield inputs

    for inputs in data_iter():
        if tokens_seen >= training_tokens:
            break

        activations.clear()
        with torch.no_grad():
            model(**{k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")})

        if not activations:
            continue

        x = activations[0].reshape(-1, d_model)  # (seq_len, d_model)
        n_tokens = x.shape[0]
        tokens_seen += n_tokens

        # SAE forward: encode → topk → decode
        z = (x - b_dec) @ W_enc.T + b_enc  # (seq, n_features)
        # TopK activation
        topk_vals, topk_idx = z.topk(k, dim=-1)
        f = torch.zeros_like(z)
        f.scatter_(1, topk_idx, torch.relu(topk_vals))
        x_hat = f @ W_dec + b_dec

        # Loss: reconstruction + L1
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = f.abs().sum(dim=-1).mean()
        loss = recon_loss + l1_coefficient * l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Normalize decoder
        with torch.no_grad():
            W_dec.data = W_dec.data / W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

        step += 1
        if step % 500 == 0:
            logger.info(f"  step={step}, tokens={tokens_seen/1e6:.1f}M, "
                        f"recon={recon_loss.item():.4f}, l1={l1_loss.item():.4f}")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    state = {
        "W_enc": W_enc.detach().cpu().float(),
        "W_dec": W_dec.detach().cpu().float(),
        "b_enc": b_enc.detach().cpu().float(),
        "b_dec": b_dec.detach().cpu().float(),
    }
    save_file(state, str(output_path / "sae_weights.safetensors"))

    config = {
        "model_name": model_name,
        "layer": layer,
        "n_features": n_features,
        "d_model": d_model,
        "training_tokens": tokens_seen,
        "l1_coefficient": l1_coefficient,
        "topk": k,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"SAE saved to {output_path} ({tokens_seen/1e6:.1f}M tokens)")

    del model
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Model: {args.model}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Features: {args.n_features}")
    logger.info(f"Training tokens: {args.training_tokens/1e6:.0f}M")
    logger.info(f"GPUs available: {n_gpus}")

    if args.no_parallel or n_gpus <= 1:
        # Sequential training
        for layer in layers:
            train_sae_single_layer(
                model_name=args.model, layer=layer,
                output_dir=str(output_dir),
                n_features=args.n_features,
                training_tokens=args.training_tokens,
                batch_size=args.batch_size, lr=args.lr,
                l1_coefficient=args.l1_coefficient,
                dataset=args.dataset, dataset_config=args.dataset_config,
                context_size=args.context_size, gpu_id=0,
            )
    else:
        # Parallel: one layer per GPU
        logger.info(f"Training {len(layers)} SAEs in parallel across {n_gpus} GPUs")
        processes = []
        for i, layer in enumerate(layers):
            gpu_id = i % n_gpus
            p = Process(
                target=train_sae_single_layer,
                kwargs=dict(
                    model_name=args.model, layer=layer,
                    output_dir=str(output_dir),
                    n_features=args.n_features,
                    training_tokens=args.training_tokens,
                    batch_size=args.batch_size, lr=args.lr,
                    l1_coefficient=args.l1_coefficient,
                    dataset=args.dataset, dataset_config=args.dataset_config,
                    context_size=args.context_size, gpu_id=gpu_id,
                ),
            )
            p.start()
            processes.append((layer, gpu_id, p))
            logger.info(f"  Started layer {layer} on GPU {gpu_id} (PID {p.pid})")

            # If all GPUs occupied, wait for one to finish
            if len([p for _, _, p in processes if p.is_alive()]) >= n_gpus:
                for _, _, proc in processes:
                    if proc.is_alive():
                        proc.join()
                        break

        # Wait for all remaining
        for layer, gpu_id, p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"Layer {layer} (GPU {gpu_id}) failed with exit code {p.exitcode}")
            else:
                logger.info(f"Layer {layer} (GPU {gpu_id}) completed successfully")

    # Verify outputs
    logger.info("\n=== Training Summary ===")
    for layer in layers:
        layer_dir = output_dir / f"layer_{layer}"
        if (layer_dir / "sae_weights.safetensors").exists():
            logger.info(f"  Layer {layer}: ✓ OK")
        else:
            logger.info(f"  Layer {layer}: ✗ MISSING")


if __name__ == "__main__":
    main()
