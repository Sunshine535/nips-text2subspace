"""
SAE-based decomposition of LoRA adapter effects.

Loads pre-trained Sparse Autoencoders (e.g., Gemma Scope, Llama Scope),
decomposes adapter activation differences into sparse feature representations,
and computes feature supports for downstream composition.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAE Architecture
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """JumpReLU Sparse Autoencoder (Gemma Scope format).

    Architecture: f = JumpReLU(W_enc @ (x - b_dec) + b_enc)
                  x_hat = W_dec @ f + b_dec

    JumpReLU: f_k = max(0, z_k) * 1[z_k > θ_k]  where z = W_enc @ x + b_enc
    Falls back to standard ReLU if no thresholds provided.
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        threshold: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.register_buffer("W_enc", W_enc.float())   # (n_features, d_model)
        self.register_buffer("W_dec", W_dec.float())   # (n_features, d_model)
        self.register_buffer("b_enc", b_enc.float())   # (n_features,)
        self.register_buffer("b_dec", b_dec.float())   # (d_model,)
        if threshold is not None:
            self.register_buffer("threshold", threshold.float())
        else:
            self.register_buffer("threshold", torch.zeros(n_features))
        self.use_jumprelu = threshold is not None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse feature coefficients.

        Args:
            x: (..., d_model) input activations
        Returns:
            f: (..., n_features) sparse feature coefficients
        """
        z = (x - self.b_dec) @ self.W_enc.T + self.b_enc  # (..., n_features)
        if self.use_jumprelu:
            f = torch.relu(z) * (z > self.threshold).float()
        else:
            f = torch.relu(z)
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space.

        Args:
            f: (..., n_features) sparse feature coefficients
        Returns:
            x_hat: (..., d_model) reconstructed activations
        """
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode.

        Returns:
            x_hat: reconstructed activations
            f: sparse feature coefficients
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "SparseAutoencoder":
        """Load a pre-trained SAE from a directory or HuggingFace repo.

        Supports Gemma Scope and Llama Scope formats.
        """
        path = Path(path)

        # Load weights
        if (path / "sae_weights.safetensors").exists():
            state = load_safetensors(str(path / "sae_weights.safetensors"), device=device)
        elif (path / "model.safetensors").exists():
            state = load_safetensors(str(path / "model.safetensors"), device=device)
        else:
            # Try loading all safetensors files
            st_files = list(path.glob("*.safetensors"))
            if st_files:
                state = load_safetensors(str(st_files[0]), device=device)
            else:
                raise FileNotFoundError(f"No safetensors files in {path}")

        # Normalize key names across formats
        key_map = {
            "W_enc": ["W_enc", "encoder.weight", "w_enc"],
            "W_dec": ["W_dec", "decoder.weight", "w_dec"],
            "b_enc": ["b_enc", "encoder.bias", "b_enc"],
            "b_dec": ["b_dec", "decoder.bias", "b_dec"],
            "threshold": ["threshold", "log_thresholds", "jumprelu_threshold"],
        }

        resolved = {}
        for canonical, aliases in key_map.items():
            for alias in aliases:
                if alias in state:
                    val = state[alias]
                    # Handle log_thresholds (Gemma Scope stores exp(threshold))
                    if alias == "log_thresholds":
                        val = torch.exp(val)
                    resolved[canonical] = val
                    break

        W_enc = resolved["W_enc"]
        W_dec = resolved["W_dec"]
        b_enc = resolved["b_enc"]
        b_dec = resolved["b_dec"]
        threshold = resolved.get("threshold")

        # Infer dimensions
        if W_enc.shape[0] > W_enc.shape[1]:
            # (n_features, d_model) — standard
            n_features, d_model = W_enc.shape
        else:
            # (d_model, n_features) — transposed
            d_model, n_features = W_enc.shape
            W_enc = W_enc.T

        # Ensure W_dec is (n_features, d_model)
        if W_dec.shape != (n_features, d_model):
            W_dec = W_dec.T

        logger.info(f"Loaded SAE: d_model={d_model}, n_features={n_features}, "
                     f"jumprelu={'yes' if threshold is not None else 'no'}")

        return cls(
            d_model=d_model,
            n_features=n_features,
            W_enc=W_enc,
            W_dec=W_dec,
            b_enc=b_enc,
            b_dec=b_dec,
            threshold=threshold,
        )

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        layer: int,
        width: str = "16k",
        sublayer: str = "res",
        device: str = "cpu",
    ) -> "SparseAutoencoder":
        """Load SAE from HuggingFace Hub (Gemma Scope format).

        Args:
            repo_id: e.g. "google/gemma-scope-9b-pt-res"
            layer: layer number
            width: SAE width, e.g. "16k", "32k", "65k", "131k"
            sublayer: "res" (residual stream), "mlp", "att"
            device: target device
        """
        from huggingface_hub import hf_hub_download, snapshot_download
        import os

        # Try to construct the subfolder path for Gemma Scope
        subfolder = f"layer_{layer}/width_{width}/average_l0_71"

        try:
            local_dir = snapshot_download(
                repo_id,
                allow_patterns=[f"{subfolder}/*"],
                local_dir_use_symlinks=False,
            )
            sae_path = os.path.join(local_dir, subfolder)
        except Exception:
            # Fallback: try direct download
            local_dir = snapshot_download(repo_id, local_dir_use_symlinks=False)
            sae_path = local_dir

        return cls.from_pretrained(sae_path, device=device)


# ---------------------------------------------------------------------------
# Activation collection with hooks
# ---------------------------------------------------------------------------

@dataclass
class ActivationCache:
    """Stores intermediate activations from model forward passes."""
    activations: Dict[str, List[torch.Tensor]] = field(default_factory=dict)

    def hook_fn(self, layer_name: str):
        """Returns a hook function that caches activations for the given layer."""
        def hook(module, input, output):
            # Handle tuple outputs (e.g., attention layers)
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            if layer_name not in self.activations:
                self.activations[layer_name] = []
            # Store on CPU to save GPU memory
            self.activations[layer_name].append(act.detach().cpu())
        return hook

    def clear(self):
        self.activations.clear()

    def get_stacked(self, layer_name: str) -> torch.Tensor:
        """Get all cached activations for a layer, stacked along batch dim."""
        return torch.cat(self.activations[layer_name], dim=0)


def collect_activations(
    model: nn.Module,
    tokenizer,
    probe_texts: List[str],
    layer_names: List[str],
    batch_size: int = 4,
    max_length: int = 256,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Collect residual-stream activations for specified layers on probe data.

    Args:
        model: the LLM (with or without LoRA applied)
        tokenizer: the tokenizer
        probe_texts: list of text samples for probing
        layer_names: layer names to hook (e.g., "model.layers.15")
        batch_size: inference batch size
        max_length: max sequence length
        device: inference device

    Returns:
        Dict mapping layer_name → (total_tokens, d_model) tensor of activations
    """
    cache = ActivationCache()
    handles = []

    # Register hooks on target layers
    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(cache.hook_fn(name)))

    if not handles:
        # Try matching by partial name
        for name, module in model.named_modules():
            for target in layer_names:
                if target in name and name.endswith(target.split(".")[-1]):
                    handles.append(module.register_forward_hook(cache.hook_fn(name)))

    # For device_map="auto" models, find the input device from the model itself
    if device == "auto" or (hasattr(model, "hf_device_map") and model.hf_device_map):
        # Multi-GPU: send inputs to the device of the first parameter
        input_device = next(model.parameters()).device
    else:
        input_device = device

    model.eval()
    with torch.no_grad():
        for i in range(0, len(probe_texts), batch_size):
            batch_texts = probe_texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(input_device)
            model(**inputs)

    # Remove hooks
    for h in handles:
        h.remove()

    # Stack and flatten to (total_tokens, d_model)
    result = {}
    for layer_name in cache.activations:
        stacked = cache.get_stacked(layer_name)  # (batch, seq, d_model)
        result[layer_name] = stacked.reshape(-1, stacked.shape[-1])

    return result


# ---------------------------------------------------------------------------
# Feature profile: decomposing LoRA effects through SAE
# ---------------------------------------------------------------------------

@dataclass
class FeatureProfile:
    """Sparse feature representation of a LoRA adapter's effect at one layer."""
    adapter_name: str
    layer_name: str
    support: torch.Tensor           # (|S|,) indices of active features
    coefficients: torch.Tensor      # (|S|,) mean absolute feature coefficients
    full_coefficients: torch.Tensor # (n_features,) full mean coefficient vector
    sparsity: float                 # |S| / n_features
    total_features: int             # n_features (SAE dictionary size)


@dataclass
class AdapterFeatureMap:
    """Full feature profile of a LoRA adapter across all SAE layers."""
    adapter_name: str
    profiles: Dict[str, FeatureProfile]  # layer_name → FeatureProfile
    global_sparsity: float               # average sparsity across layers
    total_active_features: int           # total unique active features across layers
    total_features: int                  # total features across all layers


def compute_feature_profile(
    base_activations: torch.Tensor,
    lora_activations: torch.Tensor,
    sae: SparseAutoencoder,
    adapter_name: str,
    layer_name: str,
    threshold_multiplier: float = 3.0,
    device: str = "cuda",
) -> FeatureProfile:
    """Compute the feature profile of a LoRA adapter at one layer.

    Measures which SAE features the adapter modifies and by how much.
    Uses absolute threshold (mean + k*std of all feature deltas) to avoid
    the tautological percentile-based sparsity measurement.

    Args:
        base_activations: (n_tokens, d_model) activations without LoRA
        lora_activations: (n_tokens, d_model) activations with LoRA
        sae: the Sparse Autoencoder for this layer
        adapter_name: name of the adapter
        layer_name: name of the layer
        threshold_multiplier: features with mean|Δf| > mean + k*std are "active"
        device: computation device

    Returns:
        FeatureProfile with sparse feature representation
    """
    sae = sae.to(device)

    # Process in chunks to manage memory
    chunk_size = 1024
    n_tokens = base_activations.shape[0]
    all_delta_f = []

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        base_chunk = base_activations[start:end].to(device)
        lora_chunk = lora_activations[start:end].to(device)

        # Encode both through SAE
        f_base = sae.encode(base_chunk)  # (chunk, n_features)
        f_lora = sae.encode(lora_chunk)  # (chunk, n_features)

        # Feature difference
        delta_f = f_lora - f_base  # (chunk, n_features)
        all_delta_f.append(delta_f.cpu())

    delta_f_all = torch.cat(all_delta_f, dim=0)  # (n_tokens, n_features)

    # Mean absolute feature modification across tokens
    mean_abs_delta = delta_f_all.abs().mean(dim=0)  # (n_features,)

    # Determine support: features with significant modification
    nonzero_mask = mean_abs_delta > 0
    if nonzero_mask.sum() == 0:
        # No features modified — return empty profile
        return FeatureProfile(
            adapter_name=adapter_name,
            layer_name=layer_name,
            support=torch.tensor([], dtype=torch.long),
            coefficients=torch.tensor([]),
            full_coefficients=mean_abs_delta,
            sparsity=0.0,
            total_features=sae.n_features,
        )

    # Absolute threshold: mean + k*std of all feature deltas
    # This measures TRUE sparsity, not a tautological percentile
    global_mean = mean_abs_delta.mean()
    global_std = mean_abs_delta.std()
    thresh = global_mean + threshold_multiplier * global_std

    active_mask = mean_abs_delta > thresh
    support = active_mask.nonzero(as_tuple=False).squeeze(-1)
    coefficients = mean_abs_delta[support]
    sparsity = float(support.numel()) / sae.n_features

    return FeatureProfile(
        adapter_name=adapter_name,
        layer_name=layer_name,
        support=support,
        coefficients=coefficients,
        full_coefficients=mean_abs_delta,
        sparsity=sparsity,
        total_features=sae.n_features,
    )


def compute_adapter_feature_map(
    model,
    tokenizer,
    adapter_name: str,
    adapter_path: str,
    saes: Dict[str, SparseAutoencoder],
    probe_texts: List[str],
    batch_size: int = 4,
    max_length: int = 256,
    threshold_multiplier: float = 3.0,
    device: str = "cuda",
) -> AdapterFeatureMap:
    """Compute full feature map for a LoRA adapter across all SAE layers.

    Args:
        model: base model (PEFT-compatible)
        tokenizer: tokenizer
        adapter_name: human-readable adapter name
        adapter_path: path to PEFT adapter directory
        saes: Dict mapping layer_name → SparseAutoencoder
        probe_texts: probe dataset
        batch_size: batch size for activation collection
        max_length: max sequence length
        threshold_multiplier: features with mean|Δf| > mean + k*std are "active"
        device: computation device

    Returns:
        AdapterFeatureMap with profiles for all layers
    """
    from peft import PeftModel

    layer_names = list(saes.keys())

    # Collect base activations (no adapter)
    logger.info(f"Collecting base activations for {len(layer_names)} layers...")
    base_acts = collect_activations(
        model, tokenizer, probe_texts, layer_names,
        batch_size=batch_size, max_length=max_length, device=device,
    )

    # Load adapter and collect activations
    logger.info(f"Loading adapter '{adapter_name}' from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    # Skip .to() when model uses device_map="auto" (multi-GPU)
    if not (hasattr(model, "hf_device_map") and model.hf_device_map):
        peft_model.to(device)

    logger.info("Collecting adapter activations...")
    lora_acts = collect_activations(
        peft_model, tokenizer, probe_texts, layer_names,
        batch_size=batch_size, max_length=max_length, device=device,
    )

    # Unload adapter to free memory
    del peft_model
    torch.cuda.empty_cache()

    # Compute feature profiles
    profiles = {}
    total_active = 0
    total_features = 0

    for layer_name in layer_names:
        if layer_name not in base_acts or layer_name not in lora_acts:
            logger.warning(f"Missing activations for {layer_name}, skipping")
            continue

        profile = compute_feature_profile(
            base_activations=base_acts[layer_name],
            lora_activations=lora_acts[layer_name],
            sae=saes[layer_name],
            adapter_name=adapter_name,
            layer_name=layer_name,
            threshold_multiplier=threshold_multiplier,
            device=device,
        )
        profiles[layer_name] = profile
        total_active += profile.support.numel()
        total_features += profile.total_features

    global_sparsity = total_active / max(total_features, 1)

    logger.info(
        f"Adapter '{adapter_name}': {total_active}/{total_features} active features "
        f"({global_sparsity:.4f} = {global_sparsity*100:.2f}%)"
    )

    return AdapterFeatureMap(
        adapter_name=adapter_name,
        profiles=profiles,
        global_sparsity=global_sparsity,
        total_active_features=total_active,
        total_features=total_features,
    )
