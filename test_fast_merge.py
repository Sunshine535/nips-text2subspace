#!/usr/bin/env python3
"""Quick test of fast GrassMerge."""
import sys, time
sys.path.insert(0, '.')
print("Importing...", flush=True)
import torch
from src.lora_algebra import LoRAWeights, GrassMerge

print("Loading math LoRA...", flush=True)
t0 = time.time()
lora_math = LoRAWeights.from_peft_dir('math', 'results/domain_loras/math')
print(f"  Loaded in {time.time()-t0:.2f}s: rank={lora_math.rank}, layers={len(lora_math.lora_A)}", flush=True)

print("Loading code LoRA...", flush=True)
t0 = time.time()
lora_code = LoRAWeights.from_peft_dir('code', 'results/domain_loras/code')
print(f"  Loaded in {time.time()-t0:.2f}s", flush=True)

print("Testing fast_svd...", flush=True)
t0 = time.time()
svd_math = lora_math.fast_svd()
t1 = time.time()
print(f"  fast_svd: {t1-t0:.3f}s for {len(svd_math)} layers", flush=True)
key = list(svd_math.keys())[0]
U, S, Vh = svd_math[key]
print(f"  Layer {key}: U={U.shape}, S={S.shape}, Vh={Vh.shape}", flush=True)
print(f"  Top singular values: {S[:5].tolist()}", flush=True)

print("Testing GrassMerge...", flush=True)
t0 = time.time()
merged = GrassMerge(karcher_max_iter=10).merge([lora_math, lora_code], name='math+code')
t1 = time.time()
print(f"  GrassMerge: {t1-t0:.3f}s, rank={merged.rank}, layers={len(merged.lora_A)}", flush=True)

# Quick quality check
delta_m = lora_math.to_delta_weight()
delta_c = lora_code.to_delta_weight()
delta_merged = merged.to_delta_weight()
keys = sorted(set(delta_m.keys()) & set(delta_c.keys()) & set(delta_merged.keys()))
k = keys[0]
cos_m = torch.nn.functional.cosine_similarity(delta_merged[k].flatten().unsqueeze(0), delta_m[k].flatten().unsqueeze(0)).item()
cos_c = torch.nn.functional.cosine_similarity(delta_merged[k].flatten().unsqueeze(0), delta_c[k].flatten().unsqueeze(0)).item()
print(f"  Cosine to math: {cos_m:.4f}, to code: {cos_c:.4f}", flush=True)
print("SUCCESS!", flush=True)
