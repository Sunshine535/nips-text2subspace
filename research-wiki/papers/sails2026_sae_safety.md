---
type: paper
node_id: paper:sails2026_sae_safety
title: "SAILS: Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation"
authors: ["Unknown"]
year: 2026
venue: arXiv
external_ids:
  arxiv: "2512.23260"
tags: [sae, lora, safety, interpretability, subspace]
relevance: core
origin_skill: research-lit
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# One-line thesis

Uses SAE-decoded monosemantic features to construct an interpretable safety subspace, then initializes LoRA adapters in that subspace (99.6% safety on Gemma-2-9B with 0.19% params).

## Problem / Gap

Safety fine-tuning is opaque — you don't know what features the LoRA is modifying.

## Method

1. Run SAE on base model activations
2. Identify safety-relevant SAE features
3. Construct a low-rank subspace from those features
4. Initialize LoRA in that subspace
5. Fine-tune with safety data

## Key Results

- 99.6% safety rate on Gemma-2-9B
- Proves SAE-based identification achieves arbitrarily small recovery error
- Direct identification has irreducible error floor

## Limitations / Failure Modes

- Only applied to safety (single capability)
- Only initialization direction (SAE → LoRA), not analysis direction (LoRA → SAE)
- Not used for multi-adapter composition

## Reusable Ingredients

- SAE → subspace → LoRA pipeline
- Proof that SAE features capture adapter-relevant directions
- Gemma-2-9B + Gemma Scope setup

## Open Questions

- Can this generalize beyond safety to arbitrary capabilities?
- Can you go in reverse: given an arbitrary LoRA, decompose its effect into SAE features?
- Can you COMPOSE multiple SAE-constructed LoRAs?

## Connections

[AUTO-GENERATED]

## Relevance to This Project

**Critical precursor.** Proves the SAE→LoRA direction works. We extend in two ways: (1) reverse direction (LoRA→SAE decomposition), (2) multi-adapter composition in SAE feature space.
