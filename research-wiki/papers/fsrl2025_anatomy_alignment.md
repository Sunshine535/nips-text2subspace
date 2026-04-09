---
type: paper
node_id: paper:fsrl2025_anatomy_alignment
title: "The Anatomy of Alignment: FSRL — Functional Equivalence of Adapters and Sparse SAE Feature Vectors"
authors: ["Unknown"]
year: 2025
venue: arXiv
external_ids:
  arxiv: "2509.12934"
tags: [sae, lora, alignment, sparse-features, equivalence]
relevance: core
origin_skill: research-lit
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# One-line thesis

Trains a lightweight adapter that outputs sparse SAE feature steering vectors; proves functional equivalence to a restricted class of LoRA updates.

## Problem / Gap

What is the relationship between LoRA weight updates and interpretable SAE features?

## Method

Adapter → sparse SAE feature vector → steering. Proves the adapter output is equivalent to sparse feature modification.

## Key Results

- Single adapter ≈ sparse set of SAE features (formal proof for restricted class)
- Establishes LoRA-SAE duality for individual adapters

## Limitations / Failure Modes

- Single adapter only — no multi-adapter composition
- Restricted class of LoRA updates (not arbitrary)
- No composition theorem

## Reusable Ingredients

- LoRA ↔ sparse SAE feature duality proof
- Experimental methodology for measuring feature sparsity of adapters

## Open Questions

- Does the duality extend to arbitrary LoRA updates?
- If two adapters are each sparse in SAE space, what happens when you compose them?
- Is the sparsity sufficient for interference-free composition?

## Connections

[AUTO-GENERATED]

## Relevance to This Project

**Key stepping stone.** Proves our Decomposition Theorem's premise (adapter ≈ sparse features) for single adapters. We extend to: (1) arbitrary LoRA, (2) multi-adapter composition, (3) prove interference localization.
