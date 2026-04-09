---
type: paper
node_id: paper:scaling2025_merging_laws
title: "Model Merging Scaling Laws"
authors: ["Unknown"]
year: 2025
venue: arXiv (under review)
external_ids:
  arxiv: "2509.24244"
tags: [scaling-laws, model-merging, empirical]
relevance: related
origin_skill: research-lit
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# One-line thesis

First scaling law for model merging: loss follows power law in model size and number of experts; gains taper as ~1/(k+b); larger base models lower the floor.

## Method

Empirical curve fitting across model sizes and merging methods.

## Key Results

- Merging scaling law: L(n,k) = a·n^(-α) + b·k^(-β) + c
- Holds across averaging/TA/TIES/DARE
- Larger base → lower asymptotic loss

## Limitations / Failure Modes

- Purely empirical (no theoretical derivation)
- No explanation of WHY these laws hold
- No connection to adapter subspace structure

## Connections

[AUTO-GENERATED]

## Relevance to This Project

**Our Scaling Corollary (C5) provides the theoretical foundation.** Larger models → more SAE features → sparser adapter effects → less feature overlap → better composability. This explains their empirical law from first principles.
