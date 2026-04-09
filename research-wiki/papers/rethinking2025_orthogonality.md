---
type: paper
node_id: paper:rethinking2025_orthogonality
title: "Rethinking Inter-LoRA Orthogonality: Orthogonality Does NOT Equal Composability"
authors: ["Unknown"]
year: 2025
venue: arXiv
external_ids:
  arxiv: "2510.03262"
tags: [lora, orthogonality, composability, negative-result]
relevance: core
origin_skill: research-lit
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# One-line thesis

Strict weight-space orthogonality between LoRA adapters does NOT lead to semantic disentanglement or composability — a critical negative result invalidating a common assumption.

## Problem / Gap

Many papers assume orthogonal LoRAs compose well. Is this true?

## Key Results

- Orthogonal weight updates ≠ orthogonal feature effects
- Composability needs to be measured in a different space (not weight space)

## Limitations / Failure Modes

- Identifies the problem but offers no solution
- Does not propose what DOES predict composability

## Reusable Ingredients

- Experimental methodology for testing composability
- The negative result itself (strong motivation for our work)

## Connections

[AUTO-GENERATED]

## Relevance to This Project

**Primary motivation.** This paper proves weight-space metrics are insufficient for predicting composability. Our FDS (feature overlap in SAE space) is the natural resolution: composability depends on feature overlap, not weight orthogonality.
