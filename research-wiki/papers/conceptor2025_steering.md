---
type: paper
node_id: paper:conceptor2025_steering
title: "From Steering Vectors to Conceptors: Compositional Multi-Objective Steering for LLMs"
authors: ["Unknown"]
year: 2025
venue: ICML 2025
external_ids:
  arxiv: "2410.16314"
tags: [conceptor, steering, boolean-algebra, composition, activation-space]
relevance: related
origin_skill: research-lit
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# One-line thesis

Replaces additive steering vectors with conceptor matrices (soft projections) supporting Boolean AND/OR/NOT for compositional multi-objective steering (+10-30pp over vectors).

## Problem / Gap

Steering vectors are additive only — no subtraction, intersection, or negation. Interference when combining multiple objectives.

## Method

Conceptor matrices C = R(R + α⁻²I)⁻¹ as soft subspace projections. Boolean ops: NOT C = I - C; AND = (C₁⁻¹ + C₂⁻¹ - I)⁻¹; OR = NOT(NOT C₁ AND NOT C₂).

## Key Results

- 10-30pp improvement over additive steering
- Clean Boolean algebra for multi-objective composition

## Limitations / Failure Modes

- Inference-time only (activation steering, not weight modification)
- No weight-space counterpart
- No LoRA connection

## Reusable Ingredients

- Boolean algebra framework for composition
- Conceptor matrix formalism

## Connections

[AUTO-GENERATED]

## Relevance to This Project

**Algebraic inspiration.** The Boolean algebra (AND/OR/NOT) for conceptors shows what a principled composition algebra looks like. Our SFC provides analogous operations in SAE feature space for adapter weights.
