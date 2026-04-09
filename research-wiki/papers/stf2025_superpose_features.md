---
type: paper
node_id: paper:stf2025_superpose_features
title: "Superpose Task-specific Features for Model Merging"
authors: ["Unknown"]
year: 2025
venue: EMNLP 2025
external_ids:
  arxiv: "2502.10698"
tags: [model-merging, feature-space, representation, linear-representation]
relevance: core
origin_skill: research-lit
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
---

# One-line thesis

Merges models by superposing features in representation space instead of averaging weights, using the linear representation hypothesis.

## Problem / Gap

Weight-space merging is suboptimal because it ignores feature-level structure.

## Method

Uses task-specific linear transformations to extract features, then superposes them in representation space.

## Key Results

- Outperforms weight-space merging methods
- Validates "feature space is better than weight space" thesis

## Limitations / Failure Modes

- No SAE decomposition — uses linear transformations, not monosemantic features
- No interpretability guarantees
- No formal interference theory

## Reusable Ingredients

- "Feature space > weight space" argument
- Experimental comparison methodology

## Connections

[AUTO-GENERATED]

## Relevance to This Project

**Closest competitor.** Shows feature-space merging is better. But: no SAE, no interpretability, no sparsity guarantees, no interference theory. We provide all of these.
