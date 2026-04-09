---
type: idea
node_id: idea:001
title: "Compose Features, Not Weights: Sparse Feature Composition (SFC)"
stage: proposed
outcome: null
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
tags: [sae, feature-composition, adapter-merging, interpretability, best-paper-candidate]
---

# One-line thesis

Decompose LoRA adapter effects into SAE feature space, compose by sparse feature-level operations (union/max), provably interference-free when feature supports are disjoint.

## Hypothesis

H1: Each LoRA adapter modifies a sparse set (<5%) of SAE features.
H2: Two adapters with disjoint feature supports compose without interference.
H3: SFC outperforms all weight-space merging methods (TIES, DARE, TA, ESM).
H4: Feature Disentanglement Score (FDS) predicts composition quality better than any existing metric.
H5: Composability improves with model size (more features → sparser effects → less overlap).

## Core Contributions

C1: Sparse Feature Decomposition Theorem
C2: Interference Localization Theorem
C3: SFC algorithm (zero hyperparameters)
C4: FDS diagnostic metric
C5: Composability Scaling Law from first principles

## Target Gaps

G1 (no feature-space composition), G2 (orthogonality ≠ composability), G3 (no scaling law), G4 (no mech-interp ↔ adapter bridge), G6 (no interpretable diagnostic)

## Kill Criteria

- LoRA effects not sparse in SAE space (>20% features modified) → framework fails
- SFC not better than TIES/DARE → method worthless
- FDS Spearman ρ < 0.5 → theory disconnected from practice

## Based On

paper:sails2026_sae_safety, paper:fsrl2025_anatomy_alignment, paper:rethinking2025_orthogonality, paper:stf2025_superpose_features

## Failure Notes

_(none yet — not tested)_
