# Query Pack — nips-text2subspace (2026-04-09)

## Direction
"Compose Features, Not Weights" — SAE-based sparse feature composition for provably interference-free LoRA adapter merging. NeurIPS 2026 Best Paper target.

## Top Gaps (unresolved)
- **G1**: No feature-space adapter composition (all methods work in weight space)
- **G2**: Orthogonality ≠ composability — what DOES predict it? (negative result from 2025, unsolved)
- **G4**: No formal bridge between mechanistic interpretability and adapter composition
- **G3**: No composability scaling law from first principles (only 1 empirical paper)
- **G6**: No structurally interpretable pre-merge diagnostic

## Paper Clusters
**SAE+Adapters**: SAILS (safety init via SAE), FSRL (adapter ≈ sparse features), SASFT (SAE guides fine-tuning). Direction: SAE→LoRA proved, LoRA→SAE→composition NOT done.
**Feature-Space Merging**: STF (EMNLP'25, linear features), Conceptors (ICML'25, Boolean algebra on activations). Direction: right idea but no SAE, no sparsity, no LoRA composition.
**Negative Results**: "Rethinking Orthogonality" (weight ortho ≠ composable), Raffel'26 (merging ≈ regularization). Direction: weight-space frame is wrong.

## Failed Ideas (banlist)
- **GrassMerge** (3/10): crowded geometric merging space, individual LoRAs degrade base model, 50% win rate insufficient
- **Rank Bottleneck + BAC** (7.5/10 idea, not best-paper): Eckart-Young repackaged, trivial trilemma, BAC=lightweight MoE. Lesson: repackaging known results ≠ new theory; need paradigm shift

## Active Idea
- **idea:001 SFC**: Decompose LoRA→SAE features, compose by union/max. 5 contributions (2 theorems, 1 method, 1 metric, 1 scaling law). Kill: sparsity <80%, SFC not better than TIES, FDS ρ<0.5.

## Open Unknowns
- Is LoRA effect truly sparse in SAE space for NON-safety tasks (math, code, medical)?
- Does feature-space composition scale to N>4 adapters?
- Can SAE reconstruction error be bounded tightly enough for the theorems?
