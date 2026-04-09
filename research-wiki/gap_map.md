# Field Gap Map

## G1: No Feature-Space Adapter Composition
**Status**: UNRESOLVED
**Description**: All adapter merging methods operate in weight space (TIES, DARE, TA, ESM). No one has decomposed LoRA effects into SAE feature space and composed at the feature level. SAILS (Jan 2026) showed SAE→LoRA for safety initialization only.
**Target of**: idea:001 (SFC)

## G2: Orthogonality ≠ Composability (Unexplained)
**Status**: UNRESOLVED
**Description**: "Rethinking Inter-LoRA Orthogonality" (2025) proved weight-space orthogonality does NOT predict composability. What algebraic/geometric property DOES predict it? No answer exists.
**Target of**: idea:001 (FDS metric)

## G3: No Composability Scaling Law from First Principles
**Status**: UNRESOLVED
**Description**: Model Merging Scaling Laws (2025) gave empirical curve fitting only. No theoretical derivation of how composability scales with model size, rank, feature overlap.
**Target of**: idea:001 (Scaling Corollary)

## G4: No Formal Bridge: Mechanistic Interpretability ↔ Adapter Composition
**Status**: UNRESOLVED
**Description**: Two hot fields with no rigorous connection. FSRL (2509.12934) proved single adapter ≈ sparse features but didn't compose. SASFT (ICLR 2026) used SAE to guide fine-tuning but not merging.

## G5: Weight-Space Merging Heuristics Lack Theory
**Status**: PARTIALLY RESOLVED
**Description**: TIES/DARE/TA have no optimality guarantees. Rank Bottleneck gave Eckart-Young bound but it's the "wrong" decomposition (weight space, not feature space).

## G6: No Pre-Merge Diagnostic with Structural Interpretation
**Status**: UNRESOLVED
**Description**: BGD, cosine interference, CRS are all proxy metrics. None decomposes interference into interpretable components. FDS (feature overlap) would be the first structurally interpretable diagnostic.
