# Review Summary

**Problem**: Principled LoRA adapter composition with geometric guarantees
**Initial Approach**: "LoRA Algebra" with column-subspace Grassmann operations
**Date**: 2026-03-29
**Rounds**: 3 / 5
**Final Score**: ~8.5 (Conditional READY)
**Final Verdict**: Conditional READY

## Problem Anchor
Current LoRA merging methods (Task Arithmetic, TIES, DARE) operate by naively combining parameter-space vectors, ignoring the geometric structure of the low-rank subspaces that LoRA adapters define. No mathematical framework treats LoRA adapters as geometric objects with operations that provably preserve task-relevant information.

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|-------|-------------------------|------------------------------------------|---------|----------------|
| 1     | Column-only object loses info; method too vague; scope too broad; theorems overclaimed | Upgraded to full (U,Σ,V); added exact pseudocode; cut subtract/project; rewrote theorems with assumptions | Yes | Spectral step naming; T2 credibility |
| 2     | "Rotation" not honest; weights undefined; SVD ambiguity; T2 still too strong | Renamed spectral step; defined 3 weight policies; sign-alignment; reframed T2; failure modes | Mostly | Core step still labeled wrong |
| 3     | Use projected-core form S=(U*)^T ΔW V*; T2→Proposition | Replaced spectral step entirely; T2 demoted; added adaptive BGD rule | Yes | Proof details needed |

## Overall Evolution
- **Concretization**: Vague "algebra on subspaces" → exact 8-step algorithm with pseudocode, complexity analysis, and PEFT output format
- **Focusing**: 6 operations (compose, subtract, interpolate, project, geodesic, baselines) → 1 operation (compose) + 1 diagnostic (BGD)
- **Mathematical honesty**: Column-only G(r,d) → bi-Grassmann G(r,d_out)×G(r,d_in); "parallel transport" → "projected-core averaging"; Theorem → Proposition for weaker results
- **Frontier awareness**: Positioned against 4 concurrent works (Fisher-Rao, RiemannLoRA, LoRA-S, TSPA) with clear differentiation table
- **Drift prevention**: All changes strengthened the same direction; no scope creep or problem shift

## Final Status
- Anchor status: **Preserved** throughout all rounds
- Focus status: **Tight** — one algorithm, one diagnostic, one theorem, one proposition
- Modernity status: **Appropriately frontier-aware** — no forced trendy additions
- Strongest parts: Clean algorithm design, rank-preservation insight, practical PEFT compatibility
- Remaining weaknesses: Proof needs full writeup; benefits may be small for well-aligned tasks; Proposition 2 relies on generic Lipschitz assumption
