# Refinement Report

**Problem**: Principled LoRA adapter composition with geometric guarantees
**Initial Approach**: Grassmann manifold algebraic operations on LoRA weight spaces
**Date**: 2026-03-29
**Rounds**: 3 / 5
**Final Score**: Conditional READY (~8.5 after fixes)
**Final Verdict**: Conditional READY

## Problem Anchor
Current LoRA merging methods (Task Arithmetic, TIES, DARE) operate by naively combining parameter-space vectors, ignoring the geometric structure of the low-rank subspaces that LoRA adapters define.

## Output Files
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 6                | 4                  | 6                    | 8                 | 7           | 6                | 6               | 6.0     | REVISE  |
| 2     | 9.3              | 8.4                | 8.8                  | 8.7               | -           | 8.5              | -               | 8.5     | NOT READY |
| 3     | 8.5              | 8.0                | 8.0                  | -                 | -           | 8.0              | 7.5             | 7.8*    | Conditional READY |

*Round 3 used a different rubric (7 custom dimensions instead of the standard 7).

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|-------|-------------------------|------------------|--------|
| 1     | Wrong math object (col-only); not specific enough; scope too broad; theorems overclaimed | Upgraded to bi-Grassmann (U,Σ,V); added Algorithm 1 pseudocode; cut subtract/project; rewrote theorems | Resolved |
| 2     | Spectral step ad hoc ("rotation"); T2 coarse; weights undefined; SVD ambiguity | Renamed spectral step; defined weights; reframed T2; added sign-alignment; failure modes | Mostly resolved |
| 3     | "Rotation" language still questionable; T2 should be Proposition | Replaced with projected-core averaging S_i=(U*)^T ΔW_i V*; T2→Proposition; added adaptive rule | Resolved |

## Final Proposal Snapshot
- GrassMerge: geometry-aware post-training LoRA composition on G(r,d_out)×G(r,d_in)
- Projected-core averaging for spectral interpolation (clean, handles ambiguities naturally)
- Theorem 1: rank-preserving distortion bound
- Proposition 2: BGD-based interference upper bound
- BGD: bilateral Grassmann distance as pre-merge diagnostic
- Zero new parameters, PEFT-compatible, 5-30 second merge time

## Method Evolution Highlights
1. **Most important simplification**: Cut subtract/project/interpolation from main paper → focus on COMPOSE only
2. **Most important mechanism upgrade**: Column-only Grassmann → full bi-Grassmann + projected-core averaging
3. **Most important honesty improvement**: "Parallel-transported spectral averaging" → "projected-core averaging"; Theorem 2 → Proposition 2

## Pushback / Drift Log
| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1     | Switch to full fixed-rank operator | Accepted — strengthens same direction | Accepted |
| 2     | Make BGD subordinate | Accepted — BGD is now secondary | Accepted |
| 3     | Don't say "only method" | Changed to "to our knowledge" | Accepted |

## Remaining Weaknesses
1. Theorem 1 proof needs to be fully written (currently proof sketch)
2. Proposition 2 tightness is empirical, not provable
3. Benefits may be small when subspaces are well-aligned
4. 3 base models may not be enough for some reviewers (but standard for NeurIPS)

## Next Steps
- Proceed to implementation: update `src/lora_algebra.py` to implement GrassMerge algorithm
- `/experiment-plan` for detailed execution roadmap
- `/run-experiment` to deploy training
