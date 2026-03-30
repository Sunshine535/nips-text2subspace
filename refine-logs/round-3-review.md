# Round 3 Review (GPT-5.4)

**Overall Score: 7.8/10 (different rubric) — Conditional READY**

## Key Fixes for READY

1. **Replace spectral step with projected-core averaging**: S_i = (U*)^T · ΔW_i · V* — cleaner, avoids "rotation" language, handles ambiguities naturally.
2. **State T1 assumptions explicitly**: bounded principal angles, equal rank, bounded spectra, local regime.
3. **Demote T2 to Proposition/Bound**.
4. **Clarify**: repeated singular values handled by projected-core form.
5. **Add adaptive rule**: if BGD < threshold, use linear merge; else GrassMerge.

## Reviewer said
"If you make those last two mathematical-cleanliness fixes, I would move this to plain READY."
