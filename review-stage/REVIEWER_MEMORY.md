# Reviewer Memory

## Round 1 — Score: 2/10
- **Suspicion**: Author writes theorem-strength claims first and leaves implementation at prototype quality.
- **Unresolved**: Dense full_coefficients bug, invalid factor-space baseline merging, TopK/ReLU SAE mismatch.
- **Patterns**: Overclaimed theory ("provably optimal", "zero-hyperparameter") not matched by code.
- **Track next round**: Do not trust "SFC beats baselines" unless baselines merged in delta_W space.
- **Author claims to fix**: signed deltas, zeroed full_coefficients, TopK SAE loading, delta_W baselines.

## Round 2 — Score: 3/10
- **Previous suspicions addressed?**: Yes — code fixes verified in repo. Baselines now use delta_W space.
- **New finding**: SFC empirically falsified. Static feature steering fundamentally cannot reproduce input-dependent adapter effects.
- **Verdict**: Pivot from SFC thesis entirely.
- **Suggested pivot**: Functional LoRA composition in activation space.
- **Pattern**: Author pivoted to FLC (catastrophic failure, 18% energy) then BCFF (mixed, cross-terms zero).
- **Unresolved**: No method consistently beats TA/TIES. Adapters too weak. Evaluation too noisy.
- **Track next round**: If author claims new method works, verify on pairs where adapters actually improve over base (science, philosophy only). Demand statistical significance (multiple seeds, >100 samples).
