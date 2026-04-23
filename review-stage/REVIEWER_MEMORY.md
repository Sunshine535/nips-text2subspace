# Reviewer Memory

## Round 1 — Score: 2/10
- **Suspicion**: Author writes theorem-strength claims first and leaves implementation at prototype quality.
- **Unresolved**: Dense full_coefficients bug, invalid factor-space baseline merging, TopK/ReLU SAE mismatch.
- **Patterns**: Overclaimed theory ("provably optimal", "zero-hyperparameter") not matched by code.
- **Track next round**: Do not trust "SFC beats baselines" numbers unless baselines are merged in delta_W space and raw SFC outputs are shown. Watch for grassmerge/text2subspace artifacts reused as SFC evidence.
- **Author claims to fix**: signed deltas, zeroed full_coefficients, TopK SAE loading, delta_W baselines. Verify these fixes actually appear in the code next round.
