# Auto Review Loop — SFC (Sparse Feature Composition)

**Started**: 2026-04-23
**Difficulty**: nightmare
**Target**: NeurIPS Best Paper
**Method**: Compose Features, Not Weights — Interference-Free Adapter Merging via Sparse Feature Decomposition

---

## Round 1 (2026-04-23 01:30 UTC)

### Assessment (Summary)
- Score: 2/10
- Verdict: not ready
- Key criticisms:
  - FATAL: Static unsigned steering (mean_abs_delta loses sign/context)
  - FATAL: full_coefficients not zeroed for inactive features
  - CRITICAL: SAE TopK/ReLU encoder mismatch between training and inference
  - CRITICAL: Baseline merging in LoRA factor space instead of delta_W space
  - MAJOR: No downstream results, no baseline comparison
  - MAJOR: Probe contamination from padding tokens

<details>
<summary>Click to expand full reviewer response</summary>

Score: 2/10. Verdict: not ready.

Verified claims: SAE pipeline implemented (collect activations, encode, threshold, FDS, max-pool, hooks). FDS is Jaccard distance. No completed SFC downstream results.

Unverified/false claims: "Zero-hyperparameter" false (threshold_multiplier, probe set, layers). "SFC-Exact" false (static positive offsets, not exact adapter effects). "Provably interference-free" not matched by implementation. "Zero SAE training cost" false (training Qwen SAEs locally). "Outperforms baselines" completely unverified.

FATAL #1: compute_feature_profile uses mean_abs_delta destroying sign and context. Hook adds same positive coefficients to every token.
FATAL #2: full_coefficients stores raw mean_abs_delta for ALL features, support thresholding bypassed.
CRITICAL #3: Baseline merging averages raw LoRA A/B factors, not delta_W = B@A.
CRITICAL #4: SAE trained with TopK but loaded with ReLU — encoder mismatch.
CRITICAL #5: Interference metric ignores decoder cross-terms and SAE reconstruction error.
MAJOR #6: Padding tokens contaminate probe statistics.
MAJOR #7: MCQ eval uses first-char matching instead of log-prob scoring.
MAJOR #8: Zero end-to-end SFC evidence in repo.

Path to positive results: Current "provably interference-free" thesis not credible. Weaker salvage as empirical diagnostic paper possible. Need signed context-conditioned deltas, valid baselines, real evaluation.

</details>

### Actions Taken
- Fixed FATAL #1: Use signed mean delta (mean instead of mean_abs) for coefficients
- Fixed FATAL #2: Zero inactive features in full_coefficients
- Fixed CRITICAL #3: Merge baselines in delta_W = B@A space with SVD refactoring
- Fixed CRITICAL #4: SparseAutoencoder now loads and uses topk parameter from config.json
- Fixed composition: Use max absolute magnitude with sign preservation instead of unsigned max

### Results
- Code fixes committed and pushed to pod (commit 5acedc8)
- SAE training still in progress (layer 15 at ~54%, layer 23 pending)
- Downstream evaluation queued after SAE training completes

### Status
- Continuing to round 2 after experiments complete
- Difficulty: nightmare
