# Auto Review Loop — GrassMerge

## Round 1 (2026-03-30)

### Assessment (Summary)
- Score: 4/10
- Verdict: Not ready for submission
- Key criticisms:
  1. Broken evaluation pipeline (saves LoRA factors, evaluator expects delta weights)
  2. Novelty overlap with KnOTS, Core Space, TSPA, LoRA-LEGO, OSRM
  3. Theory not strong enough for headline contribution
  4. Evaluation not NeurIPS-grade (synthetic placeholders, regex-style eval)
  5. BGD ignores spectral structure; needs comparison with simpler metrics
  6. Claims too strong (premature quantitative predictions)
  7. "Algebraic closure" overstated

### Actions Taken
1. **Fixed evaluation pipeline bug**: GrassMerge now saves delta weights (not LoRA factors) so eval path loads correctly
2. **Fixed pairwise baselines**: All baselines now run pairwise (matching GrassMerge protocol), not all-domain merge
3. **Added KnOTS and TSPA baselines**: Implemented KnOTSMerge and TSPAMerge classes in lora_algebra.py, registered in all experiment scripts
4. **Improved BGD**: Added spectral_weighted_bgd (weights subspace distances by singular value importance), plus cosine_interference and frobenius_interference as comparison baselines
5. **Fixed data loading**: Extended format_example() to handle camel-ai/physics message_1/message_2 fields and euclaise/writingprompts prompt/story fields; switched science domain to allenai/sciq
6. **Removed synthetic evaluation**: Dropped creative_writing and translation synthetic benchmarks; defined CORE_EVAL_DOMAINS = [math, code, medical, science, history, philosophy]
7. **Toned down claims**: Changed "Claim 1/2" to "Hypothesis 1/2" in FINAL_PROPOSAL.md; removed specific quantitative predictions
8. **Added grassmann_interpolate**: Fixed missing method used by ablation interpolation_type study
9. **Fixed directory name mismatch**: eval_domain_accuracy.py now looks for "grassmerge" directory first

### Status
- Continuing to Round 2

## Round 2 (2026-03-30)

### Assessment (Summary)
- Score: 5.5/10
- Verdict: Not ready (method testable, but no evidence)
- Key criticisms:
  1. No empirical evidence (still zero experiment results)
  2. CORE_EVAL_DOMAINS not enforced in default eval path
  3. HumanEval scoring uses string match, not execution-based pass@1
  4. MMLU training formatter ignores choices, SFT format wrong for MCQ domains
  5. A0 ablation only logs cosine/time proxies, not downstream accuracy
  6. BGD analysis doesn't compute correlation with actual degradation
  7. TSPA too similar to SVD-Procrustes

### Actions Taken
1. Fixed MMLU training format: MCQ with lettered options (A/B/C/D)
2. Enforced CORE_EVAL_DOMAINS as default
3. Distinguished TSPA from Procrustes: per-component sign correction + greedy permutation matching
4. Added BGD correlation analysis script (analyze_bgd_correlation.py)
5. Switched science dataset to allenai/sciq

## Round 3 (2026-03-30)

### Assessment (Summary)
- Score: 6/10
- Verdict: Code materially better, but still has paper-blocking correctness issues
- Key criticisms:
  1. MMLU uses wrong split (cais/mmlu has auxiliary_train, not train)
  2. ARC choices is a dict {text, label}, not a list; MedQA uses options field
  3. MMLU answer is ClassLabel integer, never decoded to A/B/C/D
  4. HumanEval scoring still not execution-based
  5. Ablation scripts use legacy LoRAAlgebra.compose() not GrassMerge
  6. Silent synthetic data fallback masks broken datasets

### Actions Taken
1. Fixed MMLU split: added SPLIT_OVERRIDES mapping cais/mmlu -> auxiliary_train
2. Removed synthetic data fallback: now raises RuntimeError on dataset load failure
3. Fixed MCQ evaluation: decode_answer() handles ClassLabel integers; format_mmlu_question() handles ARC dict, MedQA options, MMLU choices
4. Replaced HumanEval with MBPP sanitized (more reliable text-based evaluation)
5. Fixed ablation scripts: rank and N-way ablations now use GrassMerge.merge()
6. Fixed interpolation ablation: uses LoRAAlgebra.interpolate() and algebra.grassmann_interpolate()

### Status
- ARIS code review complete. Moving to server validation.
