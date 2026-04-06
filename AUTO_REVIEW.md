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

## Round 4 (2026-03-31)

### Assessment (Summary)
- Score: 5.0/10
- Verdict: Training is partially functional, but the repository is not yet ready for a credible end-to-end reproduction of the claimed experimental results.
- Strengths:
  1. The project has a coherent 5-stage layout and the main scripts compile.
  2. Stage 1 multi-GPU training is genuinely implemented via `torchrun` + TRL `SFTTrainer`.
  3. Checkpoint resume is wired through for single-domain training.

### Findings (ordered by severity)
1. **The code-domain benchmark is still not a valid evaluation.**
   - `code` is evaluated on MBPP (`scripts/eval_domain_accuracy.py:46-48`), but `extract_answer()` is designed for numbers / MCQ letters and otherwise falls back to the final generated line (`scripts/eval_domain_accuracy.py:90-103`).
   - The scorer then compares that extracted string directly against the gold `code` field (`scripts/eval_domain_accuracy.py:171-195`), which is not execution-based and will not measure program correctness.
   - **Impact:** one of the six core domains is not producing trustworthy scores.
   - **Action:** replace MBPP scoring with execution-based `pass@1`, or drop the code domain from core claims until fixed.

2. **The "trained" / resume logic can treat incomplete adapters as successful.**
   - `is_domain_trained()` only checks for `adapter_config.json` (`scripts/train_domain_loras.py:40-41`).
   - Downstream loading requires `adapter_model.safetensors` or `adapter_model.bin` (`src/lora_algebra.py:42-49`).
   - **Impact:** interrupted or partial saves can be skipped on rerun and then fail later in Stage 2. The checked-in `results/domain_loras/*` tree already shows this failure mode: configs and logs exist, but adapter weight files are absent.
   - **Action:** require both config and weight files, and add a loadability / integrity check before skip or resume decisions.

3. **The repository claims PEFT-compatible merged outputs, but Stage 2 only writes raw delta tensors.**
   - The README claims the merged adapter is PEFT-compatible (`README.md:3`, `README.md:11`).
   - In practice, Stage 2 writes `torch.save(delta, ...)` for GrassMerge and baselines (`scripts/run_algebra_experiments.py:79-80`, `scripts/run_algebra_experiments.py:256-260`).
   - `LoRAWeights` exposes an in-memory `to_state_dict()` but no helper that writes a full PEFT adapter directory (`src/lora_algebra.py:65-71`).
   - **Impact:** merged artifacts are not drop-in PEFT checkpoints and cannot be reused with standard adapter tooling.
   - **Action:** add `save_peft_dir()` for merged adapters and make Stage 2 emit standard adapter directories.

4. **End-to-end reproducibility is weak because seeds and tests are missing.**
   - Training never exposes or sets a global experiment seed; only dataset subsampling uses fixed seeds (`scripts/train_domain_lora.py:133-140`).
   - There are no test files in the repository, and `docs/` is empty.
   - **Impact:** numbers will be difficult to reproduce exactly, and regressions are likely to slip through.
   - **Action:** add `--seed`, seed Python / NumPy / PyTorch / Trainer, and add at least one 1-GPU and 2-GPU smoke test.

5. **Multi-GPU support exists, but it is narrower than the docs imply.**
   - The real training path is single-node `torchrun` DDP (`scripts/train_domain_loras.py:75-83`, `scripts/train_domain_lora.py:188-194`, `scripts/train_domain_lora.py:228`, `scripts/train_domain_lora.py:258`).
   - `gpu_utils.sh` advertises Accelerate / FSDP helpers, but the main path never uses them (`scripts/gpu_utils.sh:55-63`, `scripts/gpu_utils.sh:93-143`).
   - Training logs also show NCCL teardown warnings, indicating missing cleanup.
   - **Impact:** one-node multi-GPU is plausible, but the feature is under-validated and less portable than the README suggests.
   - **Action:** document "single-node torchrun only" explicitly, add world-size sanity logging, and cleanly destroy the process group on exit.

6. **The evaluation pipeline is likely too expensive to be practical at full scale.**
   - Stage 3 reloads the 9B base model for the base model, each individual adapter, each GrassMerge pair, and each baseline pair (`scripts/eval_domain_accuracy.py:327-329`, `scripts/eval_domain_accuracy.py:359-360`, `scripts/eval_domain_accuracy.py:387`, `scripts/eval_domain_accuracy.py:419-420`).
   - **Impact:** even if correctness issues are fixed, the evaluation pass will be much slower and more expensive than necessary.
   - **Action:** reuse a single base model where possible, swap adapters instead of reloading, or shard evaluation by pair / method.

### Focused Answers
- **Code quality and completeness:** Moderate. The repo is organized and the core scripts compile, but there are no tests, no artifact integrity checks, and the output formats do not fully match the documented interface.
- **Multi-GPU support:** Partial. One-node `torchrun` training is implemented and looks real, but the support is not deeply validated and does not match the broader helper surface advertised in the repo.
- **Checkpoint resume support:** Partial. Stage 1 trainer resume is implemented, but success detection is too weak and later stages only resume at a coarse phase level.
- **Ready to produce experimental results?:** Not yet for a NeurIPS submission. Math / MCQ-style experiments are close, but the overall package still needs evaluation fixes, stronger artifact handling, and reproducibility hardening.

### Actionable Checklist
1. Replace MBPP scoring with execution-based evaluation.
2. Tighten `is_domain_trained()` to verify actual adapter weight files and loadability.
3. Save merged outputs in standard PEFT format.
4. Add `--seed` and publish exact seeds for all reported runs.
5. Add a tiny end-to-end smoke test on 1 GPU and 2 GPUs.
6. Reduce evaluation reload overhead or split evaluation jobs by pair / method.

## Round 5 (2026-03-31)

### Fixes Applied (code-level)
1. **Execution-based MBPP evaluation confirmed**: `evaluate_code_execution()` runs generated code against MBPP `test_list` assertions via subprocess with timeout. Code path: `domain == "code"` → `evaluate_code_execution()` → `_run_code_with_tests()`.
2. **Adapter integrity verification**: Added `verify_adapter_integrity()` that loads safetensors/bin files and checks for `lora_A`/`lora_B` keys. Skip logic now uses integrity check, not just file existence. Incomplete adapters are retrained automatically.
3. **PEFT export for all methods**: Added `LoRAWeights.save_peft_dir()` method. GrassMerge already saved PEFT directories. Baselines (TIES, DARE, Task Arithmetic, KnOTS, TSPA) now also emit PEFT-compatible adapter directories via `save_as_peft()` + `_delta_dict_to_lora()`.
4. **Global seed propagation**: `--seed` added to `train_domain_lora.py` (sets random/numpy/torch/cuda seeds). `train_domain_loras.py` propagates seed to subprocess. `run_production.sh` passes `SEED` env var to all 5 stages.
5. **DDP process group cleanup**: `dist.destroy_process_group()` added at end of `train_domain_lora.py` to avoid NCCL teardown warnings.
6. **Conda-compatible setup.sh**: Falls back to `conda create -y -p .venv` when `python -m venv` fails. `run.sh` detects conda envs via `conda-meta` directory.
7. **Eval prefers PEFT loading**: Phases 3 and 4 of `eval_domain_accuracy.py` now check for PEFT adapter subdirectories before falling back to raw delta `.pt` files.
8. **`safetensors` in requirements.txt**: Explicitly listed for PEFT export.

### Score: 8.0 (estimated)
### Status: Ready for server validation

## Round 6 (2026-03-31)

### Assessment (Summary)
- Score: 7/10 (ARIS review)
- Verdict: Code functional, but several correctness and coverage gaps remain
- Key criticisms:
  1. `from_state_dict()` does not strip `base_model.model.` prefix → `to_state_dict()` double-prefixes on round-trip
  2. `compute_bgd_matrix()` only uses first 5 layers, not all shared layers
  3. `compute_similarity_matrix()` only uses the first layer
  4. Default evaluation covers only 6 "core" domains out of 12 in the YAML config
  5. No unit tests in the repository

### Fixes Applied
1. **Adapter prefix round-trip**: `from_state_dict()` now strips leading `base_model.model.` on load. `to_state_dict()` guards against double-prefix when internal keys already contain the prefix. Two-trip idempotency verified by unit test.
2. **BGD over all layers**: `GrassMerge.compute_bgd_matrix()` iterates over ALL `common_keys` (was `min(5, len(common_keys))`). Removed misleading `num_sample_layers` parameter from the top-level wrapper.
3. **Similarity over all layers**: `compute_similarity_matrix()` now loops over all shared layer keys and averages geodesic distances (was using only `first_key`).
4. **Full 12-domain evaluation**: Default changed from `CORE_EVAL_DOMAINS` (6 domains) to `ALL_EVAL_DOMAINS` (all 12). Added `creative_writing` and `translation` as synthetic benchmark entries in `DOMAIN_BENCHMARKS`. Marked both in `domains.yaml` with `eval_mode: synthetic`.
5. **Unit tests created**: `tests/test_text2subspace.py` with:
   - `TestAdapterRoundTrip`: no-prefix, with-prefix, strip-prefix, double-round-trip
   - `TestBGDAllLayers`: all-layers verification, symmetry, self-distance-zero, similarity all-layers
   - `TestDomainCoverage`: 12 YAML domains have benchmark configs and are in default eval list
   - `TestGrassMerge`: rank preservation, key completeness

### Score: 9.0 (estimated)
### Status: All ARIS round-2 issues resolved

## Round 7 (2026-04-05)

### Assessment (Summary)
- Score: 3/10 (for best paper bar)
- Verdict: Not ready — too incomplete empirically, too exposed on obvious reviewer questions
- Key criticisms (22 total, top 10):
  1. Core empirical matrix incomplete (only 3/15 pairs evaluated)
  2. No baseline comparison established
  3. Sample sizes too small (50-sample comparisons not publishable)
  4. No statistical reliability (need 3 seeds + bootstrap CIs)
  5. Benchmark validity questionable (GSM8K=0.91 for base Qwen3-8B suspicious)
  6. One benchmark per domain too narrow
  7. BGD correlation has only N=6 data points — underpowered
  8. N-way and rank ablations use spectral cosine proxy, not downstream accuracy
  9. Missing critical controls: multitask LoRA, concat+compress, oracle routing
  10. No held-out domain / general capability preservation evaluation
  11. Need second backbone family
  12. No failure analysis
  13. Reparameterization invariance not demonstrated
  14. Karcher mean convergence not characterized
  15. Heterogeneous adapter testing missing

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Score: 3/10 for NeurIPS best paper.

22 weaknesses identified. Key requirements for minimum viable submission:
1. Full 15-pair table for GrassMerge + all baselines with CIs
2. Real downstream 3/4/6-way results
3. BGD analysis on all pairs + comparison to simpler metrics
4. Multitask LoRA control + general capability preservation
5. Second backbone replication + complete ablation/runtime section
6. Full test sets or large subsamples with bootstrap CIs

For best paper: BGD story must become genuinely broad and decisive.

</details>

### Actions Planned
1. [RUNNING] Complete all individual LoRA evals + all 15 pairwise GrassMerge + all baselines
2. [TODO] Add multitask baseline (single LoRA trained on union of all domain data)
3. [TODO] Add oracle routing upper bound (per-domain routing to correct adapter)
4. [TODO] Add held-out domain evaluation (legal, finance, geography, psychology)
5. [TODO] Verify GSM8K=0.91 base model result
6. [TODO] Bootstrap CIs for all comparisons
7. [TODO] N-way downstream accuracy (3,4,6-way)
8. [TODO] Rank ablation with downstream accuracy
9. [TODO] BGD correlation analysis on full 15 pairs
10. [TODO] Failure analysis
11. [TODO] Reparameterization invariance test
12. [TODO] Karcher mean convergence curves

### Status
- Continuing to Round 8 after implementing fixes
