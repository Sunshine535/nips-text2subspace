# Remaining Risks

**Last updated**: 2026-04-25 (post GPT-5.5 Round 3 review, Round 4 infrastructure).

Categories: **Resolved**, **Unresolved** (Round 3 rerun will close or confirm), **Stale**
(superseded by current architecture; keep for audit).

---

## Resolved

| Risk | Resolution | Evidence |
|------|-----------|----------|
| CARR hook passes zeros instead of real hidden states | Fixed Round 2 (commit `fae727e`): `h_float = h.float()` flows into router | `src/conflict_aware_routing.py` L235 |
| `conflict_scores` not reaching router | Precomputed module-level Gram, broadcast in hook | `conflict_aware_routing.py` L242-259 |
| Training path had `no_grad` | Training mode branches without no_grad | `conflict_aware_routing.py` L250-253 |
| `top_k` / `use_base_fallback` cosmetic only | top_k masks adapters with -inf; base idx dropped when `False` | router L76-78, L132-137 |
| `configs/carr_minimal.yaml` not read | `src/carr_config_loader.py` loads yaml, scripts call it, `effective_config.json` saved per run, `assert_config_applied` guards drift | commit `63ffaa3` |
| first-char generation metric | `evaluate_model_mcq(metric_mode="logprob_mcq")` scores each option by length-normalized sum logprob | `scripts/eval_sfc_downstream.py` |
| eval sample shuffle hardcoded `seed=42` | `sample_seed` parameter; per-run `sample_manifest.jsonl` with question/label hashes | commit `63ffaa3` |
| LM loss on question text as router objective | `--objective carr_full` uses multi-term MCQ loss (L_task+L_base_KL+L_conf+L_sparse+L_cal) | `src/carr_objective.py` |
| reliability head was plain MLP with no labels | `precompute_reliability_labels` writes per-item binary correctness + logprob advantage; L_cal is BCE on these | `src/carr_objective.py` |

## Unresolved (Round 3 rerun must close)

| Risk | Severity | What needs to happen | Owner |
|------|----------|---------------------|-------|
| Full CARR not verified with logprob + multi-term + strong baselines | **Critical** | Rerun A/B/C + 3 ablations with `--config configs/carr_minimal.yaml --metric_mode logprob_mcq --objective carr_full`, 3 router × 3 sample seeds, n≥50/domain | next GPU run |
| Strongest static baseline not established | High | `static_baselines` mode runs TA/TIES/DARE and reports best; A claim should be best-of-3 | next run |
| Conflict mechanism is module-level only by default | Medium-High | `conflict_mode: "token"` implemented but not evaluated; need ablation comparing module vs token | pending |
| ECE/Brier of reliability head not verified | Medium | `--log_reliability_calibration` flag added; first run with `objective=carr_full` will emit metrics | pending |
| Checkpoint integrity not yet CI-gated | Medium | `scripts/check_checkpoint_integrity.py` implemented; must be run before any result commit | pending |
| n_samples=50 statistical power marginal | Medium | Expand to n=200/domain once methodological correctness is established | pending |
| Only science+medical pair in minimal eval | Medium | Expand to medical+math and math+science after metric/objective verified | pending |
| Official TIES/DARE comparison in same framework | Medium | `static_baselines` mode covers TA/TIES/DARE; GrassMerge not yet wired | pending |

## Stale (superseded; kept for audit)

| Risk | Status |
|------|--------|
| Existing MMLU adapters contaminated | Stale — current eval excludes MMLU domains in `safe_pairs` |
| Current adapters too weak (math/medical no improvement) | Partially stale — adapters are the existing `sfc_loras_test`; quality questions now rolled into the reliability-label pipeline |
| High novelty risk (LoRA-Flow, MixLoRA close) | Stale — novelty positioning deferred to Round 3 rerun; not a code risk |
| CARR router training may not converge | Stale — one-batch overfit passed (loss 8.02→7.48, entropy 1.33→0.19) |
| Router may collapse to domain classifier | Stale — gate stats show layer-differentiated behavior (L27 base≈0.997, L19 base≈0.01) |
| eval_domain_accuracy.py seed hardcoded | Stale — fixed in commit `3f0c57d` |

---

## What success looks like for Round 3

Run the following and land them in `results-synced/` with `effective_config.json` + `sample_manifest.jsonl` per run:

```
python scripts/eval_carr.py \
  --config configs/carr_minimal.yaml \
  --mode full \
  --domains science,medical \
  --seed {1,2,3} --sample_seed {1,2,3} \
  --metric_mode logprob_mcq \
  --output_dir results-synced/round3_seed{S}_sample{SS}
```

Then the claims ladder upgrade rule:

- Full CARR > strongest static in ≥5 of 9 (seed × sample_seed) runs → upgrade from "weak signal" to "trust minimal result".
- Full CARR > strongest static in <5 of 9 → pivot (method re-examination, not full benchmark).
