# Local Repository Scan

**Date**: 2026-04-24
**Repository**: /home/tarkoy/nips/nips-text2subspace

## Directory Map

| Component | Path | Purpose | Importance | Notes |
|-----------|------|---------|------------|-------|
| Method: LoRA algebra/baselines | `src/lora_algebra.py` | TA/TIES/DARE/GrassMerge implementations | High (baseline) | DARE uses hardcoded seed 42 |
| Method: Rank bottleneck/BAC | `src/rank_bottleneck.py` | CRS diagnostics, BAC structure | Medium | CRS may only check left subspace |
| Method: SFC | `src/sparse_feature_composition.py` | SAE feature composition, FDS, hooks | High (negative evidence) | Static steering failure documented |
| Method: SAE | `src/sae_decomposition.py` | SAE loading, feature profiling | Medium | TopK encoder fix applied |
| Method: FLC | `src/functional_composition.py` | Activation LS merge | Medium (negative evidence) | 18-20% energy retention |
| Method: BCFF | `src/cross_factor_fusion.py` | Cross-factor fusion | High (bug evidence) | Tautological target Y=y1+y2 |
| Training: domain LoRA | `scripts/train_domain_lora.py` | Single domain adapter training | High | MMLU test split leakage risk (P0 bug) |
| Training: multi-domain | `scripts/train_domain_loras.py` | Batch domain adapter training | Medium | Calls train_domain_lora |
| Training: SAE | `scripts/train_sae.py` | TopK SAE training | Medium | Sequential multi-layer |
| Eval: domain accuracy | `scripts/eval_domain_accuracy.py` | Main evaluation entry | High | Fixed sampling seed=42, crude MCQ extraction |
| Eval: SFC downstream | `scripts/eval_sfc_downstream.py` | SFC vs baselines | High (negative evidence) | Contains delta_W baseline implementations |
| Eval: FLC | `scripts/eval_flc.py` | FLC evaluation | Medium (negative) | |
| Eval: BCFF | `scripts/eval_bcff.py` | BCFF evaluation | Medium (negative) | |
| Eval: lean | `scripts/run_eval_lean.py` | Memory-optimized eval | Medium | Written for 16Gi pod |
| Eval: one pair | `scripts/eval_one_pair.py` | Subprocess-isolated pair eval | Medium | |
| Config: domains | `configs/domains.yaml` | Model/domain/eval config | High | Base model, LoRA params, domain list |
| Tests | `tests/test_text2subspace.py` | Existing tests | Low | Coverage insufficient |
| Results: synced | `results-synced/` | Pod experiment data | High (evidence) | BCFF/SFC/FLC JSONs, SAE configs, logs |
| Results: local | `results/` | Gitignored results | Medium | Pilot JSONs |
| Logs | `logs/` | Historical logs | Medium | |
| Review | `review-stage/` | Auto review loop docs | High | AUTO_REVIEW.md, REVIEWER_MEMORY, state |
| Ideas | `idea-stage/` | Idea discovery docs | Medium | RESEARCH_BRIEF.md |
| Progress | `PROGRESS.md` | Failure summary | Very High | Key evidence source |
| Proposal | `PROPOSAL.md` | SFC proposal (outdated) | Medium (historical) | Claims invalidated by PROOF_AUDIT |
| Proof audit | `PROOF_AUDIT.md` | Theory review | High | 1 FATAL + 3 CRITICAL issues |
| Results report | `RESULTS_REPORT.md` | GrassMerge old report | Low | Missing seed/command/checkpoint |
| Runbook | `README_RUN.md` | Text2Subspace pilot | Low | Stale paths |
| Experiments | `EXPERIMENTS.md` | Text2Subspace v2 plan | Medium | Proxy pilot only |
| Old scripts | `scripts/run_algebra_experiments.py`, `scripts/eval_lora_algebra.py`, etc. | Historical experiments | Low | BAC/GrassMerge era |
| Shell | `run.sh`, `collect_results.sh`, `setup.sh` | Automation | Low | Partially stale |
| Research wiki | `research-wiki/` | Literature notes | Low | |
