# Keep / Rewrite / Archive Plan

| Item | Path | Current Role | Evidence | Action | Reason | Risk |
|------|------|--------------|----------|--------|--------|------|
| Base model + adapters | pod checkpoints | Core assets | Required for all methods | KEEP | CARR needs them | None |
| TA/TIES/DARE baselines | `src/lora_algebra.py` | Static baselines | Strong empirical performance | KEEP ONLY AS BASELINE | DARE seed fix needed | Expose seed param |
| GrassMerge | `src/lora_algebra.py` | Old positive | GrassMerge report beat TA mean | KEEP ONLY AS BASELINE | Low confidence historical | Do not claim as method |
| BAC/rank bottleneck | `src/rank_bottleneck.py` | README main method | No strong downstream evidence | MERGE INTO NEW METHOD | CRS → CARR conflict diagnostic | CRS may be incomplete |
| SFC composition | `src/sparse_feature_composition.py` | Feature composition | Negative results documented | KEEP ONLY AS HISTORICAL NEGATIVE | Static steering fails | Do not present as method |
| SAE decomposition | `src/sae_decomposition.py` | SAE loading/profiling | SAEs trained but method failed | FREEZE | May be diagnostic tool | Not needed for CARR v1 |
| FLC | `src/functional_composition.py` | Activation LS merge | Energy collapse documented | KEEP ONLY AS ABLATION | Activation deltas feed CARR | Do not use rank-r merge |
| BCFF | `src/cross_factor_fusion.py` | Cross-factor merge | Tautological target confirmed | KEEP ONLY AS HISTORICAL NEGATIVE | Document bug as lesson | Add regression test |
| MMLU split in training | `scripts/train_domain_lora.py` | Domain training | MMLU test split used | REWRITE | Train/test leakage | Must retrain MMLU adapters |
| Eval sampling | `scripts/eval_domain_accuracy.py` | Main evaluation | Fixed seed=42 | REWRITE | Multi-seed meaningless | Must be parameterized |
| Domain config | `configs/domains.yaml` | Main config | Incomplete split info | REWRITE | Add split manifest | |
| RESULTS_REPORT.md | root | GrassMerge old report | Missing seed/command | ARCHIVE | Low confidence | Keep for reference |
| README_RUN.md | root | Text2Subspace runbook | Stale paths | ARCHIVE | Confuses reproduction | |
| PROPOSAL.md | root | SFC proposal | Claims invalidated | ARCHIVE | Proof audit disproved | Keep audit |
| PROGRESS.md | root | Failure summary | Critical evidence | KEEP | Evidence ledger | Append only |
| results-synced/ | root | Pod experiment data | Historical evidence | FREEZE | Do not overwrite | |
| review-stage/ | root | Review docs | Audit trail | KEEP | Shows integrity | |
| Historical scripts | `scripts/run_*`, `scripts/eval_lora_algebra.py` etc. | Old experiments | BAC/GrassMerge era | FREEZE | Not needed for CARR | Do not delete |
| New CARR code | `src/conflict_aware_routing.py` | New method | Not yet exists | CREATE | Core implementation target | High risk |
| New conflict diagnostics | `src/conflict_diagnostics.py` | New diagnostic | Not yet exists | CREATE | CARR router features | |

## Archive Plan

```
archive/20260424_pre_carr/
├── README.md            (archive description)
├── RESULTS_REPORT.md    (low-confidence GrassMerge report)
├── README_RUN.md        (stale Text2Subspace runbook)
└── PROPOSAL.md          (invalidated SFC proposal)
```
