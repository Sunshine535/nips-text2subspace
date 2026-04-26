# External Code and License Audit

## Main method files (our code, no external derivation)
- `src/conflict_aware_routing.py` — CARR router (to become ablation candidate)
- `src/carr_objective.py` — multi-term objective + MCQ utilities
- `src/carr_config_loader.py` — YAML config loader
- `src/utility_certified_routing.py` — UCAR (to be created)
- `src/utility_oracle.py` — oracle ceiling utilities (to be created)
- `scripts/eval_carr.py` — CARR evaluation harness
- `scripts/eval_ucar.py` — UCAR evaluation (to be created)
- `scripts/train_ucar_router.py` — UCAR training (to be created)
- `scripts/eval_utility_oracle.py` — oracle runner (to be created)

## Baseline / utility files (our code, standard algorithms)
- `scripts/eval_sfc_downstream.py` — TA/TIES/DARE merge + MCQ eval
- `src/cross_factor_fusion.py` — LoRA factor loading + BCFF (historical)
- `src/conflict_diagnostics.py` — activation Gram conflict analysis

## External dependencies (installed via pip, not copied)
- transformers (Apache 2.0)
- peft (Apache 2.0)
- torch (BSD-3)
- datasets (Apache 2.0)
- safetensors (Apache 2.0)
- PyYAML (MIT)

## Files forbidden from main method
- No external LoRA-Flow / MixLoRA / AdapterFusion / qa-FLoRA code copied
- No official baseline implementations in repo (baselines use standard TA/TIES/DARE from our own implementation)

## License status
- Repository: no LICENSE file currently. Should add before submission.
- All dependencies: permissive (Apache/BSD/MIT).
- No provenance risk detected.
