# Remaining Risks

| Risk | Severity | Mitigation | Status |
|------|----------|------------|--------|
| Existing MMLU adapters contaminated | High | Must retrain on auxiliary_train before using | Fix documented, retraining needed on pod |
| CARR may not beat TA/TIES | High | A/B/C comparison required | Not yet tested |
| High novelty risk (LoRA-Flow, MixLoRA close) | High | Need explicit differentiation experiments | Not yet addressed |
| Current adapters too weak (math/medical no improvement) | High | CARR test on science+medical pair only | Acknowledged |
| 50-sample evaluation noise | Medium | Need 100+ samples, 3+ seeds, CI | Eval script fix planned but not yet implemented on pod |
| MCQ first-char extraction unreliable | Medium | Logprob scorer needed | Not yet implemented |
| Router may collapse to domain classifier | Medium | Gate entropy logging + uniform-gate ablation | Logging implemented in CARR module |
| CARR router training may not converge | Medium | One-batch overfit test required first | Not yet run (needs GPU) |
| PEFT hook integration fragile | Medium | Base equivalence test passes locally | Needs real model verification |
| eval_domain_accuracy.py seed still hardcoded | Low | Fix is identified but not applied | train_domain_lora.py fixed |
