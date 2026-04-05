# LoRA Algebra — Domain Accuracy Comparison

| Method | Domain | Benchmark | Accuracy | Avg Tokens |
|--------|--------|-----------|----------|------------|
| base_model | history | mmlu_history | 0.0400 | 407.9 |
| base_model | philosophy | mmlu_philosophy | 0.0850 | 410.4 |
| base_model | science | arc_challenge | 0.1150 | 304.6 |
| individual_loras | history | mmlu_history | 0.7950 | 5.0 |
| individual_loras | philosophy | mmlu_philosophy | 0.7450 | 5.0 |
| individual_loras | science | arc_challenge | 0.9100 | 1.0 |