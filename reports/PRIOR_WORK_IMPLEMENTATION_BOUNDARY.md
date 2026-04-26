# Prior Work Implementation Boundary

## LoRA-Flow (ACL 2024)
- Role: related work / closest baseline for dynamic LoRA fusion
- Status: NOT copied; no code from thunlp/LoRAFlow in this repo
- UCAR difference: utility-certified abstention to base, not continuous token-level fusion weights

## AdapterFusion (EACL 2021)
- Role: related work / conceptual ancestor
- Status: NOT copied; no AdapterHub code
- UCAR difference: calibrated utility lower-bound selection, not learned attention-based composition

## MixLoRA / MoE-LoRA (2024)
- Role: related work
- Status: NOT copied; no TUDB-Labs/MixLoRA code
- UCAR difference: utility certification with base abstention, not MoE top-k expert routing

## qa-FLoRA (AAAI 2025/2026)
- Role: closest related work for base-vs-adapter evidence
- Status: NOT copied
- UCAR difference: calibrated utility lower-bound + abstention threshold, not distributional divergence weighting

## LORAUTER (2026)
- Role: related work for task-representation adapter routing
- Status: NOT copied
- UCAR difference: per-example utility certification with base dominance detection

## Task Arithmetic / TIES / DARE / KnOTS
- Role: required static baselines
- Status: TA/TIES/DARE implemented from descriptions (our own code in eval_sfc_downstream.py)
- No official code copied; implementations follow published algorithms

## Summary
No external method code is present in this repository. All implementations are original.
UCAR's core differentiation: **calibrated positive-utility lower-bound + abstention-to-base**.
