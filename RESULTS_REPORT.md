> **ARCHIVED — LOW CONFIDENCE HISTORICAL RESULTS** (2026-04-24)
> Missing seed, command, checkpoint manifest. Cannot be used as strong evidence.
> Archived copy: archive/20260424_pre_carr/RESULTS_REPORT.md

# GrassMerge Experiment Results (Comprehensive)

**Date**: 2026-04-07
**Model**: Qwen/Qwen3-8B (8.19B params)
**Hardware**: 1x NVIDIA H100 80GB
**Training**: 10K samples/domain, rank=16, alpha=32, 2 epochs
**Evaluation**: 200 samples/benchmark, enable_thinking=False

## 1. Individual Domain LoRA Performance

| Domain | Benchmark | Base Model | + LoRA | Change |
|--------|-----------|-----------|--------|--------|
| math | GSM8K | 0.910 | 0.775 | -0.135 |
| code | MBPP(exec) | 0.065 | 0.060 | -0.005 |
| medical | MedQA | 0.725 | 0.595 | -0.130 |
| science | ARC-Challenge | 0.895 | 0.910 | +0.015 |
| history | MMLU History | 0.870 | 0.865 | -0.005 |
| philosophy | MMLU Philosophy | 0.715 | 0.680 | -0.035 |

## 2. GrassMerge vs Baselines (All 15 Pairwise Merges, 30 Evaluations)

| Pair | Domain | Base | LoRA | GrassMerge | TaskArith | TIES | DARE | Best |
|------|--------|------|------|-----------|-----------|------|------|------|
| code+history | code | 0.065 | 0.060 | **0.060** | 0.065 | 0.070 | 0.065 | TIES |
| code+history | history | 0.870 | 0.865 | **0.870** | 0.875 | 0.870 | 0.870 | TA |
| code+math | code | 0.065 | 0.060 | **0.060** | 0.070 | 0.075 | 0.075 | TIES |
| code+math | math | 0.910 | 0.775 | **0.885** | 0.805 | 0.895 | 0.815 | TIES |
| code+medical | code | 0.065 | 0.060 | **0.070** | 0.065 | 0.065 | 0.065 | GM |
| code+medical | medical | 0.725 | 0.595 | **0.660** | 0.680 | 0.660 | 0.685 | DARE |
| code+philosophy | code | 0.065 | 0.060 | **0.055** | 0.065 | 0.080 | 0.070 | TIES |
| code+philosophy | philosophy | 0.715 | 0.680 | **0.725** | 0.730 | 0.735 | 0.745 | DARE |
| code+science | code | 0.065 | 0.060 | **0.060** | 0.070 | 0.080 | 0.070 | TIES |
| code+science | science | 0.895 | 0.910 | **0.890** | 0.920 | 0.930 | 0.925 | TIES |
| history+math | history | 0.870 | 0.865 | **0.890** | 0.870 | 0.870 | 0.875 | GM |
| history+math | math | 0.910 | 0.775 | **0.895** | 0.675 | 0.660 | 0.700 | GM |
| history+medical | history | 0.870 | 0.865 | **0.885** | 0.870 | 0.865 | 0.870 | GM |
| history+medical | medical | 0.725 | 0.595 | **0.700** | 0.675 | 0.635 | 0.700 | GM |
| history+philosophy | history | 0.870 | 0.865 | **0.880** | 0.865 | 0.860 | 0.865 | GM |
| history+philosophy | philosophy | 0.715 | 0.680 | **0.725** | 0.705 | 0.695 | 0.715 | GM |
| history+science | history | 0.870 | 0.865 | **0.885** | 0.865 | 0.870 | 0.865 | GM |
| history+science | science | 0.895 | 0.910 | **0.895** | 0.925 | 0.915 | 0.925 | TA |
| math+medical | math | 0.910 | 0.775 | **0.830** | 0.785 | 0.800 | 0.830 | GM |
| math+medical | medical | 0.725 | 0.595 | **0.720** | 0.670 | 0.625 | 0.675 | GM |
| math+philosophy | math | 0.910 | 0.775 | **0.870** | 0.675 | 0.685 | 0.660 | GM |
| math+philosophy | philosophy | 0.715 | 0.680 | **0.720** | 0.705 | 0.715 | 0.705 | GM |
| math+science | math | 0.910 | 0.775 | **0.850** | 0.750 | 0.820 | 0.760 | GM |
| math+science | science | 0.895 | 0.910 | **0.910** | 0.910 | 0.910 | 0.915 | DARE |
| medical+philosophy | medical | 0.725 | 0.595 | **0.730** | 0.645 | 0.635 | 0.640 | GM |
| medical+philosophy | philosophy | 0.715 | 0.680 | **0.730** | 0.745 | 0.730 | 0.740 | TA |
| medical+science | medical | 0.725 | 0.595 | **0.685** | 0.675 | 0.665 | 0.680 | GM |
| medical+science | science | 0.895 | 0.910 | **0.895** | 0.930 | 0.935 | 0.925 | TIES |
| philosophy+science | philosophy | 0.715 | 0.680 | **0.720** | 0.730 | 0.740 | 0.740 | TIES |
| philosophy+science | science | 0.895 | 0.910 | **0.900** | 0.910 | 0.900 | 0.910 | TA |

## 3. Aggregate Comparison

| Method | Mean Accuracy | GM Win Rate | GM Avg Advantage |
|--------|--------------|-------------|------------------|
| **GrassMerge** | **0.6883** | --- | --- |
| Task Arithmetic | 0.6642 | 15/30 (50%) | +0.0242 |
| TIES (d=0.5) | 0.6663 | 13/30 (43%) | +0.0220 |
| DARE (p=0.5) | 0.6693 | 12/30 (40%) | +0.0190 |

## 4. GrassMerge vs Base Model & Individual LoRA

- vs Base Model: mean delta = -0.0083, beats base in 16/30 evals
- vs Individual LoRA: mean delta = +0.0408, beats LoRA in 25/30 evals

## 5. Evaluation Coverage

| Phase | Entries | Status |
|-------|---------|--------|
| Base model eval | 6 | Complete |
| Individual LoRA eval | 6 | Complete |
| GrassMerge (15 pairs) | 30 | Complete |
| Task Arithmetic (15 pairs) | 30 | Complete |
| TIES (15 pairs) | 30 | Complete |
| DARE (15 pairs) | 30 | Complete |
| N-way merge | -- | Pending |
| Profiling | -- | Pending |
| BGD correlation | -- | Pending |
| **Total evaluations** | **132** | |
