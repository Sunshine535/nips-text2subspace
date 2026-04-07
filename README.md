# Text2Subspace — Grassmann 流形上的 LoRA 代数

## 项目简介

在 Grassmann 流形上对 LoRA adapter 进行代数运算（合并、插值、投影），实现 domain-specific LoRA 的几何组合。核心组件 GrassMerge 通过 Grassmann 均值在子空间层面合并 LoRA，相比传统方法 (Task Arithmetic / TIES / DARE) 保留更多功能信息。

**Review 状态**: Round 7, Score 3.0/10（实验数据不完整）

## 环境安装

```bash
cd /workspace/nips-text2subspace
python3 -m venv .venv
source .venv/bin/activate
pip install torch
pip install transformers trl peft datasets accelerate wandb tensorboard \
    safetensors scikit-learn evaluate numpy scipy matplotlib pandas pyyaml
```

## 快速开始

```bash
source .venv/bin/activate

# 训练单个 domain LoRA
python3 scripts/train_domain_lora.py --config configs/domains.yaml --domain math

# LoRA 代数运算
python3 scripts/lora_algebra_ops.py --operation merge --method grassmerge \
    --loras results/domain_loras/math results/domain_loras/code
```

## 完整实验流程

```bash
# 一键全流程
bash scripts/run_full_pipeline.sh

# 分步：
# 1. 训练全部 domain LoRAs
bash scripts/train_all_core.sh

# 2. 运行代数实验（merge/interpolation/projection）
python3 scripts/run_algebra_experiments.py --config configs/domains.yaml

# 3. 消融实验
python3 scripts/run_ablations.py --config configs/domains.yaml

# 4. 综合评估
python3 scripts/eval_comprehensive.py
```

## 断点续训

- LoRA 训练使用 HuggingFace checkpoint，支持 `--resume_from_checkpoint`
- 结果按 domain/method 存储，重跑自动跳过已有结果

## 项目结构

```
src/
  lora_algebra.py        # 核心：LoRAAlgebra, GrassmannOps, GrassMerge, KnOTSMerge
scripts/                 # 27 个实验脚本
  train_domain_lora.py   # 单 domain 训练
  eval_lora_algebra.py   # 代数运算评估
  run_bestpaper_experiments.py  # Best paper 级实验
configs/
  domains.yaml           # 全部配置
results/                 # 实验结果
```

## 已有结果

- GrassMerge vs TA/TIES/DARE: 15 pairs 完成

## 下一步

1. 完成 full eval pipeline
2. 补充 interpolation 和 projection 实验
3. 3 seeds + bootstrap CIs
