# The Rank Bottleneck of Adapter Composition

LoRA adapter 组合的 rank bottleneck 理论 + Bottleneck-Aware Composition (BAC) 方法。

核心发现: 当 N 个 rank-r adapter 的 "composition rank" r_c > r 时，任何 input-independent 静态合并都必然损失信息。损失下界 = 超出 rank r 的 spectral tail energy (tight bound)。BAC 通过仅路由 bottleneck 方向 (k = r_c - r 维) 以 <5% 开销达到接近 routing 的质量。

## Quick Start

```bash
# 1. 环境安装 (conda env 'text2subspace' + Python 3.11 + PyTorch CUDA 12.8)
bash setup.sh
conda activate text2subspace

# 2. 一键运行 (自动检测 GPU 数量，多卡并行)
bash run.sh

# 3. 或分步执行:
# 训练 6 domain LoRAs (自动多卡 DDP)
python scripts/train_domain_loras.py --config configs/domains.yaml \
    --domains math code medical science history philosophy

# 计算 Composition Rank Score
python scripts/compute_crs.py --lora_dir results/domain_loras

# 合并实验 (baselines + BAC)
python scripts/run_algebra_experiments.py --config configs/domains.yaml

# 并行评估 (自动分配到所有 GPU)
python scripts/run_parallel_eval.py --num_gpus 0  # 0 = auto-detect
```

## 环境安装

```bash
git clone <this-repo>
cd nips-text2subspace
bash setup.sh          # conda env 'text2subspace' + Python 3.11 + PyTorch (CUDA 12.8)
conda activate text2subspace
```

依赖: Python 3.11 (conda), PyTorch (CUDA 12.8), transformers, peft, datasets, safetensors.

## 项目结构

```
src/
  lora_algebra.py          # LoRA 加载/保存, GrassMerge, baselines (TIES/DARE/TA/KnOTS/TSPA)
  rank_bottleneck.py       # CRS 估计, bottleneck 识别, BAC 框架
scripts/
  train_domain_lora.py     # 单 domain LoRA 训练 (支持 torchrun DDP)
  train_domain_loras.py    # 批量训练编排 (自动检测 GPU 数量)
  compute_crs.py           # 计算所有 pair 的 CRS
  run_algebra_experiments.py  # 合并实验
  run_parallel_eval.py     # 多 GPU 并行评估
  verify_rank_bottleneck.py   # Theorem 1 synthetic verification
  eval_domain_accuracy.py  # 单次评估脚本
configs/
  domains.yaml             # domain 配置 (数据集, 训练参数, benchmarks)
```

## GPU 自适应

所有脚本自动检测可用 GPU 数量:

- **训练**: `torchrun --nproc_per_node=<auto>` DDP 并行
- **评估**: `run_parallel_eval.py` 将 domain pair 分配到所有 GPU 并行执行
- **合并**: CPU 即可 (LoRA 权重操作, 不需要 GPU)
- **CRS 计算**: CPU 即可 (SVD of Nr×Nr matrix, 毫秒级)

`--num_gpus 0` 表示自动检测, 等价于 `torch.cuda.device_count()`。

## 理论

### Adapter Composition Trilemma

组合 N 个 rank-r LoRA adapter 时，不可同时满足:

1. **Compact**: 输出 rank-r (O(dr) 参数)
2. **Static**: input-independent (零推理开销)
3. **Faithful**: 完整保留所有 domain 能力

现有方法: Merging = 1+2 (牺牲 3), Routing = 3 (牺牲 1+2), BAC = 1+3 (牺牲极少量 2)。

### Rank Bottleneck Theorem

Composition rank r_c = rank(K), 其中 K 是 Nr×Nr block Gram matrix。

对任意 rank-k 静态合并 M:

```
E(M) ≥ Σ_{j>k} σ_j(G)²
```

G = stacked whitened operator, bound is tight (Eckart-Young extension to conditional operator families)。

### Composition Rank Score (CRS)

```
CRS = 1 - (r_c - r) / ((N-1) * r)
```

CRS=1: free merge | CRS=0: max bottleneck | CRS 预测合并质量。

## API

```python
from src.lora_algebra import LoRAWeights
from src.rank_bottleneck import compute_composition_rank, BottleneckAwareComposition, BACConfig

# 加载 adapters
loras = [LoRAWeights.from_peft_dir("math", "results/domain_loras/math"),
         LoRAWeights.from_peft_dir("code", "results/domain_loras/code")]

# 计算 CRS
analysis = compute_composition_rank(loras)
print(f"CRS: {analysis.global_crs:.4f}")
for key, info in analysis.layers.items():
    if info.bottleneck_dim > 0:
        print(f"  {key}: r_c={info.composition_rank}, bottleneck_dim={info.bottleneck_dim}")

# BAC composition
bac = BottleneckAwareComposition(BACConfig(static_rank=16))
bac.analyze(loras)
static_adapter = bac.build_static_merge(loras)  # PEFT-compatible merged adapter
print(bac.summary())
```

## Baselines

已实现: Task Arithmetic, TIES, DARE, KnOTS, TSPA, GrassMerge, SVD-Procrustes。

## Citation

```bibtex
@article{rankbottleneck2026,
  title={The Rank Bottleneck of Adapter Composition},
  year={2026}
}
```
