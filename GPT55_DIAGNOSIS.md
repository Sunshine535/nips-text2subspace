# 结论先行

仓库公开可读，但当前项目不是“已有最好分支再包装一下”的状态，而是一个**方法机制断裂 + 评估可靠性不足 + 旧叙事混杂**的状态。最重要的现象不是某个正面分数，而是：**SFC、FLC、BCFF、BAC/GrassMerge/Text2Subspace pilot 都在不同形式上试图做静态合并或静态低秩/特征注入，但负面结果共同指向同一个缺失机制：没有输入条件化、可靠性校准、冲突感知的动态仲裁机制。**

我没有本地重跑大规模实验，也没有访问 checkpoint 二进制；下面的 “Verified” 表示**从仓库可见代码、日志、结果文件中可交叉确认**，不是我复现实验得出的结论。所有 publish-level 结论仍需最小验证队列确认。

唯一推荐的 MAIN METHOD PATH 是：

> **CARR：Conflict-Aware Reliability-Gated Residual Routing**
> 不再把多个 LoRA/子空间压成一个全局静态 merge；而是在安全静态共享方向之外，用校准过的 reliability gate 和 conflict-aware residual router，按输入/层/token 决定何时保留 base、何时用静态兼容方向、何时路由某个 adapter 的冲突残差方向。

Confidence: **medium-low**。理由是代码与日志给出了较强的负面机制证据，但当前适配器质量、split、metric、seed、baseline 都不足以支撑 high-confidence 方法结论。

---

# 第零部分：仓库可读性检查

| Item                       |  Found? | Location                                                                                      | Notes                                                                             |
| -------------------------- | ------: | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 仓库是否可访问                    |     Yes | GitHub public repo                                                                            | 仓库公开可读，根目录可见。([GitHub][1])                                                        |
| 完整代码                       |  Mostly | `src/`, `scripts/`, `tests/`                                                                  | 可见主代码、脚本、测试入口；未能本地执行。([GitHub][2])                                                |
| README                     |     Yes | `README.md`                                                                                   | 当前 README 主叙事是 BAC / rank bottleneck / adapter composition trilemma。([GitHub][1]) |
| 论文草稿                       | Partial | `PROPOSAL.md`, `IDEA_REPORT.md`, `EXPERIMENTS.md`, `RESULTS_REPORT.md`, `PROOF_AUDIT.md`      | 有 proposal、实验计划、proof audit、结果报告；未看到完整 LaTeX 论文稿。([GitHub][3])                    |
| 训练脚本                       |     Yes | `scripts/train_domain_lora.py`, `scripts/train_domain_loras.py`, `scripts/train_sae.py`       | 有 domain LoRA 和 SAE 训练脚本。([GitHub][4])                                            |
| 评估脚本                       |     Yes | `scripts/eval_domain_accuracy.py`, `eval_bcff.py`, `eval_one_pair.py`, `run_parallel_eval.py` | 有评估入口，但存在 split、seed、metric 可靠性问题。([GitHub][4])                                   |
| configs                    |     Yes | `configs/domains.yaml`                                                                        | 只有一个主 config；base model、LoRA、domain、eval 设置可见。([GitHub][5])                       |
| 日志和结果                      |     Yes | `logs/`, `results/`, `results-synced/`, `review-stage/`                                       | 有 SFC/FLC/BCFF/GrassMerge/Text2Subspace pilot 结果，但不少只是一轮或 50-sample。([GitHub][6]) |
| baseline                   |     Yes | README claims + `lora_algebra.py` + result logs                                               | TA/TIES/DARE/GrassMerge 等存在或被报告；official reproduction 不充分。([GitHub][1])           |
| 失败实验记录                     |     Yes | `PROGRESS.md`, `AUTO_REVIEW.md`, logs                                                         | 失败记录非常重要，尤其是 SFC/FLC/BCFF 负面结论。([GitHub][7])                                      |
| ablation                   | Partial | `scripts/run_ablations.py`, logs, `PROGRESS.md`                                               | 有失败 ablation 和多方法对比，但缺少系统 multi-seed、CI、统一 manifest。([GitHub][4])                 |
| requirements / environment |     Yes | `requirements.txt`, `environment.yml`                                                         | 环境文件存在。([GitHub][1])                                                              |

## 缺失或不足信息

| Missing Item                        | Why Needed                                                        | What You Should Upload / Add                                             |
| ----------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 完整论文草稿 `.tex` 或当前 paper draft       | claim-code-result 对齐需要完整 abstract/introduction/method/experiments | 当前最新 paper draft、overleaf zip 或 `paper/` 目录                              |
| checkpoint manifest                 | 需要判断 stale checkpoint、adapter 版本、训练 split、seed                    | 每个 LoRA / SAE / merge checkpoint 的路径、hash、训练命令、训练数据 split                |
| 原始训练日志                              | 当前结果无法确认 adapter 是否过拟合、是否训练失败                                     | training stdout/stderr、wandb/tensorboard export、seed、数据量                 |
| 统一 result registry                  | 需要确定每个数字对应哪个 config/command/seed/checkpoint                       | `results_manifest.csv/json`，包含 command/config/seed/checkpoint/git commit |
| official baseline reproduction logs | NeurIPS 需要公平 baseline；仓库内多为内部实现或部分 baselines                      | TIES/DARE/KnOTS/RegMean/LoRA-Flow/MixLoRA official code 运行记录             |
| 多 seed + CI                         | 目前 50-sample 和单 seed 不能支撑强 claim                                  | 至少 3 seeds，最好 5 seeds；bootstrap CI                                       |
| split manifest / sample ids         | 当前 MMLU train/eval leakage 风险很高                                   | 每个 domain train/calib/val/test 的 dataset name、split、indices、hash         |
| MCQ logprob metric                  | 生成式首字符 extraction 对 MMLU/ARC 不够稳定                                 | 增加 deterministic logprob-based multiple-choice scorer                    |
| Text2Subspace 真正实现路径                | README_RUN 指向的 `methods/05_text2subspace/...` 未在可见树中确认            | 若私有/遗漏，请上传该目录；否则应标为 stale                                                |

---

# 1. Repository Map

| Component                             | Path                                  | Purpose                                                  | Importance                  | Notes                                                                                          |
| ------------------------------------- | ------------------------------------- | -------------------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------- |
| 主 README / BAC 叙事                     | `README.md`                           | 声称 rank bottleneck、BAC、adapter composition trilemma      | High                        | 当前 README 主 claim 是 BAC，而不是 SFC/FLC/BCFF/Text2Subspace。([GitHub][1])                           |
| 实验总览                                  | `EXPERIMENTS.md`                      | Text2Subspace v2 实验计划、pilot 结果                           | High                        | 明确承认 pilot 是 low-rank policy-head proxy，不是真正 text-conditioned adapter generation。([GitHub][3]) |
| 历史结果报告                                | `RESULTS_REPORT.md`                   | 2026-04-07 GrassMerge 结果                                 | Medium                      | 报告 GrassMerge 均值优于 TA/TIES/DARE，但相对 base 平均为负，且缺少完整 command/seed/checkpoint。([GitHub][8])      |
| 当前失败总结                                | `PROGRESS.md`                         | 汇总 SFC/FLC/BCFF 负面结果、bug audit、pivot 结论                  | Very High                   | 最重要的现象来源：三条旧路线均失败，且 BCFF tautology、SFC static steering、FLC rank bottleneck 被指出。([GitHub][7])   |
| 自动审稿记录                                | `AUTO_REVIEW.md`                      | 多轮 reviewer-style audit                                  | High                        | 记录从 evaluation bug、baseline 缺失、proof gap 到 empirical matrix 不完整的连续问题。([GitHub][9])             |
| SFC proposal                          | `PROPOSAL.md`                         | “Compose Features, Not Weights” 早期叙事                     | Medium                      | 已被 proof audit 与实验负面结果严重削弱。([GitHub][10])                                                      |
| Proof audit                           | `PROOF_AUDIT.md`, `PROOF_SKELETON.md` | 审查 SFC 理论 claim                                          | High                        | 明确指出低秩→稀疏 SAE、max-pool optimal、feature interference→task performance 等证明缺口。([GitHub][11])      |
| LoRA algebra / GrassMerge / baselines | `src/lora_algebra.py`                 | LoRAWeights、Task Arithmetic、TIES、DARE-like、GrassMerge 等  | High                        | 是当前静态 merge/baseline 核心。([GitHub][12])                                                         |
| BAC / rank bottleneck                 | `src/rank_bottleneck.py`              | rank bottleneck diagnostics、BAC 结构                       | High                        | README 主方法相关，但当前 empirical support 不足；CRS 可能只看左子空间。([GitHub][13])                              |
| SFC                                   | `src/sparse_feature_composition.py`   | SAE feature profile、max-pool composition、activation hook | High as negative evidence   | 实现显示静态 coefficient 注入；与失败现象一致。([GitHub][14])                                                   |
| SAE                                   | `src/sae_decomposition.py`            | SAE encoder/profile                                      | Medium                      | profile 使用 signed mean coefficients；与 SFC interference 假设存在不一致。([GitHub][15])                  |
| FLC                                   | `src/functional_composition.py`       | activation-space least-squares + truncated SVD merge     | Medium as negative evidence | 实验显示能量保留低、下游崩塌。([GitHub][16])                                                                  |
| BCFF                                  | `src/cross_factor_fusion.py`          | cross-factor fusion                                      | High as bug evidence        | `Y_target = y_1 + y_2` 使 cross terms 被自目标压成零，方法退化。([GitHub][17])                               |
| 训练脚本                                  | `scripts/train_domain_lora.py`        | domain LoRA training                                     | High                        | MMLU split override 使用 test 训练，造成 leakage 风险。([GitHub][18])                                    |
| 评估脚本                                  | `scripts/eval_domain_accuracy.py`     | domain evaluation                                        | High                        | seed、dataset sampling、MCQ extraction 需要修。([GitHub][19])                                        |
| configs                               | `configs/domains.yaml`                | model/training/domain/eval config                        | High                        | config 声称 base model `Qwen/Qwen3-8B`、LoRA rank 16、target modules 等。([GitHub][20])              |
| logs                                  | `logs/`, `results-synced/`            | SFC/FLC/BCFF logs and JSON                               | High                        | 负面现象、tautology、energy collapse 的主要证据。([GitHub][21])                                            |
| Text2Subspace pilot results           | `results/text2subspace_*`             | low-rank policy-head pilot                               | Medium                      | 不是最终 text-conditioned LoRA generation。([GitHub][22])                                           |
| tests                                 | `tests/test_text2subspace.py`         | 测试入口                                                     | Low/Medium                  | 目前测试覆盖不足以保护 split/metric/result reliability。([GitHub][23])                                     |

## 当前仓库试图解决的问题

**Confirmed evidence, confidence high:** README 当前叙事是 adapter composition：多个 LoRA/adapter 组合时存在 rank bottleneck，静态 merge 在组合秩超过 rank budget 时会丢信息，BAC 试图通过 bottleneck-aware residual routing 保留关键方向。([GitHub][1])

**Likely evidence, confidence medium:** 历史上项目经历过至少四条路线：

1. **GrassMerge / LoRA algebra / BAC**：静态几何 merge + rank bottleneck 叙事。
2. **SFC**：把 LoRA 差分映射到 SAE feature coefficients 后做 max-pool composition。
3. **FLC**：用 calibration activation least-squares 找 rank-r functional merge。
4. **BCFF**：cross-factor terms `B1A2/B2A1`，试图捕捉跨 adapter 组合。
5. **Text2Subspace pilot**：但当前实验文档承认只是 policy-head proxy，不是真正 text-to-adapter generation。([GitHub][3])

## 主线 / 历史遗留 / dead code 初判

| Category           | Files                                                                                                                                                                    | Reason                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| 当前可作为 evidence 的主线 | `PROGRESS.md`, `RESULTS_REPORT.md`, `EXPERIMENTS.md`, `src/lora_algebra.py`, `src/rank_bottleneck.py`, `scripts/eval_domain_accuracy.py`, `scripts/train_domain_lora.py` | 包含主要 claim、主要结果、主要训练/评估路径                   |
| 应作为失败证据保留          | `src/sparse_feature_composition.py`, `src/functional_composition.py`, `src/cross_factor_fusion.py`, SFC/FLC/BCFF logs                                                    | 不应删除负面结果；它们是新方法约束来源                         |
| 可能 dead / stale    | `README_RUN.md` 指向的 `methods/05_text2subspace/...`                                                                                                                       | 该路径未在可见根目录中确认，应标记 stale 或补上传。([GitHub][24]) |
| 会影响实验结论            | train/eval split、metric extraction、baseline implementations、aggregation scripts、checkpoint loading                                                                       | 当前可靠性瓶颈主要在这些位置                              |

---

# 2. Result Reliability Audit

说明：`Verified` = 仓库日志/代码/文档能互相对上；不是我本地复现。

| Result ID         | Result Name                | Dataset                                 | Metric                       |                                               Claimed Value |                                                                         Logged Value | Config        | Seed    | Command       | Checkpoint       | Status                | Reliability                          | Issue                                                                                    |
| ----------------- | -------------------------- | --------------------------------------- | ---------------------------- | ----------------------------------------------------------: | -----------------------------------------------------------------------------------: | ------------- | ------- | ------------- | ---------------- | --------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------- |
| R-GM-0407         | GrassMerge 8-domain report | Qwen3-8B, 8 domains, 200 samples/domain | accuracy                     | GrassMerge mean .6883 vs TA .6642 / TIES .6663 / DARE .6693 |                                                                       Same in report | Partial       | Missing | Missing       | Missing          | Partially Verified    | low                                  | 正面只相对内部 baselines；相对 base 平均 -0.0083；缺 seed/command/checkpoint。([GitHub][8])             |
| R-GM-base         | GrassMerge vs base         | Same                                    | accuracy                     |                      Beats base 16/30, mean -0.0083 vs base |                                                                                 Same | Partial       | Missing | Missing       | Missing          | Partially Verified    | low                                  | 这不是强 positive；更像 “static merge sometimes helps but not robust”。([GitHub][8])             |
| R-T2S-pilot       | Text2Subspace pilot        | Pilot                                   | action match / acc / utility |                  action-match .775, acc .400, utility .3848 |                                                                                 Same | Partial       | Missing | Runbook stale | Missing          | Unclear               | unusable                             | 文档承认只是 low-rank policy-head proxy，不是真正 text-conditioned adapter generation。([GitHub][3]) |
| R-SFC-pilot       | SFC sparsity / FDS pilot   | SAE feature profiles                    | sparsity / FDS               |                                             Sparse features |                                                            JSON reports low sparsity | Partial       | Missing | Missing       | Missing          | Partially Verified    | medium as diagnostic / low as method | 支持“feature profile 可计算”，不支持 downstream claim；FDS 定义在不同文档中语义不一致。([GitHub][25])            |
| R-SFC-downstream  | SFC pair eval              | 50-sample domain pairs                  | accuracy                     |                             SFC fails or worse than TA/TIES |                     math+science SFC .00/.76 vs TA .00/.84; science+philosophy worse | Partial       | Missing | Log exists    | Adapters unknown | Partially Verified    | medium as negative                   | 样本少、无多 seed，但作为“静态 feature steering 失败”证据较强。([GitHub][26])                               |
| R-FLC-energy      | FLC energy retention       | Pair merges                             | retained energy              |                                   18–20% retained, collapse |                                                          Logs show ~.13–.20 retained | Partial       | Missing | Log exists    | Unknown          | Partially Verified    | medium                               | 支持“rank-r static functional merge 丢大量能量”；部分 philosophy calibration 为 0 污染。([GitHub][27]) |
| R-FLC-acc         | FLC downstream collapse    | 50-sample pairs                         | accuracy                     |                            medical .66→.16, science .74→.02 |                                                                Same in logs/progress | Partial       | Missing | Log exists    | Unknown          | Possibly Contaminated | medium as negative                   | 负面强，但部分 run calibration 数据缺失，应重跑确认。([GitHub][27])                                        |
| R-BCFF-tautology  | BCFF coeffs                | Pair merges                             | learned coefficients         |                                          coeffs ≈ [1,1,0,0] |                                                      Logs show mean_coeffs=[1,1,0,0] | Code confirms | Missing | Log exists    | Unknown          | Verified              | high as bug                          | `Y_target = y1+y2` 使 cross terms 自然被压成 0；不是有效机制。([GitHub][17])                           |
| R-BCFF-downstream | BCFF pair eval             | 50-sample pairs                         | accuracy                     |        Sometimes improves one domain but worse than TA/TIES | e.g. math+science BCFF mean .42 vs TIES .43; medical+science BCFF .70 vs TA/TIES .74 | Partial       | Missing | Log exists    | Unknown          | Partially Verified    | low as positive / medium as negative | 分数主要反映 TA-like merge；不能证明 cross-factor mechanism。([GitHub][28])                          |
| R-single-LoRA     | Individual domain adapters | 50/200 samples                          | accuracy                     |                        Many adapters weak; science positive |                                            Report/progress both显示 science 较强，math 极弱 | Partial       | Missing | Missing       | Unknown          | Partially Verified    | medium                               | 强烈说明必须做 adapter reliability calibration，不能默认每个 adapter 都该被合并。([GitHub][8])               |
| R-BAC-readme      | BAC solves rank bottleneck | Synthetic/claimed                       | CRS / rank / accuracy        |                 “routes bottleneck directions <5% overhead” |                                                             没看到完整可信 downstream table | README only   | Missing | Missing       | Missing          | Missing Log           | unusable                             | README claim 目前不能作为结果证据。([GitHub][1])                                                    |
| R-review-fixes    | Eval pipeline fixes        | Historical review                       | N/A                          |                             Multiple bugs fixed over rounds |                                       Auto review records fixes and remaining issues | Partial       | N/A     | N/A           | N/A              | Partially Verified    | medium                               | 作为“项目曾多次被 eval bug 污染”的风险证据。([GitHub][9])                                                |

## Reliability verdict

| Evidence Type                   |                        Use As Strong Evidence? |               Use As Signal? | Confidence |
| ------------------------------- | ---------------------------------------------: | ---------------------------: | ---------- |
| BCFF tautology bug              |                                            Yes |                          Yes | high       |
| Train/eval split leakage risk   |                                            Yes |                          Yes | high       |
| SFC/FLC/BCFF negative direction | No for paper numbers; Yes for method diagnosis |                          Yes | medium     |
| GrassMerge positive report      |                                             No |          Yes, low-confidence | low        |
| Text2Subspace pilot             |                                             No | Yes, as “proxy insufficient” | low        |
| README/BAC theoretical claim    |                                             No |   Yes, as intended direction | low-medium |

---

# 3. 代码正确性审查：Suspected Bug Table

| Priority | File                                               | Function/Class                           | Code Region              | Suspicion                                                                        | Evidence                                                                           | How to Verify                                                                  | Proposed Fix for Claude Code                                                                                                     | Expected Effect                                 | Confidence  |
| -------- | -------------------------------------------------- | ---------------------------------------- | ------------------------ | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------- |
| P0       | `scripts/train_domain_lora.py`                     | `SPLIT_OVERRIDES`, `load_domain_dataset` | MMLU split override      | MMLU 用 `test` split 训练；eval 也可能用 test，造成 train/test leakage                      | 代码把 `cais/mmlu` split override 到 `test`；评估脚本也按 split 加载 benchmark。([GitHub][18])   | 新增 `scripts/check_splits.py`，输出每个 domain 的 train/calib/test split 和 sample IDs | 禁止 test split 训练；增加 `configs/splits.yaml`；训练/校准/测试 index manifest；若无 train split，则用 validation/dev 做 calibration，不做 publish test | 清除最严重 contamination；旧 MMLU 相关结果降级为 low/unusable | high        |
| P0       | `src/cross_factor_fusion.py`                       | `bcff_merge`                             | `Y_target = y_1 + y_2`   | BCFF 目标是自洽 tautology，必然学到 `[1,1,0,0]`，cross terms 无效                             | 代码和日志同时显示 target 与 coefficients。([GitHub][17])                                     | 单元测试：随机 A/B 下 ridge coeff 应接近 [1,1,0,0]                                        | 从 main method 删除 BCFF；若保留，只作为 negative ablation；新 gate 必须用 downstream/calibration loss，不用 self-target                            | 防止把 TA scale=1 误包装成创新                           | high        |
| P0       | `src/functional_composition.py` / FLC eval scripts | FLC calibration collection               | calibration text loading | 部分 pair 中 philosophy calibration inputs 为 0，但仍继续 merge/eval                      | 日志显示 philosophy calibration 0 texts。([GitHub][27])                                 | 添加 fail-fast：任一 domain calibration < min_calib_samples 直接失败                    | calibration loader 强制 min samples；日志记录 sample IDs                                                                                | 避免 FLC/CARR calibration 被空数据污染                  | high        |
| P0       | `scripts/eval_domain_accuracy.py`                  | `evaluate_on_benchmark`                  | dataset sampling         | `--seed` 被设置但 sample shuffle 固定 `seed=42`，多 seed 不独立                             | 评估脚本 CLI 设置 seed，但 dataset shuffle 写死 42。([GitHub][19])                            | 用不同 `--seed` 输出 sample IDs；目前应相同                                               | 将 dataset_sample_seed 绑定 args.seed 或显式参数；保存 sample indices                                                                       | multi-seed 才有意义；旧 variance 不可信                  | high        |
| P1       | `scripts/eval_domain_accuracy.py`                  | `extract_answer`, MCQ eval               | regex extraction         | MCQ 用生成文本首字符/regex extraction，容易受 prompt/format 影响                               | 脚本包含多类 regex extraction；历史 review 也记录 MMLU/ARC 格式问题。([GitHub][19])                 | 对同一 MCQ 用 logprob scoring 与 generation scoring 对比                              | 新增 deterministic logprob MCQ scorer；generation eval 只用于 open-ended                                                               | 降低 metric 噪声；减少 false positive/negative         | medium      |
| P1       | `src/sparse_feature_composition.py`                | `SFCExactHook`                           | activation hook          | SFC 对所有 token 注入同一静态 feature offset，缺少输入条件化                                      | hook encode base features 后对 support 加固定 coeffs。([GitHub][14])                     | 对不同 inputs 记录 injected coeff 是否变化；应不变                                          | SFC 不再作为 main method；只保留 as historical ablation 或改为动态 feature diagnostic                                                         | 解释 SFC downstream 失败；避免继续调静态 offset             | high        |
| P1       | `src/sparse_feature_composition.py`                | `compute_interference`                   | interference formula     | 公式假设 non-negative coefficients，但 SAE profile 使用 signed means                     | 代码注释假设非负；SAE profile 返回 signed coeffs。([GitHub][14])                               | 构造 signed coefficients，比较 formula 与实际 decoded residual overlap                 | 改为 signed decoder Gram residual norm，或把 non-negative objective 明确化                                                               | 让 FDS/feature conflict 成为可用诊断，而非 claim          | medium-high |
| P1       | `src/sparse_feature_composition.py`                | SFC-Merge                                | reconstruction code      | 注释承认 input activation vs token index 不匹配，merge reconstruction 不完整                | 代码注释说明 “need A to operate on input activations”。([GitHub][14])                     | 尝试 roundtrip LoRA delta reconstruction；应失败或误差大                                 | Archive SFC-Merge as dead method path                                                                                            | 减少 misleading method surface                    | high        |
| P1       | `src/rank_bottleneck.py`                           | CRS / synthetic rank diagnostics         | adapter rank overlap     | CRS 似乎主要看 left/output subspace，可能忽略 right/input subspace 与 activation covariance | 代码片段显示 synthetic expected rank 基于 `U_i` overlap；LoRA ΔW 实际由左右因子共同决定。([GitHub][13]) | 构造 same-U different-V adapters；若 CRS 不变但 merge error 变，说明诊断不充分                 | 改为 full operator Gram 或 activation-conditioned Gram：`tr(ΔW_i Σ_x ΔW_j^T)`                                                        | 使 rank bottleneck 从几何口号变为可预测风险指标                | medium      |
| P1       | `src/lora_algebra.py`                              | DARE-like implementation                 | random mask seed         | DARE 使用 hardcoded seed 42，baseline variance 与 fairness 受影响                       | 代码中 generator manual seed 42。([GitHub][12])                                        | 多 seed DARE 应产生不同 masks；目前可能固定                                                 | 暴露 `seed` 参数并记录 mask seed；baseline 用多 seed mean/std                                                                              | 避免 baseline 被无意弱化或固定 cherry-pick                | high        |
| P2       | `README_RUN.md`                                    | runbook                                  | stale path               | 指向 `methods/05_text2subspace/...`，可见树中未确认                                        | runbook 路径与根目录不一致。([GitHub][24])                                                   | `test -e methods/05_text2subspace/...`                                         | 更新或 archive runbook；不要让 Claude Code 按 stale path 实现                                                                              | 降低复现混乱                                          | high        |
| P2       | `README.md`, `PROPOSAL.md`, proof docs             | claims                                   | theory/empirical claims  | “provably optimal”, “no one has done this”, “solves trilemma” 等 claim 过强         | Proof audit 已指出 theorem false / unsupported。([GitHub][11])                         | claim-code-result matrix                                                       | 删除或弱化；推迟论文 claim 到最小实验通过后                                                                                                        | 降低 reviewer rejection risk                      | high        |

---

# 4. Claim-Code-Result Matrix

| Claim                                                                                    | Source File                       | Implementation File                                             | Result Evidence                                                 | Status              | Problem                                                                            | Confidence |
| ---------------------------------------------------------------------------------------- | --------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------- | ------------------- | ---------------------------------------------------------------------------------- | ---------- |
| Static LoRA composition suffers rank bottleneck; BAC solves adapter composition trilemma | `README.md`                       | `src/rank_bottleneck.py`, `src/lora_algebra.py`                 | README + code; no strong downstream log found                   | Unsupported         | 理论/诊断可能有价值，但缺 credible empirical matrix；CRS 可能不完整。([GitHub][1])                    | medium     |
| GrassMerge outperforms TA/TIES/DARE                                                      | `RESULTS_REPORT.md`               | `src/lora_algebra.py`                                           | 2026-04-07 report                                               | Partially Supported | 只相对内部 baselines；vs base 平均为负；缺 seed/command/checkpoint。([GitHub][8])               | low        |
| SFC: low-rank LoRA differences map to sparse SAE features                                | `PROPOSAL.md`, `PROOF_AUDIT.md`   | `src/sparse_feature_composition.py`, `src/sae_decomposition.py` | SFC pilot sparsity but no task theorem                          | Unsupported         | Proof audit 指出定理不成立；sparsity 不等于 composability。([GitHub][11])                      | high       |
| SFC max-pool composition is optimal / provably reduces interference                      | `PROPOSAL.md`                     | `src/sparse_feature_composition.py`                             | Proof audit + synthetic counterexample                          | Contradicted        | `PROOF_AUDIT` 明确写 max-pool not optimal。([GitHub][11])                              | high       |
| FLC activation-space LS merge preserves functional behavior                              | `src/functional_composition.py`   | same                                                            | FLC energy retained only ~13–20%, downstream collapse           | Partially Supported | Objective 实现存在，但 rank-r truncation 破坏 functional behavior。([GitHub][16])           | medium     |
| BCFF cross terms capture transfer beyond standard composition                            | `src/cross_factor_fusion.py`      | same                                                            | coeffs ≈ [1,1,0,0]; cross terms unused                          | Contradicted        | Target tautological；method degenerates to TA-like sum。([GitHub][17])               | high       |
| Text2Subspace learns text-conditioned subspace/adapters                                  | `EXPERIMENTS.md`, `README_RUN.md` | missing/stale path                                              | pilot proxy only                                                | Unsupported         | 文档承认不是 true text-conditioned adapter generation。([GitHub][3])                      | high       |
| Evaluation pipeline is reproducible and fair                                             | README/scripts                    | `scripts/eval_domain_accuracy.py`, train scripts                | review history + code issues                                    | Unsupported         | split leakage、fixed sampling seed、MCQ extraction、missing multi-seed。([GitHub][18]) | high       |
| Existing methods already provide NeurIPS-ready positive result                           | README/progress/docs              | all                                                             | Current `PROGRESS.md` says all three attempted methods negative | Contradicted        | 正面结果不足以支撑 main claim；应 pivot。([GitHub][7])                                         | high       |

---

# 5. Phenomenon Ledger

| ID  | Observation                                                          | Type                        | Where Found                         | Setting                         | Metric                | Compared To          | Reliability | What It Suggests                                                                | What It Rules Out                                       | Confidence |
| --- | -------------------------------------------------------------------- | --------------------------- | ----------------------------------- | ------------------------------- | --------------------- | -------------------- | ----------- | ------------------------------------------------------------------------------- | ------------------------------------------------------- | ---------- |
| P1  | TA/TIES often match or beat BCFF/SFC/FLC in current pair tests       | Mixed/Positive for baseline | `PROGRESS.md`, BCFF/SFC/FLC logs    | 50-sample pairs                 | accuracy              | SFC/FLC/BCFF         | medium      | Simple static baselines are strong enough that any method must beat them fairly | “New method can be weak static merge with new name”     | high       |
| P2  | Science adapter / science direction often helps                      | Positive                    | `RESULTS_REPORT.md`, `PROGRESS.md`  | individual/pair                 | accuracy              | base/other domains   | medium      | Some adapter residuals contain real useful information                          | “All adapters are useless”                              | medium     |
| P3  | Math adapter/base scores extremely weak in current setup             | Negative                    | `PROGRESS.md`                       | GSM8K-like math                 | accuracy              | base/adapters        | medium      | Adapter reliability varies strongly by domain                                   | Uniformly merging all adapters                          | high       |
| P4  | GrassMerge old report beats TA/TIES/DARE mean but not base mean      | Mixed                       | `RESULTS_REPORT.md`                 | 30 pair-domain evals            | accuracy              | base, TA/TIES/DARE   | low         | Geometry may help in some cases, but not robust enough                          | Strong SOTA/claim from GrassMerge alone                 | high       |
| P5  | SFC feature sparsity exists, but downstream composition fails        | Negative                    | SFC JSON/logs/progress              | SAE feature profile + pair eval | sparsity/FDS/accuracy | TA/TIES              | medium      | Feature sparsity may be diagnostic, not a safe actuator                         | “Sparse SAE feature max-pool is sufficient”             | high       |
| P6  | SFC uses static feature offset for every token                       | Anomalous/Negative          | `src/sparse_feature_composition.py` | activation hook                 | mechanism             | N/A                  | high        | Missing input/token-conditioned control                                         | Static steering as main method                          | high       |
| P7  | FLC retains only ~13–20% energy and collapses downstream             | Negative                    | FLC logs/progress                   | rank-r merge                    | energy/accuracy       | TA/TIES/base         | medium      | Rank-r static compression destroys important residual directions                | “Better least-squares compression is enough”            | high       |
| P8  | FLC calibration can silently be empty for a domain                   | Anomalous                   | FLC logs                            | philosophy pairs                | calibration count     | N/A                  | high        | Pipeline needs fail-fast sanity checks                                          | Trusting old FLC results as quantitative paper evidence | high       |
| P9  | BCFF learned coefficients ≈ `[1,1,0,0]`                              | Negative/Bug                | code + logs                         | pair merge                      | coefficients          | intended cross terms | high        | Objective is self-targeting, not task-aligned                                   | BCFF as mechanism contribution                          | high       |
| P10 | BCFF sometimes improves a strong domain while hurting another        | Mixed                       | BCFF logs/progress                  | science+philosophy etc.         | accuracy              | TA/TIES/base         | medium      | Uncalibrated composition can exploit strong residual but harm weak/other domain | Global coefficients without reliability gate            | medium     |
| P11 | Static methods do not know when to leave base untouched              | Negative                    | single adapter/base tables          | multiple domains                | accuracy              | base                 | medium      | Need base-preservation / abstention                                             | Always applying a merge                                 | high       |
| P12 | Current result artifacts lack command/config/seed/checkpoint linkage | Anomalous                   | result docs/logs                    | all                             | reproducibility       | N/A                  | high        | Need result registry before paper claim                                         | Any strong empirical claim today                        | high       |
| P13 | MMLU training likely uses test split                                 | Possibly Contaminated       | train script                        | MMLU domains                    | split                 | eval test            | high        | Split control is mandatory before method discovery                              | Trusting MMLU/history/philosophy positives              | high       |
| P14 | Proof audit invalidates SFC theorem and max-pool optimality          | Negative                    | `PROOF_AUDIT.md`                    | theory                          | proof status          | claims               | high        | Method should avoid unsupported theorem-first story                             | “Provably optimal feature composition”                  | high       |
| P15 | Text2Subspace pilot is not true adapter generation                   | Negative/Unclear            | `EXPERIMENTS.md`                    | pilot                           | action match/utility  | full-rank head       | high        | Old Text2Subspace should not be main claim unless implemented                   | Claiming text-to-LoRA novelty                           | high       |
| P16 | Review history repeatedly finds eval/baseline/proof issues           | Anomalous                   | `AUTO_REVIEW.md`                    | whole project                   | review score          | N/A                  | medium      | Academic integrity must prioritize audit-first pipeline                         | Rushing to paper with current numbers                   | high       |

---

# 6. Design Constraints

| Constraint ID | Derived From Observation | Constraint Type    | Meaning                                                              | Implication for New Method                                                                   | Confidence |
| ------------- | ------------------------ | ------------------ | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------- |
| C1            | P1                       | Must Preserve      | Strong static baselines must remain as baselines/backbones           | New method must beat TA/TIES/GrassMerge under same split/env; cannot hide them               | high       |
| C2            | P2/P3                    | Must Stabilize     | Adapter competence is heterogeneous                                  | Add reliability calibration and base fallback                                                | high       |
| C3            | P5/P6                    | Must Avoid         | Static feature offset is unsafe                                      | No input-independent SAE max-pool injection as main method                                   | high       |
| C4            | P7                       | Must Avoid         | Global rank-r compression loses important directions                 | Do not force all adapter effects into one static rank-r merge                                | high       |
| C5            | P9                       | Must Fix           | Self-targeted objectives can look mathematically clean but vacuous   | New objective must optimize downstream/calibration likelihood, not reconstruct its own merge | high       |
| C6            | P10/P11                  | Must Control       | Applying a helpful adapter can hurt another domain/base              | Gate must decide when to abstain or route only local residual                                | high       |
| C7            | P12/P13                  | Must Control       | No method claim before split/seed/checkpoint manifest                | Build result registry and split checks first                                                 | high       |
| C8            | P14                      | Must Not Claim     | SFC proof claims are invalid                                         | Remove “provably optimal”, “low-rank→sparse” claims                                          | high       |
| C9            | P15                      | Must Differentiate | Text2Subspace is not implemented as claimed and has close prior work | Either implement true method or archive; do not claim text-conditioned generation            | high       |
| C10           | P4/P7                    | Must Explain       | Geometry/rank diagnostics may be useful but insufficient             | Use rank/conflict diagnostics to decide routing, not to justify static merge                 | medium     |
| C11           | P1/P10                   | Must Test          | Method must prove it is not just existing positive fragment          | Required A/B/C: existing best fragment only vs new without mechanism vs full new             | high       |
| C12           | P16                      | Must Generalize    | Reviewer risk is dominated by reproducibility and baselines          | Official baselines and confidence intervals are part of method validation                    | high       |

---

# 7. Negative-to-Insight Analysis

| Negative Observation                              | Failed Assumption                                                | Why the Assumption Failed                                                                       | What Mechanism Is Missing                            | New Design Requirement                                                                   |
| ------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| SFC downstream fails despite feature sparsity     | Sparse SAE feature profile is enough for composition             | Static feature coefficients ignore input, token, layer, SAE reconstruction error, and task loss | Input-conditioned arbitration and error-aware gating | Feature information may be diagnostic only; actuator must be dynamic and task-calibrated |
| SFC max-pool proof invalid                        | Max-pool resolves feature conflict optimally                     | Counterexample/proof audit shows max-pool can be worse than sum; no task theorem                | Objective alignment to downstream behavior           | Delete max-pool theorem; use learned/calibrated gate                                     |
| FLC energy retention 13–20% and accuracy collapse | Activation LS + truncated SVD can compress multi-adapter effects | Rank-r truncation discards residual directions that matter for task behavior                    | Residual routing beyond static rank budget           | Preserve compatible shared directions statically; route conflict residual dynamically    |
| BCFF coefficients `[1,1,0,0]`                     | Cross terms learn transfer                                       | Target is `y1+y2`, so cross terms are mathematically unnecessary                                | Supervised calibration objective                     | Any cross/residual/gate must be trained against held-out task loss/logprob               |
| Weak math / heterogeneous adapters                | Every domain adapter should be included                          | Some adapters are not better than base or not trained well                                      | Reliability calibration / abstention                 | Gate off unreliable adapters; require adapter quality manifest                           |
| TA/TIES strong vs new variants                    | More complex merge automatically improves                        | Complexity did not address true bottleneck: when to use which residual                          | Conflict-aware selection, not just merge algebra     | New method must outperform existing best fragment under preregistered selection          |
| Empty calibration texts in FLC                    | Pipeline can continue with partial data                          | Missing fail-fast checks lets invalid runs produce numbers                                      | Data sanity enforcement                              | Evaluation must stop if calibration/test data invalid                                    |
| MMLU split leakage risk                           | Domain LoRA training/eval splits are safe                        | Train script uses test split for MMLU                                                           | Split registry and sample ID audit                   | No result can be strong evidence until split isolation is fixed                          |
| Text2Subspace proxy                               | Low-rank policy head approximates text-to-adapter generation     | It does not generate checkpoint-space LoRA/subspace                                             | True adapter-space mechanism or archive              | Do not claim text-to-subspace unless implemented and compared to Text-to-LoRA            |

---

# 8. Method Synthesis Table

| Evidence Fragment          | Source in Repo      | What It Reveals                                            | Generalized Principle                             | Use in New Method?           | How to Transform It                                                                             |
| -------------------------- | ------------------- | ---------------------------------------------------------- | ------------------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------- |
| TA/TIES often strong       | Logs/progress       | Simple static composition is hard to beat                  | Keep static compatible merge as baseline/backbone | Yes, but not as final method | Use as `static_candidate`; compare against it directly                                          |
| Science direction helpful  | Single/pair results | Some adapters are reliable in some regions                 | Reliability is domain/input dependent             | Yes                          | Learn/calibrate `r_i(x)` reliability and gate science-like residual only when useful            |
| Weak math/domain adapters  | Progress            | Some adapters harm or add no signal                        | Need abstention                                   | Yes                          | Add base gate and adapter reliability threshold                                                 |
| SFC feature profile        | SFC code/results    | Feature overlap may diagnose conflict but not solve it     | Sparse features can be diagnostics                | As diagnostic only           | Convert FDS/feature overlap into conflict features for router; no static max-pool actuator      |
| FLC activation LS          | FLC code/logs       | Activation deltas expose functional conflict               | Activation-space diagnostics are useful           | Yes, transformed             | Use activation deltas to compute conflict/routing; do not truncate all into rank-r static merge |
| BCFF cross terms           | BCFF code/logs      | Self-targeted cross factor objective is vacuous            | Objective must be task-aligned                    | No as method                 | Archive BCFF; use its failure as test case for objective design                                 |
| BAC/rank bottleneck        | README/rank code    | Static rank budget may explain why merges drop information | Bottleneck directions need conditional handling   | Yes, transformed             | Correct CRS/Gram; route bottleneck residuals dynamically                                        |
| GrassMerge positive report | Results report      | Geometry may help some cases but lacks robustness          | Shared subspace can be safe static component      | Baseline/static candidate    | Use as one static candidate; not main claim                                                     |
| Text2Subspace pilot        | Experiments         | Current “text” route is proxy, not real generation         | Avoid overclaiming text-conditioned LoRA          | Archive or baseline          | Not main unless true adapter generator is implemented                                           |
| Review history             | Auto review         | Evaluation bugs repeatedly dominate                        | Audit-first pipeline is mandatory                 | Yes                          | First tasks must fix split/metric/logging before method claims                                  |

---

# 9. Missing Mechanism Diagnosis

1. **Missing Mechanism Name:**
   **Reliability-Calibrated, Input-Conditioned Conflict Arbitration**

2. **One-Sentence Diagnosis:**
   当前方法共同假设“多个 adapter 可以被一个全局静态 merge / 静态低秩子空间 / 静态 feature offset 表示”，但结果显示真正缺失的是一个按输入、层、token 和 adapter 可靠性来决定**何时保留 base、何时用共享静态方向、何时路由冲突残差方向**的机制。

3. **Evidence From Positive Results:**
   Science 相关 residual 有时明显有用；TA/TIES 静态方向在部分 pair 中强，说明并非所有合并都无效，而是需要识别“何时可安全合并”。Confidence: medium.

4. **Evidence From Negative Results:**
   SFC 静态 offset 失败、FLC rank-r truncation 崩塌、BCFF cross terms 被 tautological objective 清零，三者都说明“形式更复杂的静态组合”没有解决核心问题。Confidence: high.

5. **Evidence From Unstable Results:**
   GrassMerge 老报告有正面但相对 base 不稳定；BCFF 对 science 有时增强但伤害另一域；这说明有用信号存在，但缺少可靠性和冲突控制。Confidence: medium.

6. **Evidence From Failed Ablations:**
   max-pool SFC、FLC、BCFF 均不能成为主方法；它们分别暴露了 feature sparsity、rank compression、cross-factor algebra 与 downstream objective 的错配。Confidence: high.

7. **Why Existing Method Cannot Solve It:**
   现有方法大多输出一个固定 merged adapter 或固定 activation offset；它们无法对不同输入判断 adapter 是否可靠，也无法在冲突高时选择 base 或只路由某个 residual。

8. **Why Simple Tuning Cannot Solve It:**
   调 rank、threshold、ridge λ、merge scale 仍然是在寻找一个全局静态点；负面现象来自机制缺失，不是单个超参偏差。

9. **Why Existing Best Positive Fragment Is Insufficient:**
   TA/TIES/GrassMerge 正面片段不能解释为什么 math/medical/philosophy 会被伤害，也不能提供 per-input abstention；它们只能作为 static candidate 或 baseline。

10. **What New Mechanism Must Do:**
    新机制必须：

    1. 校准每个 adapter 对输入/任务的可靠性；
    2. 估计 adapter residual 之间的冲突；
    3. 保留安全共享方向；
    4. 对冲突方向进行动态 residual routing；
    5. 在不确定或弱 adapter 时回退 base；
    6. 用 held-out downstream loss/logprob 训练，而不是自目标 reconstruction。

11. **Confidence:**
    **medium** as mechanism diagnosis；**medium-low** as final method success prediction.

---

# 10. New MAIN METHOD PATH

## New MAIN METHOD PATH

1. **Method Name Placeholder:**
   **CARR: Conflict-Aware Reliability-Gated Residual Routing**

2. **One-Sentence Core Idea:**
   把多 adapter 组合拆成“安全共享静态方向 + 输入条件化冲突残差路由”：先用可靠性校准判断哪个 adapter 在当前输入上可信，再用 conflict/rank-bottleneck 诊断决定是否路由其 residual，失败时回退 base。

3. **Core Missing Mechanism It Adds:**
   Reliability-calibrated, input-conditioned conflict arbitration.

4. **What Phenomena It Explains:**

   * TA/TIES 强：共享兼容方向可以静态合并。
   * Science residual 有用：可靠 adapter 应被 gate 激活。
   * Math/weak adapter 伤害：可靠性低时应 gate off。
   * FLC collapse：冲突残差不能被全局 rank-r 压缩。
   * SFC failure：静态 feature offset 缺少输入条件化。
   * BCFF tautology：必须用 downstream calibration loss，而非 self-target.

5. **What Negative Results It Fixes:**
   它不再把所有 adapter 一次性压入一个静态 merge；不再对所有 token 注入同一 offset；不再用 `y1+y2` 自目标学 cross terms；不再默认每个 adapter 都有帮助。

6. **What Existing Positive Signals It Generalizes:**
   它把 TA/TIES/GrassMerge 的“兼容方向可静态合并”推广为 CARR 的 static safe component，把 science adapter 的局部正面推广为 reliability-conditioned residual route。

7. **Why Existing Best Path Is Not Enough:**
   现有最好片段最多说明某个静态 merge 在某些 pair 上有效；CARR 的 claim 是“什么时候不该合并、什么时候只该路由某个 residual”，这是机制级不同。

8. **Core Mechanism:**

   * **Reliability profiler:** 在 held-out calibration 上估计 adapter `i` 对输入 `x` 的可靠性 `r_i(x)`。
   * **Conflict estimator:** 基于 activation deltas `d_i^l,t = ΔW_i^l h_l,t` 或 corrected operator Gram 估计 adapter 间冲突 `C_ij^l,t(x)`。
   * **Static compatible component:** 用 TA/TIES/GrassMerge 或 corrected shared subspace 得到安全静态方向。
   * **Residual router:** 对高冲突/高 bottleneck 方向，不压缩成一个 rank-r merge，而是 top-k 动态路由。
   * **Base fallback:** 当 reliability 低或 conflict 高且不确定时，保留 base。

9. **New Objective / Loss:**
   `L_total = L_task + λ_base L_KL_base + λ_conf L_conflict + λ_sparse L_sparse + λ_cal L_calibration + λ_rank L_residual_budget`

10. **New Architecture or Module:**
    新增 `ConflictAwareResidualRouter`，包含 adapter reliability features、conflict features、top-k gate、base gate、optional layer/token gate。

11. **New Training Procedure:**
    不是重训所有 LoRA；先冻结 base 和 domain adapters，只训练小 router / gating head。第一阶段做 oracle/reliability upper bound，第二阶段训练 learned gate，第三阶段多 seed 小规模验证。

12. **New Evaluation Protocol:**
    必须有 A/B/C：
    A. Existing Best Positive Fragment Only
    B. New MAIN METHOD Without New Mechanism
    C. Full CARR
    并在相同 split、相同 checkpoint、相同 seeds、相同 MCQ logprob metric 下比较。

13. **What Existing Components It Reuses:**
    base model、domain LoRA checkpoints、TA/TIES/GrassMerge baseline code、FLC activation delta collection、rank bottleneck diagnostics after fix、evaluation harness after fix。

14. **What Existing Components It Deletes:**
    BCFF 作为 main method；SFC max-pool optimal theorem；Text2Subspace proxy as main claim；stale runbook path。

15. **What Existing Components It Rewrites:**
    split/eval/seed logging；rank bottleneck CRS；SFC/FDS as diagnostic only；FLC calibration loader fail-fast；README/paper claims。

16. **What Existing Components It Keeps Only as Ablation:**
    SFC static steering、FLC static LS merge、BCFF tautological merge、BAC static variant、GrassMerge.

17. **What Existing Components It Keeps Only as Baseline:**
    base model、individual LoRA、TA、TIES、DARE、GrassMerge、possibly KnOTS/RegMean official implementations.

18. **Why This Is Not Merely the Existing Best Path:**
    Existing best path outputs one static adapter. CARR outputs an input-conditioned routing decision with explicit reliability, conflict, and base abstention. Static merge is only a candidate inside CARR, not the method.

19. **Why This Could Produce Real Positive Results:**
    Current logs show useful residuals exist but are heterogeneous; a method that selectively applies them should avoid the dominant failure mode: applying harmful or conflicting updates globally.

20. **Why This Is Mechanism-Level Different from Prior Work:**
    The intended novelty is not “dynamic LoRA routing” alone, because that has close prior work. The defensible novelty must be: **rank/conflict diagnostics + reliability calibration + base-preserving residual routing derived from static merge failure modes**. This must be experimentally differentiated from LoRA-Flow/MixLoRA/AdapterFusion/LoraHub.

21. **Main Risk:**
    High novelty risk due to dynamic adapter/LoRA composition literature; high empirical risk if current adapters are too weak or router collapses to domain classifier.

22. **Minimal Falsification Experiment:**
    On two or three non-leaky domains with verified useful individual adapters, compare A/B/C over 3 seeds. If Full CARR does not beat A and B with gate diagnostics showing nontrivial reliability/conflict use, stop or pivot.

23. **Confidence:**
    **medium-low** for success; **high** that this is the right missing mechanism to test next.

---

# 11. Formal Method Description

## 11.1 Problem Setup

Given base model `M0` with frozen weights and `n` domain/task LoRA adapters `{A_i}`. For layer `l`, adapter `i` induces low-rank delta:

[
\Delta W_i^l = B_i^l A_i^l.
]

For input `x`, token `t`, hidden state `h_{l,t}`, adapter residual is:

[
d_{i}^{l,t}(x) = \Delta W_i^l h_{l,t}.
]

Goal: compose multiple adapters without global static interference, while preserving base behavior when no adapter is reliable.

## 11.2 Existing Method Failure

Static merge methods attempt:

[
\Delta W_{\text{merge}}^l = \mathcal{M}(\Delta W_1^l, \dots, \Delta W_n^l)
]

and apply it uniformly to all inputs. SFC applies a static feature coefficient offset. FLC compresses functional behavior into rank-r static updates. BCFF reconstructs a self-target. These fail because the correct update is conditional on input, adapter competence, and residual conflict.

## 11.3 New Insight

Adapter composition should be treated as **conditional residual decision-making**, not only matrix averaging. The method must decide:

[
\text{use base? use static safe merge? route adapter residual?}
]

based on reliability and conflict.

## 11.4 Method Overview

CARR decomposes composition into:

1. **Static compatible component**
   A safe low-conflict merge `ΔW_static`, e.g. TA/TIES/GrassMerge selected on calibration only.

2. **Conflict residual component**
   Per adapter residuals projected to bottleneck/conflict directions:

[
\tilde d_i^{l,t}(x) = P_{\text{conflict}}^l d_i^{l,t}(x)
]

3. **Reliability/conflict gate**
   A small router outputs:

[
g_0(x), g_{\text{static}}(x), g_1^{l,t}(x), \dots, g_n^{l,t}(x)
]

where `g0` is base fallback.

4. **Composed hidden update**

[
h'*{l,t} = h*{l,t}

* g_{\text{static}}(x) d_{\text{static}}^{l,t}(x)
* \sum_{i=1}^{n} g_i^{l,t}(x)\tilde d_i^{l,t}(x).
  ]

## 11.5 Objective

[
L_{\text{total}} =
L_{\text{task}}

* \lambda_{\text{base}} L_{\text{KL-base}}
* \lambda_{\text{conf}} L_{\text{conflict}}
* \lambda_{\text{sparse}} L_{\text{sparse}}
* \lambda_{\text{cal}} L_{\text{cal}}
* \lambda_{\text{rank}} L_{\text{budget}}.
  ]

Where:

[
L_{\text{task}} = \mathbb{E}*{(x,y)\in D*{\text{calib}}}
[-\log p_{\text{CARR}}(y|x)]
]

aligns with downstream calibration, fixing BCFF’s self-target problem.

[
L_{\text{KL-base}} =
\mathbb{E}*{x\in D*{\text{control}}}
\mathrm{KL}(p_{0}(\cdot|x)|p_{\text{CARR}}(\cdot|x))
]

prevents global harm and addresses weak-adapter failure.

[
L_{\text{conflict}} =
\mathbb{E}
\sum_{l,t}
\sum_{i<j}
g_i^{l,t}(x)g_j^{l,t}(x)
\max(0, C_{ij}^{l,t}(x))
]

discourages routing mutually conflicting residuals.

[
L_{\text{sparse}} =
\mathbb{E}\sum_{l,t}|g^{l,t}(x)|_1
]

or top-k entropy penalty forces selective routing.

[
L_{\text{cal}} =
\sum_i
\mathrm{Brier}(r_i(x), \mathbf{1}[\text{adapter } i \text{ correct or improves logprob}])
]

makes reliability meaningful.

[
L_{\text{budget}}
]

limits routed residual norm/rank to prevent “just use all adapters”.

## 11.6 Algorithm

**Algorithm: CARR**

**Input:**
Base model `M0`; frozen adapters `{A_i}`; calibration sets `{D_i^cal}`; held-out test sets `{D_i^test}`; control set `D0`; rank/residual budget `r_b`; static baseline set `{TA,TIES,GrassMerge}`.

**Output:**
A composed inference system with static component and trained residual router.

**Steps:**

1. **Validate data and checkpoints**
   Verify no train/calib/test overlap; log dataset IDs, split names, sample indices, checkpoint hashes.

2. **Profile adapter reliability**
   For each adapter `i`, compute base vs adapter logprob/accuracy advantage on calibration and control prompts. Store `r_i`, per-domain reliability, and failure cases.

3. **Compute conflict diagnostics**
   For each layer and adapter pair, collect activation deltas `d_i^{l,t}` on calibration inputs. Estimate conflict using activation-conditioned Gram:

   [
   G_{ij}^l =
   \mathbb{E}*{h\sim D*{\text{calib}}}
   \langle \Delta W_i^l h, \Delta W_j^l h \rangle.
   ]

   Derive conflict/bottleneck projectors `P_conflict^l`.

4. **Build static compatible candidate**
   Train/evaluate TA/TIES/GrassMerge on calibration only. Select a preregistered static candidate or report all.

5. **Train CARR router**
   Freeze base and adapters. Train small gate on calibration objective `L_total`.

6. **Evaluate A/B/C**
   Run:

   * A: Existing Best Positive Fragment Only
   * B: CARR Without Reliability/Conflict Mechanism
   * C: Full CARR

7. **Log mechanism evidence**
   Log gate entropy, base gate rate, adapter activation rate, conflict score before/after routing, residual norm, base KL, per-domain reliability calibration, and seed variance.

## 11.7 Required Ablations

| Ablation                             | Purpose                                         |
| ------------------------------------ | ----------------------------------------------- |
| Static TA/TIES/GrassMerge only       | Prove CARR is not just static merge             |
| Router without reliability           | Test reliability contribution                   |
| Router without conflict penalty      | Test conflict mechanism                         |
| Router without base KL               | Test base-preservation                          |
| Uniform gate                         | Test if dynamic routing matters                 |
| Oracle domain gate                   | Upper bound: is routing useful if labels known? |
| Existing best positive fragment only | Required anti-positive-anchoring control        |
| Full CARR                            | Main method                                     |

---

# 12. Related Work and Novelty Risk

| Paper              | Year / Venue | Code              | Mechanism                                                                | Why Close                                    | Difference from CARR                                                                                                      | Novelty Risk                           | Required Differentiation Experiment                                                                   |
| ------------------ | ------------ | ----------------- | ------------------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Task Arithmetic    | ICLR 2023    | Available         | Adds task vectors from finetuned models                                  | Static composition baseline                  | CARR uses static vector only as safe component; adds reliability/conflict dynamic residual route                          | medium                                 | Full CARR vs TA under same adapters/splits/seeds.([arXiv][29])                                        |
| TIES-Merging       | NeurIPS 2023 | Official repo     | Trim/elect/merge to resolve static interference                          | Strong static merging baseline               | CARR resolves conflict by conditional routing rather than one merged vector                                               | high as baseline risk                  | Must run official or faithful TIES and show CARR beats it with CI.([GitHub][30])                      |
| RegMean            | NeurIPS 2022 | Available         | Closed-form layer-wise regression mean minimizing prediction differences | Very close to FLC-style activation LS        | CARR uses activation diagnostics but avoids global static rank compression                                                | medium-high                            | Compare to RegMean/activation merge on same calibration data.([arXiv][31])                            |
| AdapterFusion      | 2021         | Available         | Learns composition over adapters after adapter training                  | Dynamic adapter composition                  | CARR adds reliability calibration, conflict/rank residual diagnostics, and base fallback                                  | high                                   | Need show conflict/rank-aware routing beats vanilla learned adapter fusion.([arXiv][32])              |
| LoraHub            | 2023         | Available         | Gradient-free composition of LoRA modules on few examples                | Composes LoRA modules using calibration data | CARR is input/layer/token conditional and conflict-aware, not one task-level coefficient vector                           | high                                   | Compare task-level optimized LoRA weights vs CARR on held-out multi-domain mixtures.([Replicate][33]) |
| LoRA-Flow          | 2024/2025    | Available/claimed | Token-level dynamic LoRA fusion with small gate                          | Very close dynamic LoRA routing              | CARR must differentiate via reliability calibration + conflict/bottleneck residual projectors + base-preserving objective | very high                              | Ablate conflict/reliability features; compare to LoRA-Flow official on same tasks.([arXiv][34])       |
| MixLoRA / MoE-LoRA | 2024         | Available/claimed | Multiple LoRA experts with router/load-balancing                         | Close top-k expert routing                   | CARR composes frozen domain adapters with explicit merge-failure diagnostics and base fallback                            | high                                   | Compare against MixLoRA trained with same data/compute; report router/load balance.([arXiv][35])      |
| KnOTS              | 2024/2025    | Available         | SVD-aligns LoRA weights into shared space before merging                 | Close to GrassMerge/BAC geometry             | CARR can use KnOTS as static candidate but adds dynamic residual arbitration                                              | high as static geometry baseline       | Official KnOTS baseline required.([arXiv][36])                                                        |
| Text-to-LoRA       | 2024/2025    | Available/claimed | Hypernetwork generates LoRA from task description                        | Close to repo name/Text2Subspace             | CARR is not text-to-adapter generation unless extended; do not claim this                                                 | very high if using Text2Subspace title | If keeping text angle, compare directly and prove true checkpoint-space generation.([arXiv][37])      |
| LoRA               | ICLR 2022    | Official          | Low-rank adaptation                                                      | Base adapter technology                      | CARR composes existing LoRAs; not replacing LoRA                                                                          | low                                    | Use standard LoRA baselines.([OpenReview][38])                                                        |

## Novelty verdict

CARR has **high novelty risk** if framed as “dynamic LoRA routing” alone. It can be defensible only if the paper’s contribution is framed as:

> Static adapter merging fails because it lacks reliability-calibrated conflict arbitration; CARR introduces a falsifiable conflict/reliability residual routing mechanism, with diagnostics showing when and why it beats static merge and vanilla dynamic LoRA routing.

Do **not** claim:

* “No one has done dynamic LoRA composition.”
* “First text-to-subspace method.”
* “Provably optimal feature composition.”
* “Solves adapter composition trilemma.”
* “SOTA” unless official baselines pass.

---

# 13. Keep / Delete / Rewrite / Archive Plan

| Item                              | Type            | File / Directory / Claim / Experiment       | Current Role            | Problem Under New MAIN PATH             | Action                                    | Reason                                                   |
| --------------------------------- | --------------- | ------------------------------------------- | ----------------------- | --------------------------------------- | ----------------------------------------- | -------------------------------------------------------- |
| Base model + domain adapters      | Code/artifacts  | checkpoint paths from configs/logs          | Core assets             | Need manifest/reliability audit         | KEEP                                      | Required for CARR                                        |
| `configs/domains.yaml`            | Config          | `configs/domains.yaml`                      | Main config             | Split/eval info insufficient            | REWRITE                                   | Add split manifest, calib/test separation, sample seed   |
| TA/TIES/DARE                      | Baseline code   | `src/lora_algebra.py`                       | Static baselines        | DARE seed issue                         | KEEP ONLY AS BASELINE                     | Must compare fairly                                      |
| GrassMerge                        | Method/baseline | `src/lora_algebra.py`, `RESULTS_REPORT.md`  | Old positive path       | Positive unreliable as main             | KEEP ONLY AS BASELINE / ABLATION          | Static candidate, not final                              |
| BAC rank bottleneck               | Method code     | `src/rank_bottleneck.py`                    | README main method      | CRS may be incomplete; no strong result | REWRITE / MERGE INTO NEW METHOD           | Use corrected conflict/bottleneck diagnostic             |
| SFC exact hook                    | Old method      | `src/sparse_feature_composition.py`         | Feature composition     | Static offset fails                     | KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE | Useful as ablation showing static feature steering fails |
| SFC-Merge                         | Old method      | `src/sparse_feature_composition.py`         | Weight reconstruction   | Incomplete / conceptually wrong         | ARCHIVE                                   | Avoid misleading method surface                          |
| SFC theorem/proposal              | Claim           | `PROPOSAL.md`, proof docs                   | Old paper story         | Proof invalid                           | ARCHIVE / DELETE CLAIM                    | Preserve audit; remove from main paper                   |
| FLC                               | Old method      | `src/functional_composition.py`             | Activation LS merge     | Static rank-r collapse                  | KEEP ONLY AS ABLATION                     | Activation deltas can feed CARR diagnostics              |
| BCFF                              | Old method      | `src/cross_factor_fusion.py`                | Cross-factor merge      | Tautological target                     | KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE | Do not present as contribution                           |
| `scripts/train_domain_lora.py`    | Training script | same                                        | Domain adapter training | MMLU leakage                            | REWRITE                                   | Add split manifest and fail-fast                         |
| `scripts/eval_domain_accuracy.py` | Eval script     | same                                        | Main evaluation         | fixed sampling seed, crude MCQ          | REWRITE                                   | Add logprob MCQ, seed/sample logging                     |
| `logs/`, `results-synced/`        | Logs            | all                                         | Evidence                | Unreliable for final claims             | FREEZE                                    | Preserve as historical evidence; don’t overwrite         |
| `RESULTS_REPORT.md`               | Result report   | root                                        | Old positive summary    | Missing reproducibility                 | ARCHIVE                                   | Useful low-confidence signal only                        |
| `PROGRESS.md`                     | Diagnosis       | root                                        | Failure summary         | Needs continuation                      | KEEP                                      | Critical evidence ledger                                 |
| `AUTO_REVIEW.md`                  | Review history  | root                                        | Reviewer-risk audit     | N/A                                     | KEEP                                      | Shows integrity and risk history                         |
| `README_RUN.md`                   | Runbook         | root                                        | Text2Subspace pilot     | stale path                              | REWRITE / ARCHIVE                         | Prevent false reproduction instructions                  |
| Current README claims             | Claim           | `README.md`                                 | BAC story               | overclaims relative to evidence         | REWRITE AFTER EXPERIMENTS                 | New paper thesis must follow CARR only if verified       |
| Text2Subspace pilot               | Experiment      | `results/text2subspace_*`, `EXPERIMENTS.md` | Pilot                   | proxy only                              | ARCHIVE                                   | Not main method evidence                                 |
| Result aggregation scripts        | Scripts         | if present under `scripts/`                 | Tables                  | missing registry/CI                     | REWRITE                                   | Need seed mean/std/CI                                    |
| New CARR code                     | New module      | `src/conflict_aware_routing.py`             | absent                  | Core missing mechanism                  | MERGE INTO NEW METHOD                     | Minimal implementation target                            |

---

# 14. Claude Code Implementation Plan

## Task 1: Freeze and label old evidence

**Purpose:** Prevent old positive/negative artifacts from being overwritten or mistaken for final evidence.
**Which Phenomenon / Constraint It Addresses:** P12, P16 / C7.
**Why It Supports New MAIN METHOD PATH:** CARR must be evaluated against frozen historical baselines, not mixed with stale logs.
**Files to Inspect:** `logs/`, `results/`, `results-synced/`, `RESULTS_REPORT.md`, `PROGRESS.md`.
**Files to Edit:** Add `results/manifest_historical.md` or `results/result_registry.json`.
**Files to Delete / Archive:** Do not delete; mark historical.
**Functions / Classes:** N/A.
**Exact Change:** Create a registry listing each result file, method, command if known, config if known, seed if known, checkpoint if known, reliability label.
**Do Not Change:** Do not edit old logs or overwrite historical JSON.
**Verification Command:**
`python scripts/validate_result_registry.py --registry results/result_registry.json`
If script missing, add it first.
**Expected Result:** Registry validates and every historical result has reliability status.
**Failure Means:** Existing evidence is still ambiguous.
**Rollback Condition:** Registry incorrectly changes or deletes old artifacts.
**Priority:** P0.
**Confidence:** high.

---

## Task 2: Add split and sample manifest checks

**Purpose:** Eliminate train/test leakage and fixed-sample ambiguity.
**Which Phenomenon / Constraint It Addresses:** P12, P13 / C7.
**Why It Supports New MAIN METHOD PATH:** CARR reliability calibration is meaningless if calibration/test overlap.
**Files to Inspect:** `scripts/train_domain_lora.py`, `scripts/eval_domain_accuracy.py`, `configs/domains.yaml`.
**Files to Edit:** Add `configs/splits.yaml`, `scripts/check_splits.py`.
**Files to Delete / Archive:** None.
**Functions / Classes:** `load_domain_dataset`, `evaluate_on_benchmark`.
**Exact Change:**

* Stop using MMLU test split for training.
* Record sample IDs for train/calib/test.
* Fail if overlap exists.
* Add `--sample_seed` separate from model seed.
  **Do Not Change:** Do not alter benchmark definitions silently.
  **Verification Command:**
  `python scripts/check_splits.py --config configs/domains.yaml --splits configs/splits.yaml --fail_on_overlap`
  **Expected Result:** No overlap; all domains have nonzero train/calib/test counts.
  **Failure Means:** Existing adapters/results cannot be used as strong evidence.
  **Rollback Condition:** Script changes dataset semantics without explicit config.
  **Priority:** P0.
  **Confidence:** high.

---

## Task 3: Fix evaluation metric and seed logging

**Purpose:** Make small experiments interpretable.
**Which Phenomenon / Constraint It Addresses:** P12, P16 / C7.
**Why It Supports New MAIN METHOD PATH:** CARR must beat baselines under deterministic, fair metric.
**Files to Inspect:** `scripts/eval_domain_accuracy.py`.
**Files to Edit:** same; optionally add `scripts/eval_mcq_logprob.py`.
**Files to Delete / Archive:** None.
**Functions / Classes:** `extract_answer`, `evaluate_on_benchmark`, model generation helper.
**Exact Change:**

* Add logprob-based MCQ scoring for MMLU/ARC-style tasks.
* Log prompt template, option order, target answer, raw output, logprobs.
* Use `args.seed` or `--sample_seed` for dataset sampling instead of hardcoded 42.
  **Do Not Change:** Do not weaken baselines or change datasets.
  **Verification Command:**
  `python scripts/eval_domain_accuracy.py --domain science --max_samples 20 --metric_mode logprob_mcq --seed 1 --sample_seed 1 --dry_run_log_samples`
  **Expected Result:** Deterministic sample manifest and metric output.
  **Failure Means:** Cannot interpret CARR vs baselines.
  **Rollback Condition:** Logprob scorer incompatible with model/tokenizer.
  **Priority:** P0.
  **Confidence:** high.

---

## Task 4: Fail-fast calibration loader

**Purpose:** Prevent empty or partial calibration data from producing results.
**Which Phenomenon / Constraint It Addresses:** P8 / C7.
**Why It Supports New MAIN METHOD PATH:** CARR depends on calibration reliability.
**Files to Inspect:** FLC scripts, future CARR scripts, `scripts/eval_one_pair.py`.
**Files to Edit:** Add shared `src/data_sanity.py`; update FLC/CARR loaders.
**Files to Delete / Archive:** None.
**Functions / Classes:** calibration data loader.
**Exact Change:** Raise error if any domain has fewer than `min_calib_samples`; log IDs.
**Do Not Change:** Do not silently synthesize fallback examples.
**Verification Command:**
`python scripts/check_calibration_data.py --domains science philosophy --min_calib_samples 50`
**Expected Result:** Nonzero calibration count or explicit failure.
**Failure Means:** That domain pair cannot be used.
**Rollback Condition:** If fail-fast blocks valid datasets due to incorrect field mapping.
**Priority:** P0.
**Confidence:** high.

---

## Task 5: Archive BCFF as main method and add regression test for tautology

**Purpose:** Prevent reuse of a vacuous objective.
**Which Phenomenon / Constraint It Addresses:** P9 / C5.
**Why It Supports New MAIN METHOD PATH:** CARR’s objective must be downstream-aligned.
**Files to Inspect:** `src/cross_factor_fusion.py`, `scripts/eval_bcff.py`.
**Files to Edit:** Add warning banner/docstring; add `tests/test_bcff_tautology.py`.
**Files to Delete / Archive:** Do not delete; move script references from main docs to historical.
**Functions / Classes:** `bcff_merge`.
**Exact Change:** Test that current objective returns approximately `[1,1,0,0]`; mark BCFF as negative ablation only.
**Do Not Change:** Do not rewrite BCFF into CARR.
**Verification Command:**
`pytest tests/test_bcff_tautology.py -q`
**Expected Result:** Test documents tautology.
**Failure Means:** Need inspect BCFF math before archiving.
**Rollback Condition:** None if only adds test/doc label.
**Priority:** P1.
**Confidence:** high.

---

## Task 6: Implement corrected conflict diagnostics

**Purpose:** Replace incomplete CRS/FDS with activation-conditioned conflict metrics.
**Which Phenomenon / Constraint It Addresses:** P5, P7, P10 / C10.
**Why It Supports New MAIN METHOD PATH:** CARR router needs conflict features.
**Files to Inspect:** `src/rank_bottleneck.py`, `src/functional_composition.py`, `src/sparse_feature_composition.py`.
**Files to Edit:** Add `src/conflict_diagnostics.py`.
**Files to Delete / Archive:** None.
**Functions / Classes:** `compute_activation_gram`, `compute_pair_conflict`, `compute_residual_projector`.
**Exact Change:**
Compute `G_ij^l = E_h <ΔW_i h, ΔW_j h>` on calibration activations; log cosine/conflict/tail energy.
**Do Not Change:** Do not alter adapter weights.
**Verification Command:**
`pytest tests/test_conflict_diagnostics.py -q`
**Expected Result:** Synthetic same-U/different-V cases produce different conflict when activation covariance differs.
**Failure Means:** Conflict diagnostic still too geometry-only.
**Rollback Condition:** Diagnostics are numerically unstable or too slow.
**Priority:** P1.
**Confidence:** medium.

---

## Task 7: Implement CARR router minimal module

**Purpose:** Add the new mechanism.
**Which Phenomenon / Constraint It Addresses:** P2/P3/P7/P10/P11 / C2/C4/C6.
**Why It Supports New MAIN METHOD PATH:** This is the core method.
**Files to Inspect:** PEFT adapter loading code, `src/lora_algebra.py`, evaluation hooks.
**Files to Edit:** Add `src/conflict_aware_routing.py`; add tests.
**Files to Delete / Archive:** None.
**Functions / Classes:** `AdapterReliabilityProfiler`, `ConflictAwareResidualRouter`, `CARRConfig`.
**Exact Change:**

* Freeze base/adapters.
* Compute reliability features.
* Add top-k gate with base fallback.
* Apply routed residuals at selected LoRA target modules.
  **Do Not Change:** Do not retrain domain adapters; do not modify baseline eval.
  **Verification Command:**
  `pytest tests/test_conflict_aware_routing.py -q`
  **Expected Result:** Router can gate off all adapters and exactly reproduce base; can activate one adapter and match that adapter path on a toy module.
  **Failure Means:** Hooking/PEFT integration unreliable.
  **Rollback Condition:** Base equivalence test fails.
  **Priority:** P1.
  **Confidence:** medium.

---

## Task 8: Add CARR train/eval scripts and A/B/C configs

**Purpose:** Make minimal verification executable.
**Which Phenomenon / Constraint It Addresses:** C11.
**Why It Supports New MAIN METHOD PATH:** Proves full method is not existing positive fragment.
**Files to Inspect:** `scripts/eval_domain_accuracy.py`, `scripts/eval_one_pair.py`, `configs/domains.yaml`.
**Files to Edit:** Add `scripts/train_carr_router.py`, `scripts/eval_carr.py`, `configs/carr_minimal.yaml`.
**Files to Delete / Archive:** None.
**Functions / Classes:** CARR module and evaluation harness.
**Exact Change:** Implement modes:

* `static_only`
* `carr_no_reliability_conflict`
* `carr_full`
  **Do Not Change:** Do not add new datasets yet.
  **Verification Command:**
  `python scripts/eval_carr.py --config configs/carr_minimal.yaml --mode static_only --domains science,medical --max_samples 20 --seed 1`
  **Expected Result:** Runs and logs result JSON with manifest.
  **Failure Means:** Cannot proceed to method experiments.
  **Rollback Condition:** If script bypasses existing eval checks.
  **Priority:** P1.
  **Confidence:** medium.

---

## Task 9: Add mechanism logging

**Purpose:** Show CARR mechanism actually occurs.
**Which Phenomenon / Constraint It Addresses:** C2/C6/C11.
**Why It Supports New MAIN METHOD PATH:** Without mechanism logs, positive results could be accidental.
**Files to Inspect:** new CARR scripts.
**Files to Edit:** `src/conflict_aware_routing.py`, `scripts/eval_carr.py`.
**Files to Delete / Archive:** None.
**Functions / Classes:** router forward/logging.
**Exact Change:** Log base gate rate, adapter gate rate, gate entropy, conflict before/after, reliability calibration curve, base KL, routed residual norm.
**Do Not Change:** Do not log only aggregate accuracy.
**Verification Command:**
`python scripts/eval_carr.py --config configs/carr_minimal.yaml --mode carr_full --max_samples 20 --seed 1 --log_mechanism_stats`
**Expected Result:** `mechanism_stats.json` produced.
**Failure Means:** Cannot support mechanism claim.
**Rollback Condition:** Logging changes model outputs.
**Priority:** P1.
**Confidence:** high.

---

## Task 10: Update README/paper claims only after minimal experiments

**Purpose:** Prevent overclaiming.
**Which Phenomenon / Constraint It Addresses:** C8/C12.
**Why It Supports New MAIN METHOD PATH:** Paper thesis must follow evidence.
**Files to Inspect:** `README.md`, `PROPOSAL.md`, `EXPERIMENTS.md`, `RESULTS_REPORT.md`.
**Files to Edit:** README only after results.
**Files to Delete / Archive:** Archive stale claims in `docs/archive/`.
**Functions / Classes:** N/A.
**Exact Change:** Replace BAC/SFC/Text2Subspace overclaims with CARR provisional thesis and reliability caveats.
**Do Not Change:** Do not claim SOTA or NeurIPS-ready.
**Verification Command:**
`python scripts/check_claims_have_evidence.py --readme README.md --registry results/result_registry.json`
**Expected Result:** Every claim links to result IDs.
**Failure Means:** Claim still overstates evidence.
**Rollback Condition:** Any claim lacks evidence.
**Priority:** P2.
**Confidence:** high.

---

# 15. Minimal Verification Experiments

如果命令引用的新脚本尚不存在，Claude Code 必须先完成上面的 implementation tasks，不能伪造运行结果。

| Priority | Experiment                                    | Hypothesis                        | Command                                                                                                                                                     | Config        | Dataset                | Seeds | Metric              | Success Criterion                                    | Failure Interpretation            |
| -------- | --------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ---------------------- | ----- | ------------------- | ---------------------------------------------------- | --------------------------------- |
| P0       | Smoke import test                             | 新模块不破坏现有 imports                  | `pytest tests/test_text2subspace.py tests/test_conflict_aware_routing.py -q`                                                                                | N/A           | toy                    | 1     | pass/fail           | all tests pass                                       | 基础集成失败，停止                         |
| P0       | Data sanity check                             | 无 train/calib/test overlap        | `python scripts/check_splits.py --config configs/domains.yaml --splits configs/splits.yaml --fail_on_overlap`                                               | splits        | all selected domains   | N/A   | overlap count       | 0 overlap, nonzero counts                            | 任何 overlap 使旧结果不可用                |
| P0       | Metric sanity check                           | logprob MCQ 与 answer key 对齐       | `python scripts/eval_domain_accuracy.py --domain science --max_samples 20 --metric_mode logprob_mcq --seed 1 --sample_seed 1 --dry_run_log_samples`         | eval          | science                | 1     | accuracy/logprob    | deterministic outputs + sample IDs                   | metric 不可信                        |
| P0       | One-batch overfit router                      | 小 router 能在 toy calib 上学习非平凡 gate | `python scripts/train_carr_router.py --config configs/carr_minimal.yaml --domains science,medical --max_steps 50 --overfit_one_batch`                       | CARR minimal  | calib small            | 1     | train CE/gate       | loss decreases, gate changes                         | router/hook broken                |
| P0       | Checkpoint loading check                      | base/static/adapter/CARR 加载一致     | `python scripts/check_checkpoint_integrity.py --config configs/carr_minimal.yaml`                                                                           | CARR minimal  | N/A                    | N/A   | hash/equivalence    | base gate=1 reproduces base                          | PEFT/hook integration不可信          |
| P0       | Reproduce current negative: BCFF tautology    | BCFF 当前机制确实 vacuous               | `pytest tests/test_bcff_tautology.py -q`                                                                                                                    | N/A           | synthetic              | 1     | coeff error         | coeff≈[1,1,0,0]                                      | 若不成立，重新审计 BCFF                    |
| P0       | Reproduce current negative: FLC energy        | 静态 rank-r compression 高损失         | `python scripts/eval_flc.py --domains science,medical --max_samples 50 --log_energy --seed 1`                                                               | old FLC       | selected pair          | 1     | retained energy     | energy low or reproduced; fail-fast if calib missing | 若能量高且准确高，重新诊断                     |
| P1       | Reproduce existing best positive fragment     | Static baseline 有公平复现             | `python scripts/eval_carr.py --config configs/carr_minimal.yaml --mode static_only --domains science,medical --seed 1 --max_samples 50`                     | static        | non-leaky pair         | 1     | accuracy/logprob    | result recorded with manifest                        | 无 baseline，不能评估 CARR              |
| P1       | New mechanism activation check                | Full CARR 使用非平凡 gate              | `python scripts/eval_carr.py --config configs/carr_minimal.yaml --mode carr_full --domains science,medical --seed 1 --max_samples 50 --log_mechanism_stats` | CARR          | same                   | 1     | gate stats          | base/adapter/static gates all non-degenerate         | gate collapse = 方法无机制             |
| P1       | A: Existing Best Positive Fragment Only       | Static candidate 是强 baseline      | same as static-only, preregistered candidate                                                                                                                | static        | same                   | 1,2,3 | accuracy/logprob    | stable baseline mean/std                             | baseline 不稳定说明数据/metric问题         |
| P1       | B: New MAIN METHOD Without New Mechanism      | 架构壳本身不应解释提升                       | `python scripts/eval_carr.py --mode carr_no_reliability_conflict ...`                                                                                       | CARR ablation | same                   | 1,2,3 | accuracy/logprob    | B ≤ Full CARR                                        | 若 B=Full，机制无贡献                    |
| P1       | C: Full New MAIN METHOD                       | reliability/conflict routing 提升   | `python scripts/eval_carr.py --mode carr_full ...`                                                                                                          | CARR full     | same                   | 1,2,3 | accuracy/logprob    | Full > A and B with CI trend                         | 若不超过 A/B，停止或 pivot                |
| P1       | Remove reliability                            | reliability 是必要项                  | `python scripts/eval_carr.py --mode no_reliability ...`                                                                                                     | ablation      | same                   | 1,2,3 | accuracy/gate ECE   | drop vs full                                         | 无 drop 则 reliability claim 弱      |
| P1       | Remove conflict                               | conflict 是必要项                     | `python scripts/eval_carr.py --mode no_conflict ...`                                                                                                        | ablation      | same                   | 1,2,3 | accuracy/conflict   | drop or higher conflict harm                         | 无 drop 则 conflict claim 弱         |
| P1       | Remove base KL/fallback                       | base preservation 防止伤害            | `python scripts/eval_carr.py --mode no_base_fallback ...`                                                                                                   | ablation      | same + control         | 1,2,3 | task acc/control KL | worse control or domain harm                         | 若无影响，base claim弱                  |
| P1       | Small baseline comparison                     | 不弱化 baseline                      | `python scripts/run_minimal_baselines.py --config configs/carr_minimal.yaml --methods base,individual,TA,TIES,DARE,GrassMerge,CARR`                         | shared        | same                   | 1,2,3 | accuracy/logprob    | Full CARR beats strongest fair static                | 否则不能继续主线                          |
| P2       | Multi-seed stability                          | 不是 best seed                      | same with `--seeds 1,2,3,4,5`                                                                                                                               | CARR          | selected pairs         | 5     | mean/std/CI         | Full improvement stable                              | 高 variance 需 stabilizer           |
| P2       | Expansion gate to full benchmark              | 小规模通过后再扩展                         | `python scripts/run_carr_benchmark.py --config configs/carr_full.yaml --seeds 1,2,3`                                                                        | full          | all valid domains      | 3     | mean/std/CI         | no broad regressions                                 | 若只一对有效，claim 必须缩小                 |
| P2       | Official TIES/DARE/KnOTS/RegMean reproduction | 机制 baselines 公平                   | `python scripts/run_official_baselines.py --config configs/baselines_official.yaml`                                                                         | official      | same                   | 3     | same metric         | official numbers logged                              | 无 official baseline 不可 claim SOTA |
| P2       | Unified environment comparison                | 避免环境差异                            | `python scripts/check_env_and_run.py --suite carr_minimal`                                                                                                  | env lock      | same                   | 3     | hash + metric       | same env hash                                        | 环境 drift 污染结果                     |
| P2       | Robustness/generalization                     | 不只是 calibration overfit           | `python scripts/eval_carr.py --split heldout_unseen --domains ...`                                                                                          | heldout       | unseen domains/prompts | 3     | accuracy/control KL | no collapse                                          | 若只 calib 有效，方法弱                   |
| P2       | Statistical CI                                | publish-level significance        | `python scripts/aggregate_results.py --registry results/result_registry.json --ci bootstrap`                                                                | registry      | all                    | 3/5   | CI/p-value          | CI excludes zero for main comparisons                | CI 不显著则 claim 降级                  |

---

# 16. Baseline and SOTA Plan

| Baseline           | Why Required                             | Official Code         | Dataset                | Metric | Reproduction Requirement                      | Fairness Risk                                   |
| ------------------ | ---------------------------------------- | --------------------- | ---------------------- | ------ | --------------------------------------------- | ----------------------------------------------- |
| Base model         | Measures whether any adapter helps       | N/A                   | all                    | same   | same prompts/splits                           | If base is strong, merged methods may only hurt |
| Individual LoRA    | Shows adapter reliability                | internal + PEFT       | each domain            | same   | same checkpoint manifest                      | Weak adapters make composition meaningless      |
| Task Arithmetic    | Simplest strong static baseline          | available             | all pairs              | same   | same scaling/tuning on calib only             | Easy to under-tune or over-tune                 |
| TIES-Merging       | Static interference baseline             | official preferred    | all pairs              | same   | official or faithful implementation, same env | Must not use weaker internal version only       |
| DARE               | Drop/rescale merge baseline              | official/faithful     | all pairs              | same   | multi seed masks                              | Hardcoded seed unfair                           |
| RegMean            | Activation regression merge baseline     | official/faithful     | same                   | same   | same calibration data                         | Very close to FLC; must include                 |
| KnOTS              | SVD/LoRA alignment baseline              | official preferred    | same                   | same   | same adapters                                 | Close to BAC/GrassMerge                         |
| GrassMerge         | Existing repo positive fragment          | internal              | same                   | same   | frozen exact implementation                   | Cannot choose best old result post hoc          |
| BAC static variant | README current claim                     | internal              | same                   | same   | after CRS fix or label old                    | Needed if README remains                        |
| AdapterFusion      | Mechanism-level dynamic adapter baseline | official/faithful     | same where feasible    | same   | same frozen adapters/training budget          | CARR novelty risk high                          |
| LoraHub            | Few-shot LoRA composition baseline       | official              | same                   | same   | same calibration budget                       | If LoraHub beats CARR, CARR claim weak          |
| LoRA-Flow          | Closest token-level dynamic LoRA         | official if available | same                   | same   | same examples/compute                         | Very high novelty risk                          |
| MixLoRA / MoE-LoRA | Expert-routing LoRA baseline             | official if available | same                   | same   | same train data budget                        | Router comparison must be fair                  |
| Text-to-LoRA       | Only if text-to-subspace claim retained  | official if available | task-description setup | same   | true task text input                          | Otherwise archive text claim                    |

---

# 17. Paper Thesis Reconstruction

1. **New Paper Thesis:**
   Static adapter merging fails not because the merge formula is insufficiently clever, but because adapter composition requires reliability-calibrated conflict arbitration. CARR converts adapter composition from one global merged update into a base-preserving, conflict-aware residual routing problem.

2. **Main Technical Contribution:**
   A calibrated dynamic residual router that combines static compatible directions with per-input conflict residual routing and base fallback.

3. **Main Empirical Claim:**
   If minimal experiments pass: CARR improves over strong static merging and vanilla dynamic/router ablations on non-leaky held-out adapter composition benchmarks, with lower negative transfer.

4. **What Previous Failures Taught Us:**

   * SFC: sparse feature support alone is not an actuator.
   * FLC: rank-r static compression destroys conflict residuals.
   * BCFF: objectives must be task-aligned, not self-targeted.
   * GrassMerge/BAC: geometry may help but cannot decide when to abstain.

5. **What We Should Not Claim:**
   No SOTA, no first dynamic LoRA routing, no provably optimal feature composition, no “solves trilemma,” no text-to-LoRA generation unless actually implemented.

6. **What We Can Claim If Experiments Pass:**
   CARR reduces negative transfer vs static LoRA merging and improves held-out composition by using reliability and conflict diagnostics.

7. **Required Baselines:**
   Base, individual LoRA, TA, TIES, DARE, RegMean, KnOTS, GrassMerge, LoraHub, AdapterFusion/LoRA-Flow/MixLoRA where feasible.

8. **Required Ablations:**
   Static only; no reliability; no conflict; no base fallback; uniform gate; oracle gate; CARR full.

9. **Required Robustness Tests:**
   Multi-seed, unseen prompts, non-MMLU leakage-free domains, control KL, adapter weakness cases.

10. **Reviewer Likely Objections:**
    “This is just AdapterFusion/LoRA-Flow/MoE-LoRA”; “baselines weak”; “datasets cherry-picked”; “router learns domain ID”; “old eval bugs persist.”

11. **How New MAIN METHOD Answers Them:**
    By using official baselines, A/B/C controls, conflict/reliability ablations, non-leaky splits, and mechanism logs.

12. **What Would Make This NeurIPS-Strong:**
    Clear empirical win over official static and dynamic baselines; diagnostics predict when static merge fails; robust multi-domain benchmark; honest negative results.

13. **What Would Make This Rejected:**
    Only beats weak internal baselines; no official LoRA-Flow/LoraHub comparison; single seed; MMLU leakage; overclaiming novelty.

14. **What Would Be Required for Oral-Level Strength:**
    Broad evidence that reliability/conflict diagnostics predict negative transfer across models, datasets, adapter ranks, and domains; strong theory or mechanistic analysis.

15. **What Would Be Required for Best-Paper-Level Strength:**
    A general adapter composition principle with predictive theory, new benchmark, strong official-baseline wins, and open reproducible artifacts.

---

# 18. Reviewer Risk Assessment

| Risk                       | Why Reviewer May Object                     | Evidence Needed                                                 | How CARR Addresses It                                            | Remaining Weakness                             |
| -------------------------- | ------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------- |
| Novelty risk               | Dynamic LoRA/adapter routing already exists | Direct comparison to LoRA-Flow, MixLoRA, LoraHub, AdapterFusion | Frames contribution as reliability/conflict residual arbitration | Still close; needs strong differentiation      |
| Incremental risk           | Could look like router + LoRA experts       | Mechanism ablations and diagnostics                             | Adds rank/conflict residual and base fallback                    | If ablations weak, incremental                 |
| Baseline weakness          | Prior repo uses internal baselines          | Official or faithful baselines                                  | Baseline plan includes official methods                          | Compute/setup may be heavy                     |
| Reproducibility risk       | Missing commands/seeds/checkpoints          | result registry, split manifest, env hash                       | First implementation tasks fix this                              | Current old results remain low confidence      |
| Cherry-picking risk        | Positive results are narrow                 | preregistered A/B/C and multi-seed                              | Requires all selected domains reported                           | Must include negative domains                  |
| Negative result hiding     | Many old failures                           | Preserve historical negative evidence                           | Archive not delete                                               | Paper must explain failures honestly           |
| Overclaiming risk          | README/proposal overstate                   | claim-code-result check                                         | Delay README rewrite until experiments pass                      | Temptation to claim “trilemma solved”          |
| Unclear mechanism          | Router may be black box                     | gate/conflict/reliability logs                                  | Mechanism logging required                                       | Logs must correlate with performance           |
| Ablation insufficiency     | Full method could equal static baseline     | A/B/C required                                                  | Full vs static-only vs no-mechanism                              | If CARR wins only via tuning, weak             |
| Dataset limitation         | Current domains noisy/weak                  | clean heldout domains, sample IDs                               | split checks                                                     | Current adapters may be poor                   |
| Compute unfairness         | Dynamic router may use extra data/params    | report params/data/compute                                      | router small, frozen adapters                                    | Must compare same calibration budget           |
| Implementation reliability | hooks/PEFT can be fragile                   | base equivalence and adapter equivalence tests                  | checkpoint integrity tests                                       | PEFT integration still risk                    |
| Related work omission      | Many close papers                           | table + official baselines                                      | related work plan                                                | Need thorough literature review in final paper |

---

# 19. Final Decision

## 1. One-Sentence Verdict

从所有正面、负面、不稳定现象推导出的唯一推荐 MAIN METHOD PATH 是：**放弃全局静态 adapter merge 作为主方法，转向 CARR：可靠性校准 + 冲突感知 + base-preserving 的输入条件化 residual routing。**

## 2. Current Most Likely Root Cause

当前失败最可能来自多因素叠加：

| Cause                     | Likelihood             | Explanation                                                           |
| ------------------------- | ---------------------- | --------------------------------------------------------------------- |
| missing mechanism         | highest                | 所有旧路线都缺少 reliability/conflict/input-conditioned arbitration           |
| evaluation bug            | high                   | MMLU split leakage、fixed sample seed、MCQ extraction、empty calibration |
| method assumption failure | high                   | 静态 merge/静态 feature offset/静态 rank-r compression 假设失败                 |
| baseline mismatch         | medium-high            | TA/TIES/official dynamic baselines未充分公平复现                             |
| weak experimental setup   | high                   | 50-sample、少 seed、无 CI、无 result manifest                               |
| novelty issue             | high                   | 动态 LoRA routing 文献接近                                                  |
| code bug                  | high for BCFF/FLC/eval | BCFF tautology、FLC empty calibration、eval seed                        |
| insufficient evidence     | high                   | 当前没有 publish-level positive evidence                                  |

## 3. Why This Is Not Just the Existing Best Path

Existing best path 是某个静态 merge 或旧 positive fragment；CARR 的核心输出不是一个静态 adapter，而是**带 base fallback 的条件化 residual routing policy**。TA/TIES/GrassMerge 在 CARR 中只是 baseline 或 static safe component，不是主贡献。

## 4. Phenomena Explained

CARR 解释：

* 为什么 TA/TIES 有时强：兼容方向可静态合并。
* 为什么 science 有时强：某些 adapter residual 可靠。
* 为什么 math/medical/philosophy 会被伤害：adapter 可靠性不均。
* 为什么 SFC 失败：静态 feature offset 缺 input/token condition。
* 为什么 FLC 崩塌：冲突 residual 不能被全局 rank-r 压缩。
* 为什么 BCFF 失败：self-target 不是 downstream objective。
* 为什么旧 positive 不稳：没有判断何时使用/不用 adapter。

## 5. Mechanism Missing in Current Method

**Reliability-calibrated, input-conditioned conflict arbitration.**

## 6. New Mechanism

**CARR router:**
在 frozen base + frozen adapters 上训练小型 gate；输入包括 adapter reliability、activation conflict、rank/bottleneck residual diagnostics；输出 base/static/residual gates；优化 downstream CE/logprob + base KL + conflict penalty + sparsity + calibration loss。

## 7. What to Delete / Archive / Rewrite

| Action                | Items                                                                                                          |
| --------------------- | -------------------------------------------------------------------------------------------------------------- |
| DELETE claim          | SFC provably optimal、low-rank→sparse SAE theorem、BCFF cross transfer claim、Text2Subspace true generation claim |
| ARCHIVE               | SFC main story, FLC main method, BCFF main method, stale Text2Subspace runbook, unreliable old result tables   |
| REWRITE               | train/eval split, metric, seed/sample logging, rank/conflict diagnostic, README/paper thesis                   |
| KEEP                  | base/adapters, TA/TIES/DARE/GrassMerge as baselines, failure logs as evidence                                  |
| MERGE INTO NEW METHOD | corrected activation conflict diagnostics, corrected rank/bottleneck residual idea                             |

## 8. First Five Claude Code Tasks

1. Freeze old results and create result registry.
2. Add split/sample manifest and fail overlap checks.
3. Fix evaluation seed and MCQ logprob metric.
4. Add fail-fast calibration loader.
5. Implement CARR conflict diagnostics and minimal router with base equivalence tests.

## 9. Minimal Experiments

Essential queue:

1. smoke/import tests
2. split sanity
3. metric sanity
4. one-batch router overfit
5. checkpoint/base equivalence
6. reproduce BCFF tautology
7. reproduce FLC negative or fail-fast
8. reproduce existing best static positive fragment
9. A: static-only
10. B: CARR without reliability/conflict
11. C: Full CARR
12. no reliability / no conflict / no base fallback ablations
13. small official baseline comparison
14. 3-seed then 5-seed stability
15. CI aggregation

## 10. Continue / Stop / Pivot Criteria

| Decision | Criteria                                                                                                                                                                                                                                                                         |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Continue | Split/metric checks pass; individual adapters include at least 2 reliable non-leaky domains; Full CARR beats A and B over 3 seeds; mechanism logs show nontrivial reliability/conflict use                                                                                       |
| Stop     | Split leakage cannot be fixed; metric unstable; base equivalence fails; no adapter beats base on calibration; Full CARR ≤ static baseline                                                                                                                                        |
| Pivot    | Oracle routing works but learned routing fails → need better router/data; CARR only works by domain labels → frame as diagnostic benchmark or task routing, not general method; official LoRA-Flow/MixLoRA clearly beats CARR → need new differentiation or abandon method claim |

## 11. NeurIPS-Level Gap

Still missing:

* clean non-leaky benchmark,
* official baselines,
* multi-seed CI,
* mechanism ablations,
* router vs dynamic LoRA related work comparison,
* reproducible registry/checkpoint manifest,
* rewritten honest paper thesis.

## 12. Oral / Best Paper Gap

Oral-level would require broad predictive diagnostics across models/ranks/tasks. Best-paper-level would require a general theory of adapter conflict and a benchmark showing static merge failures can be predicted and fixed by calibrated residual routing.

## 13. Confidence

**medium-low** for CARR becoming a strong paper result; **high** that continuing with SFC/FLC/BCFF/static-only positive anchoring is the wrong path.

---

# 20. Final Claude Code Instruction

```text
Claude Code, execute the following plan.

You must implement the New MAIN METHOD PATH defined in the GPT-5.5 Pro diagnosis report:

CARR: Conflict-Aware Reliability-Gated Residual Routing.

Do not invent a different method.
Do not optimize for superficial positive results.
Do not weaken baselines.
Do not delete negative results silently.
Do not change metrics or datasets unless explicitly instructed.
Do not rewrite unrelated files.
Do not present SFC, FLC, BCFF, BAC, GrassMerge, or Text2Subspace pilot as the final method unless the specified verification experiments support them.

Your tasks are:

1. Freeze old evidence.
   - Create a result registry for existing logs/results.
   - Mark old SFC/FLC/BCFF/GrassMerge/Text2Subspace results as historical with reliability labels.
   - Do not overwrite or delete old logs.

2. Add split and sample manifest checks.
   - Add configs/splits.yaml if missing.
   - Add scripts/check_splits.py.
   - Prevent MMLU or any dataset test split from being used for training.
   - Log sample IDs for train/calib/test.
   - Fail if overlap exists.

3. Fix evaluation reliability.
   - In scripts/eval_domain_accuracy.py, remove hardcoded dataset sampling seed=42 or expose it as --sample_seed.
   - Add deterministic logprob-based MCQ scoring.
   - Log prompt, option order, target, raw output, logprobs, sample IDs, seed, checkpoint hash.

4. Add calibration fail-fast checks.
   - Add shared data sanity code.
   - Fail if any domain has fewer than the required calibration samples.
   - Do not synthesize fallback examples silently.

5. Archive BCFF as a main method.
   - Add a regression test showing the current BCFF objective learns approximately [1,1,0,0].
   - Keep BCFF only as historical negative evidence / ablation.
   - Do not reuse its self-target objective.

6. Implement corrected conflict diagnostics.
   - Add src/conflict_diagnostics.py.
   - Compute activation-conditioned adapter conflict:
     G_ij^l = E_h <DeltaW_i^l h, DeltaW_j^l h>.
   - Include tests where same-left-space but different-right-space adapters produce different conflict under different activation covariance.

7. Implement the minimal CARR router.
   - Add src/conflict_aware_routing.py.
   - Include AdapterReliabilityProfiler, ConflictAwareResidualRouter, and CARRConfig.
   - Freeze base model and existing adapters.
   - Add base fallback, static candidate gate, and top-k residual adapter gate.
   - Router features must include adapter reliability and conflict diagnostics.
   - Base gate=1 must exactly reproduce base outputs in a test.
   - Single-adapter gate must match the corresponding adapter path in a toy test.

8. Add CARR train/eval scripts.
   - Add scripts/train_carr_router.py.
   - Add scripts/eval_carr.py.
   - Add configs/carr_minimal.yaml.
   - Required modes:
     A. static_only
     B. carr_no_reliability_conflict
     C. carr_full
     plus no_reliability, no_conflict, no_base_fallback.

9. Add mechanism logging.
   - Log base gate rate, adapter gate rate, gate entropy, reliability calibration, conflict before/after routing, residual norm, base KL, per-domain metrics, seed, sample IDs, checkpoint hashes.

10. Run only the minimal verification queue first.
   - smoke tests
   - split sanity
   - metric sanity
   - one-batch overfit
   - checkpoint/base equivalence
   - BCFF tautology regression
   - existing static baseline reproduction
   - A/B/C comparison on a small non-leaky domain pair
   - no reliability / no conflict / no base fallback ablations
   - 3-seed aggregation

For every task:
- make the smallest necessary change;
- show the diff;
- run the specified verification command;
- save logs;
- report failures;
- stop if verification fails;
- do not proceed to full benchmark until minimal tests pass.

At the end, output:
- files changed;
- files archived;
- configs added;
- commands run;
- logs;
- result table;
- failed checks;
- unresolved issues;
- whether Full New MAIN METHOD beats:
  A. Existing Best Positive Fragment Only,
  B. New MAIN METHOD Without New Mechanism,
  C. Full New MAIN METHOD.

Do not update README or paper claims until the A/B/C minimal verification and mechanism ablations pass.
```

[1]: https://github.com/Sunshine535/nips-text2subspace "GitHub - Sunshine535/nips-text2subspace: Text2Subspace: Text-Conditioned LoRA Generation (NeurIPS 2026) · GitHub"
[2]: https://github.com/Sunshine535/nips-text2subspace/tree/main/src "nips-text2subspace/src at main · Sunshine535/nips-text2subspace · GitHub"
[3]: https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/EXPERIMENTS.md "raw.githubusercontent.com"
[4]: https://github.com/Sunshine535/nips-text2subspace/tree/main/scripts "nips-text2subspace/scripts at main · Sunshine535/nips-text2subspace · GitHub"
[5]: https://github.com/Sunshine535/nips-text2subspace/tree/main/configs "nips-text2subspace/configs at main · Sunshine535/nips-text2subspace · GitHub"
[6]: https://github.com/Sunshine535/nips-text2subspace/tree/main/logs "nips-text2subspace/logs at main · Sunshine535/nips-text2subspace · GitHub"
[7]: https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/PROGRESS.md "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/RESULTS_REPORT.md "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/AUTO_REVIEW.md "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/PROPOSAL.md "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/PROOF_AUDIT.md "https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/PROOF_AUDIT.md"
[12]: https://github.com/Sunshine535/nips-text2subspace/blob/main/src/lora_algebra.py "https://github.com/Sunshine535/nips-text2subspace/blob/main/src/lora_algebra.py"
[13]: https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/src/rank_bottleneck.py "https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/src/rank_bottleneck.py"
[14]: https://github.com/Sunshine535/nips-text2subspace/blob/main/src/sparse_feature_composition.py "nips-text2subspace/src/sparse_feature_composition.py at main · Sunshine535/nips-text2subspace · GitHub"
[15]: https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/src/sae_decomposition.py "https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/src/sae_decomposition.py"
[16]: https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/src/functional_composition.py "https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/src/functional_composition.py"
[17]: https://github.com/Sunshine535/nips-text2subspace/blob/main/src/cross_factor_fusion.py "nips-text2subspace/src/cross_factor_fusion.py at main · Sunshine535/nips-text2subspace · GitHub"
[18]: https://github.com/Sunshine535/nips-text2subspace/blob/main/scripts/train_domain_lora.py "https://github.com/Sunshine535/nips-text2subspace/blob/main/scripts/train_domain_lora.py"
[19]: https://github.com/Sunshine535/nips-text2subspace/blob/main/scripts/eval_domain_accuracy.py "https://github.com/Sunshine535/nips-text2subspace/blob/main/scripts/eval_domain_accuracy.py"
[20]: https://github.com/Sunshine535/nips-text2subspace/blob/main/configs/domains.yaml "https://github.com/Sunshine535/nips-text2subspace/blob/main/configs/domains.yaml"
[21]: https://github.com/Sunshine535/nips-text2subspace/tree/main/results-synced "nips-text2subspace/results-synced at main · Sunshine535/nips-text2subspace · GitHub"
[22]: https://github.com/Sunshine535/nips-text2subspace/tree/main/results "nips-text2subspace/results at main · Sunshine535/nips-text2subspace · GitHub"
[23]: https://github.com/Sunshine535/nips-text2subspace/tree/main/tests "nips-text2subspace/tests at main · Sunshine535/nips-text2subspace · GitHub"
[24]: https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/README_RUN.md "https://raw.githubusercontent.com/Sunshine535/nips-text2subspace/main/README_RUN.md"
[25]: https://github.com/Sunshine535/nips-text2subspace/blob/main/results-synced/sfc_pilot.json "https://github.com/Sunshine535/nips-text2subspace/blob/main/results-synced/sfc_pilot.json"
[26]: https://github.com/Sunshine535/nips-text2subspace/blob/main/results-synced/sfc_eval.log "https://github.com/Sunshine535/nips-text2subspace/blob/main/results-synced/sfc_eval.log"
[27]: https://github.com/Sunshine535/nips-text2subspace/blob/main/results-synced/flc_eval.log "https://github.com/Sunshine535/nips-text2subspace/blob/main/results-synced/flc_eval.log"
[28]: https://github.com/Sunshine535/nips-text2subspace/blob/main/results-synced/bcff_eval.log "https://github.com/Sunshine535/nips-text2subspace/blob/main/results-synced/bcff_eval.log"
[29]: https://arxiv.org/abs/2212.04089 "https://arxiv.org/abs/2212.04089"
[30]: https://github.com/prateeky2806/ties-merging/blob/main/README.md "https://github.com/prateeky2806/ties-merging/blob/main/README.md"
[31]: https://arxiv.org/pdf/2212.09849 "https://arxiv.org/pdf/2212.09849"
[32]: https://arxiv.org/abs/2005.00247 "https://arxiv.org/abs/2005.00247"
[33]: https://replicate.com/cjwbw/lorahub "https://replicate.com/cjwbw/lorahub"
[34]: https://arxiv.org/abs/2402.11455 "https://arxiv.org/abs/2402.11455"
[35]: https://arxiv.org/abs/2404.15159 "https://arxiv.org/abs/2404.15159"
[36]: https://arxiv.org/abs/2410.19735 "https://arxiv.org/abs/2410.19735"
[37]: https://arxiv.org/pdf/2506.06105 "https://arxiv.org/pdf/2506.06105"
[38]: https://openreview.net/pdf?id=nZeVKeeFYf9 "https://openreview.net/pdf?id=nZeVKeeFYf9"
