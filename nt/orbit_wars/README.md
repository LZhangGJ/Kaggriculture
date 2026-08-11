# Orbit Wars 金牌方案与 Kaggriculture 迁移研究

本目录收录 Orbit Wars 金牌区（最终第 1–19 名）公开 writeup 的来源台账、逐篇方法摘要、逐篇 Kaggriculture 迁移设计，以及针对本机 `i9-12900K + 128GB RAM + RTX 3090 24GB` 的统一比较和推荐顺序。

## 先看哪几篇

1. [金牌文章覆盖台账](WRITEUP_COVERAGE_LEDGER.md)：确认哪些名次有公开文章，哪些没有。
2. [迁移共用建模底座](MIGRATION_COMMON_SPEC_ZH.md)：所有方案共用的 Kaggriculture 状态、任务动作、执行器、self-play 和评估定义。
3. [全部方案比较与推荐顺序](COMPARATIVE_MIGRATION_ANALYSIS_ZH.md)：实现难度、上分潜力、速度、风险和本机推荐顺序。
4. `sources/`：11 篇官方 writeup 的逐篇来源摘要。
5. `migrations/`：与 11 篇来源一一对应的 Kaggriculture 迁移方案。

## 目录约定

- `sources/NN_..._SOURCE_DIGEST_ZH.md`：官方文章的中文结构化研究摘要，保留排名、作者、URL、开源代码、模型、训练规模和作者报告的实验结论。
- `migrations/NN_..._TO_KAGGRICULTURE_ZH.md`：不是原作者方案的一部分，而是本项目基于该文章提出的迁移设计。
- `sources_manifest.json`：机器可读的来源清单。

## 版权与证据说明

Kaggle writeup 是第三方文章。本目录不全文复制原文，而是保存可审计的来源元数据、方法事实、少量参数和中文转述；完整原文请通过每篇文件顶部的官方链接阅读。

文档中的事实分为三类：

- **作者报告**：writeup 明确写出的模型、吞吐、训练规模或实验结果；本项目没有复跑 Orbit Wars 训练。
- **本地实测**：本项目 Kaggriculture JAX 模拟器的 parity 和 RTX 3090 benchmark，有本地 receipt。
- **迁移建议/估算**：针对 Kaggriculture 的设计与性能区间，必须在实现后重新 benchmark，不能当作已实现结果。

## 本地已验证基线

- JAX 规则核心已对 100 个未见 seed × 720 帧逐状态 parity；随机/非法动作另有差分验证。
- simulator-only，batch 4096：`1,561,634 transitions/s`。
- 608,029 参数基准策略 + simulator，batch 4096：`188,236 transitions/s`。
- 基准完整 PPO iteration，batch 1024：`57,291 transitions/s`。

证据：

- `../gpu_sim/receipts/benchmark_simulator_only_full_season.json`
- `../gpu_sim/receipts/benchmark_policy_and_ppo.json`
- `../gpu_sim/FINAL_ACCEPTANCE.md`

以上 transition 指双方共同推进一个 Kaggriculture 回合；赛季实际包含 719 次可执行推进。结构化 Transformer、复杂任务候选生成和历史对手推理会降低端到端吞吐，因此比较文档中的训练速度均按“尚未实测的方向性区间”标注。
