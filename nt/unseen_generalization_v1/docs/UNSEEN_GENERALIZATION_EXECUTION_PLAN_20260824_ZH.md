# 面向未见对手的 Agent 训练与迭代执行计划

日期：2026-08-24  
基线：FC24B / commit `a00c723681ad5b906d67a36f8fb8b73ba7eb91f5`  
实验目录：`nt/unseen_generalization_v1/`（与冻结的 FC24B 快照隔离）

## 1. 最终目标

目标不是在有限本地池上宣称“对任意 Agent 必胜”，而是建立一套能够持续发现、分类和修复未知策略弱点的系统：

```text
保留选择权的稳健开局
→ 公开状态与历史特征
→ 数据确定的决策窗口
→ 安全 Handoff Bridge
→ 兼容 continuation route 库
→ 带不确定性和拒绝机制的 selector
→ 状态驱动执行器
→ 防守清算
```

最终晋级依据必须同时包含：现有 34-Agent 非退化、whole-family holdout、独立事件 seed、冻结后生成的 adversarial agents、官方 Python/JAX 一致性以及 CPU 提交验收。

## 2. 当前已落地：UG0 数据与证据入口

### 2.1 自动同步

`sync_live_assets.py` 自动完成：

1. 查询自己的 Kaggriculture submission 列表；
2. 选择最新已完成 submission，或使用显式 submission ID；
3. 下载 submission 原文件；
4. 查询该 submission 的 episode；
5. 顺序、限速、可恢复地下载 Replay 和可选日志；
6. 输出不包含令牌的 SHA-256 receipt。

### 2.2 本地实时语料冻结

`build_ug0_manifest.py` 对 `D:\Kaggriculture` 中的公开方案、Replay、submission、报告和回执建立：

- 文件内容哈希；
- Notebook 代码哈希；
- submission 根目录 `main.py` 检查；
- Replay 第二座位增量 observation 重建；
- 行为签名、粗路线族与重复内容组；
- train / validation / whole-family holdout 分组；
- 机器可读 manifest 和中文报告。

Replay 仅提供真实状态分布和行为线索，不直接作为路线价值标签。路线价值必须在 UG3 通过本地反事实续跑生成。

## 3. UG1：CheckpointBankV1

### 输入

- 最新 own submission 的全部败局、平局、小分差胜局及随机大胜局；
- 当前高分公开玩家 Replay；
- FC24B 对冻结 34-Agent 的本地对局；
- 双座位与独立事件 seed。

### 保存内容

```text
官方 State
Events / RNG 继续状态
我方与对方策略 carry
未完成工人任务
当天喂养/浇水/照顾状态
市场提前销售债务
小麦信用
最近数日公开历史
当前经营路线和阶段
```

### 检查点触发

- 每日边界；
- 首次买动物、扩地、大量雇工、稳定产出和出售前后；
- 现金跌破安全线；
- 两条候选路线第一次发生宏观分歧；
- 停止投资和开始清算。

### 硬门

从 checkpoint 恢复并继续原 Agent 后，后续动作、每帧状态和终局现金必须 100% 一致。未通过前禁止做路线价值训练。

## 4. UG2：Handoff Bridge

先实现 identity handoff：

```text
路线 A → Bridge → 路线 A
```

必须与不经过 Bridge 完全一致。之后才允许 A→B。

Bridge 负责继承或关闭：未完成任务、随身库存、已有建筑/动物/作物、当天义务、市场债务、小麦信用和终局项目。

每条 continuation route 必须具有 Entry Contract：可进入时间、最低现金、可接受资产范围、允许的债务和最大修复成本。不满足时拒绝切换。

## 5. UG3：当前覆盖审计

建立 checkpoint × route 反事实矩阵：

```text
valid
final cash / margin
relative gain vs continue
win/loss flip
repair cost
terminal closure
hard failures
low-tail performance
```

自动将弱点分为：

1. **Bridge gap**：经济路线可能有效，但接管接口失败；
2. **Route gap**：能合法执行，但没有任何路线改善结果；
3. **Selector gap**：Oracle 有好路线，公开信息却无法可靠选中；
4. **Opening gap**：信息出现时已经超过最后可救时刻；
5. **Execution bug**：异常 PASS、非法动作或债务错位。

必须分别报告 `Cov1`（至少一条低-regret路线）和 `Cov2`（至少两条行为独立路线）。

## 6. UG4–UG5：Continuation Genome V2 与 MAP-Elites

路线语言覆盖：快速作物、番茄、草莓、瓜、鸡蛋、牛奶、羊毛、混合经营、防守清算和市场压力。

Genome 只表达语义目标与接管策略，不编码 719 步固定动作：

- 相对进入时刻的三阶段动物、作物、土地和雇工目标；
- 现金/饲料储备、出售门、停止投资和清算；
- 已有超额动物、已有作物、库存与任务如何处理；
- 接管缓冲与最大修复成本。

使用 MAP-Elites 保留不同行为生态位，而不是只优化平均现金。任何硬错误直接淘汰；排序优先级为低尾部、最差状态簇、胜率、覆盖增量、复杂度。

## 7. UG6：Optionality Opening

搜索目标不是“开局自身完整局最高分”，而是第一个决策窗口到来时仍有多条后续路线可进入，并使未见族的低尾部收益最大。

重点基因：初始牛羊鸡、种子、小麦、雇工、建筑、现金保留、首次扩地和最早路线分化日。

## 8. UG7：Selector

第一版使用 bootstrap LightGBM ensemble：

```text
公开当前状态
+ 最近 3–5 天历史
+ 候选路线描述
+ Entry Contract 余量
→ Δmargin 分位数、胜率、硬失败概率
```

在线只按收益下置信界切换；OOD、模型分歧大、Bridge 不安全或收益不足时继续基线。初版最多一次主营路线切换，随后只允许进入防守清算。

## 9. UG8：冻结后对抗测试

路线库和 selector 冻结后，使用独立 seed 进化新的强收益、克制型和欺骗型对手。最终 frozen red-team 只用于验收，不允许再直接调参。

## 10. Champion / Challenger 门

任何 Challenger 依次通过：

1. 静态检查和单元测试；
2. Checkpoint restore；
3. Identity handoff；
4. takeover 硬错误 0；
5. 开发验证；
6. whole-family holdout；
7. 现有 34-Agent 原门非退化；
8. frozen red-team；
9. 官方 Python/JAX 逐步一致；
10. CPU 时限和跨局重入。

## 11. 当前实验状态

本次已完成 UG0 代码、9 个单元测试和合成语料 smoke test。测试中发现并修复了一个实质 Replay 解析问题：后座增量 observation 不能只与上一帧合并，否则 `step` 等公共字段会停留在旧值；现在每帧都以当前前座公共状态为基底，再叠加后座增量。

当前执行环境没有挂载本机 `D:` 盘，也没有 Kaggle 网络连接，因此没有伪造以下结果：

- 最新 submission ID；
- 实际下载文件哈希；
- 真实 Replay 数量和路线族分布；
- GPU/JAX 对局结果。

在本机执行 `run_ug0_windows.ps1` 后，`live_assets\ug0_corpus_v1\UG0_CORPUS_BASELINE_V1_ZH.md` 将成为 UG1 的输入与首份真实基线。
