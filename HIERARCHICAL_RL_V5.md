# Kaggriculture 学习型候选与软意图 RL V5

V5 按依赖顺序实现三项能力：9「agent 内部 trace / 逆规划软标签」、8「专家候选注入」、
7「CNN 学习型候选 proposal」。V4 的任务状态机、真实 ETA、BFS/预约 A*、可变现净资产
奖励和每日 KEEP/SWITCH 均保留。

## 1. 统一 trace 与逆规划 Top-M 标签

`hierarchical_trace.py` 定义 `kaggriculture.agent-trace.v1`。每一步记录：

- step/day、策略模式、教师来源和原始动作；
- 每个工人的任务类型、目标格、物品、ETA、路线和证据；
- 一个归一化 Top-M 意图分布及其置信度。

本地 V5 策略直接从 `UnitTaskState` 写 confidence=1 的内部 trace。黑盒 Python/JAX
专家没有内部状态时，逆规划器在同一天的有限 horizon 内回看后续语义操作，把移动、
等待、PICKUP 和最终操作合并。候选按未来距离、路线偏差、即时操作和 PICKUP 折扣评分，
保留 Top-M 并归一化。BC 对整个概率分布做 soft cross entropy，不再丢弃歧义样本或把
移动前缀硬标成一个任务。

默认 BC 会同时输出：

- `*.traces.jsonl`：逐步 trace；
- `*.diagnostics.json`：候选闭包审计；
- `*.pt`：V5 checkpoint。

## 2. 专家候选注入

训练时把 Top-M 假设转换为 owner-bound `TaskCard`，标记为 `CandidateOrigin.EXPERT`。
64 卡池使用显式配额：最多 8 张专家卡、24 张学习卡，其余保留给 continuation、mandatory
和规则/状态候选。专家卡只保证标签进入训练候选池，不直接决定最终匹配动作。

为防止网络依赖 oracle，BC 的 expert dropout 默认从 0.10 线性增加到 0.80。每轮同时
计算两套指标：

- `oracle.soft_closure_rate`：注入完整专家卡后的可覆盖概率质量；
- `deploy.soft_closure_rate`：只用规则、状态和 CNN proposal 的可覆盖概率质量；
- `injection_gap`：两者差值，是 proposal 尚未学会复现专家候选的直接度量。

选模型应优先看 deploy closure 和 held-out agent family 表现，而不是只看 oracle closure。

## 3. CNN 学习型候选 proposal

共享双棋盘 CNN 得到 self/opponent spatial maps。proposal 分支拼接 self、opponent、差值、
绝对差值，再输出：

- 13 类空间任务热图 `[B, 13, 10, 10]`；
- 物品分布 `[B, len(ITEMS)+1, 10, 10]`；
- urgency、ETA、value、slack 四张辅助图。

推理为两阶段：先编码棋盘并产生热图，再按 task-family quota 解码 Top-K 学习卡，与规则、
状态候选合并，最后由 Candidate Transformer 做全局 worker-task matching。Top-K 在第一版
中不可微；proposal 通过 focal task loss、item cross entropy 和 ETA/value/slack Huber loss
预训练。RL 阶段继续使用已学 proposal，但离散 Top-K 暂不接 policy gradient。

学习卡会经过基本的边界、锁地、物品类型、ETA 和 pair-mask 检查。执行器仍是最终安全层，
其失败日志用于发现 proposal 与真实前置条件之间的差距。

## 4. 训练顺序

```text
专家对局 → JSONL trace → 逆规划 Top-M → CNN proposal 监督
        → 带 expert dropout 的 soft-label BC → 无专家候选自博弈 RL
```

Windows/CUDA 示例：

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe scripts\train_hierarchical_bc.py `
  --teachers starter path\to\agent.py `
  --inverse-top-m 3 --proposal-top-k 24 --device cuda `
  --output artifacts\hierarchical_bc_v5.pt

.venv\Scripts\python.exe scripts\train_hierarchical_rl.py `
  --init-checkpoint artifacts\hierarchical_bc_v5.pt `
  --proposal-top-k 24 --device cuda `
  --output artifacts\hierarchical_a2c_v5.pt
```

V3/V4 checkpoint 可用 `strict=False` 迁移，新增 proposal 参数随机初始化；V5 checkpoint
严格加载。正式训练应使用行为聚类后的分层采样，并单独报告每个 agent cluster、每个任务
类型的 deploy closure、执行成功率、路线冗余和最终现金胜率。
