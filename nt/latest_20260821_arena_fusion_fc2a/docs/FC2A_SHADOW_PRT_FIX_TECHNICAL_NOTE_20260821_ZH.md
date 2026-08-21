# FC2A Shadow-PRT 修复版详细说明

## 一句话结论

修复版**从第 1 步就调用并更新 PRT**，但在切换条件成立前，**PRT 只做影子计算，不负责输出动作**。这既保留了 PRT 完整的内部历史，又不会让它提前改变 K320/X562 的原路线。

## 原版为什么卡死

原 FC2A 的压力路线是：

```text
默认 K320
→ 观察到公开重羊开局
→ step 168 在 X562 / Legacy 中选前缀
→ step 288 视现金、YARN_STORE 和草莓市场库存决定是否切 PRT
```

问题在于 PRT 是有状态 Agent。原版直到 step 288 才第一次调用 PRT，因此 PRT 不知道此前 12 天发生了什么，也没有正确维护自己的路线、日界和市场事务状态。它在第 13 天接管后漏掉 HIRE，随后无 hand 可用，从 frame 289 到 719 连续 431 帧全 PASS。

Public 提交 `55662157` 的 10 局 Replay 中，两个败局都满足同一联合条件，并出现完全相同的 431 帧空转尾巴。该现象又在官方 1.32.7 的本地 seed `545102` 双座位逐帧复现。

## 修复后的逐步执行逻辑

### step 0--95

每一步都先正常计算 K320。开局检测窗内，如果公开盘面满足：

```text
24 <= step < 96
对手羊 >= 4
对手牛 <= 2
对手现金 <= 1000
```

就把 `sheep_pressure` 设为持久真值。

只要压力分支仍有可能启用，修复版同时调用：

- PRT：从开局同步内部状态；
- X562：同步可能使用的前缀状态；
- Legacy：同步另一条可能使用的前缀状态。

此时最终返回的动作仍按既有 FC2A 路由决定，影子调用不会直接下达动作。

### step 96 以后且从未触发重羊压力

立即走 K320 单热路径，不再继续计算无用的 X562、Legacy 和 PRT。这样普通局不会整局承担四套 CPU Agent 的开销。

### step 168

只有已经触发重羊压力的局面才选择压力前缀：

```text
有 YARN_STORE → Legacy
无 YARN_STORE → X562
```

### step 288

只在重羊压力路线中判断 PRT 接管条件：

```text
我方现金 <= 13,376
YARN_STORE 数量 <= 1
草莓市场库存 <= 9,979
```

条件成立后开始返回当步已经计算好的 `prt_action`。由于 PRT 从 step 0 起持续接收真实 observation，它拥有完整路线和日初事务状态，不再是中途冷启动。

## 修复不是哪一种做法

- 不是在第 289 步硬编码 `HIRE 5`；
- 不是提前让 PRT 控制所有动作；
- 不是从未来 Replay 恢复 PRT 状态；
- 不是按对手姓名或提交 ID 选择路线；
- 不是只让 PRT 从 step 288 开始“补算”。

## 官方复验

官方 1.32.7、seed `545102`、双方换座：

| 版本 | 我方现金 | 对手现金 | 非 PASS 帧 | 最长全 PASS | frame 289 |
|---|---:|---:|---:|---:|---|
| 原 FC2A | 13,313 | 118,351 | 261 | 431 | 全 PASS |
| Shadow-PRT | 70,382 | 68,695 | 692 | 10 | 卖牛奶、雇 5 人、取小麦 |

两种座位得到相同的语义结论。验收回执为：

- `workspace/experiments/fusion_champion_v1/receipts/fc2a_shadow_prt_fix_seed545102_20260821_v1.json`
- `workspace/submission/55663355_fc2a_shadow_prt_fix/known_failure_fix_receipt.json`

## 代码与版本

- 故障版：`workspace/submission/55662157_fc2a_public_state_fusion_BROKEN/main.py`
- 修复版：`workspace/submission/55663355_fc2a_shadow_prt_fix/main.py`
- 修复版号：`FC2A-rank14-plus-anti-mirror-cpu-v2-shadow-prt`
- 修复版 `main.py` SHA-256：`6B1B81924875F6C9207E58FE15EC72384DD46C39BC2480EFABAA148D3BC76F8C`
- 修复版提交包 SHA-256：`E6EFCC12701B62CC88A9F32C79D331E9D8A5F7F39E3CECC2480A032FBEE5A864`

## 尚未证明的事情

该修复只消除了状态型子策略晚接入导致的确定性卡死。它没有证明：

- FC2A 已经成为本地最强；
- 修复版对 28 个冻结 Agent 都达到 90%；
- CPU 融合逻辑与 JAX FC2B 的所有分支完全相同；
- Public 分数一定恢复到故障前预期。

因此后续仍需把“能正常完成经营”与“策略强度通过”分开验收。
