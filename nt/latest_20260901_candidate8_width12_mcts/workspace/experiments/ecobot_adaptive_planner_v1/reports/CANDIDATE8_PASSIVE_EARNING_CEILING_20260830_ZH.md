# Candidate8 静止对手赚钱上限测试

日期：2026-08-30  
状态：**C++ exact-future Oracle 容量测试通过；非在线可提交成绩。**

## 测试设置

- 对手：`PASSIVE_NOOP`
- 对手行为：719 步全部 `PASS`，无 hand 动作、无市场订单
- 仍保留官方城镇需求、商店和随机事件过程
- 我方：Candidate8 最新版
- 决策日：`0, 1, 3, 6, 9, 12, 18, 24`
- 深度：8
- Beam：32
- 每节点多家族候选：64
- 宏观修改持久化，Candidate8 物理上限与 R6 基线解耦
- 三个独立 seed bank，共 8 seeds、双座位、16 局

## 结果

| 指标 | 结果 |
|---|---:|
| 局数 | 16 |
| 我方平均终局现金 | **185,704.25** |
| 我方中位终局现金 | **185,838** |
| 我方最低终局现金 | **180,262** |
| 我方最高终局现金 | **190,439** |
| 静止对手终局现金 | 3,000 |
| 平均分差 | +182,704.25 |
| 完整终局续跑 | 128,213 |

三个 seed bank 的平均分差分别为：

- `+183,095`
- `+183,734`
- `+181,994`

16 局终局现金全部落在 `180,262–190,439`，不存在依靠少数极端局抬高平均值的问题。

## 结论边界

本测试证明当前候选语法、跨阶段组合和执行器在无对手干扰时，已经能稳定表达约 `18.0万–19.0万` 的经营结构，观测最高值为 `190,439`。

但每个 seed/座位都由 exact-future Oracle 分别续跑候选并挑选终局现金最高的组合，因此这不是一条固定路线，也不是线上选择器能够直接获得的成绩。它回答的是“当前规划表达上限够不够”，不能回答“只看当前公开状态是否选得出来”。

## 回执

- `candidate8_sequence_passive_seed2901001_n2x2_days01369121824_depth8_beam32_shortlist64_persistent_decoupledcaps_v1.json`
- `candidate8_sequence_passive_seed2902001_n2x2_days01369121824_depth8_beam32_shortlist64_persistent_decoupledcaps_v1.json`
- `candidate8_sequence_passive_seed2903001_n4x2_days01369121824_depth8_beam32_shortlist64_persistent_decoupledcaps_v1.json`

