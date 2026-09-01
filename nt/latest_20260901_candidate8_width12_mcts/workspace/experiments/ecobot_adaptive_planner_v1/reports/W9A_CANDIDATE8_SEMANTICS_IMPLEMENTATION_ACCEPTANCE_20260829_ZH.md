# W9A：八类通用候选语义实现验收

日期：2026-08-29（JST）  
结论：**PASS——八类能力均已进入统一 `AdaptivePlanDelta` 热路径，不是八套作者路线或 Replay 播放器。**

## 1. 边界

这次迁移的是从三代强者 Replay 中观察到的“决策自由度”，不是动作表、固定数量、固定日期或坐标。运行时接口没有作者、排名、Submission、Replay 或对手身份字段；候选只读取当前公开/自身状态、资源约束和现有计划。

三代来源审计保留在：

- `w8_rank1_55623460_candidate_capability_audit_v1.json`；
- `w8_rank1_55714246_candidate_capability_audit_v1.json`；
- `w8_rank1_55829779_candidate_capability_audit_v1.json`；
- `W8_RANK1_THREE_VERSION_CANDIDATE_CAPABILITY_EVOLUTION_20260829_ZH.md`。

它们只作为能力需求证据，不进入提交时热路径。

## 2. 八项能力与真实实现

| 能力 | 参数化表达 | 真实执行效果 | 验收 |
|---|---|---|---|
| 同计划多调度/布局 | 8 个 `ScheduleProfile` | 修改任务打分、区域均衡、全局匹配、任务链连续性、紧凑布局/服务通道 | PASS |
| 连续扩/缩产 | 8 产业 × `±1/2/3/4/6/8` | 修改作物/动物目标，重新计算现金、deadline、行动负载 | PASS |
| 单边开/停项目 | `UNILATERAL` | 可只增、不减；或释放高于不可逆下限的未来承诺 | PASS |
| 两/三产业联合调整 | 稀疏 `target_delta[8]` | 与所需 hand、土地容量在同一事务内校验 | PASS |
| NOW/延期/停止追加 | `effective_delay_days=0/1/2/-1` | 延后可逆承诺或停止未来追加，不删除已经投入资产 | PASS |
| 有序市场事务 | `MarketProfile` + `market_item` | SELL/HOLD/PARTIAL/SELL_TO_FINANCE；融资候选可和扩产、雇工、扩地组合 | PASS |
| 阶段后缀重选 | `suffix_project` + 局部增减 | 只重选未来作物后缀，不替换整条经营路线 | PASS |
| 局部偏差恢复 | issue × REPAIR/DEFER/ABANDON | 只改变受影响任务的调度，不编辑其他产业目标或容量 | PASS |

## 3. 统一流水线

```text
当前状态 + 当前经营计划
        ↓
约 300–1,000 条结构化 PlanDelta
        ↓
不可逆下限 / cap / deadline / 现金 / 土地 / 人员 / 市场槽过滤
        ↓
去重后的可行池
        ↓
反事实校准的软配额 + 解析价值 + 类别多样性
        ↓
最多 64 条线上候选
        ↓
统一计划器 → 任务池 → 调度器 → 有序事务编译器 → 原子动作
```

关键源码：

- `fast_kaggriculture/src/adaptive_candidates.hpp`：统一 schema；
- `fast_kaggriculture/src/adaptive_candidates.cpp`：生成、合法性、现金/容量联合校验、去重和软配额；
- `fast_kaggriculture/src/native_adaptive.cpp`：状态投影、真实计划/调度/市场/恢复执行、冻结前缀反事实；
- `fast_kaggriculture/src/bindings.cpp`：Python 审计接口。

## 4. 自动配额校准

初版软配额为 `[1,8,14,6,18,5,6,3,3]`。多未来反事实校准没有按 Replay 出现频率分配，而是按完整可行池的实际终局价值漏选做确定性一格坐标下降，得到：

`[1,8,15,5,18,5,6,3,3]`

对应 KEEP、调度、连续规模、单边、多产业、时间、市场、后缀、恢复。

24 个稳定校准状态中：

- 1,000 元内近最优覆盖：70.83% → 75.00%；
- 平均 regret：906.93 → 898.02；
- 4 个留出中盘状态无退化。

剩余 11 次 oracle 漏选全部属于 `MULTI_PROJECT`，说明瓶颈已不是类别配额，而是同一多产业家族内部的价值排序。继续机械增加配额不能解决。

机器凭证：`w9_candidate8_soft_quota_calibration_multifuture_v2.json`。

## 5. Schema 不是空字段

`w9_candidate8_behavior_effects_passive_v2.json` 对 8 个非 KEEP 家族逐一强制执行。每类均改变了真实 719 步联合动作轨迹，首次差异发生在候选决策后，且：

- `avoidable_crop_losses = 0`；
- `avoidable_animal_losses = 0`；
- `end_overflow = 0`。

测试套件最终 `20 passed`。其中额外验证：

- 低现金但有可出售库存时，扩产可组合 `SELL_TO_FINANCE`；
- 无库存时该融资扩产被过滤；
- 恢复候选不修改八个产业目标、hand、土地或市场配置；
- shortlist 确定、签名唯一、无来源/身份字段。

## 6. 本阶段未宣称的内容

本验收只证明“能提出并正确执行这些选择”。它不证明当前启发式或树模型已经能在每个局面选中最好的一条；选择质量在 W9C 单独验收。
