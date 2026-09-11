# W9 总验收：Rank1 抽象八类候选能力

日期：2026-08-29（JST）  
最终状态：**实现完成；候选层验收通过；线上选择器未通过，未修改当前冠军默认行为。**

## 1. 最简结论

这次已经把规划器从原来的“2/4 单位、一进一出换产业”扩展成可以提出：

1. 同一经营计划的多套人员调度/布局；
2. 八产业 `±1/2/3/4/6/8` 连续规模；
3. 只扩、只缩、保持；
4. 两/三产业 + 人员 + 土地的联合变化；
5. 立即、延期、停止追加；
6. 先卖货融资，再购买/雇工/扩地的有序事务；
7. 中期/终局后缀局部重选；
8. 杂草、部分成交、人员错位、任务断链的局部恢复。

它们都能真实改变 719 步动作轨迹，不是只加了字段，也没有复制强者路线、日期、坐标或目标数量。

## 2. 交付索引

### 核心实现

- `fast_kaggriculture/src/adaptive_candidates.hpp`
- `fast_kaggriculture/src/adaptive_candidates.cpp`
- `fast_kaggriculture/src/native_adaptive.hpp`
- `fast_kaggriculture/src/native_adaptive.cpp`
- `fast_kaggriculture/src/bindings.cpp`
- `fast_kaggriculture/python/fast_kaggriculture/__init__.py`
- `fast_kaggriculture/tests/test_adaptive_candidates.py`

### 新增审计工具

- `audit_candidate8_oracle_recall.py`
- `audit_candidate8_multifuture_oracle.py`
- `audit_candidate8_behavior_effects.py`
- `audit_candidate8_ordered_financing.py`
- `calibrate_candidate8_soft_quotas.py`
- `train_candidate8_shortlist_ranker.py`
- `benchmark_candidate8_online_budget.py`

### 验收报告

- `W9A_CANDIDATE8_SEMANTICS_IMPLEMENTATION_ACCEPTANCE_20260829_ZH.md`
- `W9B_CANDIDATE8_EXECUTOR_SAFETY_ACCEPTANCE_20260829_ZH.md`
- `W9C_CANDIDATE8_COUNTERFACTUAL_SELECTION_ACCEPTANCE_20260829_ZH.md`
- 本报告。

## 3. 最终验收表

| 验收项 | 结果 | 核心证据 |
|---|---|---|
| 8 类 schema 和生成 | PASS | 20 个测试；每类均有可行候选 |
| behavior-effect | PASS | 8 类均改变完整动作轨迹 |
| 不可逆/现金/deadline/容量 | PASS | 过滤测试 + 1,024 局硬错误 0 |
| 资源和 shed 预留 | PASS | 多单位统一容量保护；失败前缀归零 |
| 有序融资事务 | PASS | 同帧 SELL 在 BUY 前 |
| 局部恢复隔离 | PASS | 不编辑未受影响产业和容量 |
| 软配额自动校准 | PASS | 多未来价值漏选驱动，不按 Replay 频率 |
| 静止对手 shortlist | PASS（近最优） | exact 75%，1,000 内 87.5%，regret 281.83 |
| 强对手 shortlist | FAIL | G001/G017/G231 存在显著多产业漏选 |
| 线上 1 秒速度 | PASS | 生成 + 暖推理 P95 58.44 ms |
| 当前排序模型 | FAIL | 留出 regret 比 heuristic 更差 |
| 宏观 Self-play | HOLD | 先补稳定竞争标签和跨对手排序 |

## 4. 有效成果

最重要的成果不是“现在多了 64 个 if”，而是把可搜索动作单位从固定路线切换改成了统一参数化 `当前计划 + ΔPlan`。完整可行池在被测状态中相对 KEEP 的平均机会增益约为：

- 静止对手：+7.8k；
- G001：+6.4k；
- G017：+13.0k；
- G231：+17.9k。

因此候选空间扩展是真实有效的。当前失败点已经清晰移动到“如何根据对手和市场从大量多产业组合中挑对”，不是规划器无候选可选。

## 5. 未解决风险

1. 当前 C++ 多未来数据主要以我方终局现金排序，下一版必须加入胜负、分差和对手现金；
2. 32-future 标签稳定，但状态数量仍太少；
3. 多产业家族内部的解析估值与真实续跑价值偏差很大；
4. 现有 ExtraTrees 只证明推理速度可行，没有证明预测质量；
5. 强对手的小样本覆盖明显不足，不能把离线 oracle 当成可提交胜率。

## 6. 推荐下一步

保持当前 R6/冠军默认策略不变，新增一个独立工作包：

```text
多对手、前/中/后期冻结状态
→ 每状态 32 个共同未来
→ 输出胜率、分差、我方现金、硬失败
→ 训练候选排序器
→ 留来源、留对手验收
→ 通过后才启用在线候选选择
→ 再进入宏观 Self-play
```

在此之前，不再继续向候选生成器添加金牌特调规则，也不启动 PPO。
