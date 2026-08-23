# FC4 当前执行栈路线救援审计

日期：2026-08-21

## 1. 结论

本轮没有产生可晋级候选，`CURRENT_BEST_RULE_CONFIG.json` 继续指向 FC2B。

旧反事实数据来自冻结 K320，而当前最佳 FC2B 已加入 clone-aware 预售层。把旧数据得到的规则直接接入 FC2B 后：

- Rank12：111/128 降到 110/128；
- K320 镜像：114/128 降到 112/128；
- PRT：保持 107/128；
- Rank14、G04、Flex 保持不变。

因此 FC4F 被拒绝，不能用“旧基线上的反事实有效”证明“当前最佳上有效”。

## 2. 正确栈反事实

重新以 FC2B 的 clone-aware 执行栈生成：

- 训练：1,024 seeds × 双座位 = 2,048 局；
- 留出 A：256 seeds × 双座位 = 512 局；
- 留出 B：256 seeds × 双座位 = 512 局；
- 决策时刻：state step 120；
- 候选：保持 FC2B，或锁定 `route4_two_yarn`；
- 线上可见信息：仅第120步真实公开状态；
- 无对手身份、无未来事件、无终局信息进入在线输入。

结果：

| 数据集 | FC2B基线 | 二元事后Oracle | Oracle上限增量 |
|---|---:|---:|---:|
| 训练 | 85.55% | 90.43% | +4.88 pp |
| 留出A | 85.74% | 91.99% | +6.25 pp |
| 留出B | 85.55% | 90.63% | +5.08 pp |

这证明两家纱店路线确实能救部分败局，但可救样本稀少，且错误切换会破坏大量原本胜局。

## 3. 规则优先验证

在29个具有明确经济含义的公开特征中，使用训练事件库进行5折 seed-grouped OOF；双方座位始终留在同一折。两个留出库不参与特征、方向和阈值选择。

最优规则仍是：

```text
state_step == 120 且 market_inventory[WOOL] <= 9983
→ 锁定 route4_two_yarn
否则保持 FC2B
```

| 数据集 | 基线胜局 | 规则胜局 | 净变化 | 救回 | 伤害原胜局 |
|---|---:|---:|---:|---:|---:|
| 训练 | 1,752 | 1,772 | +20 | 59 | 39 |
| 留出A | 439 | 446 | +7 | 19 | 12 |
| 留出B | 438 | 440 | +2 | 11 | 9 |

规则在两个留出库方向为正，但增益小、伤害率高，并在原关键库实跑中 Rank12 -1、K320镜像 -2，所以不晋级。

## 4. LGBM后置验证

在简单规则已经充分验证后，才训练无未来泄漏的二元 LGBM：

- 目标：`route4胜负 - FC2B胜负`；
- 5折按 seed 分组；
- 2,048训练样本，209个当前公开状态特征；
- 阈值只由训练 OOF 选择；
- 留出A/B完全不调参。

| 数据集 | LGBM胜率 | 相对基线 |
|---|---:|---:|
| 训练OOF | 86.57% | +1.03 pp |
| 留出A | 87.30% | +1.56 pp |
| 留出B | 85.94% | +0.39 pp |

LGBM最重要特征仍是 `market_inventory_7`，行为与单阈值几乎相同，没有学到新的稳定条件结构，因此不接入JAX、不进入CPU提交包。

## 5. 工程发现

把“整季719步 × 双方策略 × 6路线臂”合成单个 `fori_loop` 会产生超过6分钟的CPU编译，GPU长期空闲。二元路线臂改为逐步JIT后可运行，但64环境小批造成较高Python调度成本。

后续反事实工具应改为：

1. 单路线臂大batch；
2. 路线ID和开关作为运行时张量；
3. 策略与模拟器各自持久编译；
4. 不把整季与所有路线臂融合成一个超大XLA图。

## 6. 下一步

停止在第120步路线分类器上继续调参。下一阶段直接记录 FC2B 与 PRT/Rank12 的逐日、逐商品出售意图与公开市场变化，定位具体是哪一种商品、哪一批次和哪一个出售窗口造成败局，再建立可消融的市场动作修正。

## 7. 证据文件

- `receipts/fc4f_rank14_plus_rank12_threshold_key6_seed544501_n64x2_v1.json`
- `receipts/fc4g_fc2b_rank12_binary_route4_cf_step120_train_seed546001_n1024x2_v1.json`
- `receipts/fc4h_fc2b_rank12_binary_route4_cf_step120_holdout_a_seed548001_n256x2_v1.json`
- `receipts/fc4h_fc2b_rank12_binary_route4_cf_step120_holdout_b_seed548257_n256x2_v1.json`
- `receipts/fc4i_fc2b_rank12_binary_public_threshold_rule_step120_n1024_v1.json`
- `receipts/fc4j_fc2b_rank12_binary_score_delta_lgbm_step120_n1024_v1.json`
- `artifacts/fc4j_fc2b_rank12_binary_score_delta_lgbm_step120_n1024_v1.joblib`
