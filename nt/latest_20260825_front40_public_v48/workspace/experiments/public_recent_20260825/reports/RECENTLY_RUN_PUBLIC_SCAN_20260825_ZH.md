# 2026-08-25 Recently Run 公开方案扫描与 Kaito V48 验收

## 1. 结论

- 已顺序下载 34 份公开 Notebook，保留原始 `.ipynb`、导出源码和 Kaggle 元数据。
- 16 个可直接运行的候选先用官方 `kaggle-environments==1.32.7` 做 4 seed × 双座位方向筛选；所有对局都正常完成。
- 唯一明确的新强力家族是 Kaito V48。`Adaptive Shop Guard` 与它在初筛中逐局结果完全相同，`Shops Remember` 只有很小的市场包装差异，因此按同一行为家族处理。
- Kaito V48 的 6 条 719 步生产路线均为此前 Bank 中不存在的新动作哈希，已全部加入冻结 Bank。
- Kaito V48 JAX 版本已经通过双座位逐动作、逐状态、终局奖励完全一致验收。
- 独立随机库双座位共 1,024 局，Kaito V48 对 FC24B 为 `851-173`，胜率 `83.11%`，95% Wilson 区间 `[80.69%, 85.28%]`，平均现金差 `+6,237`。

因此 Kaito V48 已达到融合供体门，不是四五局偶然获胜，也不是近似迁移。

## 2. 下载覆盖

本轮下载共 34 份，主要分为四类：

1. 新动态路线 Agent：Kaito V48、Adaptive Shop Guard、Shops Remember；
2. 已知家族的新包装或更新：Prvsiyan Soil/Moon、Mugundh V21、Tetsutani、Steven、EcoBot 等；
3. 重复单路线：Breaking the Tie、Premium Queue、Salem 3000、Amey Deterministic 等；
4. 研究/工具：KagSim、Top12 X-ray、No Yarn、Market 59、Live Meta、validator、成本和价格分析。

完整目录名和原始内容保存在：

`public_notebooks/recent_latest_20260825_scan_v1/`

## 3. 官方小样本方向筛选

官方筛选只用于排除明显弱或损坏方案，不用它宣称稳定排名。

| 候选 | 结果 | 胜率 | 平均现金差 | 判断 |
|---|---:|---:|---:|---|
| Kaito V48 | 7/8 | 87.5% | +5,717 | 新强家族，进入精确迁移 |
| Adaptive Shop Guard | 7/8 | 87.5% | +5,717 | 与 Kaito V48 同家族 |
| Shops Remember | 7/8 | 87.5% | +5,663 | Kaito 家族近变体 |
| Prvsiyan Soil V139 | 1/8 | 12.5% | -2,771 | 高赞更新，但不足以当供体 |
| Prvsiyan Moon V135 | 1/8 | 12.5% | -2,771 | 与 Soil 当前行为等价 |
| 其余 11 个可运行候选 | 0/8 | 0% | 均为负 | 不进入供体池 |

机器可读回执：`receipts/official_recent_vs_fc24b_seed1301001_n4x2_v1.json`。

## 4. Kaito V48 迁移语义

迁移保留的公开状态逻辑包括：

- 根据已经公开出现的 Yarn / Farmers / Bakery 等商店前缀选择 6 条完整路线；
- 路线切换时重置该路线自己的杂草修复事务；
- 根据公开对手盘面识别一个 Bakery capital 分支；
- 近镜像持续成立时执行两步销售抢跑，并用 4 槽债务环偿还未来销量；
- 使用官方 1.32.7 的胡萝卜、番茄、鸡蛋 hinge 市场曲线排序销售；
- 终局市场碰撞清算。

正式 Agent 不读取玩家身份、Replay ID、随机种子或未来商店。

第一次一致性检查发现 step 714 的两个销售槽顺序不同，但状态和收益完全相同。根因不是 JAX 并行误差，而是旧价格 LUT 未更新到 1.32.7 的三条 hinge 曲线。修正后对 9 种产品 × 98,304 个库存点逐项对照为 0 mismatch，随后严格一致性通过。失败回执和 debug 回执继续保留，作为规则修复审计链。

严格通过回执：`receipts/kaito_v48_jax_parity_seed1302001_n2x2_v3.json`。

## 5. 独立 1,024 局结果

| 最终路线 | 局数 | 胜局 | 胜率 | 平均现金差 |
|---|---:|---:|---:|---:|
| default | 623 | 538 | 86.4% | +7,600 |
| yarn_fast | 141 | 134 | 95.0% | +5,585 |
| farm_fast | 135 | 115 | 85.2% | +5,521 |
| yarn_second | 101 | 63 | 62.4% | +2,627 |
| yarn_third | 5 | 0 | 0% | -14,088 |
| bakery_capital | 19 | 1 | 5.3% | -3,942 |

全局结果为 83.11%，但分支并不都强。`yarn_third` 样本极少且明显失败，`bakery_capital` 也对 FC24B 退化；后续融合时不能整体照搬 V48 路由器。应优先保留 default、yarn_fast、farm_fast，把 yarn_second 作为候选对照，并对两个弱分支使用 FC24B 或新供体回退。

正式回执：`receipts/kaito_v48_vs_fc24b_seed1304001_n512x2_v1.json`。

## 6. 研究型 Notebook 的可用启发

- KagSim：公开作者声称固定动作流下可达到约 2,000 episodes/s/core，并针对 1.32.7 做 bit-exact 修补；它更适合动作流/路线搜索，不等于已支持动态 Python Agent。
- Top12 X-ray：当前前排会在中盘按商店和市场分叉，单纯固定 719 步路线已落后；物流移动约占大量动作，动态差异主要集中在有限事件点。
- No Yarn：约三分之一城镇可能没有 Yarn；商店抽样和杂草随机数消耗耦合，所以不能只由 seed 预生成未来商店并用于在线 Agent。
- Market 59：动物稳态收益率不能代替整季需求上限；产品组合和出售日期仍是核心分支。

这些结论用于设计候选和验收，不直接当作已经复现的胜率事实。

## 7. 下一步

1. 把 Kaito V48 的三个强分支加入强力供体池；
2. 把完整 V48 加入强制验收对手池，因为整体对 FC24B 远高于 10%；
3. 对其余 Recently Run 候选只继续处理真正不同且方向筛选不差的家族；
4. 新综合 Agent 必须在未参与路由选择的新随机库中逐家族达到 90%，不能只看总体平均。

