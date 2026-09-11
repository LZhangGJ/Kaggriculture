# FC12：部署端 CPU/JAX 一致性修复审计

日期：2026-08-22  
官方环境：`kaggle-environments==1.32.7`

## 结论

FC12G 第一版 CPU 打包器并没有完整复刻用于 JAX Arena 的 FC2B 市场策略，不能视为可部署候选。严格验收在第 480 步发现首个差异后，已补齐缺失的“非镜像盘面四步温和提前卖货”分支。修复后的 CPU v2 在冻结公开事件的两个座位上达到：

- 719 个决策步的全部动作字段完全一致；
- 每一帧官方公开/私有状态完全一致；
- 终局奖励完全一致；
- `action/state/reward exact rate = 100%`。

本修复只解决部署一致性，不宣称 FC12G 已达到全对手 90% 最终门。

## 首次失败证据

失败收据：

`receipts/fc12o_fc12g_jax_stepwise_parity_public_seed1811165014_v1.json`

首个差异：

- seed：`1811165014`；
- candidate seat：`0`；
- step：`480`；
- JAX 在第 8 个市场槽追加 `SELL MILK 12`；
- CPU v1 没有该订单；
- 下一帧我方现金：JAX `51,035`，CPU v1 `50,563`，立即相差 `472`。

因此，过去只比较“能完成一局”或“终局现金大致接近”的验收不足以证明部署等价。

## 根因

JAX FC2B 的 clone-aware 提前卖货逻辑分两档：

1. 当前公开农场完全镜像时，使用较强的四步提前卖货；
2. 公开农场已经分化时，使用较温和的四步提前卖货。

旧 CPU 打包器只修改了 K320 的固定常量，仍保留源码原有的 `clone_distance > 6` 时停用提前卖货逻辑。它没有实现 JAX 中第二档分支，所以在盘面分化后产生动作差异。

## 修复内容

修改：

`tools/build_fc12g_cpu_submission.py`

生成：

`artifacts/fc12g_cpu_v2/main.py`

CPU v2 根据当前可见公开农场距离选择两档参数：

| 条件 | 起始步 | horizon | distance limit | quantity cap | 商品 |
|---|---:|---:|---:|---:|---|
| 完全镜像 | 120 | 4 | 6 | 32 | 草莓、牛奶、羊毛 |
| 已分化 | 216 | 4 | 100 | 12 | 草莓、瓜、牛奶、羊毛 |

规则只读取：

- 当前公开农场；
- 当前公开市场；
- 我方当前私有库存；
- 我方冻结路线中未来计划出售的商品。

它不读取对手身份，也不读取未来随机事件。

生成文件 SHA-256：

`49C46F18CB7B035AF53391E67EFD4A53771D234E5112CF0B434BC7D1643D679A`

## 严格一致性验收

官方轨迹：

`receipts/fc12r_fc12g_cpu_v2_official_stepwise_traces_public_seed1811165014_v1.json`

JAX 对官方逐步验收：

`receipts/fc12s_fc12g_cpu_v2_jax_stepwise_parity_public_seed1811165014_v1.json`

结果：

| 指标 | 结果 |
|---|---:|
| 双座位上下文 | 2 / 2 |
| 动作完全一致 | 2 / 2 |
| 每帧状态完全一致 | 2 / 2 |
| 终局奖励完全一致 | 2 / 2 |
| 状态 | PASS |

## CPU 在线性能

收据：

`receipts/fc12t_fc12g_cpu_v2_acceptance_public_seed1811165014_v1.json`

| 指标 | 结果 |
|---|---:|
| 顺序动作调用 | 1,438 |
| 动作一致 | 100% |
| 平均推理 | 0.214 ms |
| P95 | 0.456 ms |
| P99 | 0.606 ms |
| 最大 | 1.825 ms |
| 验收门 | 1,000 ms |
| 状态 | PASS |

## 独立官方小样本

收据：

`receipts/fc12u_fc12g_cpu_v2_official_independent_seed595001_n4x2_v1.json`

对冻结 K320，4 个独立 seed、双方换座：

| 方案 | 胜/平/负 | 平均分差 |
|---|---:|---:|
| FC2B 部署源 | 6 / 0 / 2 | +776.75 |
| FC12G CPU v2 | 8 / 0 / 0 | +2,478.75 |

FC12G 救回 2 个源败局，没有伤害源胜局。该样本只作为官方执行方向验证，不能替代大样本 JAX Arena。

## 当前门控

状态：`DEPLOY_PARITY_PASS_BUT_FINAL_90_GATE_NOT_MET`

下一步仍必须完成：

1. 相同事件库的 FC2B 与 FC12G 全 28 对手逐局配对；
2. 对所有低于 90% 的唯一策略族做因果修复；
3. 候选修改后重新执行本报告的官方逐步一致性和 CPU 性能验收。
