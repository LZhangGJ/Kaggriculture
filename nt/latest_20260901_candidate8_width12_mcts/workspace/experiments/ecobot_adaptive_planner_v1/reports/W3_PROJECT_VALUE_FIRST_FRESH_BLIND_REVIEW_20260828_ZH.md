# W3 完整项目价值器：第一次新路线盲测复审

日期：2026-08-28  
结论：**工程验收通过，泛化验收失败；不得晋升。**

## 1. 本阶段验证了什么

本阶段把作物、动物、肥料收入以及种子、饲料、动物购入、动作、移动、雇工、土地和资金占用成本拆成 32 个基础字段（基线与候选各 16 个），接入 C++ 反事实数据和 LightGBM 项目排序器。

所有 310 维语料均通过以下检查：

- 原 278 维公开状态和候选字段逐值不变；
- 旧反事实终局标签逐字节不变；
- 新价值字段全部有限、非负；
- 解析总分重建最大误差为 1；
- 训练、校准和测试路线来源互不混用。

工程证据：

- `receipts/w6_project_value_breakdown310_hq60_cross4_prefix8_future64_upgrade_v42.json`
- `receipts/w6_project_value_breakdown310_calibration5_cross4_prefix8_future64_upgrade_v43.json`
- `receipts/w6_project_value_breakdown310_stable5_cross4_prefix8_future256_upgrade_v44.json`
- `receipts/w6_project_value_breakdown310_hq60_decomposition_audit_v45.json`
- `receipts/w6_project_value_breakdown310_calibration5_decomposition_audit_v46.json`
- `receipts/w6_project_value_breakdown310_stable5_decomposition_audit_v47.json`

## 2. 冻结模型与第一次独立盲测

冻结诊断模型为：

- `artifacts/w6_project_value_breakdown310_pairwise_policy_safety_v48.txt`
- `receipts/w6_project_value_breakdown310_pairwise_policy_safety_v48.json`

第一次盲测的 10 个路线家族在看到任何 EcoBot 对局结果前，仅按完整 719 步动作统计的多样性选定；选择过程不使用胜负、分差或对手身份路由：

- `receipts/w6_project_value_breakdown310_fresh_blind_opponents_10_v55.json`

随后对 10 个路线、4 个决策日、4 个前缀随机种子、双方座位、每个候选 256 次共同未来续跑，得到 264 个独立局面组：

- `artifacts/w6_project_value_breakdown310_fresh_blind10_cross4_prefix4_future256_v57.npz`
- `receipts/w6_project_value_breakdown310_fresh_blind10_cross4_prefix4_future256_v56.json`
- `receipts/w6_project_value_breakdown310_fresh_blind10_decomposition_audit_v58.json`

## 3. 盲测结果

| 指标 | 结果 | W3 门槛 | 判定 |
|---|---:|---:|---|
| 全部候选 pairwise accuracy | 64.19% | >=75% | 失败 |
| 分差 >=500 | 67.30% | 诊断 | 不足 |
| 分差 >=1,000 | 70.30% | 诊断 | 不足 |
| 分差 >=2,000 | 74.96% | 诊断 | 不足 |
| Oracle Top-4 召回 | 89.02% | >=90% | 失败 |
| Oracle Top-5 召回 | 90.15% | 参考 | 勉强通过 |
| 安全门后平均真实增益 | +886 | >0 | 通过 |
| 安全门后负增益率 | 14.39% | 越低越好 | 不足 |

结果证据：

- `receipts/w6_project_value_breakdown310_pairwise_policy_v48_fresh_blind10_raw_v59.json`
- `receipts/w6_project_value_breakdown310_pairwise_policy_v48_fresh_blind10_safety_v60.json`
- `receipts/w6_project_value_breakdown310_pairwise_policy_v48_fresh_blind10_bucket_audit_v61.json`

## 4. 失败集中在哪里

失败不是座位偏差，也不是牛羊价值完全失真：

- 双座位准确率接近，未发现明显座位泄漏；
- 牛项目约 81.5%，羊项目约 80.4%，相对稳定；
- 第 10–14 天明显较弱，其中第 10 天仅约 51.9%；
- 小麦约 57.4%，胡萝卜约 61.8%；
- 同一产业内仅改变规模的候选约 57.3%。

这说明 310 维总价值拆分能识别明显的跨产业差异，但仍没有可靠回答“已有这条产业线时，再增加或减少最后几单位是否值得”。扩大树数量没有解决该问题，因此主要瓶颈是训练覆盖与边际表达，而不是模型容量。

## 5. 门控决定与后续合同

1. V48 只保留为可回滚诊断模型，不接入最终 Agent。
2. 第一次盲测集从现在起降级为训练证据；以后不得再次称其为独立盲测。
3. 新增特征只允许由通用价值账本计算边际净收益、单位工作量回报和同产业规模变化，不允许使用路线作者、对手 ID 或固定日期特调。
4. 扩充训练路线后，必须重新预先冻结一套从未参与训练、校准、阈值选择的新路线盲集。
5. 只有新盲集同时达到 pairwise accuracy >=75% 和 Oracle Top-K 召回 >=90%，才允许进入 W4/W2 正式运行时接入。

