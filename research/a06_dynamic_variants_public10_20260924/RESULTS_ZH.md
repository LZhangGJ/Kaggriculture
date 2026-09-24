# 对战结果：两个面板给出不同答案

## 13 个改良版内部循环赛

13 份 Agent，两两 100 局（50 个种子 × 双座位），共 78 组、7,800 局。每个 Agent 打 1,200 局。全部 719 步完成、零运行错误。详细名次、逐对手矩阵和种子重采样区间见 [`internal13/REPORT_ZH.md`](evaluation/internal13/REPORT_ZH.md)、[`RANKING.json`](evaluation/internal13/RANKING.json) 和 [`HEAD_TO_HEAD.csv`](evaluation/internal13/HEAD_TO_HEAD.csv)。

内部前三：Cashflow **96.33%**、AssetTail **71.83%**、R13 NML **65.50%**。Liquidity 在这批方案里只有 **19.75%**，排第 12。

## 冻结公开前十的外部面板

使用同一批公开对手、50 个新共同种子、双座位，每个方案对每名公开对手 100 局；四个方案各 1,000 局。四轮全部 719 步完成、零运行错误。原报告及逐对手现金差见 [`cashflow_public10/REPORT_ZH.md`](evaluation/cashflow_public10/REPORT_ZH.md) 与 [`three_public10/REPORT_ZH.md`](evaluation/three_public10/REPORT_ZH.md)。

| 方案 | 胜-平-负 / 1000 | 积分率 | 相对 Cashflow | 说明 |
|---|---:|---:|---:|---|
| R14 Liquidity | 840-0-160 | **84.0%** | +54.5 个百分点 | 8/10 对手各赢 95–98 局；M4A MetaV4 32/100，Master Engine V3 29/100。 |
| R14 TL5 | 379-0-621 | 37.9% | +8.4 个百分点 | 配对种子区间跨 0，尚不能说稳定胜过 Cashflow。 |
| R14 Cashflow | 295-0-705 | 29.5% | 基准 | 内赛第一，但不适应这组公开强对手。 |
| Rule R18 | 282-0-718 | 28.2% | -1.3 个百分点 | 配对种子区间跨 0。 |

公开十入口是 2026-09-23/24 冻结资格清单前十，**不是实时天梯前十**。其中两条 `main.py` 哈希相同，另有两条在本面板逐局结果完全相同。按九份不同源码去重，Liquidity 得分率 82.44%；按八组不同结果向量去重，80.50%。六局单进程复核的双方终局现金及赢家与正式并发记录完全一致。

最重要的结论是明显的对手池交互：方案间相互克制和开局资金触发效应很强。不能用 Cashflow 的内部 96.33% 宣称已达公开方案 90%，也不能用 Liquidity 的 84% 宣称对每个公开对手均强。下一轮应针对 M4A MetaV4 与 Master Engine V3 单独复盘，再用未见种子、不同公开版本和完整 Kaggle 沙箱验证。

评测中的调用耗时受并发争用影响，不能直接按本地峰值判 Kaggle 1 秒超时；这里也没有线上提交成绩。
