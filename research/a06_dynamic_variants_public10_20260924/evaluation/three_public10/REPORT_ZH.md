# 三个 A06 改良版对冻结公开前十，各 100 局

Rule R18、R14 Liquidity、R14 TL5 均使用原交付包的冻结 `main.py` 与依赖。
公开对手与 Cashflow 轮相同，使用相同的 50 个种子、双座位；官方 Kaggriculture 1.32.7 Python 裁判，实时策略对战。
全部 3,000 局完整走满 719 步且零运行错误。

## 总体结果

| 方案 | 胜-平-负 / 1000 | 积分率 | 95% 种子区间 | 较 Cashflow 差距 | 差距 95% 区间 | 平均终局现金差 |
|---|---:|---:|---:|---:|---:|---:|
| `r14_liquidity` | 840-0-160 | 84.00% | 80.80%–86.70% | +54.5 个百分点 | +45.1～+64.1 个百分点 | +12,375.9 |
| `r14_tl5` | 379-0-621 | 37.90% | 28.00%–48.20% | +8.4 个百分点 | -5.2～+22.8 个百分点 | -2,215.2 |
| `rule_r18` | 282-0-718 | 28.20% | 18.70%–38.10% | -1.3 个百分点 | -14.5～+11.8 个百分点 | -2,668.4 |
| `r14_cashflow`（参照） | 295-0-705 | 29.50% | 见上一轮报告 | — | — | -3,674.2 |

## 逐对手胜场 / 各 100 局

| 冻结顺位 | 公开方案 | Rule R18 | R14 Liquidity | R14 TL5 | Cashflow 参照 |
|---:|---|---:|---:|---:|---:|
| 1 | [prvsiyan/kaggriculture-frontier-the-soil-remembers-rain](https://www.kaggle.com/code/prvsiyan/kaggriculture-frontier-the-soil-remembers-rain) | 29 | 98 | 41 | 34 |
| 2 | [wzhengbiao/kaggriculture-v15stack-submit](https://www.kaggle.com/code/wzhengbiao/kaggriculture-v15stack-submit) | 27 | 98 | 40 | 33 |
| 3 | [haideptry/the-2965-master-hybrid-engine](https://www.kaggle.com/code/haideptry/the-2965-master-hybrid-engine) | 27 | 95 | 37 | 32 |
| 4 | [ahmedberatozer/kaggriculture-v57-funding-order-invariant](https://www.kaggle.com/code/ahmedberatozer/kaggriculture-v57-funding-order-invariant) | 25 | 98 | 43 | 29 |
| 5 | [hosen42/kaggriculture-m4a-metav4-sr18-2690-1](https://www.kaggle.com/code/hosen42/kaggriculture-m4a-metav4-sr18-2690-1) | 27 | 32 | 25 | 35 |
| 6 | [tetsutani/demand-preserving-turn-sale-timing](https://www.kaggle.com/code/tetsutani/demand-preserving-turn-sale-timing) | 29 | 98 | 41 | 34 |
| 7 | [guruprasaathas111/kaggriculture-master-engine-v3](https://www.kaggle.com/code/guruprasaathas111/kaggriculture-master-engine-v3) | 43 | 29 | 26 | 14 |
| 8 | [dmitriigluzdov/kaggriculture-more-wheat-smarter-sales](https://www.kaggle.com/code/dmitriigluzdov/kaggriculture-more-wheat-smarter-sales) | 25 | 96 | 40 | 26 |
| 9 | [lynnsakurai/farmer-john-and-the-idle-seller](https://www.kaggle.com/code/lynnsakurai/farmer-john-and-the-idle-seller) | 25 | 98 | 43 | 29 |
| 10 | [nathanjacob/kaggriculture-pipe18-six-layers](https://www.kaggle.com/code/nathanjacob/kaggriculture-pipe18-six-layers) | 25 | 98 | 43 | 29 |

## 核验与边界

`PROTOCOL.json` 冻结候选与对手文件哈希、裁判哈希和种子；`games.jsonl` 保留全部逐局结果。
`RESULTS.json` 包含逐对手现金差、双座位成绩与配对种子重采样结果。
公开十方案是 2026-09-23/24 冻结资格面板中的前十，不代表实时天梯前十。
其中 Soil Remembers Rain 与 Demand-Preserving Sale Timing 的可运行 `main.py` 哈希相同；十条结果仍按用户要求分别呈现。
本地并发运行的单步耗时受机器争用影响，不能直接视为 Kaggle 容器超时结论。
Farmer John 与 Pipe18 Six Layers 在本面板中的逐局双方终局现金也完全相同；只能说明这些测试盘面输出一致。
按 9 份不同 `main.py` 去重后，Liquidity 积分率 82.44%；按本次 8 组不同结果向量去重后为 80.50%。
单进程复跑 6 局，双方终局现金与正式并发评测逐局完全一致。
