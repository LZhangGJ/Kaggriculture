# R14 Cashflow vs 冻结公开前十

对手是 2026-09-24 交付包中 `QUALIFIED_TOP20.json` 的前十，不代表此刻 Kaggle 的实时前十。
双方均按实时观察运行；官方 Kaggriculture 1.32.7 Python 裁判，50 个新种子、每种子交换座位。
全部 1,000 局完成 719 步且零错误。胜 295、平 0、负 705；
整体胜率 **29.50%**，胜一分、平半分的积分率 **29.50%**。
按共同种子重采样的整体积分率 95% 区间：20.50%–39.00%。

| 冻结顺位 | 公开方案 | Cashflow 胜-平-负 | 积分率 | 座位 0 / 1 胜 | 平均终局现金差 |
|---:|---|---:|---:|---:|---:|
| 1 | [prvsiyan/kaggriculture-frontier-the-soil-remembers-rain](https://www.kaggle.com/code/prvsiyan/kaggriculture-frontier-the-soil-remembers-rain) | 34-0-66 | 34.0% | 18 / 16 | -3,241.7 |
| 2 | [wzhengbiao/kaggriculture-v15stack-submit](https://www.kaggle.com/code/wzhengbiao/kaggriculture-v15stack-submit) | 33-0-67 | 33.0% | 17 / 16 | -3,564.9 |
| 3 | [haideptry/the-2965-master-hybrid-engine](https://www.kaggle.com/code/haideptry/the-2965-master-hybrid-engine) | 32-0-68 | 32.0% | 18 / 14 | -3,666.1 |
| 4 | [ahmedberatozer/kaggriculture-v57-funding-order-invariant](https://www.kaggle.com/code/ahmedberatozer/kaggriculture-v57-funding-order-invariant) | 29-0-71 | 29.0% | 16 / 13 | -3,844.5 |
| 5 | [hosen42/kaggriculture-m4a-metav4-sr18-2690-1](https://www.kaggle.com/code/hosen42/kaggriculture-m4a-metav4-sr18-2690-1) | 35-0-65 | 35.0% | 18 / 17 | -2,852.3 |
| 6 | [tetsutani/demand-preserving-turn-sale-timing](https://www.kaggle.com/code/tetsutani/demand-preserving-turn-sale-timing) | 34-0-66 | 34.0% | 18 / 16 | -3,241.7 |
| 7 | [guruprasaathas111/kaggriculture-master-engine-v3](https://www.kaggle.com/code/guruprasaathas111/kaggriculture-master-engine-v3) | 14-0-86 | 14.0% | 7 / 7 | -4,705.1 |
| 8 | [dmitriigluzdov/kaggriculture-more-wheat-smarter-sales](https://www.kaggle.com/code/dmitriigluzdov/kaggriculture-more-wheat-smarter-sales) | 26-0-74 | 26.0% | 14 / 12 | -3,951.8 |
| 9 | [lynnsakurai/farmer-john-and-the-idle-seller](https://www.kaggle.com/code/lynnsakurai/farmer-john-and-the-idle-seller) | 29-0-71 | 29.0% | 16 / 13 | -3,837.1 |
| 10 | [nathanjacob/kaggriculture-pipe18-six-layers](https://www.kaggle.com/code/nathanjacob/kaggriculture-pipe18-six-layers) | 29-0-71 | 29.0% | 16 / 13 | -3,837.1 |

## 核验与边界

`PROTOCOL.json` 固定双方文件哈希、官方规则哈希、种子和座位；`games.jsonl` 保留逐局结果。
`RESULTS.json` 和 `PER_OPPONENT.csv` 保留可机器读取的结果。
交付包中的公开方案为 2026-09-23/24 抓取版本；不能从本地胜率直接推断实时天梯胜率。
单进程重跑 4 局，终局双方现金及胜负与正式面板逐局完全一致。
其中 1 对方案的可运行 `main.py` 完全相同：`prvsiyan/kaggriculture-frontier-the-soil-remembers-rain` 与 `tetsutani/demand-preserving-turn-sale-timing`。
去重后的 9 份可运行代码，900 局积分率为 **29.00%**。
另外，在本次相同 50 种子双座位面板中，以下方案的逐局双方现金完全相同：`prvsiyan/kaggriculture-frontier-the-soil-remembers-rain` 与 `tetsutani/demand-preserving-turn-sale-timing`；`lynnsakurai/farmer-john-and-the-idle-seller` 与 `nathanjacob/kaggriculture-pipe18-six-layers`。
按本次结果向量去重后有 8 组，积分率 **29.00%**；这只证明本次测试输出一致，不证明所有盘面下的策略相同。
16 进程并发的单步耗时受机器争用影响，不能直接作为 Kaggle 容器超时判定。
