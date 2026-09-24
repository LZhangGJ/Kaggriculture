# 最新公开脚本训练语料清单（2026-09-22）

这些脚本只存在于 `work/new_public_opponents/`，未写入正式 `opponents/`，也未参与线上身份识别。
更新时间来自下载时的 Kaggle API 列表。耗时是 `FastEnv + agent/main.py` 在 192-worker 采集期间，
每个脚本 640 个独立 seed、固定 seat0 的平均单局 wall time；它反映 Python rollout 成本，不是单进程
延迟基准。

| 训练 slug | Kaggle 来源 / 版本线索 | API 更新时间 | source family | 进入 3-family 平衡集 | 平均秒/局 |
|---|---|---:|---|---:|---:|
| `market_smart_v24` | `tetsutani/market-smart-farming-kaggriculture`；v24 | 2026-09-17 13:42:19 | `ahmed-v31-descendant` | 否 | 11.857 |
| `ahmed_v56` | `ahmedberatozer/kaggriculture-v56-smarter-seeds-and-fertilizer`；v56 | 2026-09-21 15:28:15 | `ahmed-v31-descendant` | 是 | 12.900 |
| `thomas_metav4_v13` | `thomastschinkel/the-metav4-farm-submission-v13`；v13 | 2026-09-20 06:58:38 | `ahmed-v31-descendant` | 否 | 13.236 |
| `master_hybrid_2965` | `haideptry/the-2965-master-hybrid-engine`；标题分数 2965 | 2026-09-22 04:53:59 | `ahmed-v31-descendant` | 否 | 13.595 |
| `fieldcraft_2887` | `hakdevelopment/kaggriculture-2887-score-fieldcraft-agent`；标题分数 2887 | 2026-09-21 13:30:07 | `fieldcraft` | 是 | 10.223 |
| `night_harvest` | `lucifer19/kaggriculture-night-harvest`；无显式版本号 | 2026-08-07 07:50:20 | `night-harvest` | 是 | 9.447 |
| `historical_lb_2800_rescue` | `dmitriigluzdov/kaggriculture-7-turn-rescue-historical-lb-2800`；标题 LB 2800+ | 2026-09-22 06:18:23 | `ahmed-v31-descendant` | 否 | 13.652 |
| `more_wheat_smarter_sales` | `dmitriigluzdov/kaggriculture-more-wheat-smarter-sales`；无显式版本号 | 2026-09-22 04:19:52 | `ahmed-v31-descendant` | 否 | 13.533 |
| `farmer_john_idle_seller_v57` | `lynnsakurai/farmer-john-and-the-idle-seller`；导出入口 v57 | 2026-09-22 03:05:55 | `ahmed-v31-descendant` | 否 | 13.303 |
| `soil_remembers_rain` | `prvsiyan/kaggriculture-frontier-the-soil-remembers-rain`；无显式版本号 | 2026-09-22 00:47:24 | `ahmed-v31-descendant` | 否 | 12.643 |

## 采样含义

- 10 个脚本全部进入过五个 `public10` shard，每个脚本 640 局；但其中 8 个属于同一可见源码家族，
  不能按脚本名等权当成 10 种独立对手。
- 后续 `public3families` shard 固定各取一个代表：`ahmed_v56`、`fieldcraft_2887`、
  `night_harvest`。训练抽样按 `source_family` 重权，不把 opponent identity 放进网络输入。
- `night_harvest` 的隔离输出未发现明确许可，轨迹可以用于内部研究，但不得直接复制其源码进生产。
- 版本号缺失时不猜测 Kaggle revision；可复现性以隔离 `main.py` 的 SHA-256、轨迹 manifest 与下载目录为准。

## 2026-09-23 新强手补充

`master_engine_v3`、`farmer_john_wheat_seller`、`pipe18_six_layers` 均保留 Ahmed v31 的
`Chassis` 实现，应归入同一个 `ahmed-v31-descendant` 源码家族，而非三个独立训练分布。
忽略行尾差异后，Master Engine V3 对 Pipe18 的源码 diff 为 `+1/-136` 行，
Farmer John 对 Pipe18 为 `+150/-87` 行；这一归类以代码结构为证据，不以脚本标题为证据。

冻结 v131 student、argmax、seed `2653000200..2653000263`、双座、每手 128 局的
Python/FastEnv 探索性评测分别为 `83/128`、`81/128`、`83/128` 胜，全部 384 局完成且无
student fallback。结果位于 `work/student-v1/python-public-missing3-v131-argmax-seed2653000200-n64.json`。
这不是正式 `opponents/` 七强验收，也不是当前训练头的胜率；这三个变体可以用于同一家族内的
retention/泛化诊断，但不能直接各占一个对手池权重。
