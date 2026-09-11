# 第 4 名 Arman Replay 家族对 FC24B 优化与冻结报告

## 1. 结论

Rank 4 Arman 家族已经形成一个可冻结的独立强拳：`rank04_arman_v45`。

- 三个未参与路线训练的随机面板，均为 512 seeds、双座位、完整 719 步；
- 对冻结 FC24B 合计 `2771/3072 = 90.20%`；
- 座位 0 合计 `90.04%`，座位 1 合计 `90.36%`；
- 三个面板分别为 `90.53% / 89.55% / 90.53%`；
- 所有对局完成 719 步，`hard_counter_total = 0`；
- 在线 JAX Agent 与安全反事实路线选择在 256 局中，我方现金、对手现金、最终路线均逐局完全一致。

该家族已经严格超过 `>50% vs FC24B` 的入池门槛，而且继续优化到约 90%，不是刚过门槛就停止。

## 2. 来源证据

- 天梯来源：第 4 名 `Arman Tuganbaev`；
- submission：`55617399`；
- 完整获胜 Replay：143 条；
- 路线库：`artifacts/rank04_arman_trace_bank_v1.npz`；
- 来源、Episode、奖励、座位、商店序列和 Replay SHA256：`receipts/rank04_arman_trace_bank_v1.json`。

最终用到的主要路线：

| route | Episode | 原 Replay 奖励 | 用途 |
|---:|---:|---:|---|
| 50 | 94488124 | 108,844 | 六类常规第一商店基线 |
| 129 | 96718071 | 123,990 | Yarn 第一商店基线 |
| 131 | 96759131 | 105,510 | Pizza 第一商店基线 |
| 134 | 96857252 | 126,435 | 第二商店为 Yarn 时的安全后缀 |

## 3. 优化过程

### 3.1 全 143 路线筛选

开发面板 `seed=304001..304128` 对 143 条路线进行了双座位完整筛选，共 36,608 局，全部完成且硬错误为 0。

最强固定路线是 route50；随后仅用第一个公开商店建立因果路线图：

`[50, 50, 50, 50, 50, 131, 50, 129]`

它在独立 `seed=320001..320512` 面板达到 `866/1024 = 84.57%`，明显强于固定 route50 的 `76.86%`。

### 3.2 弱商店继续搜索

Bakery、PET、Pizza 是第一商店地图的主要弱项。我们从全 143 路线中取 37 条候选，在新 `seed=328001..328512` 面板完成 37,888 局筛选。

根据两个开发面板提出的三套第一商店替换，在独立 `seed=336001..336512` 上全部退化：

| 候选 | 独立胜率 |
|---|---:|
| 原稳定地图 | 88.28% |
| weak minimax | 87.40% |
| weak stable | 86.62% |
| weak Wilson | 87.60% |

因此没有保留单库过拟合的 Bakery/Pizza 替换。

### 3.3 第二商店安全切换

路线只能在切换前保持同一生产承诺时更换。候选后缀必须在 step 72–143 的八组原始动作字段逐字节相同：

- unit op/item/amount/count；
- market op/item/amount/count。

41 条 exact-prefix-compatible 后缀在两个开发面板上筛选后，route134 对以下公开条件有效：

- Brunch → Yarn；
- Farmers Market → Yarn；
- Ice Cream → Yarn；
- Pizza → Yarn；
- Smoothie → Yarn。

路线切换只读取已经公开的第一、第二商店，不读取未来事件、Replay ID 或对手身份。

## 4. 找到并修复的二级索引漏洞

旧二级地图按 `base_route × second_shop` 索引。由于六个第一商店共享 route50，一个本应只属于 `Brunch → Yarn` 的条件，会被错误扩大到所有 route50 开局。

修复后：

- 新地图严格按 `first_shop × second_shop` 索引；
- JAX 在线 router 支持明确的 8×8 公共商店矩阵；
- 旧的 route-index 地图仍兼容，但不会与新语义混用；
- 验证工具同时报告使用的是新索引还是旧索引。

这不是数值微调，而是修复了真实的条件表达错误。

## 5. 独立结果

| 面板 | 胜场/局数 | 胜率 | 座位0 | 座位1 | 相对第一商店基线 | 硬错误 |
|---|---:|---:|---:|---:|---:|---:|
| seed336001 | 927/1024 | 90.53% | 90.82% | 90.23% | +23 胜 | 0 |
| seed344001 | 917/1024 | 89.55% | 89.06% | 90.04% | 0 胜 | 0 |
| seed352001 | 927/1024 | 90.53% | 90.23% | 90.82% | +28 胜 | 0 |
| 合计 | 2771/3072 | 90.20% | 90.04% | 90.36% | +51 胜 | 0 |

第二后缀主要提高严格胜率，但平均领先从约 54,922 降至约 51,621。最终目标是击败 FC24B，因此以胜率为第一目标；该分差代价已被明确记录，不伪装成全面改善。

## 6. 在线语义验收

在线 one-step-JIT Agent 在 `seed=352001..352128` 双座位共 256 局中：

- 第一商店基线：`219/256 = 85.55%`；
- exact5：`230/256 = 89.84%`；
- 全部完成 719 步；
- `hard_counter_total = 0`；
- 在线结果与离线 compatible-route screen 的候选现金、FC24B现金和最终路线全部逐局一致，三项 mismatch 均为 0。

`invalid_intent_total` 是 Replay 原始意图在新状态下被动作守卫提前拦截并修复的诊断量，不是提交给环境的非法动作；实际送入 JAX 环境的动作经过合法性守卫，未产生硬错误。

## 7. 冻结交付物

- 冻结配置：`configs/rank04_arman_v45_frozen_best.json`；
- 第一商店地图：`artifacts/rank04_arman_stable_win_unrestricted_map_dev640_v8.json`；
- 第二商店地图：`artifacts/rank04_arman_secondshop_exact5_dev640_v38.json`；
- 在线候选回执：`receipts/rank04_arman_online_final_pair_vs_fc24b_seed352001_n128x2_v43.json`；
- 在线/反事实逐局一致回执：`receipts/rank04_arman_online_exact5_screen_parity_seed352001_n128x2_v44.json`；
- 三个独立面板：`receipts/rank04_arman_v45_eval_seed{336,344,352}_n512x2.json`。

Rank 4 V45 现在进入 FC24B 克星池。下一阶段继续寻找行为不同的其他家族；最终综合 Agent 仍须对完整去重本地池逐一达到 90%，不能用本家族对单一 FC24B 的结果代替最终门检。
