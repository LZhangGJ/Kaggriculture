# 第 6 名 KANTA Replay 家族对 FC24B 优化与冻结报告

## 1. 结论

第 6 名 `Izzoudine Mohamed KANTA` 家族已经冻结为 `rank06_kanta_v30`。

- ACD 路由锁定后的两个 512-seed 双座位独立面板合计 `1465/2048 = 71.53%`；
- 座位 0 为 `732/1024 = 71.48%`，座位 1 为 `733/1024 = 71.58%`；
- 两个独立面板分别为 `72.56% / 70.51%`；
- 所有对局完成 719 步，`hard_counter_total = 0`；
- 在线 one-step-JIT Agent 另测 256 局为 `176/256 = 68.75%`；
- 在线与离线的我方现金、FC24B 现金和最终路线 mismatch 全部为 0。

该家族固定路线最高只有 47.27%，经过第一商店重路由、弱商店修复、86 条安全后缀筛选和 15 种组合消融后才稳定超过 70%，并非刚过 50% 就停止。

## 2. 经营风格与来源

KANTA 是牛羊均衡、大规模草莓家族：

- 典型规模约 6 牛、6 羊；
- 中前期同时兑现牛奶与羊毛；
- 主要作物是草莓，并用甜瓜和小麦支撑现金及饲料；
- 相比 MiMi 的重牛路线，动物比例更均衡；
- 相比 Arman 的羊优先路线，牛奶更早进入现金循环。

来源：

- 天梯第 6 名 `Izzoudine Mohamed KANTA`；
- submission `55630633`；
- 136 条完整获胜 Replay，编译拒绝数为 0；
- 路线银行：`artifacts/rank06_kanta_trace_bank_v1.npz`；
- 来源 Episode、奖励、座位、商店序列和 SHA256：`receipts/rank06_kanta_trace_bank_v1.json`。

## 3. 第一商店路线

在 `seed=400001..400128` 上对全部 136 条路线完成 34,816 局双座位筛选：

- `all_done = true`；
- `hard_counter_total = 0`；
- 最强固定路线 route91 仅 `121/256 = 47.27%`；
- 单面板第一商店地图在 seed408 达到 `68.65%`，但 Brunch 仅 41.67%。

合并 seed400 与 seed408 后，稳健地图收敛为：

`[2, 2, 2, 91, 91, 91, 91, 35]`

即：

- Bakery / Brunch / Farmers → route2；
- Ice / PET / Pizza / Smoothie → route91；
- Yarn → route35。

它在 seed416 达到 `725/1024 = 70.80%`，八类第一商店均超过 50%。

主要路线来源：

| route | Episode | 原 Replay 奖励 | 用途 |
|---:|---:|---:|---|
| 2 | 94627562 | 106,368 | Bakery / Brunch / Farmers |
| 91 | 95383771 | 90,015 | Ice / PET / Pizza / Smoothie |
| 35 | 94681309 | 110,785 | Yarn |

## 4. 第二商店安全后缀

候选必须在 step 72–143 的 unit 与 market 八组原始动作逐字节相同，防止把已有投资接到不匹配的后半程。

- 136 条来源路线共完成 246 次精确前缀校验；
- PET 的全部兼容后缀均被保留；
- 其他商店组合保留开发库前 5 名；
- 在 seed408 对尚未运行的 81 条后缀补做 82,944 局，全部完成且硬错误为 0；
- seed416 对 4 条核心条件做全部 15 种组合消融；
- seed424 复核后，最终保留 ACD 三条。

最终切换：

| 第一商店 → 第二商店 | 基线 route | 后缀 route | Episode |
|---|---:|---:|---:|
| Bakery → Yarn | 2 | 64 | 94731722 |
| PET → Bakery | 91 | 3 | 94629239 |
| PET → Yarn | 91 | 86 | 95207949 |

被淘汰的 `Brunch → Ice Cream` 在 seed416 没增加胜场，seed424 又落后 ACD，因此没有写入最终版。

所有路由仅使用已经公开的第一、第二商店，不使用未来事件、Replay ID、对手身份或隐藏状态。

## 5. 最终独立验证

ACD 在确定后不再修改，随后运行两个新随机面板：

| 面板 | 第一商店基线 | ACD | ACD 胜率 | 座位0 | 座位1 | 净增胜场 | 硬错误 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed432001 | 729/1024 | 743/1024 | 72.56% | 72.27% | 72.85% | +14 | 0 |
| seed448001 | 716/1024 | 722/1024 | 70.51% | 70.70% | 70.31% | +6 | 0 |
| 合计 | 1445/2048 | 1465/2048 | 71.53% | 71.48% | 71.58% | +20 | 0 |

## 6. 在线语义验收

`seed=440001..440128` 双座位 256 局：

- `176/256 = 68.75%`；
- 两个座位均为 `88/128 = 68.75%`；
- 全部完成 719 步；
- `hard_counter_total = 0`。

同 seed 离线路线矩阵逐局核对：

- candidate cash mismatch = 0；
- FC24B cash mismatch = 0；
- final route mismatch = 0。

`invalid_intent_total` 是来源 Replay 原始意图被合法守卫修复的诊断量，不是环境收到非法动作；实际环境动作未产生硬错误或卡死。

## 7. 冻结交付物

- 冻结配置：`configs/rank06_kanta_v30_frozen_best.json`；
- 第一商店地图：`artifacts/rank06_kanta_stable_win_map_dev640_v6.json`；
- 第二商店地图：`artifacts/rank06_kanta_secondshop_acd_dev640_v18.json`；
- 全量筛选：`receipts/rank06_kanta_all136_vs_fc24b_screen_seed400001_n128x2_v1.json`；
- 安全后缀筛选：`receipts/rank06_kanta_compatible86_merged_seed408001_n512x2_v11.json`；
- 最终独立面板：`receipts/rank06_kanta_final_acd_eval_seed432_full512x2_v23.json`、`rank06_kanta_final_acd_eval_seed448_full512x2_v29.json`；
- 在线回执：`receipts/rank06_kanta_online_acd_vs_fc24b_seed440001_n128x2_v25.json`；
- 在线/离线一致性：`receipts/rank06_kanta_online_acd_screen_parity_seed440001_n128x2_v27.json`。

Rank6 V30 进入 FC24B 克星池。它补充牛羊均衡与草莓大规模能力，后续融合时应作为区别于 Rank4 羊优先和 Rank5 重牛混作的第三种动物经营骨架。
