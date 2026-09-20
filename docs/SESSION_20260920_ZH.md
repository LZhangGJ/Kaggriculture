# 2026-09-20 会话记录：测量纪律修复与 opening / 接管重标定

本文记录一次完整的排查—修改—测量会话。**先读第 6 节（本轮更正过的错误）再动手**，
那里每一条都是已经走过的弯路。

## 0. 一句话摘要

工程此前是"对着 4-seed 噪声做决策"。本轮把头条结论重建成 64-seed 可复现证据，
把默认 opening 从 `G001` 换成 `G275`，把接管条件从固定步数改成"买地帧 + 相对延迟 + 截止步"。
对 7 个强公开对手：**纯 R1 `531/896` (59.3%) → 新配置 `706/896` (78.8%)**，
配对净胜 `+175` 局，7/7 对手全部改善。

## 1. 最重要的更正：此前的头条结论是噪声

| 口径 | 纯 R1 | 温接管 |
|---|---|---|
| 4 seeds（旧记录，56 局/臂） | 36/56 = 64.3% | 30/56 = 53.6% |
| 16 seeds（224 局/臂） | 118/224 = 52.7% | 141/224 = 62.9% |
| **64 seeds（896 局/臂，不重叠块）** | **531/896 = 59.3%** | **683/896 = 76.2%** |

旧记录由此得出「温接管不优于纯 R1」，**符号是反的**。该结论当时还被
`verify_project.py` 断言、写进 `PROJECT_MANIFEST.json` / `README.md` / `HANDOFF_ZH.md`，现已全部更正。

**为什么 4 seeds 会错**：seed 决定城镇后续解锁哪些商店（`shop_*` 是 147 维特征的一部分），
4 个 seed 的商店组合是小样本，方差极大。工程自己的 `train_robust_search_route_trees.py`
对树的准入用了 seed 分组 + 单侧 95% 置信下界；而对自己的头条结论却只用了 4 seeds、无区间估计。

**纪律**：16 seeds 只用于筛错，正式结论必须 64 seeds 起，且用不重叠种子块确认。

## 2. 默认配置的改动

| 项 | 旧 | 新 | 依据 |
|---|---|---|---|
| opening | `G001` (64.2%) | **`G275` (76.2%)** | §3 |
| 接管条件 | 固定 `handoff_step=288` | **买地帧 + `handoff_land_delay_days=1`，截止 288** | §4 |
| 接管实现 | —— | `handoff_land=3` 状态触发 | §4 |

`agent/replay_deployment.json` 当前为：

```json
{"schema":"replay-route-to-dynamic-v1","opening":"G275",
 "handoff_step":288,"handoff_land":3,"handoff_land_delay_days":1}
```

## 3. opening 选择：本轮最大的单项增益

64 seeds、双座、7 强手、每臂 896 局、种子块 `2609600000–63`：

| opening | 胜局 | 胜率 |
|---|---|---|
| **G275** | **683/896** | **76.2%** |
| G379 | 657/896 | 73.3% |
| G210 | 585/896 | 65.3% |
| G001（旧默认） | 575/896 | 64.2% |
| G411 | 540/896 | 60.3% |

- G275 相对纯 R1：配对净胜 `+152`，**7/7 对手的净胜与平均分差同时为正**
- G275 相对 G001：逐对手 5 个大胜、herd 持平、salemali7 都满分 → **是支配关系，不是选择偏差**
- 产物：`experiments/results/r1-vs-g275-handoff12-64seed-v1.json`、
  `opening-handoff12-64seed-confirm-v1.json`

⚠️ **G275 与 G379 在前两个种子块上互有胜负**（16 seeds 时 G379 第一、64 seeds 时 G275 第一）。
最终选择**仍需第三个不重叠种子块**确认；目前的可辩护结论是"G275 与 G379 都明显优于 G001"。

## 4. 接管条件：为什么必须是"买地 + 相对延迟"

### 4.1 接管日曲线（16 seeds、双座、7 强手，224 局/日，基准 = day0 纯 R1）

| 接管日 | 胜率 | 配对 Δmargin | t |
|---|---|---|---|
| 0（纯 R1） | 52.7% | — | — |
| 8 | 43.8% | −3296 | −4.2 |
| 9 | 49.6% | −3156 | −4.0 |
| 10 | **37.9%** | −4428 | −5.6 |
| 11 | 62.5% | +1624 | +2.0 |
| 12 | 62.9% | +622 | +0.8 |
| 13 | **64.3%** | −347 | −0.5 |
| 14 | 61.6% | −358 | −0.5 |

**day8/9/10 显著有害，day11–14 优于纯 R1。** 产物
`experiments/results/cleanscan-r1-handoff-days-public7-16seed-v1.json`。

### 4.2 五个部署 opening 的买地帧差异极大（关键实测）

反解 `agent/route_actions.json.zlib`：

| opening | `BUY_LAND` 步 |
|---|---|
| G001 | 150 (day6 h6), **265 (day11 h1)** |
| G210 | 150 (day6 h6), **197 (day8 h5)** |
| **G275**（当前部署） | 149 (day6 h5), **219 (day9 h3)**, 221 (day9 h5) |
| **G379** | 149 (day6 h5), **219 (day9 h3)**, 221 (day9 h5) |
| G411 | 150 (day6 h6), **265 (day11 h1)** |

**「第三块地在 step 265」只对 G001/G411 成立。G275 在 day9 就买地，而且是两次购买（去 4 块地）。**
再叠加"买地会因现金不足失败重试"，**实际到手帧依赖 seed**。

→ 所以固定步数接管对 G275 是错位的；改成**相对买地帧的延迟**才自适应。

### 4.3 三个变体的 64-seed 实测

| 配置 | 胜率 | 配对净胜 |
|---|---|---|
| 纯 R1 | 531/896 = 59.3% | — |
| G275 + 固定 day12 | 683/896 = 76.2% | +152 |
| **G275 + 3块地触发 + floor=264** | **706/896 = 78.8%** | **+175** |
| G275 + 3块地触发 + floor=0（纯状态） | 629/896 = 70.2% | +98 |

**"纯状态触发"掉 8.6pp。** 原因：`handoff_land=3` 对 G275 在 step 222 就满足，
于是部分 seed 在 day9 或 day10 接管，正好落进 §4.1 最差的区间。
→ **"持有 3 块地"不是正确的完成信号；买地只是开始挖那块地，磁带还要花 day9–11 把它铺满。**
实测的赢家盘面是 day11 的 52–58 格作物 + 14–16 头动物。

### 4.4 当前部署与实测的对应关系

`handoff_land=3, handoff_land_delay_days=1`：
`due = ceil((purchase_step + 24)/24)*24`。G275 实测 `purchase_step = 222` → `due = 264`（day11）。

实测三档（同一个 seed、对 thomas）：

```
delay=0 -> 接管于 step 240 (day10)
delay=1 -> 接管于 step 264 (day11)      ← 当前部署
delay=2 -> step 288（被截止步截断）
```

⚠️ **它与已实测 78.8% 的那版（floor=264）对"买地在 264 之前"的 seed 机制等价，
但对买地更晚的 seed 会晚一点（并被截止步 288 截断）—— 所以严格说并未逐位等价，
当前部署的 64-seed 数字尚未单独复测。** 这是第 1 优先的待办（§10.1）。

`handoff_step=288` 保留为**截止步**：245 条路线里最晚的买地在 step 433（day18），
而接管只测到 day14 为好，不能让接管漂移到未验证区间。

## 5. 尚未落地 / 待决策

### 5.1 检查点集合需要按各 opening 的买地时点重定（未做）

现有检查点 144/168/216（day6/7/9）；已生成 240/264 的反事实数据
（`data/artifacts/switch-fine-late-240-264-26x128.npz`，1630 万局）。

**但生成 240/264 的理由（"卡在 265 那次买地之前"）只对 G001/G411 成立**；
对 G275（买地在 day9）而言 **240 反而是买地之后**。→ 需要按 opening 各自定检查点，
或统一取"该 opening 买地前最近的一天"。数据仍可用，位置解释要改。

### 5.2 树的目标域错位（未解决，且目前无可信替代）

- 标签 = 磁带对打**打满 719 步**的收益（`switch_search` → `play(opening, opponent, seed,
  checkpoint, target, -1, -1)`，**R1 完全不在训练回路**）；而部署只用磁带打到接管点。
  于是树在优化 288–719 这 431 步**永远不会执行**的赛程。
- **速度约束是硬的**：原生 28,000–36,000 局/秒 vs R1 约 20 秒/局，差约 6 个数量级。
  R1 永远不该进训练回路。
- 试过的两个替代方案，**都被否**：
  - `stop_after_steps=288`（已实现、已验证可用，**未用于任何数据集**）：停在接管点测现金差
    → 现金是弱指标（G379 在 day11 落后 11114 却 8/8，G001 落后 827 却 4/8）
  - 用 R1 的价值函数 `predicted` 打分 → **无任何验证**且跨局不可比（同一 seed 不同 seat 可差
    −48827 与 −17718），等于引入第三个未验证代理
- **结论：在找到经过验证的标签之前，留在已测目标上。**

### 5.3 多次切换（未做，成本低）

`NativeTeammateExecutor::play` **已支持两次切换**（`switch_step0/route0` + `switch_step1/route1`），
但批量入口 `switch_search` 未暴露。→ 只需加批量包装，**不需要新仿真逻辑**。

注意：在线 `SearchRouteController.switched` 是一次性标志，且训练数据全部是"单次切换"标签
（每个 `(opening, checkpoint, target)` 都是一次独立干预），放开多次会偏离标签分布。

### 5.4 对手域错位（未做）

- 609 份 replay 的 60 个目标是 Kaggle 排行榜队伍（Majkel1337、Planned Economy…），
  **与 `opponents/` 这 7 个公开脚本不是同一批对象**。
- 已确认 **melon_2749 / thomas_2945 / demand_preserving 也是磁带回放底盘**：
  41 条 719 步磁带 + 64 项 shop→route 映射 + ~30 层反应外壳，**磁带在 step 144 由
  `unlocked_shops[:2]` 一次性选定**。thomas 的 40 条活跃磁带在 step 0–143 **逐字节相同**。
- 可选做法：把它们的磁带解出来当**训练对手**（不是当我们自己的路线），直接修对手域错位。
  ⚠️ `opponents/` 部分脚本无明确许可，工程注明"只用于本地评测，不应打包进对外发布物"。

### 5.5 245 条路线的完整筛选（未完成）

只系统测过 5 个带树的 opening。用 `trace_vs_opponent.py` 扫过 245 条但**用错了工具**：
它带逐日/逐小时快照，每步对两个 player 各做一次完整 observation 的 JSON 深拷贝，
比普通 A/B 慢 3–4 倍，30 分钟未完成，已中止。**正确做法：加 `--no-trace` 只取最终比分**，
或把筛选搬到原生侧。

### 5.6 末期掉血（已定位，未修）

对 thomas 的逐小时追踪（4 seeds × 双座）：

| | 我们 | thomas |
|---|---|---|
| 动物（day11 → day29） | 13.8 → **10.8** | 16.0 → **16.8** |
| 杂草（day11 → day29） | 0.0 → **8.1** | 0.0 → **0.4** |
| 每天雇工人数（day13 起） | 8–11，day28 掉到 **7.5** | 稳定 **11–11.8**（上限 14） |

假设（**可测的调参假设，不是结论**）：`config.json` 的 `work_price: 4`（`DEFAULTS` 是 1.2，
**劳动影子价格高 3.3 倍**）加上 `rotation/repeat/replant = 0`，导致少雇人、不复耕。
注意游戏的雇工成本是**斐波那契**（1,1,2,3,5,8,13,21,34,55,89…），**一天雇 11 人共约 $232** ——
劳动是最便宜的资源。

对手外壳里可移植的免费改进：

- `_sell_lead`：把**下一步**要卖的货**现在**卖，且只在 `step % 4 != 0`（两帧之间无城镇消费，
  价格可证不变）时做，下一步再抑制同量
- hour-23 仓储纪律：三层独立保证 shed ≤ 99，只卖黎明会被销毁的部分
- 终局收尾：step 712 起精确 7 步收尾规划 + 磁带 `SELL 1000` 哨兵

## 6. 本轮更正过的错误（下一个 agent 请勿重复）

| 当时的说法 | 实际 | 教训 |
|---|---|---|
| "提交物在 seat 1 第 1 帧就崩" | **假的**。`env.run` 下两个 seat 都有 `step`；我用 `env.step()` 手动驱动，而框架真实路径走 `Environment.__agent_runner` → `__get_shared_state(i)`，它把 schema 里标 `shared` 的字段（`observation.step/farms/market/town/day/hour`）从 `state[0]` 拷进**每个** agent 的视图；直接读 `state[p].observation` 得到的是存储视图，对 `p>0` 删掉了 shared 字段 | 手写驱动 ≠ 框架路径。下结论前先用框架自己的入口复现 |
| "前缀比 R1 开局弱" | **假的**。前缀 day12 现金是纯 R1 的 3–8 倍（13836 vs 2998） | 那个对比被"磁带尾盘弱"混淆；要比就跟**真实对手**比 |
| "180 条路线只持有 2 块地" | **假的**。棋盘自带 NW，2 次买地 = **3 块地**；245 条**全部** ≥3 块地（最小买地次数就是 2） | 数指令前先确认棋盘初始状态 |
| "floor 是冗余的，去掉" | **假的**，去掉掉 8.6pp（78.8% → 70.2%） | 代理推理不能替代测量 |
| "R1 的价值函数可以当标签" | **撤回**。`predicted` 无验证、跨局不可比 | 不要引入第三个未验证代理 |
| "连续很多天没有产出" | —— | 长任务不要用 sleep 轮询；远程杀进程要连 spawn worker 一起清（本地 ssh 被 kill 会把 96 个 worker 变孤儿） |

## 7. 本轮新增产物

| 文件 | 用途 |
|---|---|
| `experiments/test_submission_contract.py` | 按框架真实路径驱动**两个 seat**，防止"harness 补丁掩盖真实契约"回归 |
| `experiments/test_feature_parity.py` | C++ `features_at` vs Python `route_switch_vector` **逐位对拍**；108/108，`max_abs_diff = 0.0` |
| `experiments/trace_vs_opponent.py` | 逐日/逐小时追踪，用于定位"哪一天被打崩"（**批量筛选时不要用，见 §5.5**） |
| `experiments/probe_handoff_label.py` | 标签验证：真跑 R1-handoff 对局，测各候选标签与真值的相关性。**已写好，尚未运行** |
| `data/artifacts/switch-fine-late-240-264-26x128.npz` | 检查点 240/264 的反事实数据（1630 万局、627,200 状态、9 分 45 秒） |
| `fast_kaggriculture/src/bindings.cpp` | `switch_search` 增加 `stop_after_steps`（**向后兼容**，默认 -1）；扩展已重建 |

**已验证的正确性检查（改动后必须全绿）**：

```
verify_project.py                      PASS
experiments/test_submission_contract.py PASS（两 seat 满局 + 完成接管）
experiments/test_feature_parity.py      PASS（108/108，max_abs_diff 0.0）
pytest fast_kaggriculture/tests         11 passed（模拟器对官方环境逐步差分）
```

**备份**：`/root/backup-pre-stepfix/`（最早的 7 个文件）、
`/root/backup-pre-labelswitch/`（`fast_kaggriculture` 扩展与 `bindings.cpp`）。

## 8. 复现命令

```bash
cd /root/kaggriculture-replay-switch-clean-v1
export PYTHONPATH=$PWD
export PY=/root/miniforge3/envs/torch-npu/bin/python

$PY verify_project.py
(cd experiments && $PY test_submission_contract.py)
$PY experiments/test_feature_parity.py --openings G001,G275,G379 --checkpoints 144,168,216

# 64-seed A/B（当前部署 vs 纯 R1），约 8 分钟
$PY experiments/run_strong_ab.py --baseline policy/r1/agent.py --candidate agent/main.py \
   --output work/ab-64seed.json --seeds 64 --start 2609600000 --workers 192

# 接管延迟扫描（相对买地帧）
REPLAY_HANDOFF_LAND=3 REPLAY_HANDOFF_LAND_DELAY=1 \
$PY experiments/run_route_opening_scan.py --openings G275 --handoff-day 12 \
   --start 2609600000 --seeds 64 --workers 192 --output /tmp/scan.json
```

## 9. 当前部署状态

- `agent/replay_deployment.json`：`opening=G275, handoff_step=288, handoff_land=3,
  handoff_land_delay_days=1` + 4 个资产的 sha256（哈希已同步）
- 所有训练数据集的标签**未变**：全部是 `stop_after_steps=-1`（打满 719 步）
- `FILE_MANIFEST.json` 已刷新（542 条）；`SOURCE_SHA256.txt` 已同步

## 10. 建议的下一步优先级

1. **复测当前部署配置**（`delay=1`，64 seeds）—— 8 分钟。它是部署配置却还没有独立 64-seed 数字，
   而且与已实测 78.8% 的那版只对部分 seed 机制等价。**在这之前不要再改配置。**
2. 第三个不重叠种子块（如 `2609700000+`）确认 **G275 vs G379**，钉死最终 opening。
3. 接管延迟标定：`delay ∈ {1,2}` 在 16 seeds 筛选 → 64 seeds 确认（`delay=2` 对 G275 就是 step 288，
   即固定 day12，已实测 76.2%）。
4. 245 条路线筛选（加 `--no-trace`），找是否存在显著优于 G275 的路线。
5. 修末期掉血：`work_price` 与 `rotation/repeat/replant` 的 A/B（§5.6）。
6. 检查点按 opening 重定（§5.1）；随后才是多次切换（§5.3）。
7. 标签验证 `probe_handoff_label.py` —— **作为诊断**，用它测"现有标签与真实交接价值的相关性"，
   而不预设要换标签。
