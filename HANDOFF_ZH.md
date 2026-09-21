# 交接：Replay 路线树 → 无开局模板 R1

更新时间：2026-09-21。

## 当前唯一性能研究对象：前期路线切换 + 温接管

按用户最新决定，后续只优化真实部署链：固定 G275 起步，147 维浅树在前期切换 replay 路线，
随后 delay=1 温接管无模板 JointAFS R1。纯 R1、冷接管不再做性能探索，至多保留默认行为兼容检查。
候选最终只以未参与训练/筛选的七强同 seed 双座多 seed warm 结果准入，并要求逐对手均严格高于 80%。

### 当前已部署：G114 叶回退 G275，保留其余浅树分支

新 A8 消融中，默认树 `98/112`，固定 stay G275 仅 `84/112`。G024/G316 共 41 局全胜并相对
G275 净救 28 胜；G114 的 46 局包含全部 14 败局并相对 G275 净损 14 胜。因此不可禁用整棵树。
G114→G275 在独立 B16 七强逐手 `Δwin=+2/+4/+8/+6/+4/+2/0`。原生 stop288 proxy 曾把
G396 排第一，小块 D16 相对 G275 为 Thomas `+2`、Melon `0`；预注册 H32 完整部署 A/B 随后
明确反转：G275 相对 G396 为 Thomas `39→46/64`、Melon `41→46/64`，平均分差
`+1399/+1748`。`agent/route_policy.json` 因而显式部署 `target_fallbacks: {G114: G275}`，
控制器验证 source/target 均存在；默认其余行为不变。两手 H32 都只有 71.9%，不得写成最终完成。

合并 G16+H32 后，G114 条件域有 16 个独立 seeds；去除双座重复并做 leave-one-seed-out，
depth1–3 门控均不如静态 G275，根特征和叶路线在折间明显漂移，故不做新门控。阶段 trace 显示败局到
day12 已多落后约 1.1k–1.4k，day15 后才持续放大，下一轮继续研究温前缀状态支配，不回到纯 R1。
H32 的六路线 oracle 上限也只有 Thomas `53/64`、Melon `54/64`。表内对 Thomas 最强的 G249
在全新七强 I16 上相对 G275 为 `-4/-2/-2/-4/-6/-4/0`，七手平均分差全负，故停止全局路线替代。

### 2026-09-21 阶段归因与成交顺序证伪

当前抓手排序是：**接管首日交易与资本配置 > 中后期 DP 的商品结构/价格冲击放大 > 前期路线切换**。
正式 warm 64-seed 中五个未达标对手分别为 Thomas `93/128`、Melon `86/128`、Demand
`96/128`、Ahmed `94/128`、Pipe `98/128`；严格超过 80% 还各差 10/17/7/9/5 胜。
day10 的现金下降对应第三块地和约 20.5 株新增作物；day11 我方仍多约 17.1 株作物、约 39.8
日产，因此 replay 前缀完成了资本形成。Thomas 却在 day11→12 的相对现金差继续恶化约 862，
而另外四手改善约 596–1459。Thomas/Melon 的 day12 胜败现金差已分离约 2530/2109，到 day15
才扩大到约 3003/3894，符合“接管首错、后期复利放大”。

审计随后撤回了对 seed `2610100049` 的 sale-timing 归因。旧 `td_prepare_observation()` 从
`h.policy.base` 生成 diagnostic，而 Python 所称 default 是已评分 proposal 的 argmax。该局实际
default 为 id1（`discount=.08, capital_power=.5`），旧 `competitive_sale` 却是
（`.005,.4,delay_sale=1`）；所以“默认卖 EGG/WHEAT、买 STRAWBERRY，而 candidate 卖 MELON、
取消采购并翻盘”混入了资本估值变化，不是单变量证据。旧 formal selector 的胜负仍是有效的组合
策略 A/B，但其中的 `competitive_sale` 类名不能再解释为纯卖出时机。

根因已在 `policy/r1/bridge.cpp` 修复：两个 diagnostic 都从当局最高分 nondiagnostic proposal 的
settings 派生，再只改各自命名的设置；测试固定复现 seed49 的 default=id1，并核对 discount 与
capital_power 完全一致。修正后的新 warm A8：Thomas `8/16→8/16`、均分差 `+181`，Melon
`8/16→8/16`、均分差 `+178`，无救败/致败。seed49 双座定点复核也仅为 Thomas
`-2565→-2391`、Melon `-12009→-11841`，原来的翻盘消失。因此不再测试从旧污染样本后验得到的
`买草莓 && 对手现金>=17000` 门控。

为了分离污染中真正起作用的资本方案，又在全新 seeds `2615000000..7`、Thomas/Melon、双座 warm
比较 auto 与原始 base proposal。32/32 个 handoff 状态的 auto 都选 id1。base 在 Thomas 上
胜数 `10/16→10/16`（救 2、伤 2，均分差 `+921`），在 Melon 上 `12/16→10/16`、均分差
`-597`。所以不能把修正简化成“handoff 首日统一保守投资”，也不继续扫 discount/capital_power。
产物为 `work/warm-base-plan-vs-auto-A8-2615000000.npz`；纯化 diagnostic 产物为
`work/pure-diagnostic-warm-A8-2614900000.npz` 与 `work/pure-diagnostic-warm-seed49.npz`。

已完成三个只看 warm 的 DP 时序实验，均停止扩大：

- 对手日产全部放在我方成交前：新 A8 Thomas 净 `+2/16`、均分差 `+1019`，但 Melon 净
  `-2/16`，虽均分差 `+1170` 仍不满足逐手不退。
- 同时把城镇需求全部放到成交后：两手净胜均为 0，均分差分别 `-1753/-2761`。
- 把公开资产当前待售上界加入当日对手供给：weight=1 的 A8 曾为 Thomas `+2/16`、Melon
  `+4/16`，但全新 B16 反向为 Thomas `-1/32`、Melon `-4/32`，均分差也为负。

默认宏的双座 1,438 帧动作一致性通过；供给顺序实验宏随后已从正式源码删除。结论不是 50/50
假设“精确”，而是统一顺序无法同时解释 Thomas 与 Melon，且公开资产上界会把不可见库存当成待售量。
下一步只研究 handoff 首次重规划的评分误差；任何公开状态门控都必须在独立 seed 上先复现，
不扫连续参数、不扩大已失败候选。

后续评分审计没有发现 `SearchController::score()` 的实现 bug：候选使用同一 horizon、同一竞争口径和
base tail；当前现金与 `tail.predicted` 不重叠，跨日恢复 base 也是“一日干预”的定义。B8 中 h1/h2
从不选 base；h3–h5 虽开始换选，却让 Thomas 胜数 `10→8`，不能靠延长 rollout 修复。h1 对激进
方案的 terminal predicted 优势平均约 6.7k，跑完一天后仍约 1.8k，而真实终局方向接近随机，缺口是
尾值商品/资本估值误差，不是 horizon 边界。

动作级归因也已更正：seed49 auto/base 在 step264 的卖货完全相同，auto 只多买 1–2 份草莓；
随后形成更多 STRAWBERRY/TOMATO、更少 MELON，day12 工人数无差。拆分 id1 后，discount-only
在 28/32 个 fresh handoff 计划中完全复现 id1，capital_power 只是少数 tie-break；但独立 D8 上
discount-only 对 Melon 32/32 与 auto 相同，对 Thomas 仅一个 joint seed 不同且分差更差。

`id3→id0` guard 在 seeds `2615100000`、`2615200006` 曾对 Melon 共救 4 局、Thomas 救 2 局且
不伤胜局，但 fresh 七强 A16 (`2615400000..15`) 对五个未达标对手 160 对局全部无动作差，只影响
本已全胜的 Salemali 分差，因此发生率不足以作为抓手，实验开关已删除。对应产物：
`work/warm-score-horizons-B8-2615100000.npz`、`work/warm-id3-guard-C16-2615200000.npz`、
`work/warm-animal-heavy-guard-public7-A16-2615400000.json`、
`work/warm-discount-only-D8-2615500000.npz`。

未来商店的精确 Jensen 探针只得到单地块 Tomato `+$4.15`、Strawberry `-$6.28`、Melon `$0`；
`feed_cover=2` 的七强 A8 全部 `Δwin=0` 且多数分差下降。这两条均已停止，不进入正式源码。

### 最新因果定位：Thomas 的优势在共享价格冲击，但路线后验不可在线预测

- 禁用 Thomas 的 RACE/RACEPX/RACEGATE（生产路线不变）后，同 8 seeds 双座我方从
  `12/16` 到 `16/16`，平均分差 `+2769`。终局现金分解为我方 `+2100`、Thomas `-668`。
- 分阶段差分在 day12 只有我方 `+92`、Thomas `-1171`，我方收益主要在后续价格传导中形成；
  因而不能把 no-race 诊断误读为“R1 没有提前卖”。`settle_market()` 本来就逐帧清算全部非保留库存。
- 关闭 R1 local-sale hold：Thomas B16 `+1/32` 但平均分差 `-910`；只在价高于 base 时释放：
  `0/32`、`-144`。`delay_sale=1` 和两日 horizon 分别 `0/32,-420`、`-4/32,-2102`。
- replay 侧复用已有 `CausalLeadSaleManager` 提前 5 步，以及扫描未来 40 步并做 base-price/debt
  门控，Thomas B16 均 `Δwin=0`（分差 `-4.5/-89`）。不是已有库存卖慢，而是对手先卖自己的
  未来批次，改变了我方后续产出的价格。
- 147 维历史没有累计市场成交流。主切换在 step144/168，早于 Thomas step192 开始的 race。
  真实部署树前缀跑到 step216 后做 12 路、Thomas+Melon、A32 双座 warm 反事实：前缀分布为
  `G114=80,G275=32,G024=12,G316=4`；G114/G275 的 direct-payoff seed-CV 都不如 stay。
  训练内 `G275→G190` 为 `26/32→28/32`，但全新 B16 对 Thomas/Melon 都 `Δwin=0`，拒绝。

该公开成交流账本已完成：对七种不可购买商品逐 tick 做库存守恒，并在报价可能触底 1 时拒绝
不精确样本。Thomas/Melon 的 A32 双座在 step216 前却全部是同一个累计向量
`MILK=12,WOOL=12`，其余为 0，因此没有任何可学习信息，不扩入 147 维线上状态。

warm direct-payoff 树曾在 B16 净 `+14/224`，加公开状态的 Herd 保守门控后 E16 净 `+18/224`
且七手均过 80%，但独立 F32 对五弱手反向净 `-10/320`，Thomas 仅 59.4%，拒绝。混合 seat
训练和逐对手 maximin 叶目标也在独立 8-seed pilot 中伤 Thomas/Herd。step192 的
`G114→G190` 训练内 `14/24→18/24`，全新 B16 却使 Thomas/Melon 分别 `-6/-4` 胜。
结论是 warm 反事实 oracle 主要包含 checkpoint 后不可观测的商店/RNG 后验；继续加树、换时间点
或扩大同类标签不会解决可识别性。纯 R1/冷接管仍不做性能实验，默认部署未改。

进一步只保留 8 个商店位和 2 个历史现金特征训练 maximin 树：I8 对五弱手净 `+10/80`，加
Herd/Salem 保守门控后的独立 J16 仅净 `+3/160`，Melon `-1`、Thomas `0`，Thomas 胜率仍
62.5%。低方差特征只能降低回撤，不能创造 Thomas 增益。下一机制实验应校准 DP 对公开对手资产
未来供给的成交时序（Thomas 的 step192 race），不再扩 replay 树标签域。

2026-09-21 的正式 64-seed 路线追踪已把主要失败定位到错误标签域：当前树的训练目标是另一条
replay 路线一直跑到 day29 的胜负，但部署只让它运行到约 day12。五个弱对手各有 62/128 局
被选为 `G114`，对应胜率仅 51.6%–61.3%；hard15 的 150 局中 120 局落到该路线。该条件统计
不能单独证明禁用 G114（存在选择偏差），但足以把下一实验限定为：复用 147 维特征和成熟鲁棒树
训练器，用真实 replay-prefix → delay1 warm-R1-suffix 终局结果重标当前活跃路线。

该实验现已完成。候选固定为旧树活跃叶子的 9 条路线，检查点 144/168；A32 暴露并修复了旧
`probe_handoff_label.py` 的阻断 bug（step0 会覆盖预装 schedule），随后以完整 observation+history
哈希保证反事实前缀一致。扩至 128 个独立训练 seed 后，warm oracle 对五个弱手均为 95% 以上，
训练内新树相对旧树逐手全正。冻结树的 B32 盲测净增 53/448；但再独立 C64 只从 767/896
到 779/896（+12），逐手 Thomas -1、Melon -1、Demand +6、Ahmed -3、Pipe +6、Herd +5、
Salem 0，平均分差 +214。按“弱手逐个不退”门槛，**不部署**。

失败归因：成熟训练器先把每局 `(win, margin)` 的 oracle target 压成分类标签，再拟合分类树；
这会丢掉各路线收益差，并把 checkpoint 后不可观测的未来商店/RNG噪声当作可预测类别。下一步若
继续路线树，应复用同一完整反事实矩阵，直接按叶节点的配对期望胜负、再按分差选择 route，并用
seed-group/leave-bot-out 评估策略收益；不要继续扩样、加深树或根据 B/C 重选阈值。

产物：`work/warm-route-labels-g275-32seed-2610800000.npz`、
`work/warm-route-labels-g275-96seed-seat0-2611000000.npz`、
`work/warm-route-tree-ab-public7-32seed-2610900000.json`、
`work/warm-route-tree128-ab-public7-32seed-2611100000.json`、
`work/warm-route-tree128-confirm-public7-64seed-2611200000.json`。

## 统一边际价值实验：保留开关，不部署

新增 `R1_CONFIG_OVERRIDES={"marginal_value":1}`，把 9 商品未来各日的完整组合价值中心差分
接入生命周期候选 DP，并保持完整价值比较与执行准入。默认 0；旧/新二进制双座共 1,438 帧动作一致。
2,808 个差分对拍通过，现行交易模式和精确积分/交易时序模式均通过。

七强同 seed 双座、16 seeds 的纯 R1 筛查：基线 145/224，最终 v2 86/224，
净胜 -59；448 条记录无运行异常。数值一致性没有转化为策略收益，开关不进默认。
本轮没有冷/温接管成绩；完整 replay 二进制资产未能在本地取得，不得借用此前 warm 成绩作为本轮基线。
详细实现、局限、复现命令和原始实验记录见 [统一边际价值实验](docs/MARGINAL_VALUE_EXPERIMENT_ZH.md)。

## 2026-09-21 实验：联合作物/动物 portfolio 局部搜索（保留开关，不部署）

R1 新增严格 opt-in 的 `R1_CONFIG_OVERRIDES={"portfolio_swaps":1}`：只对本轮计划中新建、
现金购买且位于空地的项目，用现有完整 `Planner::value` 尝试单地块作物/动物替换；不碰存量资产、
库存已购承诺或 joint bundle。默认 `portfolio_swaps=0`，与改动前二进制连续 320 帧动作完全一致。
`portfolio_swap_min_gain` 可设非负收益门槛，默认 0。模式 2 在 8 个代表地块中先按单换价值筛到
4 个地块，每地保留 3 个替代，再一次性评价双地块组合；每次 plan 的联合候选硬上限为 54。

七强、同 seed 双座的两批独立 16-seed warm A/B：首批 Thomas 净 `+2` 胜，其余为 0；第二批
Thomas 净 `-2` 胜，其余为 0，合并后净胜为 **0**。收益门槛 20 和 50 在首批均没有改变任何
胜负。因此 1-swap 不是当前强对手瓶颈，**不进入默认部署，也不扩大到 64 seed**。双地块模式在
全新 8-seed pilot 中 Thomas 净 `-1` 胜，其余六个对手胜负不变；除 Melon 近乎持平（`+11`）外，
另外六个对手平均分差全部下降，因此也立即停止。结果位于
`work/warm-portfolio-swaps1-public7-16seed-2610300000.json`、
`work/warm-portfolio-swaps1-public7-16seed-2610300100.json`、两个 `gain20/gain50` 文件，以及
`work/warm-portfolio-pairs2-public7-8seed-2610400100.json`。

模型核对：当前 `Planner::value` 已包含已开放商店需求、未来商店期望、双方公开存量资产的未来产出、
库存价格曲线、劳动/工资、预算和竞争收益；成交顺序已按“对手预计供给一半在我方之前、一半在之后”
处理。更强的双换反而更差，说明缺口不是这些变量缺失或局部搜索深度，而是模型误差下离散组合
改动不稳定；不要继续扫 swap 门槛或扩大组合枚举。下一步必须先做败局阶段归因。

## 2026-09-20 正式四臂验证：当前默认 80.02%，handoff selector 不部署

全新未参与训练/筛选的 seeds `2610100000..2610100063`，七个公开强对手、同 seed 双座，
每臂 896 局：

| 臂 | 胜局 | 胜率 | 平均分差 |
|---|---:|---:|---:|
| 纯无模板 R1 | 528 | 58.93% | +4,893 |
| cold：replay 后接未观察前史的 R1 | 705 | 78.68% | +11,109 |
| **warm-off：当前默认** | **717** | **80.02%** | **+11,123** |
| warm-on：实验 candidate-diff selector | 718 | 80.13% | +11,388 |

warm-on 相对当前默认只有 `+1` 胜：救败 28、致败 27，McNemar `p=1.0`；平均分差
`+265`（双侧 t-test `p=0.051`）。逐对手净胜为 Thomas `-5`、Melon `+4`、Demand `+2`、
Ahmed `+4`、Pipe8 `+2`、Herd `-6`、Salemali `0`。因此 candidate-diff **不进入默认部署**；
`REPLAY_HANDOFF_SELECTOR` 仍只是显式 opt-in 实验入口，缺省/`0` 时旧动作不变。

机制结论：真实 handoff 后缀的三类 oracle 可由 186/224 提到 218/224，说明候选覆盖有价值；
但 147 维状态树在 pilot 中退化为近似全局 sale 开关。使用
`[competitive-default, succession-default]` 的 712 维候选计划差分后，独立 16-seed pilot 从
182/224 提到 192/224（12 次救败、2 次致败，`p=0.0129`），却没有在 64-seed 上稳定转化为
胜率收益。下一步若继续，应校准 DP 的 opponent-flow/value，而不是再扫全局参数或部署该树。

正式产物：`work/formal-pure-vs-cold-public7-64seed-2610100000.json`、
`work/formal-warm-selector-candidate-diff-public7-64seed-2610100000.json`。

> **本轮完整会话记录见 [docs/SESSION_20260920_ZH.md](docs/SESSION_20260920_ZH.md)** ——
> 其中第 6 节列出本轮更正过的错误，动手前请先读。

## 2026-09-20 实测：245 条路线的买地时机，以及接管条件为何是纯状态触发

反解 `agent/route_actions.json.zlib` 里 245 条磁带的 `BUY_LAND` 指令：

- 买地次数分布 `{2次:180, 3次:40, 4次:9, 5次:3, 6次:6, 8次:1, 9次:4, 11次:2}`。
  棋盘开局自带 NW 一个象限、`LAND_PRICES=(1000,2000,4000)`，所以 **2 次买地 = 3 块地**。
  **最小值是 2 —— 245 条路线全部达到 ≥3 块地，没有一条停在 1–2 块地。**
- 买地时点主峰在 **day8–9（步 195–225）**，而 **只有 33/245 条在 step 265 买地**。
  也就是说「第三块地在 step 265 买入」**只适用于我们这 5 个部署 opening**，不是文库的普遍属性。
- 因此 **264 不是可用的检查点**：在 264 切换意味着第 265 步的动作来自新磁带，而只有 33/245 条
  target 在 265 有 `BUY_LAND` —— 对绝大多数 target，这次买地会直接丢掉。检查点取
  `{144,168,216,240}`，全部严格在 265 之前。
- **拼接不破坏正确性**：标签由同一个拼接生成（`play(opening, opponent, seed, checkpoint, target, -1, -1)`），
  所以训练与服务的构造一致，一次糟糕的拼接会被标签如实测成更差。

**接管条件改为纯状态触发。** `agent/replay_deployment.json` 现在只有
`opening=G275, handoff_land=3, handoff_step=288` —— **去掉了 `handoff_floor=264` 这个时间门槛**。
门槛是早期为了绕开"day8/9/10 接管有害"而加的，但那个观测测的是**固定步数**接管（当时盘面只有 2 块地、
扩地未完成）；纯状态触发永远不会在扩地中途触发，所以理由不成立。`handoff_step=288` 保留为**截止步**，
因为个别路线最晚到 step 433（day18）才买够象限，而接管只测到 day14 为好。G275 实测在 **step 266** 触发。

## 2026-09-20 部署变更：opening G001 → G275，接管改为三块地状态触发

**64 seed 不重叠确认（每个 opening 896 局）。** 同 seed、双座、7 强手：

| 配置 | 胜局 | 胜率 |
|---|---|---|
| 纯 R1 | 531/896 | 59.3% |
| **G275 + day12 温接管** | **683/896** | **76.2%** |
| G275 + 三块地状态触发 | 706/896 | 78.8% |
| G001 + day12（旧默认） | 575/896 | 64.2% |

G275 相对纯 R1：配对净胜 `+152` 局，**7/7 个对手的净胜与平均分差同时为正**
（thomas +29、melon +32、demand +24、ahmed +31、pipe8 +21、herd +14、salemali7 +1）。
G275 相对旧默认 G001：逐对手 5 个大胜、herd 持平、salemali7 同样满分，是稳定的支配关系，
不是选择偏差。产物 `experiments/results/r1-vs-g275-handoff12-64seed-v1.json` 与
`opening-handoff12-64seed-confirm-v1.json`。

**接管实现改为状态触发。** `agent/replay_deployment.json` 现在写
`opening=G275, handoff_step=288（截止）, handoff_land=3, handoff_floor=264`：
在 step 264 之后、一旦自有农场解锁到 3 块地就交给 R1，最晚不超过 step 288。
实测 G275 下第三块地在 step 266 被买入，触发即落在该帧。

状态触发相对固定 day12 是 `+2.6pp`（706 vs 683），**McNemar 精确 p = 0.12，未达显著**，
配对分差 −261（t = −0.80, p = 0.42）。采用它的理由是**鲁棒性而非这点增益**：
day8/9/10 接管被证明显著有害（t = −4.0 ~ −5.6），断点正是第三块地那次购买；
换 opening 会改变三块地的时点，写死 day12 就不再对准它，而状态触发自动跟随。

**verify_project.py 已更新**：新增对 64-seed 头条结果的断言与部署配置断言；
原先对 4-seed `36/56`/`30/56` 的断言保留，但降级为历史产物完整性检查。

## 2026-09-20 更正：温接管优于纯 R1；接管时机存在 day11 断点

**作废结论。** 本文下面“可信结果”里写的「G001 浅树路线 day12 温接管 R1 `30/56`，低于纯 R1 `36/56`，
因此温接管不优」**是 4 seeds 的噪声，已作废**。16 seeds、双座、7 强手（224 局/臂，
seeds 2609500000..2609500015）复测：

| 臂 | 胜局 | 胜率 |
|---|---|---|
| 纯 R1 | `118/224` | 52.7% |
| G001 + day12 温接管 | `141/224` | **62.9%** |

配对净胜 `+23` 局；逐对手（baseline→candidate）：thomas 9→9、melon 11→12、demand 12→18、
ahmed 14→21、pipe8 14→19、herd 26→30、salemali7 32→32。产物：
`experiments/results/r1-vs-hybrid-16seed-v1.json`。

**接管时机的断点。** 同一批 seeds 扫接管日（`cleanscan-r1-handoff-days-public7-16seed-v1.json`），
配对比较（基准 = day0 纯 R1，224 对）：

| 接管日 | 胜局 | 胜率 | 配对 Δmargin | t |
|---|---|---|---|---|
| 0（纯 R1） | 118/224 | 52.7% | 0 | — |
| 8 | 98/224 | 43.8% | −3296 | −4.2 |
| 9 | 111/224 | 49.6% | −3156 | −4.0 |
| 10 | 85/224 | **37.9%** | −4428 | −5.6 |
| 11 | 140/224 | 62.5% | **+1624** | +2.0 |
| 12 | 141/224 | 62.9% | +622 | +0.8 |
| 13 | 144/224 | **64.3%** | −347 | −0.5 |
| 14 | 138/224 | 61.6% | −358 | −0.5 |

day10 → day11 是断崖（85 → 140 胜），而 day11 小时 1（约 step265）正是 replay 路线买第三块地的
时点。**结论：day8–10 接管显著有害（早于第三块地），day11–14 才优于纯 R1。**
配对 margin 上只有 day11 (+1624, t=2.0) 明显为正，day12–14 接近零：赢面来自胜局增加，
而非分差。1v1 排名看胜率，故 day11–14 平台期都可用，day13 胜率最高但三者在噪声内。

**仍需注意。** thomas（9/32 = 28%）与 melon（12/32 = 38%）是剩余主要缺口，且接管对 thomas
净胜为 0 —— 这两个对手的改进不来自 replay 前缀。salemali7 对所有配置都是 32/32，
不提供区分度。

## 2026-09-20 记录：observation 视图保真度（含一次错误判断的撤回）

**机制。** 框架把 observation 交给 agent 走的是 `Environment.__agent_runner` →
`__get_shared_state(position)`：它把 state schema 里标记为 `shared` 的字段从 `state[0]`
拷进该 agent 的视图。kaggriculture 的 shared 字段是
`observation.step / farms / market / town / day / hour`。**所以两个 seat 在真实运行
（`env.run` / Kaggle）下都能拿到 `step`。**

**坑点。** 直接读 `state[p].observation` 得到的是**存储视图**：`__get_state` 对
`position > 0` 删掉了 shared 字段（core.py:602）。本工程的 `run_strong_ab.py` /
`run_handoff_scan.py` / `run_route_opening_scan.py` 正是这么读的，于是 p1 丢 `step`，
三个 harness 各自加了一行 `if obs["step"] is None: obs["step"] = day*24+hour`。
该补丁与 `__get_shared_state` 的逐帧取值一致，**因此历史全部 A/B 结果保真，无需重跑**。

**曾据此误判（已撤回）。** 用 `env.step()` 手动驱动（而非 `env.run`）复现时 p1 确实没有
`step`，一度被记成「提交物在 seat 1 第 1 帧就崩」的严重 bug。该结论**错误**：
`env.run([thomas_2945, thomas_2945])` 两 seat 均正常 DONE，seat 1 拿到的 step 为
0,1,2,3… 与 seat 0 一致；且强公开对手本身硬索引 `observation["step"]` 也能跑通。
**`agent/main.py` / `policy/r1/agent.py` 在 Kaggle 上从未因此失效。**

**保留的改动。** 为让代码在两种调用约定下都正确，`observed_step` / `_step_of` /
`observation_step` 统一从公开时钟 `day*24+hour` 解析 step（时钟缺失才回退 `step`）。
已验证行为保持：同 seed、双座、4 seeds，纯 R1 `36/56`、温接管 `30/56`，
逐对手配对分差与改动前逐位相同。`agent/teammate_base.py` 哈希已同步更新。

**值得复用的产物。** `experiments/test_submission_contract.py` 可按任一视图驱动 agent，
作为 harness 保真度回归保留。

## 2026-09-20 实测：day12 前缀盘面明显强于纯 R1 开局

对 thomas_2945、4 seeds、双座，各自跑到 step 288（day12）后记录自身盘面：

| | 现金 | 地 | 动物 | 作物 |
|---|---|---|---|---|
| replay 前缀（hybrid） | **13836** | 3 | 15.8 | 54.0 |
| 纯 R1 自己开局 | 2998 | 3 | 17.5 | 55.0 |

**前缀现金是纯 R1 的 3–8 倍**（逐局：12242/3085、18571/2250、12475/4414）。所以
「前缀比 R1 开局弱」**不成立**——项目前提没有被证伪。

同时必须说明：全程用路线树（树开）打强公开对手只有 `G210 99/224`、`G275 95/224`、
`G411 84/224`、`G379 79/224`、`G001 78/224`，明显低于纯 R1 的 `36/56`（4 seeds, 64.3%）。
但这**不能**用来判断前缀强弱：树的检查点只有 144/168/216（day6/7/9），且最多切换一次，
**day9–day29 这 20 天是在冻结执行一条他人录像的动作序列**，无任何适应。全程成绩基本由这段
尾盘决定，几乎不含前缀质量信息。

用 R1 自身价值函数 (`plan()` 的 `predicted`) 给两个 day12 盘面打分，只在 4/8 局偏好前缀，
但该量包含对手预测与行情、只在同局内可比，**不适合跨局当裁判**。判定前缀优劣仍需
held-out 的多 seed A/B，而不是跨局比较 `predicted`。

## 用户确定的方法

目标不是给现有动态规划器塞一个“骨架”，而是复用成熟方法：下载近期强者 replay，按经营路线聚类；路线之间大规模互打；在公开真实状态上训练浅层决策树切换 replay 路线；前期走 replay 路线并保留成熟的杂草、交易、现金和喂养修补；中期模块化转交真正的动态策略。公开验证只看七个强对手，使用大量 seed、同 seed 双座和 192 核游戏级并行。不使用 Nash 开局混合，先固定主流且对强脚本较好的 opening。

## 已完成并已收拢

- 609 份近期完整 replay，60 个排行榜目标；提取 752 个参赛方路线。
- 按生产、布局、日程和土地/劳工显式距离聚成 411 家族，保留 245 代表。
- 245 路线、64 seeds、双座互打 3,825,920 局。
- 五个 opening、三个前期检查点、26 targets、245 对手、128 seeds、双座切换搜索 24,460,800 局，940,800 个 147 维状态。
- 浅树和深树产物均保存；当前部署仍用浅树。rollout 可直接复用。
- 成熟 replay 执行器已确认不是裸动作磁带，含杂草修补、市场/现金保护、喂养价值保护等。
- 修复外部观察对 `['NOOP']` 的崩溃：官方环境把未知操作静默当 no-op，桥现在将未知/空动作规范成 PASS，未知 item 为 `-1`。
- 精确定位真正纯动态为 `jointafs-equivalent-speed-v1` 的 JointAFS R1；它没有 opening template。此前使用 `template-decoder-v1` 原生模板分支做“纯动态”是错误对照。
- 给原版 R1 增加了最薄外部公开状态桥，未把 opening-template 代码带入 R1。
- 新工程官方烟测完整运行 719 步并跨 day12 接管。示例 Thomas seed 2609400000 seat0 为 `109546:107155`，无异常。
- 高速仿真器从新工程独立加载成功，识别 245 路线并完成原生短局。

## 可信结果

冻结 R1 在原纠正后的 4-seed、双座、7 强手基线是 `36/56`。本工程重跑得到同样 `36/56`，平均分差 `+5191`。

本工程当前 G001 浅树路线、day12 温接管 R1：`30/56` **[已作废：见上节 16-seed 更正] **，平均分差 `+3302`。逐对手胜局为 Thomas 4/8、Melon 2/8、Demand 2/8、Ahmed 4/8、Pipe8 2/8、Herd 8/8、Salemali7 8/8。配对平均分差相对纯 R1 为：Thomas -2487、Melon -1817、Demand -2676、Ahmed -1298、Pipe -2338、Herd +1177、Salemali7 -3789。

旧冷接管（R1 完全不看前史，day12 才新建）也是 `30/56`，旧批次平均分差约 `+3742`。温接管没有改善胜局，当前甚至没有稳定改善分差，因此状态桥仍有问题或 replay 前缀本身损害 R1 后续。

旧路线树跑满全局的 16-seed结果：G001 78/224，G210 99/224，G275 95/224，G379 79/224，G411 84/224。G210 胜局最多，G275 平均分差最好，但它们都不是强基线；后续应把 day11–14 的 R1 接管重新对这几个 opening 做正式对照。

曾得到 day11–14 很强的接管扫描（例如 day13 144/224），但该扫描接的是**带 opening-template 的错误动态分支**，只可用于发现扩地边界和 bug，不能证明 R1 的最佳接管日。不要在后续文档引用为最终 R1 成绩。

## 当前主要困难与可能 bug

1. **温接管状态不完整。** 当前桥只维护公开交易、作物时钟和联合项目事件。R1 的 `Controller` 内还有 book、既有计划、项目承诺、模型派生量及日界初始化。它们不是全部能从动作历史增量恢复。接管时直接把 `previous_step` 设为当前前一帧后，R1 的 `choose()` 可能在尚未以当前真实盘面完整初始化的 `live` 上生成 proposal。
2. **冷启动与温启动都只有 30/56。** 这说明“桥缺状态”不是唯一解释；G001 replay 前缀可能比 R1 自己开局更差，或者浅树目标适合路线对打但不适合交给 R1。
3. **路线树训练域和最终对手域不同。** 树来自 245 replay 路线互打，不是七个强公开脚本；迁移是否成立必须以强对手 held-out A/B 判断。
4. **一次切换限制。** 在线 `SearchRouteController` 当前只记录一次切换。成熟脚本已有序列搜索入口，可研究两三次切换，但必须复用已有 rollout 状态/前缀，不应从头重复所有比赛。
5. **日级 replay 提取不完整。** 609 中 432 局被旧 receipt 工具因空/异常动作隔离。宏观路线资产仍已生成，但完整质量审计应统一 NOOP/PASS 语义后重建。
6. **样本量。** 4 seeds 只用于快速判错，不足以准入。任何候选先 16 seeds 筛选，再用不重叠 64+ seeds 确认；报告逐 bot 双座。
7. **第三块地边界。** G001→G009 路线在约 step265（day11 hour1）买第三块地；旧错误分支扫描在 day11–13 较好，提示接管应围绕扩地/大额购买完成状态建模，而不是硬编码固定日。

## 下一步优先顺序

1. 做同一真实 handoff 观察的三分对照：冷 R1、当前公开账本温 R1、从真实状态显式重建 `Controller` 后的 R1。逐字段比较第一次 `choose/plan` 前后的 book、plans、core.day、planned_land、joint 和销售记忆，找到首个分歧，而不是继续扫日期。
2. 冻结接管实现后，对 G001/G210/G275 在 day11/12/13/14 跑 16 seeds；只晋级在多个强 bot 上改善而非靠 Salemali 拉高均值的组合。
3. 将“动态接管”作为浅树 target 一起训练：标签比较继续 replay 路线与接入 R1 的真实 suffix，而不是用 replay→replay 胜负代理 R1 接管价值。已有前缀状态和大量 route rollout 可复用，新增部分仅是候选 suffix。
4. 再研究两三次 replay 切换。先用 `search_native_route_tree_sequences.py` 证明序列增益，并对状态分布做 held-out；不要直接放宽在线标志。
5. 修复 432 个日级隔离 replay 的动作正规化，重跑质量标签，检查代表路线是否因执行硬失败应被替换。

## 不要再次混淆

- `policy/r1/` 才是无 opening-template 的动态基线。
- `experiments/results` 中 `replay609-G001-shallow-handoff-*` 的大部分日期扫描来自错误模板动态分支；文件保留用于审计，结论已降级。
- `agent/main.py` 默认固定 G001 且 day12 接 R1；这是当前可复现实验配置，不是最终最优。
- `trees-deep-d8-16.json` 是后台训练产物，不是已验证部署。
- 旧“8 seeds 56%”重复了默认 seed，已判无效；可信纯 R1 快速基线是纠正后的 4 seeds 36/56。
